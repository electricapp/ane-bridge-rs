//! Emitter for Apple MIL programs targeting the Neural Engine.
//!
//! MIL text is the input contract of `ane-bridge`: there is no converter, so a
//! DSP graph has to be written out directly. This module builds one
//! statement-at-a-time in SSA form and pairs it with a [`BlobWriter`] holding
//! the weight payloads.
//!
//! Constraints this emitter is built around, all of them learned from Apple's
//! shipped models and from `_ANECompiler`'s behaviour:
//!
//! * **fp16 compute only.** An fp32 arithmetic graph is rejected with
//!   `CompilationFailure`. fp32 is fine at the function boundary as long as it
//!   is `cast` to fp16 immediately.
//! * **Static shapes only.** Any `?` dimension yields `InvalidMILProgram`.
//! * **Rank 4 `[N, C, H, W]`.** The framework canonicalizes the I/O schema to
//!   NCHW regardless of the rank declared here, so a 1-D signal is idiomatically
//!   carried as `[1, C, 1, W]`.
//! * **No transcendentals.** There is no `sin`/`cos`/`exp`/`sqrt`. Everything is
//!   composed from `mul`/`add`/`sub`/`relu`/`real_div`.
//! * **The graph must be ANE-eligible.** A purely elementwise program can be
//!   left on the CPU and refused at load, so anchor a graph with a `conv`,
//!   `linear` or `matmul`.
//! * **`buildInfo` must stay on one line** — the framework's parser rejects a
//!   line break inside it.
//! * The compile cache is keyed on a content hash of the MIL bytes, so each
//!   program embeds a nonce to avoid colliding with a structurally identical one.
//!
//! Because [`Val`] carries a static shape, the emitter can price a program
//! before it is dispatched: see [`Graph::traffic`]. Elementwise ops do not fuse
//! here, so a graph's runtime is essentially the bytes its intermediates move,
//! which makes it possible to compare two formulations without building
//! either.

use crate::weights::{BlobOffset, BlobWriter};
use core::fmt::Write as _;
use std::path::Path;

/// Element type of a MIL tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    /// IEEE binary16 — the only type the ANE computes in.
    Fp16,
    /// IEEE binary32 — boundary use only.
    Fp32,
    /// Signed 32-bit integer — shapes, axes and other attributes.
    Int32,
}

impl Dtype {
    /// The spelling used in MIL text.
    #[must_use]
    pub const fn mil(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Fp32 => "fp32",
            Self::Int32 => "int32",
        }
    }
}

/// A value in the program: an SSA name plus its static type.
#[derive(Clone, Debug)]
pub struct Val {
    /// SSA name this value is bound to.
    name: String,
    /// Element type.
    dtype: Dtype,
    /// Static shape, in MIL's declared order.
    shape: Vec<i64>,
}

impl Val {
    /// The SSA name this value is bound to.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The element type.
    #[must_use]
    pub const fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// The static shape.
    #[must_use]
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// Total element count.
    #[must_use]
    pub fn numel(&self) -> i64 {
        self.shape.iter().product()
    }

    /// The MIL type annotation, e.g. `tensor<fp16, [1, 2, 1, 16]>`.
    #[must_use]
    fn ty(&self) -> String {
        format!("tensor<{}, [{}]>", self.dtype.mil(), join(&self.shape))
    }
}

/// Widest convolution kernel the framework compiles. `KW >= 16` fails.
pub const MAX_CONV_KERNEL: i64 = 15;

/// Required alignment, and minimum, of a tensor's width at the program
/// boundary.
///
/// Violating it is silent: rows are laid out at the next multiple of 32 while
/// the schema still reports the requested width, so a `[1, 100, 1, 4082]`
/// output comes back with a row stride of 4096 and every channel reads from
/// where the next one begins. [`Graph::input`] and [`Graph::finish`] reject it.
pub const WIDTH_ALIGN: i64 = 32;

/// Explicit padding for a convolution, in `[top, bottom, left, right]` order.
pub type Pad = [i64; 4];

/// Geometry of a convolution.
#[derive(Clone, Copy, Debug)]
pub struct ConvOpts {
    /// `[stride_h, stride_w]`.
    pub strides: [i64; 2],
    /// `[dilation_h, dilation_w]`.
    pub dilations: [i64; 2],
    /// `[top, bottom, left, right]`.
    pub pad: Pad,
    /// Number of convolution groups. `groups == channels` is depthwise.
    pub groups: i64,
}

impl Default for ConvOpts {
    fn default() -> Self {
        Self {
            strides: [1, 1],
            dilations: [1, 1],
            pad: [0, 0, 0, 0],
            groups: 1,
        }
    }
}

impl ConvOpts {
    /// A 1-D convolution along `W` with the given left/right padding.
    #[must_use]
    pub const fn conv1d(pad_left: i64, pad_right: i64) -> Self {
        Self {
            strides: [1, 1],
            dilations: [1, 1],
            pad: [0, 0, pad_left, pad_right],
            groups: 1,
        }
    }

    /// Decimate along `W` by `factor`, padding on the left for a causal FIR of
    /// `taps` length.
    ///
    /// `groups` must be more than 1 when `factor` is: see [`Graph::conv`].
    #[must_use]
    pub const fn decimating_fir(taps: i64, factor: i64, groups: i64) -> Self {
        Self {
            strides: [1, factor],
            dilations: [1, 1],
            pad: [0, 0, taps.saturating_sub(1), 0],
            groups,
        }
    }
}

/// One emitted op: what it produced, and what it read to do so.
///
/// Kept alongside the statement text purely so [`Graph::traffic`] can price the
/// program without dispatching it.
struct Op {
    /// SSA name of the result.
    result: String,
    /// SSA names of the tensor operands, one entry per use.
    reads: Vec<String>,
}

/// Bytes a program moves per dispatch.
///
/// The ANE does not fuse elementwise ops: each writes its whole result and the
/// next reads it back, and runtime tracks those bytes rather than the
/// arithmetic between them.
///
/// **These are not necessarily main-memory bytes.** Activations stay in on-chip
/// SRAM while they fit and stream to DRAM once they do not, which is a cliff
/// rather than a slope — an elementwise op over a 0.82 MB tensor moves its
/// bytes at ~200 GB/s, the same op over a 6.55 MB tensor at ~106 GB/s. A byte
/// count alone therefore does not give a time; divide it by the bandwidth that
/// applies at the tensor size in question, and add the per-dispatch floor
/// (~60 us on an M5) separately. A fit across graphs at ~1.6 MB gives a tighter
/// `50 us + total / 136 GB/s`. Both over-predict badly for a graph dense enough
/// to be compute-bound — a `1x1` convolution over many channels is `O(C^2)`
/// work for `O(C)` output.
///
/// [`Traffic::total`] against the compulsory traffic — input read once, output
/// written once — gives the materialization factor, a property of how the
/// algorithm is expressed rather than of the code.
///
/// This over-counts one case: a `mul` or `add` against a scalar constant folds
/// into the preceding convolution's weights and bias and is free.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Traffic {
    /// Bytes of op results written.
    pub written: usize,
    /// Bytes read by ops, counted once per use. Includes weights, which are
    /// re-read on every dispatch.
    pub read: usize,
    /// Ops emitted, not counting attribute and weight constants.
    pub ops: usize,
}

impl Traffic {
    /// Bytes crossing the memory interface per dispatch.
    #[must_use]
    pub const fn total(&self) -> usize {
        self.written.saturating_add(self.read)
    }
}

/// A MIL program under construction.
pub struct Graph {
    /// Nonce mixed into `buildInfo` so the compile cache key is unique.
    nonce: String,
    /// Declared function parameters, in order.
    params: Vec<Val>,
    /// Emitted statements, in order.
    stmts: Vec<String>,
    /// Structure behind `stmts`, for [`Graph::traffic`].
    ops: Vec<Op>,
    /// Byte size of every named tensor: parameters, weights and op results.
    sizes: std::collections::HashMap<String, usize>,
    /// Weight payloads referenced by `BLOBFILE` consts.
    blob: BlobWriter,
    /// Monotonic counter backing SSA name generation.
    next_id: usize,
    /// Set by [`Graph::raw_stmt`], whose reads are opaque to the dead-op check.
    has_raw_stmts: bool,
}

impl core::fmt::Debug for Graph {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Graph")
            .field("nonce", &self.nonce)
            .field("params", &self.params.len())
            .field("stmts", &self.stmts.len())
            .field("blobs", &self.blob.len())
            .finish_non_exhaustive()
    }
}

/// Join a shape into the comma-separated form MIL uses.
fn join(dims: &[i64]) -> String {
    dims.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Escape a string for a MIL `string("...")` literal.
///
/// The emitter mints most of its own strings, but the nonce and parameter
/// names come from the caller, and a stray quote there would close the literal
/// early and produce a program that fails to parse a long way from its cause.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

/// Whether `s` is usable as a bare MIL identifier.
///
/// Unlike a string literal, an identifier cannot be escaped — it appears
/// unquoted on the left of an assignment and in every operand list — so a name
/// that is not one has to be rejected outright.
fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The four NCHW dimensions of `v`.
///
/// Returning an array rather than reading `shape[i]` leaves the rank check as
/// the only place rank can go wrong.
///
/// # Panics
/// Panics unless `v` is rank 4.
fn nchw(what: &str, v: &Val) -> [i64; 4] {
    let mut dims = [1i64; 4];
    assert_eq!(
        v.shape.len(),
        dims.len(),
        "{what} must be rank 4 NCHW, got {:?}",
        v.shape
    );
    for (slot, &d) in dims.iter_mut().zip(&v.shape) {
        *slot = d;
    }
    dims
}

/// Panic unless a convolution is one the ANE will run, and run at speed.
///
/// Returns the dimensions it checked, as `(input, weight)`, so the caller
/// computes the output geometry from values that have already been validated.
fn check_conv(x: &Val, weight: &Val, opts: ConvOpts) -> ([i64; 4], [i64; 4]) {
    let x_dims = nchw("conv: input", x);
    let w_dims = nchw("conv: weight", weight);
    let [_, c_in, _, _] = x_dims;
    let [c_out, w_in, _, kw] = w_dims;
    assert_eq!(
        x.dtype, weight.dtype,
        "conv: input is {:?} but weight is {:?}",
        x.dtype, weight.dtype
    );
    assert!(
        x.dtype != Dtype::Fp32,
        "conv: the ANE computes in fp16 only; cast at the boundary"
    );
    assert!(
        opts.groups > 0,
        "conv: groups must be positive, got {}",
        opts.groups
    );
    assert!(
        opts.strides.iter().all(|&s| s > 0) && opts.dilations.iter().all(|&d| d > 0),
        "conv: strides {:?} and dilations {:?} must be positive",
        opts.strides,
        opts.dilations
    );
    assert!(
        kw <= MAX_CONV_KERNEL,
        "conv: the ANE rejects kernels wider than {MAX_CONV_KERNEL}, got {kw}"
    );
    // Measured at ~45 GB/s against ~110 for everything else. Slicing to the
    // stride and convolving unstrided produces the same output.
    let [_, stride_w] = opts.strides;
    assert!(
        stride_w == 1 || opts.groups > 1,
        "conv: a strided dense convolution runs at a fraction of the \
         machine's bandwidth; slice to the stride and convolve unstrided, \
         or group the convolution"
    );

    // The grouping rule is where a depthwise bank goes wrong quietly: a
    // weight shaped [C, C, 1, K] with groups = C compiles as something
    // else entirely rather than as C independent FIRs.
    assert_eq!(
        c_in.checked_rem(opts.groups),
        Some(0),
        "conv: {c_in} input channels do not divide into {} groups",
        opts.groups
    );
    assert_eq!(
        c_out.checked_rem(opts.groups),
        Some(0),
        "conv: {c_out} output channels do not divide into {} groups",
        opts.groups
    );
    let per_group = c_in.checked_div(opts.groups);
    assert_eq!(
        Some(w_in),
        per_group,
        "conv: weight expects {w_in} input channels per group, but {c_in} channels \
         in {} groups gives {per_group:?}",
        opts.groups
    );

    (x_dims, w_dims)
}

/// A matmul operand's batch dimensions and its trailing `[rows, cols]`.
///
/// # Panics
/// Panics unless `v` is rank 2 or higher.
fn mat_dims<'val>(what: &str, v: &'val Val) -> (&'val [i64], [i64; 2]) {
    let mut last = [1i64; 2];
    assert!(
        v.shape.len() >= last.len(),
        "{what}: operands must be rank >= 2, got {:?}",
        v.shape
    );
    let (batch, tail) = v.shape.split_at(v.shape.len().saturating_sub(last.len()));
    for (slot, &d) in last.iter_mut().zip(tail) {
        *slot = d;
    }
    (batch, last)
}

/// Output extent of a convolution along one axis.
///
/// The kernel's dilated span is subtracted from the padded input, and what is
/// left is stepped over by `stride`. A non-positive `stride` yields 1, which
/// [`Graph::conv`] then rejects as a degenerate output.
fn conv_extent(extent: i64, pad_lo: i64, pad_hi: i64, dilation: i64, kernel: i64, stride: i64) -> i64 {
    let span = dilation
        .saturating_mul(kernel.saturating_sub(1))
        .saturating_add(1);
    extent
        .saturating_add(pad_lo)
        .saturating_add(pad_hi)
        .saturating_sub(span)
        .checked_div(stride)
        .unwrap_or(0)
        .saturating_add(1)
}

/// Panic unless `shape`'s last dimension satisfies [`WIDTH_ALIGN`].
fn assert_width_aligned(what: &str, name: &str, shape: &[i64]) {
    let w = shape.last().copied().unwrap_or(0);
    assert!(
        w >= WIDTH_ALIGN && w.checked_rem(WIDTH_ALIGN) == Some(0),
        "{what} {name:?}: width {w} must be a positive multiple of \
         {WIDTH_ALIGN}, or the rows come back mis-strided"
    );
}

impl Graph {
    /// Start an empty program.
    ///
    /// `nonce` must be unique per distinct program: `aned` keys its compile
    /// cache on a content hash of the MIL bytes, so two structurally identical
    /// programs that mean different things would otherwise collide and the
    /// second would silently run the first's lowering.
    ///
    /// # Panics
    /// Panics if `nonce` is empty, which would defeat the point.
    #[must_use]
    pub fn new<N: Into<String>>(nonce: N) -> Self {
        let text: String = nonce.into();
        assert!(
            !text.is_empty(),
            "Graph::new: the cache-busting nonce cannot be empty"
        );
        Self {
            nonce: text,
            params: Vec::new(),
            stmts: Vec::new(),
            ops: Vec::new(),
            sizes: std::collections::HashMap::new(),
            blob: BlobWriter::new(),
            next_id: 0,
            has_raw_stmts: false,
        }
    }

    /// Mint a fresh SSA name with the given prefix.
    fn fresh(&mut self, prefix: &str) -> String {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        format!("{prefix}_{id}")
    }

    /// Record a tensor's size so traffic through it can be priced later.
    fn size_of(&mut self, v: &Val) {
        let bytes: usize = match v.dtype {
            Dtype::Fp16 => 2,
            Dtype::Fp32 | Dtype::Int32 => 4,
        };
        // A shape too large to count in bytes is priced at the maximum rather
        // than wrapping, so an absurd graph reads as absurd instead of cheap.
        let elems = usize::try_from(v.numel().max(1)).unwrap_or(usize::MAX);
        self.sizes
            .insert(v.name.clone(), elems.saturating_mul(bytes));
    }

    /// Record an emitted op and the tensors it reads.
    fn note(&mut self, out: &Val, reads: &[&Val]) {
        self.size_of(out);
        self.ops.push(Op {
            result: out.name.clone(),
            reads: reads.iter().map(|v| v.name.clone()).collect(),
        });
    }

    /// Bytes this program moves through memory in one dispatch.
    ///
    /// Elementwise ops do not fuse: each writes its whole result and the next
    /// reads it back. Whether those bytes reach DRAM or stay in on-chip SRAM
    /// depends on tensor size; see [`Traffic`]. So the cost of a graph is essentially the sum of
    /// its intermediate tensors, and this counts them — every op's output
    /// written once, every operand read once per use, weights included, since
    /// those are re-read on every dispatch too.
    ///
    /// Attribute constants (strides, axes, pad vectors) are not counted; they
    /// are a handful of bytes folded into the lowered program.
    ///
    /// Measured against a dispatch's wall time this gives achieved bandwidth,
    /// which is the honest denominator for "how close to the machine are we" —
    /// as opposed to comparing against the compulsory traffic, which measures
    /// the algorithm rather than the implementation.
    #[must_use]
    pub fn traffic(&self) -> Traffic {
        let mut t = Traffic {
            ops: self.ops.len(),
            ..Traffic::default()
        };
        for op in &self.ops {
            t.written = t
                .written
                .saturating_add(self.sizes.get(&op.result).copied().unwrap_or(0));
            for r in &op.reads {
                t.read = t.read.saturating_add(self.sizes.get(r).copied().unwrap_or(0));
            }
        }
        t
    }

    /// Declare a function parameter.
    ///
    /// The name reaches the framework as the port name, and ports are resolved
    /// by name rather than by position, so it has to be both a legal MIL
    /// identifier and unique within the program.
    ///
    /// # Panics
    /// Panics if `name` is not a bare identifier, collides with an existing
    /// parameter, or if `shape` has a non-positive dimension. A dynamic
    /// dimension is not expressible: the framework rejects the program.
    pub fn input(&mut self, name: &str, dtype: Dtype, shape: &[i64]) -> Val {
        assert!(is_ident(name), "input: {name:?} is not a MIL identifier");
        assert!(
            !self.params.iter().any(|p| p.name == name),
            "input: parameter {name:?} is already declared"
        );
        assert!(
            shape.iter().all(|&d| d > 0),
            "input {name:?}: every dimension must be static and positive, got {shape:?}"
        );
        assert_width_aligned("input", name, shape);
        let v = Val {
            name: name.to_owned(),
            dtype,
            shape: shape.to_vec(),
        };
        self.size_of(&v);
        self.params.push(v.clone());
        v
    }

    /// Emit a raw statement. Escape hatch for ops without a helper.
    ///
    /// Its operands are invisible to [`Graph::traffic`] and disable the
    /// dead-op check in [`Graph::finish`].
    pub fn raw_stmt<S: Into<String>>(&mut self, stmt: S) {
        self.stmts.push(stmt.into());
        self.has_raw_stmts = true;
    }

    /// Panic if any emitted op's result is unreachable from `outputs`.
    ///
    /// The framework compiles dead subgraphs rather than pruning them, and
    /// fails outright on some of them, so an unused op is a bug either way.
    fn assert_no_dead_ops(&self, outputs: &[&Val]) {
        if self.has_raw_stmts {
            return;
        }
        let mut live: std::collections::HashSet<&str> =
            outputs.iter().map(|v| v.name.as_str()).collect();
        for op in self.ops.iter().rev() {
            if live.contains(op.result.as_str()) {
                live.extend(op.reads.iter().map(String::as_str));
            }
        }
        for op in &self.ops {
            assert!(
                live.contains(op.result.as_str()),
                "finish: {:?} is not reachable from any output; the framework \
                 compiles dead subgraphs rather than pruning them",
                op.result
            );
        }
    }

    // ---------------------------------------------------------------- consts

    /// A scalar `int32` attribute constant.
    fn const_i32(&mut self, value: i64) -> String {
        let n = self.fresh("ci");
        self.stmts.push(format!(
            "int32 {n} = const()[name = string(\"{n}\"), val = int32({value})];"
        ));
        n
    }

    /// A scalar `bool` attribute constant.
    fn const_bool(&mut self, value: bool) -> String {
        let n = self.fresh("cb");
        self.stmts.push(format!(
            "bool {n} = const()[name = string(\"{n}\"), val = bool({value})];"
        ));
        n
    }

    /// A scalar `string` attribute constant.
    fn const_str(&mut self, value: &str) -> String {
        let n = self.fresh("cs");
        let escaped = escape(value);
        self.stmts.push(format!(
            "string {n} = const()[name = string(\"{n}\"), val = string(\"{escaped}\")];"
        ));
        n
    }

    /// A small inline `int32` vector constant (strides, pads, axes).
    fn const_i32_vec(&mut self, values: &[i64]) -> String {
        let n = self.fresh("cv");
        let len = values.len();
        self.stmts.push(format!(
            "tensor<int32, [{len}]> {n} = const()[name = string(\"{n}\"), val = tensor<int32, [{len}]>([{}])];",
            join(values)
        ));
        n
    }

    /// A scalar fp16 constant, usable as a broadcast operand.
    pub fn scalar_fp16(&mut self, value: f32) -> Val {
        let n = self.fresh("k");
        self.stmts.push(format!(
            "fp16 {n} = const()[name = string(\"{n}\"), val = fp16({value:?})];"
        ));
        let v = Val {
            name: n,
            dtype: Dtype::Fp16,
            shape: vec![],
        };
        self.size_of(&v);
        v
    }

    /// An fp16 weight tensor stored in the weights blob.
    ///
    /// `data` is in C order and must have exactly `shape.iter().product()`
    /// elements.
    ///
    /// # Panics
    /// Panics if `data`'s length does not match `shape`.
    pub fn weight_fp16(&mut self, shape: &[i64], data: &[f32]) -> Val {
        let expect = usize::try_from(shape.iter().product::<i64>()).unwrap_or(usize::MAX);
        assert_eq!(
            data.len(),
            expect,
            "weight_fp16: shape {shape:?} needs {expect} elements, got {}",
            data.len()
        );
        let BlobOffset(off) = self.blob.write_fp16(data);
        let n = self.fresh("w");
        let ty = format!("tensor<fp16, [{}]>", join(shape));
        self.stmts.push(format!(
            "{ty} {n} = const()[name = string(\"{n}\"), val = {ty}(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({off})))];"
        ));
        let v = Val {
            name: n,
            dtype: Dtype::Fp16,
            shape: shape.to_vec(),
        };
        self.size_of(&v);
        v
    }

    // ------------------------------------------------------------------ ops

    /// `cast(x, dtype)` — the only legal way across the fp32/fp16 boundary.
    pub fn cast(&mut self, x: &Val, dtype: Dtype) -> Val {
        let d = self.const_str(dtype.mil());
        let out = Val {
            name: self.fresh("cast"),
            dtype,
            shape: x.shape.clone(),
        };
        self.stmts.push(format!(
            "{} {} = cast(dtype = {d}, x = {})[name = string(\"{}\")];",
            out.ty(),
            out.name,
            x.name,
            out.name
        ));
        self.note(&out, &[x]);
        out
    }

    /// Elementwise binary op with numpy-style broadcasting.
    ///
    /// MIL has no implicit numeric promotion: mixing an fp32 operand with an
    /// fp16 one is a type error at parse time, not a silent widening, so the
    /// mismatch is caught here instead of as an `InvalidMILProgram` with no
    /// indication of which statement caused it.
    fn binary(&mut self, op: &str, x: &Val, y: &Val) -> Val {
        assert_eq!(
            x.dtype, y.dtype,
            "{op}: operand types {:?} and {:?} disagree; insert a cast",
            x.dtype, y.dtype
        );
        assert!(
            x.dtype != Dtype::Fp32,
            "{op}: the ANE computes in fp16 only; cast at the boundary"
        );
        let shape = broadcast(op, &x.shape, &y.shape);
        let out = Val {
            name: self.fresh(op),
            dtype: x.dtype,
            shape,
        };
        self.stmts.push(format!(
            "{} {} = {op}(x = {}, y = {})[name = string(\"{}\")];",
            out.ty(),
            out.name,
            x.name,
            y.name,
            out.name
        ));
        self.note(&out, &[x, y]);
        out
    }

    /// `add(x, y)`.
    pub fn add(&mut self, x: &Val, y: &Val) -> Val {
        self.binary("add", x, y)
    }

    /// `sub(x, y)`.
    pub fn sub(&mut self, x: &Val, y: &Val) -> Val {
        self.binary("sub", x, y)
    }

    /// `mul(x, y)`.
    pub fn mul(&mut self, x: &Val, y: &Val) -> Val {
        self.binary("mul", x, y)
    }

    /// `real_div(x, y)` — the only division available.
    pub fn real_div(&mut self, x: &Val, y: &Val) -> Val {
        self.binary("real_div", x, y)
    }

    /// Elementwise unary op preserving shape.
    fn unary(&mut self, op: &str, x: &Val) -> Val {
        let out = Val {
            name: self.fresh(op),
            dtype: x.dtype,
            shape: x.shape.clone(),
        };
        self.stmts.push(format!(
            "{} {} = {op}(x = {})[name = string(\"{}\")];",
            out.ty(),
            out.name,
            x.name,
            out.name
        ));
        self.note(&out, &[x]);
        out
    }

    /// `relu(x)`.
    pub fn relu(&mut self, x: &Val) -> Val {
        self.unary("relu", x)
    }

    /// `sigmoid(x)`.
    pub fn sigmoid(&mut self, x: &Val) -> Val {
        self.unary("sigmoid", x)
    }

    /// `tanh(x)`.
    pub fn tanh(&mut self, x: &Val) -> Val {
        self.unary("tanh", x)
    }

    /// `maximum(x, y)`, composed as `y + relu(x - y)`.
    ///
    /// MIL has a `maximum` op but it does not appear in any ANE-verified model,
    /// whereas `relu`, `add` and `sub` all do. Building it this way keeps the
    /// graph inside the proven op set.
    pub fn max(&mut self, x: &Val, y: &Val) -> Val {
        let d = self.sub(x, y);
        let r = self.relu(&d);
        self.add(y, &r)
    }

    /// `minimum(x, y)`, composed as `x - relu(x - y)`.
    pub fn min(&mut self, x: &Val, y: &Val) -> Val {
        let d = self.sub(x, y);
        let r = self.relu(&d);
        self.sub(x, &r)
    }

    /// `conv(x, weight, bias)` over `[N, C, H, W]`.
    ///
    /// `weight` is `[C_out, C_in / groups, KH, KW]`. Set `opts.groups` equal to
    /// the channel count for a depthwise bank of independent per-channel FIRs.
    ///
    /// # Panics
    /// Panics if either tensor is not rank 4, if `groups` does not divide both
    /// channel counts, if the weight's input-channel dimension disagrees with
    /// `C_in / groups`, if `bias` is not `[C_out]`, or if the geometry produces
    /// an empty output.
    pub fn conv(&mut self, x: &Val, weight: &Val, bias: Option<&Val>, opts: ConvOpts) -> Val {
        let ([n, _, h, w], [c_out, _, kh, kw]) = check_conv(x, weight, opts);
        if let Some(b) = bias {
            assert_eq!(
                b.shape,
                vec![c_out],
                "conv: bias must be [C_out] = [{c_out}], got {:?}",
                b.shape
            );
        }
        let [pt, pb, pl, pr] = opts.pad;
        let [dil_h, dil_w] = opts.dilations;
        let [stride_h, stride_w] = opts.strides;
        let h_out = conv_extent(h, pt, pb, dil_h, kh, stride_h);
        let w_out = conv_extent(w, pl, pr, dil_w, kw, stride_w);
        assert!(
            h_out > 0 && w_out > 0,
            "conv: degenerate output {h_out}x{w_out}"
        );

        let dil = self.const_i32_vec(&opts.dilations);
        let grp = self.const_i32(opts.groups);
        let padv = self.const_i32_vec(&opts.pad);
        let pad_type = self.const_str(if opts.pad == [0, 0, 0, 0] {
            "valid"
        } else {
            "custom"
        });
        let strides = self.const_i32_vec(&opts.strides);

        let out = Val {
            name: self.fresh("conv"),
            dtype: Dtype::Fp16,
            shape: vec![n, c_out, h_out, w_out],
        };
        let bias_arg = bias.map_or(String::new(), |b| format!("bias = {}, ", b.name));
        self.stmts.push(format!(
            "{} {} = conv({bias_arg}dilations = {dil}, groups = {grp}, pad = {padv}, pad_type = {pad_type}, strides = {strides}, weight = {}, x = {})[name = string(\"{}\")];",
            out.ty(),
            out.name,
            weight.name,
            x.name,
            out.name
        ));
        let reads: Vec<&Val> = core::iter::once(x)
            .chain(core::iter::once(weight))
            .chain(bias)
            .collect();
        self.note(&out, &reads);
        out
    }

    /// `matmul(x, y)` with optional operand transposes.
    ///
    /// Batched rank-4 `[B, H, M, K] x [B, H, K, N]` is supported, as is rank-2.
    ///
    /// # Panics
    /// Panics if either operand is rank 1, if the batch dimensions disagree, or
    /// if the inner dimensions do not match after the requested transposes.
    pub fn matmul(&mut self, x: &Val, y: &Val, transpose_x: bool, transpose_y: bool) -> Val {
        assert_eq!(
            x.dtype, y.dtype,
            "matmul: operand types {:?} and {:?} disagree",
            x.dtype, y.dtype
        );
        let (x_batch, [x_rows, x_cols]) = mat_dims("matmul: x", x);
        let (y_batch, [y_rows, y_cols]) = mat_dims("matmul: y", y);

        // MIL broadcasts batch dims, but the ANE lowering does not: a mismatch
        // here compiles and then produces a differently shaped tensor than the
        // rest of the graph was built against. Equal batch dims also imply
        // equal rank.
        assert_eq!(
            x_batch, y_batch,
            "matmul: batch dims {x_batch:?} and {y_batch:?} disagree"
        );

        let (m, kx) = if transpose_x {
            (x_cols, x_rows)
        } else {
            (x_rows, x_cols)
        };
        let (ky, n) = if transpose_y {
            (y_cols, y_rows)
        } else {
            (y_rows, y_cols)
        };
        assert_eq!(kx, ky, "matmul: inner dims {kx} and {ky} disagree");

        let mut shape = x_batch.to_vec();
        shape.push(m);
        shape.push(n);

        let tx = self.const_bool(transpose_x);
        let ty = self.const_bool(transpose_y);
        let out = Val {
            name: self.fresh("mm"),
            dtype: Dtype::Fp16,
            shape,
        };
        self.stmts.push(format!(
            "{} {} = matmul(transpose_x = {tx}, transpose_y = {ty}, x = {}, y = {})[name = string(\"{}\")];",
            out.ty(),
            out.name,
            x.name,
            y.name,
            out.name
        ));
        self.note(&out, &[x, y]);
        out
    }

    /// `reshape(x, shape)`. The element count must be preserved.
    ///
    /// # Panics
    /// Panics if `shape` has a non-positive dimension — MIL's `-1` inference
    /// is deliberately not supported, since a static shape is knowable here and
    /// tracking it is the whole point — or if it changes the element count.
    pub fn reshape(&mut self, x: &Val, shape: &[i64]) -> Val {
        assert!(
            shape.iter().all(|&d| d > 0),
            "reshape: every dimension must be positive and static, got {shape:?}"
        );
        let want: i64 = shape.iter().product();
        assert_eq!(
            x.numel(),
            want,
            "reshape: {:?} -> {shape:?} changes element count",
            x.shape
        );
        let s = self.const_i32_vec(shape);
        let out = Val {
            name: self.fresh("rs"),
            dtype: x.dtype,
            shape: shape.to_vec(),
        };
        self.stmts.push(format!(
            "{} {} = reshape(shape = {s}, x = {})[name = string(\"{}\")];",
            out.ty(),
            out.name,
            x.name,
            out.name
        ));
        self.note(&out, &[x]);
        out
    }

    /// `transpose(x, perm)`.
    ///
    /// # Panics
    /// Panics unless `perm` is a genuine permutation of `x`'s axes.
    pub fn transpose(&mut self, x: &Val, perm: &[i64]) -> Val {
        assert_eq!(
            perm.len(),
            x.shape.len(),
            "transpose: perm rank must match input"
        );
        // One check covers negative, duplicate and out-of-range axes, and it is
        // what makes the lookup below total.
        let mut sorted = perm.to_vec();
        sorted.sort_unstable();
        assert!(
            sorted
                .iter()
                .enumerate()
                .all(|(i, &p)| i64::try_from(i).is_ok_and(|want| want == p)),
            "transpose: {perm:?} is not a permutation of 0..{}",
            perm.len()
        );
        let shape: Vec<i64> = perm
            .iter()
            .filter_map(|&p| usize::try_from(p).ok())
            .filter_map(|a| x.shape.get(a).copied())
            .collect();
        let p = self.const_i32_vec(perm);
        let out = Val {
            name: self.fresh("tp"),
            dtype: x.dtype,
            shape,
        };
        self.stmts.push(format!(
            "{} {} = transpose(perm = {p}, x = {})[name = string(\"{}\")];",
            out.ty(),
            out.name,
            x.name,
            out.name
        ));
        self.note(&out, &[x]);
        out
    }

    /// `concat(first, rest, axis)`.
    ///
    /// The leading operand is separate because a concatenation of nothing has
    /// no shape and no dtype: there would be nothing to build a result from.
    ///
    /// # Panics
    /// Panics if the operands disagree in rank, dtype, or in any dimension
    /// other than `axis`, or if `axis` is out of range.
    pub fn concat(&mut self, first: &Val, rest: &[&Val], axis: i64) -> Val {
        let rank = first.shape.len();
        // A negative axis counts from the end. Anything that does not land in
        // `0..rank` — including one that does not survive the conversion —
        // saturates to a value the assert rejects.
        let from_end = i64::try_from(rank).unwrap_or(i64::MAX);
        let resolved = if axis < 0 { axis.saturating_add(from_end) } else { axis };
        let ax = usize::try_from(resolved).unwrap_or(usize::MAX);
        assert!(
            ax < rank,
            "concat: axis {axis} is out of range for rank {rank}"
        );

        // Everything but the concatenated axis has to line up, and the
        // concatenated one accumulates. MIL will catch a mismatch eventually,
        // but by then the error names a generated identifier and not the call
        // that built it.
        let mut cat = first.shape.get(ax).copied().unwrap_or_default();
        // Numbered from 1: operand 0 is `first`, which sets the shape the
        // others are checked against.
        for (i, v) in (1..).zip(rest) {
            assert_eq!(
                v.shape.len(),
                rank,
                "concat: operand {} has rank {} but operand 0 has {rank}",
                i,
                v.shape.len()
            );
            assert_eq!(
                v.dtype, first.dtype,
                "concat: operand {} is {:?} but operand 0 is {:?}",
                i,
                v.dtype,
                first.dtype
            );
            for (d, (&a, &b)) in first.shape.iter().zip(&v.shape).enumerate() {
                if d == ax {
                    cat = cat.saturating_add(b);
                } else {
                    assert_eq!(
                        a,
                        b,
                        "concat on axis {ax}: operand {} shape {:?} disagrees with {:?} at dim {d}",
                        i,
                        v.shape,
                        first.shape
                    );
                }
            }
        }

        let shape: Vec<i64> = first
            .shape
            .iter()
            .enumerate()
            .map(|(d, &n)| if d == ax { cat } else { n })
            .collect();

        let axis_c = self.const_i32(axis);
        let interleave = self.const_bool(false);
        let out = Val {
            name: self.fresh("cat"),
            dtype: first.dtype,
            shape,
        };
        let operands: Vec<&Val> = core::iter::once(first).chain(rest.iter().copied()).collect();
        let list = operands
            .iter()
            .map(|v| v.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        self.stmts.push(format!(
            "{} {} = concat(axis = {axis_c}, interleave = {interleave}, values = ({list}))[name = string(\"{}\")];",
            out.ty(),
            out.name,
            out.name
        ));
        self.note(&out, &operands);
        out
    }

    /// `slice_by_index(x, begin, end)` with unit stride on every axis.
    ///
    /// # Panics
    /// Panics if the bounds are not the input's rank, run outside it, or
    /// describe an empty result.
    pub fn slice(&mut self, x: &Val, begin: &[i64], end: &[i64]) -> Val {
        assert_eq!(
            begin.len(),
            x.shape.len(),
            "slice: begin rank must match input"
        );
        assert_eq!(end.len(), x.shape.len(), "slice: end rank must match input");
        for (d, ((&b, &e), &n)) in begin.iter().zip(end).zip(&x.shape).enumerate() {
            assert!(
                0 <= b && b < e && e <= n,
                "slice: dim {d} range {b}..{e} is outside 0..{n}"
            );
        }
        let shape: Vec<i64> = begin
            .iter()
            .zip(end)
            .map(|(b, e)| e.saturating_sub(*b))
            .collect();

        let b = self.const_i32_vec(begin);
        let e = self.const_i32_vec(end);
        let stride = self.const_i32_vec(&vec![1; x.shape.len()]);
        let mask = self.fresh("cm");
        let rank = x.shape.len();
        self.stmts.push(format!(
            "tensor<bool, [{rank}]> {mask} = const()[name = string(\"{mask}\"), val = tensor<bool, [{rank}]>([{}])];",
            vec!["false"; rank].join(", ")
        ));

        let out = Val {
            name: self.fresh("sl"),
            dtype: x.dtype,
            shape,
        };
        self.stmts.push(format!(
            "{} {} = slice_by_index(begin = {b}, begin_mask = {mask}, end = {e}, end_mask = {mask}, squeeze_mask = {mask}, stride = {stride}, x = {})[name = string(\"{}\")];",
            out.ty(),
            out.name,
            x.name,
            out.name
        ));
        self.note(&out, &[x]);
        out
    }

    // --------------------------------------------------------------- output

    /// Serialize the program to MIL text plus its weights blob.
    ///
    /// # Panics
    /// Panics if `outputs` is empty or names the same value twice — the
    /// framework would hand back two ports with one name, and ports are
    /// resolved by name.
    #[must_use]
    pub fn finish(&self, outputs: &[&Val]) -> (String, Vec<u8>) {
        assert!(
            !outputs.is_empty(),
            "finish: a program needs at least one output"
        );
        for (i, a) in outputs.iter().enumerate() {
            assert!(
                outputs.iter().take(i).all(|b| b.name != a.name),
                "finish: {:?} is listed as an output twice",
                a.name
            );
            assert_width_aligned("finish: output", &a.name, &a.shape);
        }
        self.assert_no_dead_ops(outputs);

        // The framework's MIL parser rejects a line break inside buildInfo, so
        // this dict must stay on exactly one line.
        let build_info = format!(
            "dict<string, string>({{{{\"coremlc-component-MIL\", \"3510.2.1\"}}, \
             {{\"coremlc-version\", \"3505.4.1\"}}, \
             {{\"coremltools-component-milinternal\", \"\"}}, \
             {{\"coremltools-version\", \"9.0\"}}, \
             {{\"ane-bridge-nonce\", \"{}\"}}}})",
            escape(&self.nonce)
        );

        let params = self
            .params
            .iter()
            .map(|p| format!("{} {}", p.ty(), p.name))
            .collect::<Vec<_>>()
            .join(", ");

        let mut mil = String::with_capacity(self.stmts.len().saturating_mul(128).saturating_add(4096));
        line(&mut mil, format_args!("program(1.3)"));
        line(&mut mil, format_args!("[buildInfo = {build_info}]"));
        line(&mut mil, format_args!("{{"));
        line(&mut mil, format_args!("    func main<ios18>({params}) {{"));
        for s in &self.stmts {
            line(&mut mil, format_args!("        {s}"));
        }
        let outs = outputs
            .iter()
            .map(|v| v.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        line(&mut mil, format_args!("    }} -> ({outs});"));
        line(&mut mil, format_args!("}}"));

        (mil, self.blob.finish())
    }

    /// Write `model.mil` and `weights.bin` into `dir`, creating it if needed.
    ///
    /// Returns the two paths, ready to hand to `ane_bridge::OpenOptions::new`.
    /// # Errors
    /// Returns the underlying [`std::io::Error`] if the directory cannot be
    /// created or either file cannot be written.
    pub fn save<D: AsRef<Path>>(
        &self,
        dir: D,
        outputs: &[&Val],
    ) -> std::io::Result<(std::path::PathBuf, std::path::PathBuf)> {
        let root = dir.as_ref();
        std::fs::create_dir_all(root)?;
        let (mil, weights) = self.finish(outputs);
        let mil_path = root.join("model.mil");
        let w_path = root.join("weights.bin");
        std::fs::write(&mil_path, mil)?;
        std::fs::write(&w_path, weights)?;
        Ok((mil_path, w_path))
    }
}

/// Append `args` and a newline. `String`'s `Write` is infallible, so there is
/// no error to propagate.
fn line(out: &mut String, args: core::fmt::Arguments<'_>) {
    if out.write_fmt(args).is_ok() {
        out.push('\n');
    }
}

/// Numpy-style broadcast of two shapes.
///
/// # Panics
/// Panics if the shapes are incompatible, naming `op` as the operation that
/// asked for them.
fn broadcast(op: &str, a: &[i64], b: &[i64]) -> Vec<i64> {
    // Right-align by padding the shorter shape with leading 1s, which is what
    // numpy broadcasting means.
    fn align(s: &[i64], rank: usize) -> impl Iterator<Item = i64> + '_ {
        core::iter::repeat_n(1, rank.saturating_sub(s.len())).chain(s.iter().copied())
    }
    let rank = a.len().max(b.len());
    align(a, rank)
        .zip(align(b, rank))
        .map(|(da, db)| {
            assert!(
                da == db || da == 1 || db == 1,
                "{op}: shapes {a:?} and {b:?} do not broadcast"
            );
            da.max(db)
        })
        .collect()
}

#[cfg(test)]
#[expect(
    clippy::inline_modules,
    reason = "these tests cover private helpers, which only a module inside the \
              file can reach"
)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::float_cmp,
        reason = "in a test a panic is the failure report, and an exact MIL string \
                  is checked with an exact comparison"
    )]
    use super::*;

    /// A minimal program must carry the header, a one-line buildInfo, the
    /// parameter list and the output tuple.
    #[test]
    fn emits_well_formed_program() {
        let mut g = Graph::new("test-1");
        let x = g.input("x", Dtype::Fp16, &[1, 2, 1, 32]);
        let y = g.relu(&x);
        let (mil, _) = g.finish(&[&y]);

        assert!(mil.starts_with("program(1.3)\n"), "{mil}");
        assert!(
            mil.contains("func main<ios18>(tensor<fp16, [1, 2, 1, 32]> x) {"),
            "{mil}"
        );
        assert!(mil.contains("-> (relu_0);"), "{mil}");

        let build_line = mil.lines().nth(1).unwrap();
        assert!(build_line.starts_with("[buildInfo = "), "{build_line}");
        assert!(
            build_line.ends_with(']'),
            "buildInfo must be a single line: {build_line}"
        );
        assert!(
            build_line.contains("\"ane-bridge-nonce\", \"test-1\""),
            "{build_line}"
        );
    }

    /// A `BLOBFILE` const must point at `@model_path/weights/weight.bin` — the
    /// fixed location `_ANECompiler` materializes the blob to — and carry the
    /// record offset the blob writer handed back.
    #[test]
    fn weight_const_references_blob() {
        let mut g = Graph::new("test-2");
        let x = g.input("x", Dtype::Fp16, &[1, 2, 1, 32]);
        let w = g.weight_fp16(&[2, 1, 1, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = g.conv(
            &x,
            &w,
            None,
            ConvOpts {
                pad: [0, 0, 2, 0],
                groups: 2,
                ..ConvOpts::default()
            },
        );
        let (mil, blob) = g.finish(&[&y]);

        assert!(
            mil.contains(
                "BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64))"
            ),
            "{mil}"
        );
        assert!(mil.contains("tensor<fp16, [2, 1, 1, 3]>"), "{mil}");
        assert_eq!(blob.len(), 64 + 64 + 12, "header + one record + 6 fp16");
    }

    /// Convolution output geometry must follow the padding/stride/dilation rule.
    #[test]
    fn conv_output_shape() {
        let mut g = Graph::new("test-3");
        let x = g.input("x", Dtype::Fp16, &[1, 4, 1, 64]);
        let w = g.weight_fp16(&[4, 1, 1, 8], &[0.0; 32]);

        // Causal depthwise FIR: 8 taps, left-padded, no decimation.
        let same = g.conv(&x, &w, None, ConvOpts::decimating_fir(8, 1, 4));
        assert_eq!(
            same.shape(),
            &[1, 4, 1, 64],
            "causal padding preserves length"
        );

        // Same filter decimating by 4.
        let dec = g.conv(&x, &w, None, ConvOpts::decimating_fir(8, 4, 4));
        assert_eq!(dec.shape(), &[1, 4, 1, 16], "stride 4 decimates by 4");

        // Unpadded: loses taps-1 samples.
        let valid = g.conv(
            &x,
            &w,
            None,
            ConvOpts {
                groups: 4,
                ..ConvOpts::default()
            },
        );
        assert_eq!(valid.shape(), &[1, 4, 1, 57]);
    }

    /// Matmul must agree on the inner dimension and keep the batch dims.
    #[test]
    fn matmul_shapes() {
        let mut graph = Graph::new("test-4");
        let lhs = graph.input("a", Dtype::Fp16, &[1, 1, 16, 32]);
        let rhs = graph.input("b", Dtype::Fp16, &[1, 1, 32, 64]);
        let product = graph.matmul(&lhs, &rhs, false, false);
        assert_eq!(product.shape(), &[1, 1, 16, 64]);

        // transpose_y turns [64, 32] into an effective [32, 64].
        let rhs_t = graph.input("bt", Dtype::Fp16, &[1, 1, 64, 32]);
        let transposed = graph.matmul(&lhs, &rhs_t, false, true);
        assert_eq!(transposed.shape(), &[1, 1, 16, 64]);
    }

    /// `max`/`min` lower to relu arithmetic, which stays inside the op set that
    /// is proven to compile on the ANE.
    #[test]
    fn minmax_lowers_to_relu() {
        let mut g = Graph::new("test-5");
        let a = g.input("a", Dtype::Fp16, &[1, 1, 1, 32]);
        let b = g.input("b", Dtype::Fp16, &[1, 1, 1, 32]);
        let m = g.max(&a, &b);
        let (mil, _) = g.finish(&[&m]);

        assert!(mil.contains("relu("), "{mil}");
        assert!(
            !mil.contains("maximum("),
            "must not emit an unverified op: {mil}"
        );
    }

    /// A quote in a caller-supplied string must not be able to close the MIL
    /// literal early, and a newline must not break `buildInfo` across lines —
    /// the framework's parser rejects both.
    #[test]
    fn caller_strings_cannot_break_out_of_literals() {
        let program = |nonce: &str| {
            let mut g = Graph::new(nonce);
            let x = g.input("x", Dtype::Fp16, &[1, 1, 1, 32]);
            let y = g.relu(&x);
            g.finish(&[&y]).0
        };
        let mil = program("nasty \" nonce\nwith a newline");

        let build_line = mil.lines().nth(1).unwrap();
        assert!(
            build_line.ends_with(']'),
            "buildInfo must stay on one line: {build_line}"
        );
        assert!(
            build_line.contains("nasty \\\" nonce\\nwith"),
            "{build_line}"
        );
        assert_eq!(
            mil.lines().count(),
            program("clean").lines().count(),
            "the nonce added a line:\n{mil}"
        );
    }

    /// A weight shaped for the wrong number of groups is the classic way a
    /// depthwise bank turns into something else, so it must not reach MIL.
    #[test]
    #[should_panic(expected = "weight expects")]
    fn conv_rejects_a_weight_that_does_not_match_its_groups() {
        let mut g = Graph::new("test-6");
        let x = g.input("x", Dtype::Fp16, &[1, 4, 1, 64]);
        // Depthwise wants [4, 1, 1, 8]; this asks for a dense filter instead.
        let w = g.weight_fp16(&[4, 4, 1, 8], &vec![0.0; 128]);
        drop(g.conv(
            &x,
            &w,
            None,
            ConvOpts {
                groups: 4,
                ..ConvOpts::default()
            },
        ));
    }

    /// Batch dims are not broadcast by the ANE lowering, so a mismatch has to
    /// fail here rather than produce an unexpectedly shaped tensor.
    #[test]
    #[should_panic(expected = "batch dims")]
    fn matmul_rejects_mismatched_batch_dims() {
        let mut g = Graph::new("test-7");
        let a = g.input("a", Dtype::Fp16, &[1, 2, 16, 32]);
        let b = g.input("b", Dtype::Fp16, &[1, 3, 32, 32]);
        drop(g.matmul(&a, &b, false, false));
    }

    /// Concat only joins along one axis; everything else must already agree.
    #[test]
    #[should_panic(expected = "disagrees with")]
    fn concat_rejects_mismatched_dims() {
        let mut g = Graph::new("test-8");
        let a = g.input("a", Dtype::Fp16, &[1, 2, 1, 32]);
        let b = g.input("b", Dtype::Fp16, &[1, 3, 1, 64]);
        drop(g.concat(&a, &[&b], 1));
    }

    /// MIL has no implicit promotion, so a mixed-type elementwise op is a
    /// parse error with no indication of which statement caused it.
    #[test]
    #[should_panic(expected = "insert a cast")]
    fn elementwise_rejects_mixed_dtypes() {
        let mut g = Graph::new("test-9");
        let a = g.input("a", Dtype::Fp32, &[1, 1, 1, 32]);
        let b = g.input("b", Dtype::Fp16, &[1, 1, 1, 32]);
        drop(g.mul(&a, &b));
    }

    /// A cast changes the tracked dtype, and elementwise ops carry it through
    /// rather than assuming everything is fp16.
    #[test]
    fn dtype_follows_the_values() {
        let mut g = Graph::new("test-10");
        let a = g.input("a", Dtype::Fp32, &[1, 1, 1, 32]);
        let h = g.cast(&a, Dtype::Fp16);
        assert_eq!(h.dtype(), Dtype::Fp16);
        assert_eq!(g.mul(&h, &h).dtype(), Dtype::Fp16);
    }

    /// fp32 is a boundary type: arithmetic in it is refused at compile, so the
    /// builder refuses it first, where the offending op is still identifiable.
    #[test]
    #[should_panic(expected = "fp16 only")]
    fn fp32_arithmetic_is_refused() {
        let mut g = Graph::new("test-10b");
        let a = g.input("a", Dtype::Fp32, &[1, 1, 1, 32]);
        drop(g.add(&a, &a));
    }

    /// An unaligned input width runs and returns mis-strided rows, so it is
    /// refused where the shape is still traceable to a caller.
    #[test]
    #[should_panic(expected = "multiple of 32")]
    fn input_rejects_an_unaligned_width() {
        let mut g = Graph::new("test-w1");
        drop(g.input("x", Dtype::Fp16, &[1, 4, 1, 4082]));
    }

    /// The same for an output, which is where the mis-striding is observed.
    #[test]
    #[should_panic(expected = "multiple of 32")]
    fn finish_rejects_an_unaligned_output() {
        let mut g = Graph::new("test-w2");
        let x = g.input("x", Dtype::Fp16, &[1, 4, 1, 64]);
        let y = g.slice(&x, &[0, 0, 0, 0], &[1, 4, 1, 48]);
        drop(g.finish(&[&y]));
    }

    /// A kernel of 16 or more taps fails to compile, with no line number.
    #[test]
    #[should_panic(expected = "wider than 15")]
    fn conv_rejects_an_oversized_kernel() {
        let mut g = Graph::new("test-k");
        let x = g.input("x", Dtype::Fp16, &[1, 1, 1, 64]);
        let w = g.weight_fp16(&[1, 1, 1, 16], &[0.0; 16]);
        drop(g.conv(&x, &w, None, ConvOpts::conv1d(15, 0)));
    }

    /// Striding a dense convolution is correct and runs at a fraction of the
    /// machine's bandwidth, which no test downstream would notice.
    #[test]
    #[should_panic(expected = "strided dense convolution")]
    fn conv_rejects_a_strided_dense_kernel() {
        let mut g = Graph::new("test-s");
        let x = g.input("x", Dtype::Fp16, &[1, 4, 1, 64]);
        let w = g.weight_fp16(&[4, 4, 1, 1], &[0.0; 16]);
        drop(g.conv(&x, &w, None, ConvOpts::decimating_fir(1, 8, 1)));
    }

    /// The same convolution grouped is the normal way to decimate.
    #[test]
    fn conv_allows_a_strided_grouped_kernel() {
        let mut g = Graph::new("test-s2");
        let x = g.input("x", Dtype::Fp16, &[1, 4, 1, 256]);
        let w = g.weight_fp16(&[4, 1, 1, 1], &[0.0; 4]);
        let y = g.conv(&x, &w, None, ConvOpts::decimating_fir(1, 8, 4));
        assert_eq!(y.shape(), &[1, 4, 1, 32]);
    }

    /// A dead subgraph is compiled rather than pruned, and some of them fail
    /// the whole program, so an unreachable op is always a mistake.
    #[test]
    #[should_panic(expected = "not reachable from any output")]
    fn finish_rejects_an_unreachable_op() {
        let mut g = Graph::new("test-d");
        let x = g.input("x", Dtype::Fp16, &[1, 1, 1, 64]);
        let wanted = g.relu(&x);
        let _orphan = g.tanh(&x);
        drop(g.finish(&[&wanted]));
    }

    /// A slice running past the end of its input is caught before MIL sees it.
    #[test]
    #[should_panic(expected = "outside 0..64")]
    fn slice_rejects_out_of_range_bounds() {
        let mut g = Graph::new("test-11");
        let x = g.input("x", Dtype::Fp16, &[1, 1, 1, 64]);
        drop(g.slice(&x, &[0, 0, 0, 32], &[1, 1, 1, 96]));
    }

    /// Ports are resolved by name, so two outputs with one name is unusable.
    #[test]
    #[should_panic(expected = "twice")]
    fn finish_rejects_a_repeated_output() {
        let mut g = Graph::new("test-12");
        let x = g.input("x", Dtype::Fp16, &[1, 1, 1, 32]);
        let y = g.relu(&x);
        drop(g.finish(&[&y, &y]));
    }

    /// Traffic must count every intermediate written and every operand read,
    /// which is what the ANE actually pays for — elementwise ops do not fuse.
    #[test]
    fn traffic_counts_every_materialized_tensor() {
        let mut graph = Graph::new("test-13");
        let x = graph.input("x", Dtype::Fp16, &[1, 4, 1, 1024]); // 8 KiB
        let squared = graph.mul(&x, &x);
        let out = graph.add(&squared, &x);
        let traffic = graph.traffic();

        assert_eq!(traffic.ops, 2, "two ops, and const attributes are not ops");
        // Each op writes 8192 bytes; mul reads x twice, add reads y and x.
        assert_eq!(traffic.written, 2 * 8192);
        assert_eq!(traffic.read, 4 * 8192);
        assert_eq!(traffic.total(), 6 * 8192);

        // Compulsory traffic is x in, out out: 2 * 8192. So this program moves
        // three times what the computation requires.
        let compulsory = 2 * 8192;
        assert_eq!(traffic.total(), compulsory * 3, "materialization factor");
        drop(out);
    }

    /// Weights are re-read on every dispatch, so they are traffic too.
    #[test]
    fn traffic_includes_weights() {
        let mut g = Graph::new("test-14");
        let x = g.input("x", Dtype::Fp16, &[1, 4, 1, 64]); // 512 bytes
        let w = g.weight_fp16(&[4, 1, 1, 8], &[0.0; 32]); // 64 bytes
        drop(g.conv(&x, &w, None, ConvOpts::decimating_fir(8, 1, 4)));
        let t = g.traffic();

        assert_eq!(t.ops, 1);
        assert_eq!(
            t.written, 512,
            "length-preserving conv writes as much as it read"
        );
        assert_eq!(t.read, 512 + 64, "the weight is read too");
    }

    /// Broadcasting must follow numpy rules, including scalar operands.
    #[test]
    fn broadcast_rules() {
        assert_eq!(broadcast("mul", &[1, 512], &[512]), vec![1, 512]);
        assert_eq!(broadcast("mul", &[1, 4, 1, 8], &[]), vec![1, 4, 1, 8]);
        assert_eq!(
            broadcast("mul", &[1, 4, 1, 8], &[1, 4, 1, 1]),
            vec![1, 4, 1, 8]
        );
    }

    /// Two shapes that line up in neither direction are a build-time error,
    /// not a reshaped result.
    #[test]
    #[should_panic(expected = "do not broadcast")]
    fn broadcast_rejects_incompatible_shapes() {
        drop(broadcast("mul", &[4], &[8]));
    }
}
