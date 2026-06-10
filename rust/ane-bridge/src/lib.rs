//! Safe Rust wrapper for ane-bridge.
//!
//! Compile a MIL program, dispatch evaluations on the Apple Neural Engine,
//! and read results back — without manually managing `IOSurface`s, retain
//! counts, or `objc_msgSend` casts.
//!
//! # Example
//!
//! ```no_run
//! use ane_bridge::{Model, OpenOptions, QoS};
//!
//! let opts = OpenOptions::new("model.mil", "weights.bin");
//! let model = Model::open(&opts).unwrap();
//! // Schema is derived from the loaded model — query it back:
//! let inp = model.input(0).unwrap();
//! let out = model.output(0).unwrap();
//! let in_nbytes  = inp.nbytes();
//! let out_nbytes = out.nbytes();
//!
//! let mut req = model.request().unwrap();
//! let x = vec![0.5_f32; in_nbytes / 4];
//! let mut y = vec![0.0_f32; out_nbytes / 4];
//!
//! // Byte-slice fast path: library memcpys into an internal IOSurface.
//! let x_bytes: &[u8] = unsafe { core::slice::from_raw_parts(x.as_ptr().cast(), in_nbytes) };
//! req.set_input_bytes(0, x_bytes).unwrap();
//! req.run(QoS::Default).unwrap();
//! let y_bytes: &mut [u8] = unsafe { core::slice::from_raw_parts_mut(y.as_mut_ptr().cast(), out_nbytes) };
//! req.get_output_bytes(0, y_bytes).unwrap();
//! ```
//!
//! # Threading model
//!
//! * [`Model`] is `Send + Sync` — share it across threads via `Arc` (it
//!   already wraps its inner handle in one internally; cloning is cheap).
//! * [`Request`] is `Send` but not `Sync`: each request maintains its own
//!   bindings and dispatch queue, so one request handles one in-flight
//!   submission at a time.
//! * [`Buffer`] is `Send`. Concurrent CPU access (host-side `lock`) is the
//!   caller's responsibility; binding the same buffer into multiple
//!   in-flight requests is undefined behaviour and not prevented by the
//!   library — protect with your own synchronization.

#![forbid(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::ptr_as_ptr,
    reason = "FFI-heavy code legitimately needs lossy/widening casts at the C boundary; \
              the library's pointer/owner discipline is documented per-callsite via \
              `// SAFETY:` comments"
)]

extern crate alloc;

use alloc::ffi::CString;
use alloc::sync::Arc;
use core::ffi::{CStr, c_char, c_void};
use core::ptr;
use std::path::Path;

/// Convert a `&Path` to a `CString`, going through the OS-string bytes
/// directly (so non-UTF-8 paths still work on Unix). On macOS this is
/// effectively a `as_os_str().as_bytes()` round-trip.
///
/// # Panics
/// Panics if the path bytes contain an interior NUL — POSIX paths
/// cannot contain NUL by definition, so this is a programmer error.
#[expect(
    clippy::expect_used,
    reason = "NUL in a filesystem path is a programmer error and is documented to panic"
)]
fn path_to_cstring(p: &Path) -> CString {
    use std::os::unix::ffi::OsStrExt as _;
    CString::new(p.as_os_str().as_bytes()).expect("path must not contain NUL byte")
}

pub use ane_bridge_sys as sys;

// =============================================================
// Error reporting
// =============================================================

/// All errors produced by the bridge surface here. `status` is the raw
/// FFI status code; `message` is the thread-local error string captured
/// from the C side at the time of failure.
#[derive(Debug, Clone)]
pub struct Error {
    /// Raw C-side status.
    pub status: sys::AneStatus,
    /// Human-readable detail. May be empty.
    pub message: String,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.message.is_empty() {
            f.write_str(self.status.short_name())
        } else {
            write!(f, "{}: {}", self.status.short_name(), self.message)
        }
    }
}
impl core::error::Error for Error {}

/// Crate-wide `Result` alias.
pub type Result<T> = core::result::Result<T, Error>;

/// Convert a C-side status into a `Result`, capturing the thread-local
/// error message on the failure branch.
fn check(status: sys::AneStatus) -> Result<()> {
    if matches!(status, sys::AneStatus::Ok) {
        return Ok(());
    }
    // SAFETY:
    // `ane_last_error` is a no-arg C function with no preconditions and
    // returns either a null pointer or a pointer to a thread-local UTF-8
    // buffer owned by the library, valid until the next library call on
    // this same thread.
    let p = unsafe { sys::ane_last_error() };
    let message = if p.is_null() {
        String::new()
    } else {
        // SAFETY:
        // `p` was just produced by `ane_last_error` and is non-null per
        // the check above. The library guarantees a NUL-terminated UTF-8
        // string at that address, stable until the next library call.
        // We immediately copy the contents out of the C buffer via
        // `into_owned()`, so the pointer is not retained beyond this
        // statement — a subsequent library call invalidating it cannot
        // affect us.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    };
    Err(Error { status, message })
}

// Re-export FFI enums under nicer names.
pub use sys::AneBufferAccess as BufferAccess;
pub use sys::AneDtype as Dtype;
pub use sys::AneQoS as QoS;

/// Trait extension giving every status/dtype/QoS enum a short string
/// name suitable for error messages and logging.
pub trait ShortName {
    /// Return a brief, lowercase, human-readable name for the value.
    fn short_name(&self) -> &'static str;
}

impl ShortName for sys::AneStatus {
    fn short_name(&self) -> &'static str {
        match *self {
            Self::Ok => "ok",
            Self::InvalidArg => "invalid argument",
            Self::Io => "I/O error",
            Self::Compile => "MIL compile failed",
            Self::Load => "model load failed",
            Self::Eval => "evaluation failed",
            Self::Oom => "out of memory",
            Self::Unsupported => "unsupported",
            Self::Timeout => "wait timed out",
            Self::Busy => "request busy",
            Self::NotDone => "not done",
            Self::Internal => "internal error",
        }
    }
}

impl ShortName for sys::AneDtype {
    fn short_name(&self) -> &'static str {
        match *self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
            Self::Int32 => "i32",
            Self::Int64 => "i64",
            Self::UInt8 => "u8",
            Self::Int8 => "i8",
        }
    }
}

impl ShortName for sys::AneQoS {
    fn short_name(&self) -> &'static str {
        match *self {
            Self::Default => "default",
            Self::UserInteractive => "user-interactive",
            Self::UserInitiated => "user-initiated",
            Self::Utility => "utility",
            Self::Background => "background",
        }
    }
}

/// Helper struct so we can `impl Display` for the foreign enums without
/// the orphan rule biting us. Use as `format!("{}", Display::new(dt))`.
pub struct Display<T>(pub T);
impl<T: Copy + ShortName> core::fmt::Display for Display<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.0.short_name())
    }
}

/// Format a foreign enum value via its short name.
///
/// ```
/// use ane_bridge::{Dtype, display};
/// assert_eq!(display(Dtype::Fp16).to_string(), "fp16");
/// ```
#[must_use]
pub const fn display<T: Copy + ShortName>(v: T) -> Display<T> {
    Display(v)
}

/// Bridge version string.
#[must_use]
pub fn version() -> &'static str {
    // SAFETY:
    // `ane_bridge_version` is a no-arg C function returning a pointer
    // to a `static const char[]` string literal in the library
    // (`#define ANE_BRIDGE_VERSION "..."`). It is never null.
    let p = unsafe { sys::ane_bridge_version() };
    debug_assert!(
        !p.is_null(),
        "ane_bridge_version must return a static string"
    );
    // SAFETY:
    // `p` points to a `static const char[]` with program lifetime, so
    // promoting the borrow to `&'static str` is sound. UTF-8 is
    // guaranteed by the literal we emit at the C side; we fall back
    // to "?" if a future library somehow ships an invalid version.
    unsafe { CStr::from_ptr(p) }.to_str().unwrap_or("?")
}

// =============================================================
// Tensor schema
// =============================================================

/// Framework-derived schema for one tensor. Returned by
/// [`Model::input`] / [`Model::output`].
#[derive(Clone, Debug)]
pub struct TensorSpec {
    /// Framework-supplied tensor name (matches the MIL IR identifier).
    name: String,
    /// Element type of the tensor's contiguous buffer.
    dtype: Dtype,
    /// Logical shape in element units. May contain -1 for ANE-internal axes.
    shape: Vec<i64>,
}

/// Convert a `*const sys::AneTensorSpec` into an owned [`TensorSpec`].
fn tensor_spec_from_raw(p: *const sys::AneTensorSpec) -> Option<TensorSpec> {
    if p.is_null() {
        return None;
    }
    // SAFETY:
    // `p` was produced by `ane_model_{input,output}_spec`, which
    // either returns NULL (caught above) or a pointer into the
    // model's internal storage that remains valid until the model
    // is closed. We hold a `&Model` borrow at the call site, so
    // the pointee outlives this call.
    let s = unsafe { &*p };
    let rank = usize::try_from(s.rank.max(0)).ok()?;
    let name = if s.name.is_null() {
        String::new()
    } else {
        // SAFETY: per the C contract, `name` is a NUL-terminated
        // UTF-8 string owned by the model.
        unsafe { CStr::from_ptr(s.name) }
            .to_string_lossy()
            .into_owned()
    };
    let shape: Vec<i64> = if s.shape.is_null() || rank == 0 {
        Vec::new()
    } else {
        // SAFETY: `shape` points to `rank` `i64` elements held by
        // the model.
        unsafe { core::slice::from_raw_parts(s.shape, rank) }.to_vec()
    };
    Some(TensorSpec {
        name,
        dtype: s.dtype,
        shape,
    })
}

impl TensorSpec {
    /// Construct a stand-alone spec. The library does not consume
    /// user-supplied specs at `Model::open` (those are derived from
    /// the loaded model), but this constructor is useful for tests
    /// and for cross-checking against [`Model::input`] /
    /// [`Model::output`].
    #[must_use]
    pub fn new(name: &str, dtype: Dtype, shape: &[i64]) -> Self {
        Self {
            name: name.to_owned(),
            dtype,
            shape: shape.to_vec(),
        }
    }
    /// Element type.
    #[must_use]
    pub const fn dtype(&self) -> Dtype {
        self.dtype
    }
    /// Borrowed shape as `[i64]`.
    #[must_use]
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }
    /// Informational tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Total bytes required to hold a dense tensor of this shape +
    /// dtype, or `0` if the shape is invalid (negative dimension,
    /// or product overflows `usize`).
    ///
    /// Returning `0` on overflow is a deliberate sentinel: every
    /// downstream caller treats `nbytes() == 0` as "reject this
    /// spec." If we instead wrapped, a caller could allocate a tiny
    /// `Buffer` for what should be a huge tensor and let the ANE
    /// write past its end.
    #[must_use]
    pub fn nbytes(&self) -> usize {
        // SAFETY:
        // `ane_dtype_size` is a pure function over a `#[repr(C)]`
        // enum value; no pointers involved.
        let elt = unsafe { sys::ane_dtype_size(self.dtype) };
        self.shape
            .iter()
            .try_fold(elt, |acc, &d| {
                let dim = usize::try_from(d).ok()?;
                acc.checked_mul(dim)
            })
            .unwrap_or(0)
    }
}

// =============================================================
// OpenOptions
// =============================================================

/// Builder describing what to open and how.
///
/// Input/output schemas are **derived from the loaded model** —
/// callers no longer (and cannot) declare them. After [`Model::open`]
/// returns, query them via [`Model::input`] / [`Model::output`].
#[derive(Clone, Debug)]
pub struct OpenOptions {
    /// Filesystem path to the `.mil` program.
    mil_path: CString,
    /// Filesystem path to the weights blob referenced by the program.
    weights_path: CString,
    /// `QoS` class used while the model compiles + loads on the daemon.
    qos: QoS,
}

impl OpenOptions {
    /// Build a new options struct with the given MIL + weights paths.
    ///
    /// # Panics
    /// Panics if a path contains a NUL byte — that is always a
    /// programmer error since filesystem paths cannot contain NULs.
    pub fn new<P: AsRef<Path>, Q: AsRef<Path>>(mil_path: P, weights_path: Q) -> Self {
        Self {
            mil_path: path_to_cstring(mil_path.as_ref()),
            weights_path: path_to_cstring(weights_path.as_ref()),
            qos: QoS::Default,
        }
    }
    /// Override the `QoS` used during compile + load.
    #[must_use]
    pub const fn qos(mut self, qos: QoS) -> Self {
        self.qos = qos;
        self
    }
}

// =============================================================
// Model
// =============================================================

/// Internal owned handle. Separated from `Model` so `Model` can be
/// cheaply cloneable via `Arc`.
struct ModelInner {
    /// Owning FFI handle. Released by `Drop` on the last `Arc` decrement.
    raw: *mut sys::AneModel,
}

// SAFETY:
// The C-side `AneModel` is effectively immutable after `ane_model_open`
// returns: it owns immutable `NSData` MIL + weights blobs, an immutable
// schema, and the compiled+loaded `_ANEInMemoryModel` + `_ANEClient`.
// The only two fields that change post-open are `perf_stats_mask` (via
// `Model::set_perf_stats_mask`) and `loaded` (via `Model::unload`);
// both are `_Atomic` on the C side, so concurrent `&self` access to
// them from multiple threads is not a data race, and the underlying
// objc messages are serialized by the framework. Apple's framework
// serializes hardware access internally when distinct threads submit
// through distinct `_ANERequest`s, which is the discipline we enforce.
// The destructor (`ane_model_close`) runs exactly once because
// `Arc::drop` only fires on the last reference.
unsafe impl Send for ModelInner {}
// SAFETY: see the `Send` impl just above — every post-open mutation
// goes through an atomic field, and the C side handles concurrent eval
// safely against the same model.
unsafe impl Sync for ModelInner {}

impl Drop for ModelInner {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY:
            // `self.raw` is the pointer returned by a successful
            // `ane_model_open`. By the `ModelInner` invariant we have
            // not freed it before, and `Arc<ModelInner>` guarantees
            // this `Drop` runs exactly once when the last reference is
            // released. `ane_model_close` accepts the resulting
            // exclusive ownership and is documented to handle null
            // (we check anyway for defensive symmetry).
            unsafe {
                sys::ane_model_close(self.raw);
            }
        }
    }
}

/// A compiled, loaded model. Cheap to clone (`Arc`); share across threads.
#[derive(Clone)]
pub struct Model {
    /// Reference-counted owning handle — clones share the same model.
    inner: Arc<ModelInner>,
}

impl core::fmt::Debug for Model {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Model")
            .field("inner", &self.inner.raw)
            .field("num_inputs", &self.num_inputs())
            .field("num_outputs", &self.num_outputs())
            .finish()
    }
}

impl Model {
    /// Compile and load a model.
    ///
    /// # Errors
    /// Returns the framework-reported error if the MIL fails to compile,
    /// the weights blob can't be read, or `aned` rejects the load.
    pub fn open(opts: &OpenOptions) -> Result<Self> {
        let copts = sys::AneModelOpenOptions {
            mil_path: opts.mil_path.as_ptr(),
            weights_path: opts.weights_path.as_ptr(),
            compile_qos: opts.qos,
        };

        let mut raw: *mut sys::AneModel = ptr::null_mut();

        // SAFETY:
        // - `&copts` points to a stack value of `#[repr(C)]` matching
        //   the C struct exactly.
        // - The strings, shape, and spec arrays referenced from
        //   `copts` are owned by `opts` / by `in_specs` / `out_specs`,
        //   each of which lives until end-of-function.
        // - `&mut raw` is a valid writable `*mut *mut AneModel`.
        // - `ane_model_open` writes to `*out` only on success and
        //   leaves `raw` untouched on failure (its contract; verified
        //   by inspection of `ane_bridge.m`). We treat the success
        //   case as transferring ownership of `raw` to `ModelInner`.
        let status = unsafe { sys::ane_model_open(&raw const copts, &raw mut raw) };
        check(status)?;
        if raw.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: "ane_model_open returned OK but null handle".into(),
            });
        }
        Ok(Self {
            inner: Arc::new(ModelInner { raw }),
        })
    }

    /// Number of declared inputs.
    #[must_use]
    pub fn num_inputs(&self) -> i32 {
        // SAFETY:
        // `inner.raw` is a valid model handle for the lifetime of
        // `self.inner` (released only in `ModelInner::drop`).
        // `ane_model_num_inputs` takes a `const`-qualified pointer
        // and performs no mutation; the function is reentrant.
        unsafe { sys::ane_model_num_inputs(self.inner.raw) }
    }
    /// Number of declared outputs.
    #[must_use]
    pub fn num_outputs(&self) -> i32 {
        // SAFETY: see `num_inputs`.
        unsafe { sys::ane_model_num_outputs(self.inner.raw) }
    }
    /// Total bytes of input `idx`. 0 if `idx` is out of range.
    #[must_use]
    pub fn input_nbytes(&self, idx: i32) -> usize {
        // SAFETY: see `num_inputs`; the C side range-checks `idx`.
        unsafe { sys::ane_model_input_nbytes(self.inner.raw, idx) }
    }
    /// Total bytes of output `idx`. 0 if `idx` is out of range.
    #[must_use]
    pub fn output_nbytes(&self, idx: i32) -> usize {
        // SAFETY: see `num_inputs`; the C side range-checks `idx`.
        unsafe { sys::ane_model_output_nbytes(self.inner.raw, idx) }
    }

    /// True if `Model::open` reused a cached lowered artifact instead of
    /// running a fresh compile.
    ///
    /// Apple's `aned` daemon caches the ANE-lowered program keyed by the
    /// descriptor's content hash. On a cache hit, `open` skips the
    /// `compileWithQoS:` call entirely (which costs tens of seconds for
    /// large graphs) and goes straight to `loadWithQoS:`. The cache
    /// survives across processes but not aned restarts or its opaque
    /// eviction policy, so a `false` here just means "we paid the full
    /// compile this time" — not that anything went wrong.
    #[must_use]
    pub fn was_cached(&self) -> bool {
        // SAFETY: see `num_inputs`; the C accessor is a pure read of a
        // bool set at open time and never mutated afterwards.
        unsafe { sys::ane_model_was_cached(self.inner.raw) }
    }

    /// Framework-derived schema for input `idx`. Returns `None` if
    /// the index is out of range.
    #[must_use]
    pub fn input(&self, idx: i32) -> Option<TensorSpec> {
        // SAFETY: const pointer to a read-only model; C does the range check.
        let p = unsafe { sys::ane_model_input_spec(self.inner.raw, idx) };
        tensor_spec_from_raw(p)
    }
    /// Framework-derived schema for output `idx`.
    #[must_use]
    pub fn output(&self, idx: i32) -> Option<TensorSpec> {
        // SAFETY: see `input`.
        let p = unsafe { sys::ane_model_output_spec(self.inner.raw, idx) };
        tensor_spec_from_raw(p)
    }

    /// Number of declared resident state tensors (`MLState`). 0 for a
    /// stateless model. A state is a third tensor category alongside
    /// inputs and outputs; it does not count against [`Self::num_inputs`]
    /// or [`Self::num_outputs`].
    #[must_use]
    pub fn num_states(&self) -> i32 {
        // SAFETY: see `num_inputs`.
        unsafe { sys::ane_model_num_states(self.inner.raw) }
    }
    /// Total bytes of state `idx`. 0 if `idx` is out of range.
    #[must_use]
    pub fn state_nbytes(&self, idx: i32) -> usize {
        // SAFETY: see `num_inputs`; the C side range-checks `idx`.
        unsafe { sys::ane_model_state_nbytes(self.inner.raw, idx) }
    }
    /// Framework-derived schema for state `idx`. Returns `None` if the
    /// index is out of range.
    #[must_use]
    pub fn state(&self, idx: i32) -> Option<TensorSpec> {
        // SAFETY: see `input`.
        let p = unsafe { sys::ane_model_state_spec(self.inner.raw, idx) };
        tensor_spec_from_raw(p)
    }

    /// Allocate a fresh request bound to this model. The model is kept
    /// alive (via `Arc`) for the lifetime of the request.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::Oom`] if allocation fails or a framework
    /// error if `ane_request_create` rejects the model.
    pub fn request(&self) -> Result<Request> {
        let n_in = usize::try_from(self.num_inputs().max(0)).unwrap_or(0);
        let n_out = usize::try_from(self.num_outputs().max(0)).unwrap_or(0);
        let n_state = usize::try_from(self.num_states().max(0)).unwrap_or(0);
        let mut raw: *mut sys::AneRequest = ptr::null_mut();
        // SAFETY:
        // `inner.raw` is a valid model handle. `ane_request_create`
        // writes the new handle into `*out` only on success.
        let status = unsafe { sys::ane_request_create(self.inner.raw, &raw mut raw) };
        check(status)?;
        if raw.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: "ane_request_create returned OK but null handle".into(),
            });
        }
        let mut bound_inputs = Vec::with_capacity(n_in);
        bound_inputs.resize_with(n_in, || None);
        let mut bound_outputs = Vec::with_capacity(n_out);
        bound_outputs.resize_with(n_out, || None);
        let mut bound_states = Vec::with_capacity(n_state);
        bound_states.resize_with(n_state, || None);
        Ok(Request {
            raw,
            _model: self.clone(),
            callback: None,
            bound_inputs,
            bound_outputs,
            bound_states,
            weights_buf: None,
            perf_stats: None,
            shared_events: None,
        })
    }

    /// Allocate a fresh `IOSurface`-backed buffer of `nbytes`.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::Oom`] if the kernel refuses the
    /// `IOSurfaceCreate` request, or a framework error otherwise.
    pub fn buffer(&self, nbytes: usize) -> Result<Buffer> {
        let mut raw: *mut sys::AneBuffer = ptr::null_mut();
        // SAFETY:
        // `ane_buffer_create` allocates an IOSurface; no pointer
        // preconditions beyond `out` being a valid `**`.
        let status = unsafe { sys::ane_buffer_create(nbytes, &raw mut raw) };
        check(status)?;
        if raw.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: "ane_buffer_create returned OK but null handle".into(),
            });
        }
        Ok(Buffer { raw })
    }

    /// Allocate a buffer sized for input `idx`.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::InvalidArg`] for an out-of-range `idx`,
    /// or [`sys::AneStatus::Oom`] / framework error otherwise.
    pub fn input_buffer(&self, idx: i32) -> Result<Buffer> {
        let mut raw: *mut sys::AneBuffer = ptr::null_mut();
        // SAFETY: see `buffer`. `idx` is range-checked by the C side.
        let status = unsafe { sys::ane_buffer_create_for_input(self.inner.raw, idx, &raw mut raw) };
        check(status)?;
        if raw.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: "ane_buffer_create_for_input returned OK but null handle".into(),
            });
        }
        Ok(Buffer { raw })
    }

    /// Allocate a buffer sized for output `idx`.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::InvalidArg`] for an out-of-range `idx`,
    /// or [`sys::AneStatus::Oom`] / framework error otherwise.
    pub fn output_buffer(&self, idx: i32) -> Result<Buffer> {
        let mut raw: *mut sys::AneBuffer = ptr::null_mut();
        // SAFETY: see `buffer`. `idx` is range-checked by the C side.
        let status =
            unsafe { sys::ane_buffer_create_for_output(self.inner.raw, idx, &raw mut raw) };
        check(status)?;
        if raw.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: "ane_buffer_create_for_output returned OK but null handle".into(),
            });
        }
        Ok(Buffer { raw })
    }

    /// Allocate a buffer sized for state slot `idx`. Bind it once with
    /// [`Request::bind_state`]; it then persists and is updated in place
    /// across submits. Initialize its contents (e.g. zero) before the
    /// first run.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::InvalidArg`] for an out-of-range `idx`,
    /// or [`sys::AneStatus::Oom`] / framework error otherwise.
    pub fn state_buffer(&self, idx: i32) -> Result<Buffer> {
        let mut raw: *mut sys::AneBuffer = ptr::null_mut();
        // SAFETY: see `buffer`. `idx` is range-checked by the C side.
        let status = unsafe { sys::ane_buffer_create_for_state(self.inner.raw, idx, &raw mut raw) };
        check(status)?;
        if raw.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: "ane_buffer_create_for_state returned OK but null handle".into(),
            });
        }
        Ok(Buffer { raw })
    }
}

// =============================================================
// Buffer
// =============================================================

/// `IOSurface`-backed buffer. Drop releases the underlying surface.
pub struct Buffer {
    /// Owning FFI handle to an `IOSurface`-backed buffer.
    raw: *mut sys::AneBuffer,
}

// SAFETY:
// `AneBuffer` wraps an `IOSurface` plus an `_ANEIOSurfaceObject`. Both
// are kernel-shared resources with no thread affinity, so the buffer
// can be moved freely between threads. We deliberately do NOT impl
// `Sync` because `lock`/`unlock` are stateful (lock count + last
// access flags); concurrent host-side mutation of those fields would
// race. Callers wanting cross-thread sharing must wrap in a Mutex.
unsafe impl Send for Buffer {}

impl Buffer {
    /// Allocate a fresh `IOSurface`-backed buffer of `nbytes`, independent of
    /// any open model. Same allocation as [`Model::buffer`], but usable on its
    /// own — e.g. to hand an `IOSurface` to [`StateModel::wrap_buffer`] for
    /// zero-copy E5RT I/O.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::Oom`] if the kernel refuses the
    /// `IOSurfaceCreate` request, or a framework error otherwise.
    pub fn new(nbytes: usize) -> Result<Self> {
        let mut raw: *mut sys::AneBuffer = ptr::null_mut();
        // SAFETY: `ane_buffer_create` allocates an IOSurface; `out` need only
        // be a valid `**`. Model-independent.
        let status = unsafe { sys::ane_buffer_create(nbytes, &raw mut raw) };
        check(status)?;
        if raw.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: "ane_buffer_create returned OK but null handle".into(),
            });
        }
        Ok(Self { raw })
    }

    /// Buffer size in bytes.
    #[must_use]
    pub fn nbytes(&self) -> usize {
        // SAFETY: `self.raw` is non-null and owned by `self`.
        unsafe { sys::ane_buffer_nbytes(self.raw) }
    }
    /// Underlying `IOSurfaceID` for advanced inter-process use.
    #[must_use]
    pub fn iosurface_id(&self) -> u32 {
        // SAFETY: see `nbytes`.
        unsafe { sys::ane_buffer_iosurface_id(self.raw) }
    }

    /// Run `f` while the buffer is locked for host access. The slice
    /// passed to `f` is the entire mapped `IOSurface`; do not let it
    /// escape the closure — it becomes invalid after unlock.
    ///
    /// # Errors
    /// Forwards any error from [`Self::lock`].
    pub fn with_locked<R, F>(&mut self, access: BufferAccess, body: F) -> Result<R>
    where
        F: FnOnce(&mut [u8]) -> R,
    {
        let mut guard = self.lock(access)?;
        Ok(body(&mut guard))
    }

    /// Acquire an exclusive lock and return a RAII guard that derefs
    /// to a `&mut [u8]` view of the mapped `IOSurface`.
    ///
    /// The guard releases the lock when dropped. This is the
    /// recommended API when you need to read or write the buffer
    /// across more than one statement; for one-shot access, prefer
    /// [`Self::with_locked`].
    ///
    /// # Errors
    /// Forwards any `IOSurfaceLock` failure as
    /// [`sys::AneStatus::Internal`].
    pub fn lock(&mut self, access: BufferAccess) -> Result<LockGuard<'_>> {
        let mut base_ptr: *mut c_void = ptr::null_mut();
        // SAFETY:
        // `self.raw` is a non-null buffer handle owned by `self`, valid
        // for the duration of the `&mut self` borrow. `ane_buffer_lock`
        // writes the mapped base address into its out-pointer on success
        // (`ANE_OK`); on failure the pointer is left null (verified in C).
        let status = unsafe { sys::ane_buffer_lock(self.raw, access, &raw mut base_ptr) };
        check(status)?;
        let size = self.nbytes();
        debug_assert!(
            !base_ptr.is_null(),
            "ane_buffer_lock returned OK with null base pointer"
        );
        Ok(LockGuard {
            buf: self,
            base: base_ptr.cast::<u8>(),
            len: size,
        })
    }

    /// Internal: raw handle for binding into a request.
    pub(crate) const fn raw(&self) -> *mut sys::AneBuffer {
        self.raw
    }
}

/// RAII guard returned by [`Buffer::lock`]. Derefs to `&mut [u8]`;
/// drops release the underlying `IOSurface` lock.
///
/// The guard borrows the buffer mutably, so the borrow checker
/// prevents concurrent locking or rebinding for its lifetime.
pub struct LockGuard<'buf> {
    /// Borrowed buffer; the `&mut` excludes any concurrent lock.
    buf: &'buf mut Buffer,
    /// Mapped base address returned by `ane_buffer_lock`.
    base: *mut u8,
    /// Mapped byte length (matches `Buffer::nbytes`).
    len: usize,
}

impl LockGuard<'_> {
    /// Length of the locked mapping in bytes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }
    /// Returns `true` if the mapping is zero-sized (would only happen
    /// for a buffer allocated with `nbytes == 0`).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl core::ops::Deref for LockGuard<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        // SAFETY:
        // `self.base` was produced by `ane_buffer_lock` on
        // `self.buf.raw`, which we hold via `&'buf mut Buffer`. The
        // mapping is valid for `self.len` bytes for the entire
        // lifetime of `self` (the C lock count is non-zero until
        // `Drop` calls `ane_buffer_unlock`). The `&mut Buffer`
        // borrow prevents any concurrent lock attempt.
        unsafe { core::slice::from_raw_parts(self.base, self.len) }
    }
}

impl core::ops::DerefMut for LockGuard<'_> {
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: see `Deref::deref`. The exclusive mutable borrow of
        // `self` upholds the unique-mutable-borrow rule on the slice.
        unsafe { core::slice::from_raw_parts_mut(self.base, self.len) }
    }
}

impl Drop for LockGuard<'_> {
    fn drop(&mut self) {
        // SAFETY:
        // `self.buf.raw` is still the live buffer we locked in
        // `Buffer::lock`. The matching `ane_buffer_unlock` returns
        // the IOSurface to the unlocked state. We deliberately
        // ignore any error: a failure here is rare and there is
        // nowhere to surface it inside `Drop`. The next `lock`
        // call on the same buffer would surface it.
        let _unlock_status: sys::AneStatus = unsafe { sys::ane_buffer_unlock(self.buf.raw) };
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY:
            // `self.raw` was allocated by one of `ane_buffer_create*`
            // and has not been released — release happens exactly
            // here when the `Buffer` is dropped, which `Buffer`'s
            // single-owner type (no `Clone` impl) makes a one-shot.
            unsafe {
                sys::ane_buffer_release(self.raw);
            }
        }
    }
}

// =============================================================
// Request
// =============================================================

/// Type-erased completion closure. Boxed and pointed to by the C-side
/// `user` arg; freed when the `Request` is dropped or a new callback
/// replaces it.
type Callback = Box<dyn FnMut(core::result::Result<(), Error>) + Send + 'static>;

/// One in-flight (or idle) inference unit. Each request owns a serial
/// dispatch queue on the C side; submit dispatches an eval onto it.
pub struct Request {
    /// Owning FFI handle.
    raw: *mut sys::AneRequest,
    /// Keep the parent model alive for at least as long as the request.
    _model: Model,
    /// Owning box for the registered completion closure (if any). We
    /// keep a `*mut` so the address handed to C is stable across moves;
    /// freed in `Drop` or when replaced.
    callback: Option<*mut Callback>,
    /// Bindings: Rust must keep each `Buffer` alive for the entire
    /// lifetime of the request, since the C side stores raw pointers
    /// into them in `r->input_bound` and dereferences those on every
    /// `submit`. Without this, a caller could drop the buffer
    /// immediately after `bind_*` returned and trigger a UAF.
    ///
    /// The vector is sized once at request creation and is append-replaced
    /// (each `bind_input(idx, buf)` overwrites slot `idx`).
    bound_inputs: Vec<Option<Buffer>>,
    /// Output bindings — same lifetime/append-replace discipline as
    /// `bound_inputs`, mirroring `r->output_bound` on the C side.
    bound_outputs: Vec<Option<Buffer>>,
    /// Resident state bindings — same lifetime/append-replace discipline
    /// as `bound_inputs`, mirroring `r->state_bound` on the C side. A
    /// bound state buffer persists and is updated in place across submits.
    bound_states: Vec<Option<Buffer>>,
    /// Optional separate weights buffer, retained so the C side's
    /// borrowed pointer stays valid for the request's lifetime.
    weights_buf: Option<Buffer>,
    /// Optional perf-stats sink bound for the request's lifetime.
    perf_stats: Option<PerfStats>,
    /// Optional shared-events handle bound for the request's lifetime.
    shared_events: Option<SharedEvents>,
}

// SAFETY:
// The request is a self-contained set of bindings + a serial dispatch
// queue on the C side. Bindings reference `AneBuffer`s (themselves
// `Send`) and a parent `AneModel` kept alive both by the C side's own
// refcounts and by the Rust `_model: Model` field. We deliberately do
// NOT impl `Sync`: submit+wait must be serialized — the C atomic
// `in_flight` only rejects concurrent submits, it does not protect a
// concurrent wait from freeing state the worker thread still reads.
unsafe impl Send for Request {}

impl Request {
    /// Bind a buffer for input `idx`. The Request takes ownership of
    /// the buffer; you can access it again via [`Self::input_buffer_mut`]
    /// to read or rewrite its contents between runs. Replacing a
    /// binding drops the previous buffer.
    ///
    /// Taking ownership is deliberate: the C side stores a raw pointer
    /// into the buffer and dereferences it on every submit, so the
    /// Rust safety story requires the Buffer to live for the whole
    /// remaining request lifetime.
    ///
    /// Validation note: the C `ane_request_bind_input` is the single
    /// source of truth for `idx` validity and buffer-size checks. We
    /// trust its return: if it accepted `idx`, then `idx` is in
    /// `[0, bound_inputs.len())` — the same invariant the Vec was
    /// sized with at `Request::create`.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::InvalidArg`] for out-of-range `idx`,
    /// dtype/shape mismatch, or a negative input index.
    pub fn bind_input(&mut self, idx: i32, buf: Buffer) -> Result<()> {
        // SAFETY:
        // - `self.raw` is a valid request handle.
        // - `buf.raw()` is the buffer's stable handle; we move `buf`
        //   into `self.bound_inputs[..]` on success so the pointer
        //   stays valid for the Request's lifetime.
        // - On error we drop `buf` here and the C side never recorded
        //   any pointer to it (the C check happens before the store).
        unsafe {
            check(sys::ane_request_bind_input(self.raw, idx, buf.raw()))?;
        }
        // The C check just confirmed `0 <= idx < num_inputs`, and
        // `bound_inputs.len() == num_inputs`, so the conversion + index
        // is in-bounds. We use `debug_assert` to keep a non-panic
        // tripwire if those invariants ever drift.
        let slot = usize::try_from(idx).map_err(|_neg| Error {
            status: sys::AneStatus::InvalidArg,
            message: alloc::format!("negative input index {idx}"),
        })?;
        debug_assert!(
            slot < self.bound_inputs.len(),
            "input slot {slot} out of range (have {})",
            self.bound_inputs.len()
        );
        // Replace last so the previous Buffer's Drop fires only after
        // the C side has rebound to the new pointer.
        if let Some(cell) = self.bound_inputs.get_mut(slot) {
            *cell = Some(buf);
        }
        Ok(())
    }

    /// Bind a buffer for output `idx`. Same ownership rules as
    /// [`Self::bind_input`].
    ///
    /// # Errors
    /// Same as [`Self::bind_input`].
    pub fn bind_output(&mut self, idx: i32, buf: Buffer) -> Result<()> {
        // SAFETY: see `bind_input`.
        unsafe {
            check(sys::ane_request_bind_output(self.raw, idx, buf.raw()))?;
        }
        let slot = usize::try_from(idx).map_err(|_neg| Error {
            status: sys::AneStatus::InvalidArg,
            message: alloc::format!("negative output index {idx}"),
        })?;
        debug_assert!(
            slot < self.bound_outputs.len(),
            "output slot {slot} out of range (have {})",
            self.bound_outputs.len()
        );
        if let Some(cell) = self.bound_outputs.get_mut(slot) {
            *cell = Some(buf);
        }
        Ok(())
    }

    /// Bind a persistent buffer to resident state slot `idx`. The buffer
    /// persists and is updated in place across every submit until rebound
    /// or the request is dropped — it never crosses the host boundary per
    /// call, so a streaming cache stays engine-resident. Initialize the
    /// buffer (e.g. zero) before the first submit. Same ownership rules as
    /// [`Self::bind_input`]: the Request takes the buffer.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::InvalidArg`] for an out-of-range `idx`,
    /// or [`sys::AneStatus::Unsupported`] until the private-framework
    /// state-binding path is wired.
    pub fn bind_state(&mut self, idx: i32, buf: Buffer) -> Result<()> {
        // SAFETY: see `bind_input`.
        unsafe {
            check(sys::ane_request_bind_state(self.raw, idx, buf.raw()))?;
        }
        let slot = usize::try_from(idx).map_err(|_neg| Error {
            status: sys::AneStatus::InvalidArg,
            message: alloc::format!("negative state index {idx}"),
        })?;
        debug_assert!(
            slot < self.bound_states.len(),
            "state slot {slot} out of range (have {})",
            self.bound_states.len()
        );
        if let Some(cell) = self.bound_states.get_mut(slot) {
            *cell = Some(buf);
        }
        Ok(())
    }

    /// Mutable access to a previously bound input buffer (e.g. to
    /// refill it between runs).
    ///
    /// Returns `None` if:
    ///   * `idx` is out of range,
    ///   * no buffer is bound at that slot, or
    ///   * **a submit is currently in flight** — `IOSurfaceLock`
    ///     synchronizes only CPU accesses, not ANE/DMA, so handing
    ///     out a `&mut Buffer` mid-flight would let the caller race
    ///     the ANE on the same `IOSurface`. Call [`Self::wait`] first.
    pub fn input_buffer_mut(&mut self, idx: i32) -> Option<&mut Buffer> {
        if !self.is_done() {
            return None;
        }
        let slot = usize::try_from(idx).ok()?;
        self.bound_inputs.get_mut(slot)?.as_mut()
    }

    /// Mutable access to a previously bound output buffer (e.g. to
    /// read it after `run`). Same in-flight rules as
    /// [`Self::input_buffer_mut`].
    pub fn output_buffer_mut(&mut self, idx: i32) -> Option<&mut Buffer> {
        if !self.is_done() {
            return None;
        }
        let slot = usize::try_from(idx).ok()?;
        self.bound_outputs.get_mut(slot)?.as_mut()
    }

    /// Byte-slice fast path: library memcpys `data` into an internal
    /// `IOSurface` for input `idx`. Convenient but copies; use
    /// [`Self::bind_input`] for hot loops.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::InvalidArg`] for out-of-range `idx` or
    /// `data.len()` not matching the input's `nbytes`.
    pub fn set_input_bytes(&mut self, idx: i32, data: &[u8]) -> Result<()> {
        // SAFETY:
        // - `self.raw` valid (see `bind_input`).
        // - `data.as_ptr()` is a valid pointer to `data.len()` bytes
        //   for the duration of this call (Rust borrow rules).
        // - The C side memcpys synchronously before returning.
        unsafe {
            check(sys::ane_request_set_input_bytes(
                self.raw,
                idx,
                data.as_ptr().cast::<c_void>(),
                data.len(),
            ))
        }
    }

    /// Byte-slice fast path: copy output `idx` into `out`. Only valid
    /// after the request has completed.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::NotDone`] if called before completion,
    /// or [`sys::AneStatus::InvalidArg`] for an out-of-range `idx` or
    /// `out.len()` not matching the output's `nbytes`.
    pub fn get_output_bytes(&mut self, idx: i32, out: &mut [u8]) -> Result<()> {
        // SAFETY:
        // - `self.raw` valid (see `bind_input`).
        // - `out.as_mut_ptr()` is exclusively borrowed for `out.len()`
        //   bytes for the duration of this call.
        // - The C side memcpys synchronously.
        unsafe {
            check(sys::ane_request_get_output_bytes(
                self.raw,
                idx,
                out.as_mut_ptr().cast::<c_void>(),
                out.len(),
            ))
        }
    }

    /// Enqueue an evaluation. Non-blocking; returns immediately.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::Busy`] if a previous submission has not
    /// yet drained, or the framework error from the dispatch call.
    pub fn submit(&mut self, qos: QoS) -> Result<()> {
        // SAFETY:
        // `self.raw` is valid. The C-side atomic `in_flight` rejects
        // concurrent submits with `Busy`. The dispatched block keeps
        // the request alive until its own internal flag is set, and
        // `Drop` flushes the queue before releasing.
        unsafe { check(sys::ane_request_submit(self.raw, qos)) }
    }

    /// Block until the in-flight submission completes.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::Timeout`] if the timeout elapses, or
    /// the eval's recorded error on completion.
    pub fn wait(&mut self, timeout_ms: i32) -> Result<()> {
        // SAFETY: `self.raw` valid; semaphore wait is internally safe.
        unsafe { check(sys::ane_request_wait(self.raw, timeout_ms)) }
    }

    /// Non-blocking completion check.
    #[must_use]
    pub fn is_done(&self) -> bool {
        // SAFETY: const pointer to a request we own; atomic load only.
        unsafe { sys::ane_request_is_done(self.raw) }
    }

    /// Submit + wait. Returns when the eval is done (or has failed).
    ///
    /// # Errors
    /// Same conditions as [`Self::submit`] and [`Self::wait`].
    pub fn run(&mut self, qos: QoS) -> Result<()> {
        // SAFETY: see `submit` / `wait`.
        unsafe { check(sys::ane_request_run(self.raw, qos)) }
    }

    /// Submit an evaluation and return a [`EvalFuture`] that resolves
    /// when it completes. Executor-agnostic — works under `tokio`,
    /// `async-std`, `smol`, or a hand-rolled `block_on`.
    ///
    /// The returned future borrows `self` mutably, so the request
    /// cannot be touched again until the future resolves (or is
    /// dropped). After `await` you can read outputs via
    /// [`Self::get_output_bytes`] or the bound buffers.
    ///
    /// # Errors
    /// Forwards any error from installing the internal callback
    /// (`Busy` if a prior submission hasn't drained yet) or from the
    /// underlying [`Self::submit`].
    pub fn submit_async(&mut self, qos: QoS) -> Result<EvalFuture<'_>> {
        let shared = Arc::new(EvalShared::new());
        let shared_cb = Arc::<EvalShared>::clone(&shared);
        self.on_complete(move |result| {
            // Store the result first, then publish `done`. The future
            // uses Acquire on `done` to synchronize-with this Release,
            // so it always observes a fully-written `result`.
            // Poison recovery: the only way these mutexes could be poisoned
            // is if a previous holder panicked while holding the guard. The
            // worker callback is wrapped in `catch_unwind` (see
            // `ane_bridge_eval_trampoline`) so this is unreachable, but
            // recovering the inner guard regardless keeps the lint clean
            // and the runtime resilient.
            *shared_cb
                .result
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(result);
            shared_cb
                .done
                .store(true, core::sync::atomic::Ordering::Release);
            // Take the waker (if any) outside the lock, then wake.
            let waker = shared_cb
                .waker
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take();
            if let Some(w) = waker {
                w.wake();
            }
        })?;
        self.submit(qos)?;
        Ok(EvalFuture {
            _request: self,
            shared,
        })
    }

    /// Install a completion callback fired from the library's worker
    /// thread when an in-flight eval completes.
    ///
    /// The closure receives `Ok(())` on a successful eval or
    /// `Err(Error)` with the per-request error message captured by the
    /// worker. Replacing an existing callback drops the previous box.
    ///
    /// Calling this while an eval is in flight will block until the
    /// in-flight callback has returned — the underlying C side does
    /// a `dispatch_sync` drain of the request's queue before swapping
    /// `(fn, user)`. That drain is the single source of truth for
    /// "the previous callback is no longer using its box," so the
    /// safe wrapper does not pre-check `is_done`; doing so would be
    /// a duplicate guard that could fall out of sync with the C
    /// invariant.
    ///
    /// # Errors
    /// Returns the framework error if `ane_request_on_complete` rejects
    /// the swap.
    pub fn on_complete<F>(&mut self, f: F) -> Result<()>
    where
        F: FnMut(core::result::Result<(), Error>) + Send + 'static,
    {
        let new_box: Box<Callback> = Box::new(Box::new(f));
        let new_ptr: *mut Callback = Box::into_raw(new_box);
        let user_ptr = new_ptr.cast::<c_void>();

        // Order matters here:
        //   1. Call `ane_request_set_completion` first. The C side
        //      drains the request's serial dispatch queue (via
        //      `dispatch_sync`) *before* swapping `(fn, user)`. So
        //      after this call returns, any callback firing under
        //      the previous `(fn, user)` has fully completed, and
        //      any future submission will pick up the new pointer.
        //   2. Only THEN is it safe to free the previous box —
        //      freeing it earlier would let the in-flight callback
        //      dereference an already-freed Rust allocation.
        //
        // SAFETY: `self.raw` is valid for the duration of `&mut self`.
        // `user_ptr` is a fresh `Box::into_raw` allocation; on success
        // ownership transfers to `self.callback`, on failure we
        // immediately reclaim and drop.
        let status = unsafe {
            sys::ane_request_set_completion(self.raw, Some(completion_trampoline), user_ptr)
        };
        if !matches!(status, sys::AneStatus::Ok) {
            // SAFETY: `new_ptr` was just produced by `Box::into_raw`
            // and never stored anywhere — we own it exclusively.
            drop(unsafe { Box::from_raw(new_ptr) });
            return check(status);
        }
        // Now the old box is definitely unreferenced by C — free it.
        if let Some(prev) = self.callback.take() {
            // SAFETY: `prev` came from `Box::into_raw` in a previous
            // `on_complete` call, was stored in `self.callback`, and
            // has not been freed since. The drain in
            // `ane_request_set_completion` above guarantees no C-side
            // worker thread is currently dereferencing it.
            drop(unsafe { Box::from_raw(prev) });
        }
        self.callback = Some(new_ptr);
        Ok(())
    }

    /// Remove any installed completion callback. Blocks if an eval is
    /// in flight (via the C `dispatch_sync` drain — same single-
    /// source-of-truth contract as [`Self::on_complete`]).
    ///
    /// # Errors
    /// Returns the framework error if `ane_request_set_completion` fails.
    pub fn clear_completion(&mut self) -> Result<()> {
        // SAFETY: see `on_complete`. Pass `None` + `null` to clear.
        // The drain inside `ane_request_set_completion` ensures any
        // pending callback has finished before we drop its box below.
        let status =
            unsafe { sys::ane_request_set_completion(self.raw, None, core::ptr::null_mut()) };
        check(status)?;
        if let Some(prev) = self.callback.take() {
            // SAFETY: same justification as in `on_complete`.
            drop(unsafe { Box::from_raw(prev) });
        }
        Ok(())
    }

    /// Per-request error message captured by the worker thread on the
    /// most recent eval. Empty string if no error or no submission yet.
    #[must_use]
    pub fn last_error(&self) -> String {
        // SAFETY:
        // `self.raw` is a valid request. `ane_request_last_error`
        // returns either a static empty string or a pointer to the
        // request's per-instance `last_err_msg`, valid until the next
        // submit on this request — which cannot happen concurrently
        // because we hold `&self`.
        let p = unsafe { sys::ane_request_last_error(self.raw) };
        if p.is_null() {
            String::new()
        } else {
            // SAFETY:
            // `p` was just produced by the call above and is non-null
            // per the check. The library guarantees a NUL-terminated
            // UTF-8 string at that address.
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    }
}

// ============================================================
// Async / Future support
// ============================================================

/// Internal state shared between an [`EvalFuture`] and the completion
/// callback closure that wakes it.
struct EvalShared {
    /// Release-stored once the worker callback has written `result`.
    /// Polled with `Acquire` by [`EvalFuture::poll`].
    done: core::sync::atomic::AtomicBool,
    /// Park-slot for the latest pending waker. Replaced (not stacked)
    /// on each poll: only the most-recent task is woken.
    waker: std::sync::Mutex<Option<core::task::Waker>>,
    /// Final result captured by the callback. `Some` exactly once after
    /// `done` flips, then taken by the future on its first wake-poll.
    result: std::sync::Mutex<Option<core::result::Result<(), Error>>>,
}

impl EvalShared {
    /// Construct an idle shared cell with all slots empty.
    const fn new() -> Self {
        Self {
            done: core::sync::atomic::AtomicBool::new(false),
            waker: std::sync::Mutex::new(None),
            result: std::sync::Mutex::new(None),
        }
    }
}

/// Future returned by [`Request::submit_async`].
///
/// Resolves to `Ok(())` on a successful eval or `Err(Error)` with the
/// per-request error message captured by the worker thread.
///
/// Dropping the future before it resolves does **not** cancel the
/// in-flight evaluation — the ANE runs it to completion regardless.
/// The borrowed `&mut Request` is released, but the C side's
/// `in_flight` flag stays set until the worker finishes, so a
/// subsequent [`Request::submit`] / [`Request::submit_async`] may
/// return [`sys::AneStatus::Busy`].
#[must_use = "futures do nothing until awaited"]
pub struct EvalFuture<'request> {
    /// Borrowed request whose `&mut` excludes any concurrent submission.
    _request: &'request mut Request,
    /// Shared cell holding the `done` flag, waker, and final result.
    shared: Arc<EvalShared>,
}

impl core::future::Future for EvalFuture<'_> {
    type Output = core::result::Result<(), Error>;

    fn poll(
        self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Self::Output> {
        // Fast path: the callback already ran before this poll. Poison
        // recovery is harmless here — see `submit_async`.
        if self.shared.done.load(core::sync::atomic::Ordering::Acquire) {
            let r = self
                .shared
                .result
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
                .unwrap_or(Ok(()));
            return core::task::Poll::Ready(r);
        }
        // Register the current waker so the callback can wake us when
        // it runs. We re-check `done` after the store to close the
        // race where the callback fires between the first check and
        // the waker store.
        *self
            .shared
            .waker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(cx.waker().clone());
        if self.shared.done.load(core::sync::atomic::Ordering::Acquire) {
            let r = self
                .shared
                .result
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
                .unwrap_or(Ok(()));
            return core::task::Poll::Ready(r);
        }
        core::task::Poll::Pending
    }
}

/// `extern "C"` trampoline that fans out to the boxed Rust closure.
///
/// # Safety
/// `user` must be a `*mut Callback` previously installed via
/// `Request::on_complete`, or null. The library promises to either
/// pass null or the value we stored.
extern "C" fn completion_trampoline(
    req: *mut sys::AneRequest,
    status: sys::AneStatus,
    user: *mut c_void,
) {
    if user.is_null() {
        return;
    }
    // SAFETY:
    // - `user` was set by `Request::on_complete` as a pointer to a
    //   `Box<Callback>` on the Rust heap. It remains valid until the
    //   Rust side either replaces it (which requires `is_done`) or
    //   drops the request (which drains the queue first). The C side
    //   never invokes the trampoline after a `ane_request_release`,
    //   so we cannot race with `Drop`.
    // - We borrow the closure exclusively for the duration of the
    //   call. The C side serializes callbacks on the request's
    //   dispatch queue, so two concurrent invocations against the
    //   same `user` cannot occur.
    let cb_ref: &mut Callback = unsafe { &mut *(user.cast::<Callback>()) };

    let result = if matches!(status, sys::AneStatus::Ok) {
        Ok(())
    } else {
        let msg = if req.is_null() {
            String::new()
        } else {
            // SAFETY:
            // `req` is the request that the C side passed back; it is
            // valid for the duration of this callback (the worker holds
            // it). `ane_request_last_error` reads the per-request error
            // string, stable until the next submit on this request.
            let p = unsafe { sys::ane_request_last_error(req.cast_const()) };
            if p.is_null() {
                String::new()
            } else {
                // SAFETY: `p` is non-null and points to a NUL-terminated
                // UTF-8 string owned by the request, stable for the
                // duration of this callback.
                unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
            }
        };
        Err(Error {
            status,
            message: msg,
        })
    };

    // Trap closure panics so unwinding never crosses the FFI boundary
    // into Obj-C, which would be undefined behavior.
    // We deliberately drop the `Result` from `catch_unwind`: a panicking
    // user callback is logged and swallowed so unwinding never crosses
    // the FFI boundary into Objective-C (UB).
    drop(std::panic::catch_unwind(core::panic::AssertUnwindSafe(
        || (cb_ref)(result),
    )));
}

impl Drop for Request {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY:
            // The C `ane_request_release` first drains the dispatch
            // queue via `dispatch_sync`, ensuring the worker thread
            // has stopped touching this request — including any
            // pending callback — before any state is freed. After
            // this returns we know the C side will never read the
            // callback box again, so it is safe to drop below.
            unsafe {
                sys::ane_request_release(self.raw);
            }
            self.raw = core::ptr::null_mut();
        }
        if let Some(prev) = self.callback.take() {
            // SAFETY:
            // `prev` came from `Box::into_raw` in `on_complete` and
            // was not freed since. The C side has been released
            // above; no aliases exist.
            drop(unsafe { Box::from_raw(prev) });
        }
    }
}

// =============================================================
// Device info
// =============================================================

/// Hardware facts about the ANE on this machine, populated from
/// `_ANEVirtualClient`. Returned by [`device_info`].
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    /// Number of ANE compute cores (e.g. 16 on M4 H16G).
    pub num_cores: i32,
    /// Number of ANE units (typically 1).
    pub num_anes: i32,
    /// True if the framework reports an ANE present.
    pub has_ane: bool,
    /// Opaque board identifier.
    pub board_type: i64,
    /// Architecture string (e.g. `"H16G"`).
    pub arch_type: String,
}

/// Query the ANE device facts. Cheap; backed by `_ANEVirtualClient`.
///
/// # Errors
/// Returns [`sys::AneStatus::Unsupported`] when no ANE-capable client is
/// available, or the framework error from `_ANEVirtualClient`.
pub fn device_info() -> Result<DeviceInfo> {
    let mut raw = sys::AneDeviceInfo {
        num_cores: 0,
        num_anes: 0,
        has_ane: false,
        board_type: 0,
        arch_type: [0; 32],
    };
    // SAFETY: `&mut raw` points to a stack-allocated `#[repr(C)]` matching
    // the C struct. The call writes fields and never reads from the buffer.
    let status = unsafe { sys::ane_device_info(&raw mut raw) };
    check(status)?;
    let arch_type = {
        // SAFETY: `raw.arch_type` is an inline `[c_char; 32]` whose
        // backing storage lives on the same stack frame as `raw`.
        let bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(raw.arch_type.as_ptr().cast::<u8>(), raw.arch_type.len())
        };
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        // `end` is either a byte position within `bytes` (from `position`) or
        // `bytes.len()` itself, so `get(..end)` always returns `Some`.
        String::from_utf8_lossy(bytes.get(..end).unwrap_or(bytes)).into_owned()
    };
    Ok(DeviceInfo {
        num_cores: raw.num_cores,
        num_anes: raw.num_anes,
        has_ane: raw.has_ane,
        board_type: raw.board_type,
        arch_type,
    })
}

// =============================================================
// Cache management
// =============================================================

/// True if `aned` has a compiled artifact for the given hex hash.
#[must_use]
pub fn cache_exists_for_hash(hex_hash: &str) -> bool {
    let Ok(c) = CString::new(hex_hash) else {
        return false;
    };
    // SAFETY: `c` lives until the call returns; the C side reads it once.
    unsafe { sys::ane_cache_exists_for_hash(c.as_ptr()) }
}

/// Evict the compiled artifact for the given hex hash from `aned`.
///
/// # Errors
/// Returns [`sys::AneStatus::InvalidArg`] if `hex_hash` contains a NUL
/// byte, or the framework error from the daemon if the eviction fails.
pub fn cache_purge_for_hash(hex_hash: &str) -> Result<()> {
    let c = CString::new(hex_hash).map_err(|_nul| Error {
        status: sys::AneStatus::InvalidArg,
        message: "hex_hash contains NUL".into(),
    })?;
    // SAFETY: see `cache_exists_for_hash`.
    let status = unsafe { sys::ane_cache_purge_for_hash(c.as_ptr()) };
    check(status)
}

/// Query an ANE network cache for the model at `model_path` (a compiled
/// `.mlmodelc` / `.bundle` URL), via the Espresso framework. A path that was
/// never compiled yields `Ok(false)`, not an error.
///
/// Which cache this reads is not yet confirmed: it is distinct from the
/// in-memory MIL hash cache behind [`cache_exists_for_hash`], and currently
/// reports `false` for every model tried (including ones the OS has ANE-run),
/// so do not rely on a positive result. See `c/include/espresso.h` for the
/// E5RT-compiler-cache hypothesis and `tests/espresso_cache.rs`.
///
/// # Panics
/// Panics if `model_path` contains a NUL byte.
///
/// # Errors
/// Returns [`sys::AneStatus::Internal`] if the Espresso query reports a
/// non-zero status.
pub fn espresso_cache_has_network<P: AsRef<Path>>(model_path: P) -> Result<bool> {
    let c = path_to_cstring(model_path.as_ref());
    let mut exists = false;
    // SAFETY: `c` lives until the call returns and is read once; `exists`
    // is a writable local the C side writes once. Both outlive the call.
    let rc = unsafe { sys::espresso::espresso_ane_cache_has_network(c.as_ptr(), &raw mut exists) };
    if rc != 0 {
        return Err(Error {
            status: sys::AneStatus::Internal,
            message: alloc::format!("espresso_ane_cache_has_network returned status {rc}"),
        });
    }
    Ok(exists)
}

/// Evict the compiled network cached under `model_path` from aned, via the
/// Espresso framework. Path-keyed counterpart to [`cache_purge_for_hash`].
///
/// # Panics
/// Panics if `model_path` contains a NUL byte.
///
/// # Errors
/// Returns [`sys::AneStatus::Internal`] if the purge reports a non-zero
/// status.
pub fn espresso_cache_purge_network<P: AsRef<Path>>(model_path: P) -> Result<()> {
    let c = path_to_cstring(model_path.as_ref());
    // SAFETY: `c` lives until the call returns and is read once by the C side.
    let rc = unsafe { sys::espresso::espresso_ane_cache_purge_network(c.as_ptr()) };
    if rc != 0 {
        return Err(Error {
            status: sys::AneStatus::Internal,
            message: alloc::format!("espresso_ane_cache_purge_network returned status {rc}"),
        });
    }
    Ok(())
}

// =============================================================
// Real-time scheduling class
// =============================================================

/// Enter the ANE real-time scheduling class for the current task.
/// Pair with [`realtime_task_end`].
///
/// # Errors
/// Returns [`sys::AneStatus::Unsupported`] if the binary lacks the
/// required entitlement, or [`sys::AneStatus::Internal`] if the kernel
/// rejects the transition.
pub fn realtime_task_begin() -> Result<()> {
    // SAFETY: parameter-less FFI call.
    check(unsafe { sys::ane_realtime_task_begin() })
}

/// Leave the real-time scheduling class.
///
/// # Errors
/// Same conditions as [`realtime_task_begin`].
pub fn realtime_task_end() -> Result<()> {
    // SAFETY: parameter-less FFI call.
    check(unsafe { sys::ane_realtime_task_end() })
}

// =============================================================
// Extended open options
// =============================================================

/// One named weight blob for the multi-blob [`OpenOptionsEx`] path.
#[derive(Clone, Debug)]
pub enum WeightSource {
    /// Read the blob from a filesystem path.
    Path(CString),
    /// Use an owned in-memory blob.
    Bytes(Vec<u8>),
}

/// Builder for a single named weight entry.
#[derive(Clone, Debug)]
pub struct WeightEntry {
    /// Identifier matched against MIL weight references.
    name: CString,
    /// Where the blob lives (filesystem or in-memory).
    source: WeightSource,
}

impl WeightEntry {
    /// Build a weight entry that reads from `path`.
    ///
    /// # Panics
    /// Panics if `name` or `path` contain a NUL byte.
    #[expect(
        clippy::expect_used,
        reason = "NUL in a weight name is a programmer error and is documented to panic"
    )]
    pub fn from_path<S: Into<Vec<u8>>, P: AsRef<Path>>(name: S, path: P) -> Self {
        Self {
            name: CString::new(name.into()).expect("name contains NUL"),
            source: WeightSource::Path(path_to_cstring(path.as_ref())),
        }
    }

    /// Build a weight entry from an in-memory blob.
    ///
    /// # Panics
    /// Panics if `name` contains a NUL byte.
    #[expect(
        clippy::expect_used,
        reason = "NUL in a weight name is a programmer error and is documented to panic"
    )]
    pub fn from_bytes<S: Into<Vec<u8>>>(name: S, bytes: Vec<u8>) -> Self {
        Self {
            name: CString::new(name.into()).expect("name contains NUL"),
            source: WeightSource::Bytes(bytes),
        }
    }
}

/// Extended open options: in-memory MIL bytes + multiple named weight blobs.
#[derive(Clone, Debug)]
pub struct OpenOptionsEx {
    /// The MIL program itself (path or owned bytes).
    mil: WeightSource,
    /// Named weight blobs referenced by the MIL program.
    weights: Vec<WeightEntry>,
    /// `QoS` class used while the model compiles + loads on the daemon.
    qos: QoS,
    /// When false the `mil` bytes are a legacy `NetworkDescription`.
    is_mil_model: bool,
}

impl OpenOptionsEx {
    /// Build options with MIL read from `mil_path` and no weights.
    pub fn from_path<P: AsRef<Path>>(mil_path: P) -> Self {
        Self {
            mil: WeightSource::Path(path_to_cstring(mil_path.as_ref())),
            weights: Vec::new(),
            qos: QoS::Default,
            is_mil_model: true,
        }
    }

    /// Build options with MIL supplied as an in-memory blob.
    #[must_use]
    pub const fn from_bytes(mil_bytes: Vec<u8>) -> Self {
        Self {
            mil: WeightSource::Bytes(mil_bytes),
            weights: Vec::new(),
            qos: QoS::Default,
            is_mil_model: true,
        }
    }

    /// Append a named weight blob.
    #[must_use]
    pub fn weight(mut self, entry: WeightEntry) -> Self {
        self.weights.push(entry);
        self
    }

    /// Replace the full weights list.
    #[must_use]
    pub fn weights(mut self, weights: Vec<WeightEntry>) -> Self {
        self.weights = weights;
        self
    }

    /// Override the compile + load `QoS`.
    #[must_use]
    pub const fn qos(mut self, qos: QoS) -> Self {
        self.qos = qos;
        self
    }

    /// When false the MIL bytes are interpreted as a legacy
    /// `NetworkDescription` rather than MIL.
    #[must_use]
    pub const fn is_mil_model(mut self, yes: bool) -> Self {
        self.is_mil_model = yes;
        self
    }

    /// Project this builder into the FFI-layout structs the C side expects.
    /// Returns the borrowed weight-entry list alongside the main struct so
    /// the caller can keep both alive for the duration of `ane_model_open_ex`.
    ///
    /// The fields `mil_path` / `mil_bytes` / `mil_nbytes` mirror the
    /// `AneModelOpenOptionsEx` C struct; renaming for the lint would lose
    /// the 1:1 correspondence.
    #[expect(
        clippy::similar_names,
        clippy::pattern_type_mismatch,
        reason = "C field names + default-binding-mode patterns are clearer here"
    )]
    fn to_sys(&self) -> (sys::AneModelOpenOptionsEx, Vec<sys::AneWeightEntry>) {
        let (mil_path, mil_bytes, mil_nbytes) = match &self.mil {
            WeightSource::Path(p) => (p.as_ptr(), core::ptr::null::<c_void>(), 0usize),
            WeightSource::Bytes(b) => (
                core::ptr::null::<c_char>(),
                b.as_ptr().cast::<c_void>(),
                b.len(),
            ),
        };
        let entries: Vec<sys::AneWeightEntry> = self
            .weights
            .iter()
            .map(|w| match &w.source {
                WeightSource::Path(p) => sys::AneWeightEntry {
                    name: w.name.as_ptr(),
                    path: p.as_ptr(),
                    bytes: core::ptr::null(),
                    nbytes: 0,
                },
                WeightSource::Bytes(b) => sys::AneWeightEntry {
                    name: w.name.as_ptr(),
                    path: core::ptr::null(),
                    bytes: b.as_ptr().cast::<c_void>(),
                    nbytes: b.len(),
                },
            })
            .collect();
        let copts = sys::AneModelOpenOptionsEx {
            mil_path,
            mil_bytes,
            mil_nbytes,
            weights: if entries.is_empty() {
                core::ptr::null()
            } else {
                entries.as_ptr()
            },
            n_weights: i32::try_from(entries.len()).unwrap_or(i32::MAX),
            compile_qos: self.qos,
            is_mil_model: self.is_mil_model,
            options_plist_json: core::ptr::null(),
        };
        (copts, entries)
    }
}

/// Options for the file-based open path (`_ANEModel modelAtURL:key:`).
#[derive(Clone, Debug)]
pub struct OpenFileOptions {
    /// URL/path to the compiled `.mlmodelc` or `.bundle`.
    model_url: CString,
    /// Optional override for the daemon cache key.
    cache_key: Option<CString>,
    /// Optional override for the daemon cache URL identifier.
    cache_url_identifier: Option<CString>,
    /// Selector controlling how the cache identity is derived.
    identifier_source: sys::AneIdentifierSource,
    /// `QoS` class used while the model compiles + loads on the daemon.
    qos: QoS,
}

impl OpenFileOptions {
    /// Build options pointing at a compiled `.mlmodelc` or `.bundle` URL/path.
    pub fn new<P: AsRef<Path>>(model_url: P) -> Self {
        Self {
            model_url: path_to_cstring(model_url.as_ref()),
            cache_key: None,
            cache_url_identifier: None,
            identifier_source: sys::AneIdentifierSource::Default,
            qos: QoS::Default,
        }
    }

    /// Set an explicit cache key.
    ///
    /// # Panics
    /// Panics if `key` contains a NUL byte.
    #[must_use]
    #[expect(
        clippy::expect_used,
        reason = "NUL in a cache key is a programmer error and is documented to panic"
    )]
    pub fn cache_key<S: Into<Vec<u8>>>(mut self, key: S) -> Self {
        self.cache_key = Some(CString::new(key.into()).expect("cache_key contains NUL"));
        self
    }

    /// Set an explicit cache URL identifier.
    ///
    /// # Panics
    /// Panics if `id` contains a NUL byte.
    #[must_use]
    #[expect(
        clippy::expect_used,
        reason = "NUL in a cache URL identifier is a programmer error and is documented to panic"
    )]
    pub fn cache_url_identifier<S: Into<Vec<u8>>>(mut self, id: S) -> Self {
        self.cache_url_identifier =
            Some(CString::new(id.into()).expect("cache_url_identifier contains NUL"));
        self
    }

    /// Choose how the cache identity is derived.
    #[must_use]
    pub const fn identifier_source(mut self, source: sys::AneIdentifierSource) -> Self {
        self.identifier_source = source;
        self
    }

    /// Override the compile + load `QoS`.
    #[must_use]
    pub const fn qos(mut self, qos: QoS) -> Self {
        self.qos = qos;
        self
    }
}

impl Model {
    /// Open via the extended path: in-memory MIL + multi-blob weights.
    ///
    /// # Errors
    /// Same conditions as [`Self::open`], plus
    /// [`sys::AneStatus::InvalidArg`] if a weight name is empty or NUL.
    pub fn open_ex(opts: &OpenOptionsEx) -> Result<Self> {
        let (copts, _entries) = opts.to_sys();
        let mut raw: *mut sys::AneModel = core::ptr::null_mut();
        // SAFETY:
        // `copts` and the `entries` Vec it points into are owned by this
        // stack frame and live across the call. `ane_model_open_ex`
        // copies what it needs and writes a non-null handle on success.
        let status = unsafe { sys::ane_model_open_ex(&raw const copts, &raw mut raw) };
        check(status)?;
        if raw.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: "ane_model_open_ex returned OK but null handle".into(),
            });
        }
        Ok(Self {
            inner: Arc::new(ModelInner { raw }),
        })
    }

    /// Open via the file-based path (`_ANEModel modelAtURL:key:`).
    ///
    /// # Errors
    /// Returns the framework error if the bundle URL fails to parse or
    /// the compiled artifact cannot be loaded by `aned`.
    pub fn open_file(opts: &OpenFileOptions) -> Result<Self> {
        let copts = sys::AneModelFileOpenOptions {
            model_url: opts.model_url.as_ptr(),
            cache_key: opts
                .cache_key
                .as_ref()
                .map_or(core::ptr::null(), |c| c.as_ptr()),
            cache_url_identifier: opts
                .cache_url_identifier
                .as_ref()
                .map_or(core::ptr::null(), |c| c.as_ptr()),
            identifier_source: opts.identifier_source,
            compile_qos: opts.qos,
        };
        let mut raw: *mut sys::AneModel = core::ptr::null_mut();
        // SAFETY: see `open_ex`.
        let status = unsafe { sys::ane_model_open_file(&raw const copts, &raw mut raw) };
        check(status)?;
        if raw.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: "ane_model_open_file returned OK but null handle".into(),
            });
        }
        Ok(Self {
            inner: Arc::new(ModelInner { raw }),
        })
    }

    /// Open with the real-time priority class.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::Unsupported`] without the required
    /// entitlement, or the same conditions as [`Self::open`].
    pub fn open_realtime(opts: &OpenOptions) -> Result<Self> {
        let copts = sys::AneModelOpenOptions {
            mil_path: opts.mil_path.as_ptr(),
            weights_path: opts.weights_path.as_ptr(),
            compile_qos: opts.qos,
        };
        let mut raw: *mut sys::AneModel = core::ptr::null_mut();
        // SAFETY: see `Model::open`.
        let status = unsafe { sys::ane_model_open_realtime(&raw const copts, &raw mut raw) };
        check(status)?;
        if raw.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: "ane_model_open_realtime returned OK but null handle".into(),
            });
        }
        Ok(Self {
            inner: Arc::new(ModelInner { raw }),
        })
    }

    /// Real-time variant of [`Model::open_ex`].
    ///
    /// # Errors
    /// Same conditions as [`Self::open_ex`] plus the entitlement
    /// requirement noted on [`Self::open_realtime`].
    pub fn open_realtime_ex(opts: &OpenOptionsEx) -> Result<Self> {
        let (copts, _entries) = opts.to_sys();
        let mut raw: *mut sys::AneModel = core::ptr::null_mut();
        // SAFETY: see `open_ex`.
        let status = unsafe { sys::ane_model_open_realtime_ex(&raw const copts, &raw mut raw) };
        check(status)?;
        if raw.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: "ane_model_open_realtime_ex returned OK but null handle".into(),
            });
        }
        Ok(Self {
            inner: Arc::new(ModelInner { raw }),
        })
    }

    /// Number of procedures (entry points) declared by the loaded model. Always >= 1.
    #[must_use]
    pub fn num_procedures(&self) -> i32 {
        // SAFETY: read-only access to the model handle.
        unsafe { sys::ane_model_num_procedures(self.inner.raw) }
    }

    /// Number of inputs for procedure `proc_idx`.
    #[must_use]
    pub fn num_inputs_for_procedure(&self, proc_idx: i32) -> i32 {
        // SAFETY: see `num_inputs`.
        unsafe { sys::ane_model_num_inputs_for_procedure(self.inner.raw, proc_idx) }
    }

    /// Number of outputs for procedure `proc_idx`.
    #[must_use]
    pub fn num_outputs_for_procedure(&self, proc_idx: i32) -> i32 {
        // SAFETY: see `num_outputs`.
        unsafe { sys::ane_model_num_outputs_for_procedure(self.inner.raw, proc_idx) }
    }

    /// Schema of input `idx` of procedure `proc_idx`.
    #[must_use]
    pub fn input_for_procedure(&self, proc_idx: i32, idx: i32) -> Option<TensorSpec> {
        // SAFETY: see `input`.
        let p = unsafe { sys::ane_model_input_spec_for_procedure(self.inner.raw, proc_idx, idx) };
        tensor_spec_from_raw(p)
    }

    /// Schema of output `idx` of procedure `proc_idx`.
    #[must_use]
    pub fn output_for_procedure(&self, proc_idx: i32, idx: i32) -> Option<TensorSpec> {
        // SAFETY: see `output`.
        let p = unsafe { sys::ane_model_output_spec_for_procedure(self.inner.raw, proc_idx, idx) };
        tensor_spec_from_raw(p)
    }

    /// Byte size of input `idx` of procedure `proc_idx`.
    #[must_use]
    pub fn input_nbytes_for_procedure(&self, proc_idx: i32, idx: i32) -> usize {
        // SAFETY: see `input_nbytes`.
        unsafe { sys::ane_model_input_nbytes_for_procedure(self.inner.raw, proc_idx, idx) }
    }

    /// Byte size of output `idx` of procedure `proc_idx`.
    #[must_use]
    pub fn output_nbytes_for_procedure(&self, proc_idx: i32, idx: i32) -> usize {
        // SAFETY: see `output_nbytes`.
        unsafe { sys::ane_model_output_nbytes_for_procedure(self.inner.raw, proc_idx, idx) }
    }

    /// Number of resident state tensors for procedure `proc_idx`.
    #[must_use]
    pub fn num_states_for_procedure(&self, proc_idx: i32) -> i32 {
        // SAFETY: see `num_inputs`.
        unsafe { sys::ane_model_num_states_for_procedure(self.inner.raw, proc_idx) }
    }

    /// Schema of state `idx` of procedure `proc_idx`.
    #[must_use]
    pub fn state_for_procedure(&self, proc_idx: i32, idx: i32) -> Option<TensorSpec> {
        // SAFETY: see `input`.
        let p = unsafe { sys::ane_model_state_spec_for_procedure(self.inner.raw, proc_idx, idx) };
        tensor_spec_from_raw(p)
    }

    /// Byte size of state `idx` of procedure `proc_idx`.
    #[must_use]
    pub fn state_nbytes_for_procedure(&self, proc_idx: i32, idx: i32) -> usize {
        // SAFETY: see `input_nbytes`.
        unsafe { sys::ane_model_state_nbytes_for_procedure(self.inner.raw, proc_idx, idx) }
    }

    /// Static hardware queue depth advertised for this model (max in-flight).
    #[must_use]
    pub fn queue_depth(&self) -> i32 {
        // SAFETY: read-only access.
        unsafe { sys::ane_model_queue_depth(self.inner.raw) }
    }

    /// Live count of evaluations currently in flight against this model.
    #[must_use]
    pub fn in_flight(&self) -> i64 {
        // SAFETY: read-only access.
        unsafe { sys::ane_model_in_flight(self.inner.raw) }
    }

    /// `hexStringIdentifier` of the loaded program (empty if absent).
    #[must_use]
    pub fn program_id(&self) -> String {
        // SAFETY: const accessor on a live model handle.
        let raw = unsafe { sys::ane_model_program_id(self.inner.raw) };
        // SAFETY: null or NUL-terminated C string owned by the library.
        unsafe { sys_str_to_string(raw) }
    }

    /// `weightsHash` of the descriptor (empty if absent).
    #[must_use]
    pub fn weights_hash(&self) -> String {
        // SAFETY: const accessor on a live model handle.
        let raw = unsafe { sys::ane_model_weights_hash(self.inner.raw) };
        // SAFETY: null or NUL-terminated C string owned by the library.
        unsafe { sys_str_to_string(raw) }
    }

    /// Opaque driver `programHandle` exposed by `_ANEModel`. `None` if absent.
    #[must_use]
    pub fn program_handle(&self) -> Option<u64> {
        let mut out: u64 = 0;
        // SAFETY: writes to a stack scalar.
        unsafe { sys::ane_model_program_handle(self.inner.raw, &raw mut out) }.then_some(out)
    }

    /// Opaque `intermediateBufferHandle`. `None` if absent.
    #[must_use]
    pub fn intermediate_buffer_handle(&self) -> Option<u64> {
        let mut out: u64 = 0;
        // SAFETY: writes to a stack scalar.
        unsafe { sys::ane_model_intermediate_buffer_handle(self.inner.raw, &raw mut out) }
            .then_some(out)
    }

    /// Set the per-model bitmask telling the driver which hardware counters to populate.
    ///
    /// # Errors
    /// Returns the framework error if `_ANEModel` rejects the new mask.
    pub fn set_perf_stats_mask(&self, mask: u32) -> Result<()> {
        // SAFETY: the C side reads `mask` and may mutate model-internal state.
        check(unsafe { sys::ane_model_set_perf_stats_mask(self.inner.raw, mask) })
    }

    /// Read back the current perf-stats mask.
    #[must_use]
    pub fn perf_stats_mask(&self) -> u32 {
        // SAFETY: read-only access.
        unsafe { sys::ane_model_get_perf_stats_mask(self.inner.raw) }
    }
}

/// Copy a NUL-terminated C string into an owned [`String`]. Returns an
/// empty string on null input. Lossy on non-UTF-8 bytes.
///
/// # Safety
/// `p` must be either null or point at a NUL-terminated C string owned
/// by the library and valid for the duration of this call.
unsafe fn sys_str_to_string(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: caller guarantees `p` points at a NUL-terminated C string
    // owned by the library; we copy it into a fresh `String`.
    unsafe { core::ffi::CStr::from_ptr(p) }
        .to_string_lossy()
        .into_owned()
}

impl Buffer {
    /// Borrowed `IOSurfaceRef` backing this buffer, exposed as a raw pointer
    /// for Metal interop (cast to `IOSurfaceRef`). Do not release.
    #[must_use]
    pub fn iosurface_ref(&self) -> *mut c_void {
        // SAFETY: read-only accessor.
        unsafe { sys::ane_buffer_iosurface_ref(self.raw) }
    }

    /// Wrap a caller-supplied `IOSurfaceRef` (as a raw pointer) in a fresh
    /// [`Buffer`]. The surface is `CFRetain`ed; you may release your own
    /// reference after this call returns `Ok`.
    ///
    /// # Safety
    /// `surface` must be a live `IOSurfaceRef`. `nbytes` should be the
    /// logical payload size; the framework validates it against the
    /// surface metadata.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::InvalidArg`] if `surface` is null or
    /// `nbytes` exceeds the surface payload, or the framework error.
    pub unsafe fn adopt_iosurface(surface: *mut c_void, nbytes: usize) -> Result<Self> {
        let mut raw: *mut sys::AneBuffer = core::ptr::null_mut();
        // SAFETY: see method docs.
        let status = unsafe { sys::ane_buffer_adopt_iosurface(surface, nbytes, &raw mut raw) };
        check(status)?;
        if raw.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: "ane_buffer_adopt_iosurface returned OK but null handle".into(),
            });
        }
        Ok(Self { raw })
    }
}

// =============================================================
// Perf stats
// =============================================================

/// Hardware-counters sink. Attach to a [`Request`] before eval; read
/// `hw_execution_ns()` after wait/run.
pub struct PerfStats {
    /// Owning FFI handle to the underlying `_ANEPerformanceStats`.
    raw: *mut sys::AnePerfStats,
}

// SAFETY: the wrapped `_ANEPerformanceStats` is plain Obj-C state with no
// thread affinity; we serialize host-side mutation via `&mut self`.
unsafe impl Send for PerfStats {}

impl PerfStats {
    /// Allocate a fresh perf-stats sink.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::Oom`] if the framework allocation fails.
    pub fn new() -> Result<Self> {
        let mut raw: *mut sys::AnePerfStats = core::ptr::null_mut();
        // SAFETY: writes the new handle on success.
        let status = unsafe { sys::ane_perf_stats_create(&raw mut raw) };
        check(status)?;
        if raw.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: "ane_perf_stats_create returned OK but null handle".into(),
            });
        }
        Ok(Self { raw })
    }

    /// Hardware execution time of the most recent eval, in nanoseconds.
    /// Returns 0 before any eval has populated the sink.
    #[must_use]
    pub fn hw_execution_ns(&self) -> u64 {
        // SAFETY: read-only access.
        unsafe { sys::ane_perf_stats_hw_execution_ns(self.raw) }
    }

    /// Copy the raw counter blob into a freshly-allocated `Vec<u8>`.
    #[must_use]
    pub fn counters(&self) -> Vec<u8> {
        // SAFETY: read-only access.
        let n = unsafe { sys::ane_perf_stats_counters_nbytes(self.raw) };
        let mut buf = vec![0u8; n];
        // SAFETY: `buf` has exactly `n` writable bytes.
        let copied =
            unsafe { sys::ane_perf_stats_counters_copy(self.raw, buf.as_mut_ptr().cast(), n) };
        buf.truncate(copied);
        buf
    }

    /// Raw FFI handle, for binding into the C API.
    pub(crate) const fn raw(&self) -> *mut sys::AnePerfStats {
        self.raw
    }
}

impl Drop for PerfStats {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: handle came from `ane_perf_stats_create` and is
            // owned uniquely (no `Clone` impl).
            unsafe { sys::ane_perf_stats_release(self.raw) };
        }
    }
}

// =============================================================
// Shared events (GPU↔ANE sync)
// =============================================================

/// Container of signal + wait events for GPU↔ANE synchronization.
pub struct SharedEvents {
    /// Owning FFI handle to the underlying `_ANESharedEvents`.
    raw: *mut sys::AneSharedEvents,
}

// SAFETY: the wrapped `_ANESharedEvents` is plain Obj-C state with no
// thread affinity; host-side mutation is serialized via `&mut self`.
unsafe impl Send for SharedEvents {}

impl SharedEvents {
    /// Allocate a fresh event container.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::Oom`] if the framework allocation fails.
    pub fn new() -> Result<Self> {
        let mut raw: *mut sys::AneSharedEvents = core::ptr::null_mut();
        // SAFETY: writes the new handle on success.
        let status = unsafe { sys::ane_shared_events_create(&raw mut raw) };
        check(status)?;
        if raw.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: "ane_shared_events_create returned OK but null handle".into(),
            });
        }
        Ok(Self { raw })
    }

    /// Append a signal event. `mtl_shared_event` is the raw `id` pointer
    /// to a Metal `MTLSharedEvent`. The container retains it.
    ///
    /// # Safety
    /// `mtl_shared_event`, if non-null, must be a live Obj-C object pointer.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::InvalidArg`] if the C side rejects the
    /// shared-event handle.
    pub unsafe fn add_signal(
        &mut self,
        value: u64,
        symbol_index: u32,
        event_type: sys::AneEventType,
        mtl_shared_event: *mut c_void,
        agent_mask: u64,
    ) -> Result<()> {
        // SAFETY: see method docs.
        check(unsafe {
            sys::ane_shared_events_add_signal(
                self.raw,
                value,
                symbol_index,
                event_type,
                mtl_shared_event,
                agent_mask,
            )
        })
    }

    /// Append a wait event.
    ///
    /// # Safety
    /// Same contract as [`SharedEvents::add_signal`] on `mtl_shared_event`.
    ///
    /// # Errors
    /// Same conditions as [`SharedEvents::add_signal`].
    pub unsafe fn add_wait(
        &mut self,
        value: u64,
        mtl_shared_event: *mut c_void,
        event_type: sys::AneEventType,
    ) -> Result<()> {
        // SAFETY: see method docs.
        check(unsafe {
            sys::ane_shared_events_add_wait(self.raw, value, mtl_shared_event, event_type)
        })
    }

    /// Number of signal events queued.
    #[must_use]
    pub fn num_signals(&self) -> i32 {
        // SAFETY: read-only.
        unsafe { sys::ane_shared_events_num_signals(self.raw) }
    }

    /// Number of wait events queued.
    #[must_use]
    pub fn num_waits(&self) -> i32 {
        // SAFETY: read-only.
        unsafe { sys::ane_shared_events_num_waits(self.raw) }
    }

    /// Raw FFI handle, for binding into the C API.
    pub(crate) const fn raw(&self) -> *mut sys::AneSharedEvents {
        self.raw
    }
}

impl Drop for SharedEvents {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: handle came from `ane_shared_events_create` and is
            // owned uniquely.
            unsafe { sys::ane_shared_events_release(self.raw) };
        }
    }
}

// =============================================================
// Extended request configuration
// =============================================================

impl Request {
    /// Per-request weight override. The buffer is retained by `self` and
    /// kept alive until cleared or the request drops.
    ///
    /// # Errors
    /// Returns the framework error if the C side rejects the binding.
    pub fn set_weights(&mut self, weights: Option<Buffer>) -> Result<()> {
        let ptr = weights.as_ref().map_or(core::ptr::null_mut(), Buffer::raw);
        // SAFETY: `self.raw` is live; the C side just stores the pointer.
        let status = unsafe { sys::ane_request_set_weights(self.raw, ptr) };
        check(status)?;
        self.weights_buf = weights;
        Ok(())
    }

    /// Select which procedure (entry point) this request targets.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::InvalidArg`] if `proc_idx` is outside
    /// `[0, num_procedures)`.
    pub fn set_procedure_index(&mut self, proc_idx: i32) -> Result<()> {
        // SAFETY: integer setter on a live request.
        check(unsafe { sys::ane_request_set_procedure_index(self.raw, proc_idx) })
    }

    /// Currently-set procedure index. Defaults to 0.
    #[must_use]
    pub fn procedure_index(&self) -> i32 {
        // SAFETY: read-only access.
        unsafe { sys::ane_request_procedure_index(self.raw) }
    }

    /// Attach a [`PerfStats`] sink for the next eval. The sink is retained
    /// by `self` until cleared.
    ///
    /// # Errors
    /// Returns the framework error if the C side rejects the binding.
    pub fn set_perf_stats(&mut self, ps: Option<PerfStats>) -> Result<()> {
        let ptr = ps.as_ref().map_or(core::ptr::null_mut(), PerfStats::raw);
        // SAFETY: pointer setter on a live request.
        let status = unsafe { sys::ane_request_set_perf_stats(self.raw, ptr) };
        check(status)?;
        self.perf_stats = ps;
        Ok(())
    }

    /// Borrow the currently-attached perf-stats sink, if any.
    #[must_use]
    pub const fn perf_stats(&self) -> Option<&PerfStats> {
        self.perf_stats.as_ref()
    }

    /// Attach a [`SharedEvents`] container for GPU↔ANE synchronization.
    ///
    /// # Errors
    /// Returns the framework error if the C side rejects the binding.
    pub fn set_shared_events(&mut self, ev: Option<SharedEvents>) -> Result<()> {
        let ptr = ev.as_ref().map_or(core::ptr::null_mut(), SharedEvents::raw);
        // SAFETY: pointer setter on a live request.
        let status = unsafe { sys::ane_request_set_shared_events(self.raw, ptr) };
        check(status)?;
        self.shared_events = ev;
        Ok(())
    }

    /// Set the transaction handle used to group this request with others.
    ///
    /// # Errors
    /// Returns the framework error if `ane_request_set_transaction` fails.
    pub fn set_transaction(&mut self, handle: u64) -> Result<()> {
        // SAFETY: scalar setter on a live request.
        check(unsafe { sys::ane_request_set_transaction(self.raw, handle) })
    }

    /// Currently-set transaction handle.
    #[must_use]
    pub fn transaction(&self) -> u64 {
        // SAFETY: read-only access.
        unsafe { sys::ane_request_transaction(self.raw) }
    }
}

// =============================================================
// Chained dispatch
// =============================================================

/// A multi-stage pipeline submitted as one ANE dispatch.
///
/// Build one stage per [`Request`] (with its bindings, procedure index,
/// and shared events already configured), call [`Chain::prepare`] once,
/// then [`Chain::enqueue`] repeatedly.
pub struct Chain {
    /// Owning FFI handle to the underlying `_ANEChainingRequest` array.
    raw: *mut sys::AneChain,
    /// Retained `Request` stages — keep alive for the chain's lifetime so
    /// the buffer/perf-stats borrows the C side stored stay valid.
    stages: Vec<Request>,
}

// SAFETY: the wrapped `_ANEChainingRequest` array is Obj-C state with no
// thread affinity; host-side use is single-owner.
unsafe impl Send for Chain {}

impl core::fmt::Debug for Chain {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Chain")
            .field("raw", &self.raw)
            .field("stages", &self.stages.len())
            .finish()
    }
}

/// Parameters wiring one stage into a [`Chain`].
#[derive(Clone, Debug, Default)]
pub struct ChainLink {
    /// Loopback input symbol id matching the previous stage's output.
    pub lb_input_symbol_id: i64,
    /// Loopback output symbol id feeding the next stage's input.
    pub lb_output_symbol_id: i64,
    /// Firmware enqueue delay in nanoseconds.
    pub fw_enqueue_delay: u64,
    /// Shared intermediate memory pool id. 0 = no pool.
    pub memory_pool_id: u64,
}

impl Chain {
    /// Build a chain from already-configured requests + their inter-stage links.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::InvalidArg`] if `stages` is empty, or
    /// the framework error if `ane_chain_create` rejects the layout.
    pub fn new(stages: Vec<(Request, ChainLink)>) -> Result<Self> {
        if stages.is_empty() {
            return Err(Error {
                status: sys::AneStatus::InvalidArg,
                message: "Chain::new: empty stages".into(),
            });
        }
        let mut requests: Vec<Request> = Vec::with_capacity(stages.len());
        let mut steps: Vec<sys::AneChainStep> = Vec::with_capacity(stages.len());
        for (req, link) in stages {
            steps.push(sys::AneChainStep {
                request: req.raw,
                lb_input_symbol_id: link.lb_input_symbol_id,
                lb_output_symbol_id: link.lb_output_symbol_id,
                fw_enqueue_delay: link.fw_enqueue_delay,
                memory_pool_id: link.memory_pool_id,
            });
            requests.push(req);
        }
        let mut raw: *mut sys::AneChain = core::ptr::null_mut();
        // SAFETY: `steps` lives across the call; the C side reads it once.
        let status = unsafe {
            sys::ane_chain_create(
                steps.as_ptr(),
                i32::try_from(steps.len()).unwrap_or(i32::MAX),
                &raw mut raw,
            )
        };
        check(status)?;
        Ok(Self {
            raw,
            stages: requests,
        })
    }

    /// One-time preparation: maps intermediate buffers and resolves
    /// loopback symbols.
    ///
    /// # Errors
    /// Returns the framework error if a stage's loopback symbols can't
    /// be resolved or a memory pool can't be allocated.
    pub fn prepare(&mut self, qos: QoS) -> Result<()> {
        // SAFETY: live handle owned by `self`.
        check(unsafe { sys::ane_chain_prepare(self.raw, qos) })
    }

    /// Submit the chain. May be called repeatedly after a successful prepare.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::Busy`] if a prior enqueue is still in
    /// flight, or the framework error from the dispatch call.
    pub fn enqueue(&mut self, qos: QoS) -> Result<()> {
        // SAFETY: live handle owned by `self`.
        check(unsafe { sys::ane_chain_enqueue(self.raw, qos) })
    }

    /// Block up to `timeout_ms` for the most recent enqueue to complete.
    /// Negative = forever, zero = poll.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::Timeout`] if the timeout elapses, or
    /// the per-stage eval error on completion.
    pub fn wait(&mut self, timeout_ms: i32) -> Result<()> {
        // SAFETY: live handle owned by `self`.
        check(unsafe { sys::ane_chain_wait(self.raw, timeout_ms) })
    }
}

impl Drop for Chain {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: handle came from `ane_chain_create` and is owned uniquely.
            unsafe { sys::ane_chain_release(self.raw) };
        }
    }
}

// =============================================================
// Weight decompression
// =============================================================

// =============================================================
// Connection management
// =============================================================

/// Number of `aned` connections currently open in this process for loading models.
#[must_use]
pub fn num_client_connections() -> i32 {
    // SAFETY: parameter-less FFI call returning a scalar.
    unsafe { sys::ane_client_num_connections() }
}

// =============================================================
// Session hints
// =============================================================

/// Load-tuning hint passed to `_ANEVirtualClient.sessionHintWithModel:`.
pub struct SessionHint {
    /// Owning FFI handle to the underlying `_ANESessionHint`.
    raw: *mut sys::AneSessionHint,
}

// SAFETY: the inner C struct is a plain enum-carrying box with no thread affinity.
unsafe impl Send for SessionHint {}

impl SessionHint {
    /// Allocate a fresh session hint of the given kind.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::Unsupported`] for an unknown kind or
    /// [`sys::AneStatus::Oom`] on allocation failure.
    pub fn new(kind: sys::AneSessionHintKind) -> Result<Self> {
        let mut raw: *mut sys::AneSessionHint = core::ptr::null_mut();
        // SAFETY: writes the new handle on success.
        let status = unsafe { sys::ane_session_hint_create(kind, &raw mut raw) };
        check(status)?;
        Ok(Self { raw })
    }
}

impl Drop for SessionHint {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: handle came from `ane_session_hint_create`.
            unsafe { sys::ane_session_hint_release(self.raw) };
        }
    }
}

// =============================================================
// Extended file open (mpsConstants / modelAttributes overrides)
// =============================================================

/// Extended file-open options. Wraps [`OpenFileOptions`] and adds
/// pointers to caller-owned Obj-C `NSDictionary*` objects for MPS
/// constants and `modelAttributes` overrides.
pub struct OpenFileOptionsEx {
    /// File-based options the extended struct decorates.
    base: OpenFileOptions,
    /// Caller-owned `NSDictionary*` of MPS constant overrides, or null.
    mps_constants_id: *mut c_void,
    /// Caller-owned `NSDictionary*` of model attribute overrides, or null.
    model_attributes_id: *mut c_void,
}

impl OpenFileOptionsEx {
    /// Wrap a base set of file-open options.
    #[must_use]
    pub const fn new(base: OpenFileOptions) -> Self {
        Self {
            base,
            mps_constants_id: core::ptr::null_mut(),
            model_attributes_id: core::ptr::null_mut(),
        }
    }

    /// Set the Metal Performance Shaders constants dictionary.
    ///
    /// # Safety
    /// `id` must be a live Obj-C `NSDictionary*` (or null) for the
    /// duration of the open call.
    #[must_use]
    pub const unsafe fn mps_constants(mut self, id: *mut c_void) -> Self {
        self.mps_constants_id = id;
        self
    }

    /// Override `modelAttributes` with a caller-supplied dictionary.
    ///
    /// # Safety
    /// Same contract as [`Self::mps_constants`].
    #[must_use]
    pub const unsafe fn model_attributes(mut self, id: *mut c_void) -> Self {
        self.model_attributes_id = id;
        self
    }
}

// =============================================================
// Extended model accessors / lifecycle
// =============================================================

impl Model {
    /// `_ANEModel.UUID` string. Empty when absent.
    #[must_use]
    pub fn uuid(&self) -> String {
        // SAFETY: const accessor on a live model handle.
        let raw = unsafe { sys::ane_model_uuid(self.inner.raw) };
        // SAFETY: null or NUL-terminated C string owned by the library.
        unsafe { sys_str_to_string(raw) }
    }
    /// Absolute string of the original source URL, when known.
    #[must_use]
    pub fn source_url(&self) -> String {
        // SAFETY: const accessor on a live model handle.
        let raw = unsafe { sys::ane_model_source_url(self.inner.raw) };
        // SAFETY: null or NUL-terminated C string owned by the library.
        unsafe { sys_str_to_string(raw) }
    }
    /// Absolute string of the compiled-bundle URL, when loaded from disk.
    #[must_use]
    pub fn model_url(&self) -> String {
        // SAFETY: const accessor on a live model handle.
        let raw = unsafe { sys::ane_model_model_url(self.inner.raw) };
        // SAFETY: null or NUL-terminated C string owned by the library.
        unsafe { sys_str_to_string(raw) }
    }
    /// Cache key currently associated with the loaded model.
    #[must_use]
    pub fn key(&self) -> String {
        // SAFETY: const accessor on a live model handle.
        let raw = unsafe { sys::ane_model_key(self.inner.raw) };
        // SAFETY: null or NUL-terminated C string owned by the library.
        unsafe { sys_str_to_string(raw) }
    }
    /// `cacheURLIdentifier` of the loaded model, when set.
    #[must_use]
    pub fn cache_url_identifier(&self) -> String {
        // SAFETY: const accessor on a live model handle.
        let raw = unsafe { sys::ane_model_cache_url_identifier(self.inner.raw) };
        // SAFETY: null or NUL-terminated C string owned by the library.
        unsafe { sys_str_to_string(raw) }
    }
    /// Raw `identifierSource` enum value reported by the framework.
    #[must_use]
    pub fn identifier_source(&self) -> i64 {
        // SAFETY: scalar accessor.
        unsafe { sys::ane_model_identifier_source(self.inner.raw) }
    }
    /// True if the underlying `_ANEClient` is a virtual (per-process) client.
    #[must_use]
    pub fn is_virtual_client(&self) -> bool {
        // SAFETY: scalar accessor.
        unsafe { sys::ane_model_is_virtual_client(self.inner.raw) }
    }

    /// Request that transient state be reset on the next unload.
    ///
    /// # Errors
    /// Returns the framework error if `_ANEModel` rejects the request.
    pub fn reset_on_unload(&self) -> Result<()> {
        // SAFETY: takes a live model handle; framework mutates internal state.
        check(unsafe { sys::ane_model_reset_on_unload(self.inner.raw) })
    }

    /// Explicitly unload the model (without freeing the handle).
    ///
    /// # Errors
    /// Returns the framework error if `_ANEModel` rejects the unload.
    pub fn unload(&self) -> Result<()> {
        // SAFETY: takes a live model handle.
        check(unsafe { sys::ane_model_unload(self.inner.raw) })
    }

    /// Load a fresh runtime instance sharing the same compiled program.
    /// Useful for the train-from-checkpoint pattern.
    ///
    /// # Errors
    /// Returns [`sys::AneStatus::InvalidArg`] if `key` contains a NUL
    /// byte, or the framework error if a fresh instance can't be
    /// instantiated from the compiled program.
    pub fn new_instance(&self, key: Option<&str>, qos: QoS) -> Result<Self> {
        let key_c = key
            .map(|k| {
                CString::new(k).map_err(|_nul| Error {
                    status: sys::AneStatus::InvalidArg,
                    message: "key contains NUL".into(),
                })
            })
            .transpose()?;
        let params = sys::AneModelInstanceParams {
            key: key_c.as_ref().map_or(core::ptr::null(), |c| c.as_ptr()),
            flags: 0,
        };
        let mut raw: *mut sys::AneModel = core::ptr::null_mut();
        // SAFETY: writes the new handle on success; reads `params` once.
        let status = unsafe {
            sys::ane_model_new_instance(self.inner.raw, &raw const params, qos, &raw mut raw)
        };
        check(status)?;
        if raw.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: "ane_model_new_instance returned OK but null handle".into(),
            });
        }
        Ok(Self {
            inner: Arc::new(ModelInner { raw }),
        })
    }

    /// Apply a session hint to this model. Returns the framework's
    /// per-hint report as a UTF-8 JSON string (empty if none).
    ///
    /// # Errors
    /// Returns the framework error if the hint is incompatible with the
    /// loaded model or `_ANEModel` rejects the request.
    pub fn apply_session_hint(&self, hint: &SessionHint) -> Result<String> {
        let mut report_ptr: *mut c_char = core::ptr::null_mut();
        // SAFETY: writes a freshly-malloc'd C string into `report_ptr` on success.
        let status = unsafe {
            sys::ane_model_apply_session_hint(self.inner.raw, hint.raw, &raw mut report_ptr)
        };
        check(status)?;
        if report_ptr.is_null() {
            return Ok(String::new());
        }
        // SAFETY: `report_ptr` was malloc'd by C; we copy and free here.
        let s = unsafe { CStr::from_ptr(report_ptr) }
            .to_string_lossy()
            .into_owned();
        // SAFETY: pair the C-side `malloc` with `free`.
        unsafe {
            unsafe extern "C" {
                fn free(p: *mut c_void);
            }
            free(report_ptr.cast::<c_void>());
        }
        Ok(s)
    }

    /// Open via the extended file path with MPS constants / attribute overrides.
    ///
    /// # Errors
    /// Same conditions as [`Self::open_file`] plus framework rejection of
    /// the supplied MPS constants or model-attribute overrides.
    pub fn open_file_ex(opts: &OpenFileOptionsEx) -> Result<Self> {
        let base = sys::AneModelFileOpenOptions {
            model_url: opts.base.model_url.as_ptr(),
            cache_key: opts
                .base
                .cache_key
                .as_ref()
                .map_or(core::ptr::null(), |c| c.as_ptr()),
            cache_url_identifier: opts
                .base
                .cache_url_identifier
                .as_ref()
                .map_or(core::ptr::null(), |c| c.as_ptr()),
            identifier_source: opts.base.identifier_source,
            compile_qos: opts.base.qos,
        };
        let copts = sys::AneModelFileOpenOptionsEx {
            base,
            mps_constants_id: opts.mps_constants_id,
            model_attributes_id: opts.model_attributes_id,
        };
        let mut raw: *mut sys::AneModel = core::ptr::null_mut();
        // SAFETY: `copts` lives across the call; the C side reads it once.
        let status = unsafe { sys::ane_model_open_file_ex(&raw const copts, &raw mut raw) };
        check(status)?;
        if raw.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: "ane_model_open_file_ex returned OK but null handle".into(),
            });
        }
        Ok(Self {
            inner: Arc::new(ModelInner { raw }),
        })
    }
}

// =============================================================
// Perf counter naming + signposts
// =============================================================

/// Human-readable name of perf counter `counter_idx`.
#[must_use]
pub fn perf_counter_name(counter_idx: i32) -> String {
    // SAFETY: parameter-less FFI call.
    let raw = unsafe { sys::ane_perf_counter_name(counter_idx) };
    // SAFETY: null or NUL-terminated thread-local C string.
    unsafe { sys_str_to_string(raw) }
}

impl PerfStats {
    /// Emit `os_signpost` events keyed by `model_string_id` for the
    /// counters captured by this sink.
    ///
    /// # Errors
    /// Returns the framework error if `os_signpost` emission fails.
    pub fn emit_signpost(&self, model_string_id: u64) -> Result<()> {
        // SAFETY: read-only access to a live handle.
        check(unsafe { sys::ane_perf_stats_emit_signpost(self.raw, model_string_id) })
    }
}

impl Request {
    /// Number of per-stage perf-stats slots on this request.
    /// For a single-stage request this is 0 or 1; for a chained request
    /// it equals the number of stages.
    #[must_use]
    pub fn num_perf_stats(&self) -> i32 {
        // SAFETY: read-only access.
        unsafe { sys::ane_request_num_perf_stats(self.raw) }
    }
}

// =============================================================
// Weight decompression
// =============================================================

/// Route a compressed weight blob through the framework's parallel
/// decompressor and return the decompressed bytes.
///
/// # Errors
/// Returns [`sys::AneStatus::InvalidArg`] if `compressed` does not match
/// the framework's expected weight-compression container, or the
/// framework error if decompression fails.
pub fn decompress_weights(compressed: &[u8]) -> Result<Vec<u8>> {
    let mut out_ptr: *mut c_void = core::ptr::null_mut();
    let mut out_n: usize = 0;
    // SAFETY: `compressed` lives for the call; the C side either fills the
    // out-params or returns an error and leaves them unchanged.
    let status = unsafe {
        sys::ane_decompress_weights(
            compressed.as_ptr().cast(),
            compressed.len(),
            &raw mut out_ptr,
            &raw mut out_n,
        )
    };
    check(status)?;
    if out_ptr.is_null() {
        return Ok(Vec::new());
    }
    // SAFETY: the C side allocated `out_n` bytes via `malloc`; we copy out
    // and free the original allocation to match the documented contract.
    let bytes = unsafe { core::slice::from_raw_parts(out_ptr.cast::<u8>(), out_n).to_vec() };
    // SAFETY: `out_ptr` was returned by `malloc`; freeing matches the contract.
    unsafe {
        unsafe extern "C" {
            fn free(p: *mut c_void);
        }
        free(out_ptr);
    }
    Ok(bytes)
}

// =============================================================
// Stateful inference (CoreML MLModel + MLState) — ANE-resident state
// =============================================================

/// Like [`check`], but reads the `ane_state_*` thread-local error source.
fn state_check(status: sys::AneStatus) -> Result<()> {
    if matches!(status, sys::AneStatus::Ok) {
        return Ok(());
    }
    // SAFETY: no-arg C function returning null or a thread-local NUL-terminated
    // UTF-8 buffer valid until the next `ane_state_*` call on this thread; we
    // copy it out immediately.
    let p = unsafe { sys::ane_state_last_error() };
    let message = if p.is_null() {
        String::new()
    } else {
        // SAFETY: `p` is non-null per the check and points at a NUL-terminated
        // string owned by the library; `into_owned` copies before any further call.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    };
    Err(Error { status, message })
}

/// A CoreML-backed model that carries ANE-resident state (e.g. a KV cache).
///
/// This is a separate backend from [`Model`]: state ops cannot compile through
/// the private `_ANE` path, so this goes through `CoreML`'s `MLModel` →
/// `MLE5Engine` / `E5RT`, which keeps the state on the Neural Engine across calls.
/// Only the (small) inputs/outputs cross the host boundary per
/// [`Self::predict`]; the state never does. Needs no ANE entitlement.
pub struct StateModel {
    /// Owning FFI handle (`MLModel`).
    raw: *mut sys::AneStateModel,
}

// SAFETY: `MLModel` is documented thread-safe for predictions; the handle owns
// no thread-affine state of ours. Not `Sync` — concurrent `predict` on one
// model would race the shared model object, so callers serialize via `&mut`.
unsafe impl Send for StateModel {}

/// A resident state buffer (`MLState`) bound to a [`StateModel`] — the KV cache
/// that lives on the ANE across [`StateModel::predict`] calls.
pub struct State {
    /// Owning FFI handle (`MLState`).
    raw: *mut sys::AneState,
}

// SAFETY: see [`StateModel`]. `predict` takes `&mut State`, so in-place updates
// are exclusively borrowed; the handle is safe to move between threads.
unsafe impl Send for State {}

impl core::fmt::Debug for StateModel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StateModel").finish_non_exhaustive()
    }
}

impl core::fmt::Debug for State {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("State").finish_non_exhaustive()
    }
}

impl StateModel {
    /// Open a compiled `.mlmodelc` (or `.mlpackage`) that declares state,
    /// pinned to CPU+ANE.
    ///
    /// # Panics
    /// Panics if `path` contains a NUL byte.
    ///
    /// # Errors
    /// [`sys::AneStatus::Compile`] if a `.mlpackage` fails to compile,
    /// [`sys::AneStatus::Load`] if the model fails to load.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let c = path_to_cstring(path.as_ref());
        let mut raw: *mut sys::AneStateModel = core::ptr::null_mut();
        // SAFETY: `c` lives for the call; `raw` is written on success.
        let status = unsafe { sys::ane_state_model_open(c.as_ptr(), &raw mut raw) };
        state_check(status)?;
        if raw.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: "ane_state_model_open returned OK but null handle".into(),
            });
        }
        Ok(Self { raw })
    }

    /// Sorted input feature names.
    #[must_use]
    pub fn input_names(&self) -> Vec<String> {
        // SAFETY: `self.raw` is live; `num`/`name` are bounds-checked in C.
        let num = unsafe { sys::ane_state_model_num_inputs(self.raw) };
        (0..num)
            .filter_map(|i| {
                // SAFETY: `i` in `[0, num)`; returns a borrowed C string or null.
                let p = unsafe { sys::ane_state_model_input_name(self.raw, i) };
                (!p.is_null()).then(|| {
                    // SAFETY: non-null borrowed NUL-terminated name owned by the model.
                    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
                })
            })
            .collect()
    }

    /// Sorted output feature names.
    #[must_use]
    pub fn output_names(&self) -> Vec<String> {
        // SAFETY: as in `input_names`.
        let num = unsafe { sys::ane_state_model_num_outputs(self.raw) };
        (0..num)
            .filter_map(|i| {
                // SAFETY: `i` in `[0, num)`.
                let p = unsafe { sys::ane_state_model_output_name(self.raw, i) };
                (!p.is_null())
                    // SAFETY: non-null borrowed name owned by the model.
                    .then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
            })
            .collect()
    }

    /// Sorted state (e.g. KV-cache) buffer names.
    #[must_use]
    pub fn state_names(&self) -> Vec<String> {
        // SAFETY: as in `input_names`.
        let num = unsafe { sys::ane_state_model_num_states(self.raw) };
        (0..num)
            .filter_map(|i| {
                // SAFETY: `i` in `[0, num)`.
                let p = unsafe { sys::ane_state_model_state_name(self.raw, i) };
                (!p.is_null())
                    // SAFETY: non-null borrowed name owned by the model.
                    .then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
            })
            .collect()
    }

    /// Declared element count for a named input (0 if unknown).
    ///
    /// # Panics
    /// Panics if `name` contains a NUL byte.
    #[must_use]
    #[expect(
        clippy::expect_used,
        reason = "a NUL in a feature name is a programmer error and is documented to panic"
    )]
    pub fn input_count(&self, name: &str) -> usize {
        let c = CString::new(name).expect("input name contains NUL");
        // SAFETY: `self.raw` live; `c` lives for the call.
        unsafe { sys::ane_state_model_input_count(self.raw, c.as_ptr()) }
    }

    /// Declared element count for a named output (0 if unknown).
    ///
    /// # Panics
    /// Panics if `name` contains a NUL byte.
    #[must_use]
    #[expect(
        clippy::expect_used,
        reason = "a NUL in a feature name is a programmer error and is documented to panic"
    )]
    pub fn output_count(&self, name: &str) -> usize {
        let c = CString::new(name).expect("output name contains NUL");
        // SAFETY: `self.raw` live; `c` lives for the call.
        unsafe { sys::ane_state_model_output_count(self.raw, c.as_ptr()) }
    }

    /// Allocate a fresh resident state (e.g. a zeroed KV cache) for this model.
    ///
    /// # Errors
    /// [`sys::AneStatus::Internal`] if the framework returns no state object.
    pub fn new_state(&self) -> Result<State> {
        let mut raw: *mut sys::AneState = core::ptr::null_mut();
        // SAFETY: `self.raw` live; `raw` written on success.
        let status = unsafe { sys::ane_state_create(self.raw, &raw mut raw) };
        state_check(status)?;
        if raw.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: "ane_state_create returned OK but null handle".into(),
            });
        }
        Ok(State { raw })
    }

    /// Run one inference step using the resident `state`, binding `inputs` and
    /// filling `outputs` by name (flat row-major f32, converted to/from the
    /// model's dtype). The state is read and updated in place on the ANE — its
    /// contents never cross the host boundary.
    ///
    /// # Panics
    /// Panics if any feature name contains a NUL byte.
    ///
    /// # Errors
    /// [`sys::AneStatus::InvalidArg`] if a name is unknown or a buffer length
    /// differs from the model's declared count; [`sys::AneStatus::Eval`] if the
    /// prediction fails.
    #[expect(
        clippy::expect_used,
        reason = "a NUL in a feature name is a programmer error and is documented to panic"
    )]
    pub fn predict(
        &self,
        state: &mut State,
        inputs: &[(&str, &[f32])],
        outputs: &mut [(&str, &mut [f32])],
    ) -> Result<()> {
        // Field access (`.0`/`.1`) rather than tuple-pattern closures keeps the
        // `pattern_type_mismatch` restriction lint happy on the borrowed pairs.
        let in_names: Vec<CString> = inputs
            .iter()
            .map(|pair| CString::new(pair.0).expect("input name contains NUL"))
            .collect();
        let out_names: Vec<CString> = outputs
            .iter()
            .map(|pair| CString::new(pair.0).expect("output name contains NUL"))
            .collect();
        let in_name_ptrs: Vec<*const c_char> = in_names.iter().map(|c| c.as_ptr()).collect();
        let in_data_ptrs: Vec<*const f32> = inputs.iter().map(|pair| pair.1.as_ptr()).collect();
        let in_counts: Vec<usize> = inputs.iter().map(|pair| pair.1.len()).collect();
        let out_name_ptrs: Vec<*const c_char> = out_names.iter().map(|c| c.as_ptr()).collect();
        let out_counts: Vec<usize> = outputs.iter().map(|pair| pair.1.len()).collect();
        let mut out_data_ptrs: Vec<*mut f32> =
            outputs.iter_mut().map(|pair| pair.1.as_mut_ptr()).collect();

        let n_in = i32::try_from(inputs.len()).map_err(|_too_many| Error {
            status: sys::AneStatus::InvalidArg,
            message: "too many inputs".into(),
        })?;
        let n_out = i32::try_from(outputs.len()).map_err(|_too_many| Error {
            status: sys::AneStatus::InvalidArg,
            message: "too many outputs".into(),
        })?;

        // SAFETY: every pointer array has the matching length passed alongside
        // it; each `in_data[k]`/`out_data[k]` is valid for `*_counts[k]` f32s
        // for the duration of the call (borrowed from `inputs`/`outputs`).
        // `self.raw` and `state.raw` are live owning handles.
        let status = unsafe {
            sys::ane_state_predict_f32(
                self.raw,
                state.raw,
                in_name_ptrs.as_ptr(),
                in_data_ptrs.as_ptr(),
                in_counts.as_ptr(),
                n_in,
                out_name_ptrs.as_ptr(),
                out_data_ptrs.as_mut_ptr(),
                out_counts.as_ptr(),
                n_out,
            )
        };
        state_check(status)
    }

    /// Borrow the live `e5rt_execution_stream` `CoreML` built for this model —
    /// the `E5RT` runtime under `MLE5Engine` — as a raw pointer, or null for a
    /// non-MLE5 model or before the first [`Self::predict`] has built it.
    ///
    /// Escape hatch for driving the raw `e5rt_*` C-ABI (priority/QoS, async
    /// events, …) on the same entitlement-free stream. The pointer is borrowed
    /// (owned by the engine); do not release it, and do not use it past the
    /// lifetime of this [`StateModel`].
    #[must_use]
    pub fn e5rt_stream_handle(&self) -> *mut c_void {
        // SAFETY: `self.raw` is a live model handle; the C side returns a
        // borrowed pointer or null.
        unsafe { sys::ane_state_e5rt_stream(self.raw) }
    }

    /// Stream id of this model's live E5RT execution stream, or `None` if no
    /// stream is available yet (see [`Self::e5rt_stream_handle`]).
    #[must_use]
    pub fn e5rt_stream_id(&self) -> Option<u64> {
        let stream = self.e5rt_stream_handle();
        if stream.is_null() {
            return None;
        }
        // SAFETY: `stream` is a live, non-null `e5rt_execution_stream*` just
        // borrowed from the engine; `get_stream_id` only reads it.
        Some(unsafe { sys::espresso::e5rt_execution_stream_get_stream_id(stream) })
    }

    /// Error unless this model's E5RT runtime is warm — i.e. at least one
    /// [`Self::predict`] has built the execution stream. The
    /// `e5rt_buffer_object_*` factory dereferences an uninitialized provider
    /// (and crashes) when called cold, so every buffer constructor gates here.
    fn ensure_e5rt_warm(&self) -> Result<()> {
        if self.e5rt_stream_handle().is_null() {
            return Err(Error {
                status: sys::AneStatus::Unsupported,
                message: "E5RT runtime not warm: call predict() at least once before \
                          creating E5RT buffers"
                    .into(),
            });
        }
        Ok(())
    }

    /// Wrap a [`Buffer`]'s `IOSurface` as a zero-copy [`E5rtBuffer`] — the same
    /// ANE-resident bytes, no copy. The result borrows both this model (whose
    /// warm runtime owns the E5RT context) and `buf` (which owns the surface),
    /// so it cannot outlive either.
    ///
    /// # Errors
    /// [`sys::AneStatus::Unsupported`] if the runtime is not warm (predict
    /// first); [`sys::AneStatus::Internal`] if the framework rejects the surface.
    pub fn wrap_buffer<'src>(&'src self, buf: &'src Buffer) -> Result<E5rtBuffer<'src>> {
        // SAFETY: `buf.iosurface_ref()` is a live surface owned by `buf`, which
        // the returned `E5rtBuffer<'src>` borrows for `'src` (via `buf: &'src _`).
        unsafe { self.wrap_iosurface(buf.iosurface_ref()) }
    }

    /// Wrap a raw `IOSurfaceRef` (as a pointer) as a zero-copy [`E5rtBuffer`].
    ///
    /// # Safety
    /// `surface` must be a live `IOSurfaceRef` that outlives the returned
    /// buffer (tie it to a value of lifetime `'b`). Prefer
    /// [`Self::wrap_buffer`].
    ///
    /// # Errors
    /// As [`Self::wrap_buffer`].
    pub unsafe fn wrap_iosurface(&self, surface: *mut c_void) -> Result<E5rtBuffer<'_>> {
        self.ensure_e5rt_warm()?;
        let mut raw: sys::espresso::BufferObject = ptr::null_mut();
        // SAFETY: runtime is warm (checked above); `surface` is a live
        // IOSurfaceRef per the contract; `raw` is written on success.
        let rc = unsafe {
            sys::espresso::e5rt_buffer_object_create_from_iosurface(&raw mut raw, surface)
        };
        E5rtBuffer::from_create(rc, raw)
    }

    /// Wrap a raw `id<MTLBuffer>` (as a pointer) as a zero-copy [`E5rtBuffer`]
    /// for GPU interop.
    ///
    /// # Safety
    /// `mtlbuffer` must be a live `id<MTLBuffer>` outliving the returned buffer.
    ///
    /// # Errors
    /// As [`Self::wrap_buffer`].
    pub unsafe fn wrap_mtlbuffer(&self, mtlbuffer: *mut c_void) -> Result<E5rtBuffer<'_>> {
        self.ensure_e5rt_warm()?;
        let mut raw: sys::espresso::BufferObject = ptr::null_mut();
        // SAFETY: runtime warm; `mtlbuffer` live per contract; `raw` written
        // on success.
        let rc = unsafe {
            sys::espresso::e5rt_buffer_object_create_from_mtlbuffer(&raw mut raw, mtlbuffer)
        };
        E5rtBuffer::from_create(rc, raw)
    }

    /// Wrap a host byte region as a zero-copy [`E5rtBuffer`].
    ///
    /// # Safety
    /// `data` must be valid for `len` bytes and outlive the returned buffer.
    ///
    /// # Errors
    /// As [`Self::wrap_buffer`].
    pub unsafe fn wrap_data(&self, data: *mut c_void, len: usize) -> Result<E5rtBuffer<'_>> {
        self.ensure_e5rt_warm()?;
        let mut raw: sys::espresso::BufferObject = ptr::null_mut();
        // SAFETY: runtime warm; `data`/`len` valid per contract; `raw` written
        // on success.
        let rc = unsafe {
            sys::espresso::e5rt_buffer_object_create_from_data_pointer(&raw mut raw, data, len)
        };
        E5rtBuffer::from_create(rc, raw)
    }

    /// Allocate a fresh ANE-resident [`E5rtBuffer`] of `nbytes` (type-0
    /// backing — the only buffer type verified). Useful as resident scratch /
    /// I/O that never leaves the ANE.
    ///
    /// # Errors
    /// [`sys::AneStatus::Unsupported`] if the runtime is not warm;
    /// [`sys::AneStatus::Internal`] on framework failure.
    pub fn alloc_buffer(&self, nbytes: usize) -> Result<E5rtBuffer<'_>> {
        self.ensure_e5rt_warm()?;
        let mut raw: sys::espresso::BufferObject = ptr::null_mut();
        // SAFETY: runtime warm; `raw` written on success; type 0 is the
        // verified backing class.
        let rc = unsafe { sys::espresso::e5rt_buffer_object_alloc(&raw mut raw, nbytes, 0) };
        E5rtBuffer::from_create(rc, raw)
    }

    /// Create a named E5RT timeline [`E5rtEvent`] on this model's warm runtime —
    /// the synchronization primitive for ordering stream work.
    ///
    /// # Errors
    /// [`sys::AneStatus::InvalidArg`] if `name` contains a NUL byte;
    /// [`sys::AneStatus::Unsupported`] if the runtime is not warm (predict
    /// first); [`sys::AneStatus::Internal`] (with the framework message) if
    /// creation fails.
    pub fn create_event(&self, name: &str) -> Result<E5rtEvent<'_>> {
        self.ensure_e5rt_warm()?;
        let c = CString::new(name).map_err(|_nul| Error {
            status: sys::AneStatus::InvalidArg,
            message: "event name contains a NUL byte".into(),
        })?;
        let mut raw: sys::espresso::AsyncEvent = ptr::null_mut();
        // SAFETY: runtime warm; `c` lives for the call; `raw` written on
        // success; flags 0 is the verified option word.
        let rc = unsafe { sys::espresso::e5rt_async_event_create(&raw mut raw, c.as_ptr(), 0) };
        e5rt_check(rc, "e5rt_async_event_create")?;
        if raw.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: "e5rt_async_event_create returned OK but null handle".into(),
            });
        }
        Ok(E5rtEvent {
            raw,
            _src: core::marker::PhantomData,
        })
    }

    /// Set the scheduling quality-of-service of this model's borrowed E5RT
    /// execution stream, influencing how subsequent ANE work on it is
    /// scheduled. `qos_class` is a Darwin `qos_class_t` (`0x21`
    /// user-interactive, `0x19` user-initiated, `0x15` default, `0x11` utility,
    /// `0x09` background).
    ///
    /// # Errors
    /// [`sys::AneStatus::Unsupported`] if the runtime is not warm (predict
    /// first); [`sys::AneStatus::Internal`] (with the framework message) on
    /// failure.
    pub fn set_stream_qos(&self, qos_class: u32) -> Result<()> {
        let stream = self.e5rt_stream_handle();
        if stream.is_null() {
            return Err(Error {
                status: sys::AneStatus::Unsupported,
                message: "E5RT stream not available: call predict() at least once first".into(),
            });
        }
        // SAFETY: `stream` is the live execution stream just borrowed from the
        // engine; the setter only adjusts its scheduling hint.
        let rc = unsafe {
            sys::espresso::e5rt_execution_stream_set_quality_of_service(stream, qos_class)
        };
        e5rt_check(rc, "e5rt_execution_stream_set_quality_of_service")
    }

    /// Set the ANE execution priority of this model's borrowed E5RT execution
    /// stream. The priority enum is not reversed (`0` verified).
    ///
    /// # Errors
    /// [`sys::AneStatus::Unsupported`] if the runtime is not warm (predict
    /// first); [`sys::AneStatus::Internal`] (with the framework message) on
    /// failure.
    pub fn set_stream_ane_priority(&self, priority: u32) -> Result<()> {
        let stream = self.e5rt_stream_handle();
        if stream.is_null() {
            return Err(Error {
                status: sys::AneStatus::Unsupported,
                message: "E5RT stream not available: call predict() at least once first".into(),
            });
        }
        // SAFETY: `stream` is the live execution stream just borrowed from the
        // engine; the setter only adjusts its priority hint.
        let rc = unsafe {
            sys::espresso::e5rt_execution_stream_set_ane_execution_priority(stream, priority)
        };
        e5rt_check(rc, "e5rt_execution_stream_set_ane_execution_priority")
    }

    /// Borrow a read-only view of the live `e5rt_program_library` the engine
    /// loaded for this model — its functions and e5 bundle path — or `None`
    /// before the first [`Self::predict`] / for a non-MLE5 model.
    #[must_use]
    pub fn e5rt_program_library(&self) -> Option<E5rtProgramLibrary<'_>> {
        // SAFETY: `self.raw` is a live model handle; the C side returns a
        // borrowed, engine-owned `e5rt_program_library*` or null.
        let raw = unsafe { sys::ane_state_e5rt_program_library(self.raw) };
        (!raw.is_null()).then_some(E5rtProgramLibrary {
            raw,
            _src: core::marker::PhantomData,
        })
    }

    /// Read-only views of the live `e5rt_execution_stream_operation`s the engine
    /// built on this model's stream (empty before the first [`Self::predict`] /
    /// for a non-MLE5 model). Each exposes its op name and I/O names.
    #[must_use]
    pub fn e5rt_operations(&self) -> Vec<E5rtOperation<'_>> {
        // SAFETY: `self.raw` is a live model handle.
        let count = unsafe { sys::ane_state_e5rt_operation_count(self.raw) };
        (0..count)
            .filter_map(|i| {
                // SAFETY: `i` is in `[0, count)`; the C side returns a borrowed,
                // engine-owned operation handle or null.
                let raw = unsafe { sys::ane_state_e5rt_operation_at(self.raw, i) };
                (!raw.is_null()).then_some(E5rtOperation {
                    raw,
                    _src: core::marker::PhantomData,
                })
            })
            .collect()
    }

    /// Build an [`E5rtRunner`] that re-drives this model's compiled operation on
    /// the ANE with your own [`E5rtBuffer`] I/O — entitlement-free inference with
    /// the resident state kept on-device, interleavable with [`Self::predict`].
    ///
    /// Uses the model's first operation. Requires at least one prior
    /// [`Self::predict`] (which builds + loads the operation).
    ///
    /// # Errors
    /// [`sys::AneStatus::Unsupported`] if the runtime is not warm or no
    /// operation is available yet (predict first).
    pub fn e5rt_runner(&self) -> Result<E5rtRunner<'_>> {
        if self.e5rt_stream_handle().is_null() {
            return Err(Error {
                status: sys::AneStatus::Unsupported,
                message: "E5RT stream not available: call predict() at least once first".into(),
            });
        }
        // SAFETY: `self.raw` is a live model handle; returns a borrowed,
        // engine-owned operation handle or null.
        let op = unsafe { sys::ane_state_e5rt_operation_at(self.raw, 0) };
        if op.is_null() {
            return Err(Error {
                status: sys::AneStatus::Unsupported,
                message: "no E5RT operation available: call predict() at least once first".into(),
            });
        }
        Ok(E5rtRunner {
            model: self,
            op,
            ports: Vec::new(),
        })
    }
}

impl Drop for StateModel {
    fn drop(&mut self) {
        // SAFETY: `self.raw` came from `ane_state_model_open` and is dropped once.
        unsafe { sys::ane_state_model_close(self.raw) };
    }
}

impl Drop for State {
    fn drop(&mut self) {
        // SAFETY: `self.raw` came from `ane_state_create` and is dropped once.
        unsafe { sys::ane_state_release(self.raw) };
    }
}

// =============================================================
// E5RT buffer objects (zero-copy ANE I/O)
// =============================================================

/// Human-readable string for a raw `e5rt_error_code_t`.
///
/// Via Espresso's own `e5rt_error_code_get_string` (e.g. `0` → `"OK"`, `3` →
/// `"MEM ALLOC FAILURE"`). Falls back to a placeholder for null / non-UTF-8.
#[must_use]
pub fn e5rt_error_string(code: i32) -> &'static str {
    // SAFETY: always safe to call; returns a pointer to a `'static` string
    // literal owned by the framework, or null.
    let p = unsafe { sys::espresso::e5rt_error_code_get_string(code) };
    if p.is_null() {
        return "unknown e5rt error";
    }
    // SAFETY: `p` is a non-null, NUL-terminated static C string from the
    // framework; valid for the program's lifetime, so `'static` is sound.
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .unwrap_or("invalid e5rt error string")
}

/// Map a non-zero `e5rt_error_code_t` to an [`Error`] carrying the framework's
/// own message (via [`e5rt_error_string`]); `Ok(())` on `0`. `what` names the
/// failing call.
fn e5rt_check(rc: sys::espresso::E5rtErrorCode, what: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    Err(Error {
        status: sys::AneStatus::Internal,
        message: format!("{what} failed: {} (code {rc})", e5rt_error_string(rc)),
    })
}

/// E5RT surface format for a `CVPixelBuffer` `OSType` 4CC (e.g. `'BGRA'` ==
/// `0x4247_5241` maps to `2`), or `None` if the framework does not recognize
/// it. A pure lookup — needs no warm runtime.
#[must_use]
pub fn surface_format_for_fourcc(fourcc: u32) -> Option<u32> {
    let mut out: u32 = 0;
    // SAFETY: `out` is a writable `u32`; the converter is a cold-safe pure
    // lookup that writes four bytes.
    let rc = unsafe { sys::espresso::e5rt_cvpb_4cc_to_surface_format(fourcc, &raw mut out) };
    (rc == 0).then_some(out)
}

/// `CVPixelBuffer` `OSType` 4CC for an E5RT surface format, or `None` if
/// unrecognized. Inverse of [`surface_format_for_fourcc`]; needs no warm
/// runtime.
#[must_use]
pub fn fourcc_for_surface_format(format: u32) -> Option<u32> {
    let mut out: u32 = 0;
    // SAFETY: `out` is a writable `u32`; cold-safe pure lookup.
    let rc = unsafe { sys::espresso::e5rt_surface_format_to_cvpb_4cc(format, &raw mut out) };
    (rc == 0).then_some(out)
}

/// An E5RT count getter: `(handle, *mut u64) -> error_code`.
type E5rtCountFn = unsafe extern "C" fn(*mut c_void, *mut u64) -> sys::espresso::E5rtErrorCode;
/// An E5RT name-list getter: `(handle, count, *mut *const c_char) -> error_code`.
type E5rtNamesFn =
    unsafe extern "C" fn(*mut c_void, u64, *mut *const c_char) -> sys::espresso::E5rtErrorCode;

/// Read an E5RT name list — `count` then a caller-allocated array of that many
/// library-owned C strings — into owned `String`s. Empty on any failure.
///
/// # Safety
/// `handle` must be a live, engine-owned handle accepted by both `count` and
/// `names`; the two functions must be a matching `get_num_* `/`get_*_names`
/// pair for it.
unsafe fn e5rt_name_list(
    handle: *mut c_void,
    count: E5rtCountFn,
    names: E5rtNamesFn,
) -> Vec<String> {
    let mut n: u64 = 0;
    // SAFETY: `handle` is live; `&raw mut n` is a writable `*mut u64`.
    if unsafe { count(handle, &raw mut n) } != 0 || n == 0 {
        return Vec::new();
    }
    let len = usize::try_from(n).unwrap_or(0);
    let mut ptrs: Vec<*const c_char> = vec![ptr::null(); len];
    // SAFETY: `handle` live; `ptrs` holds `len == n` writable slots, matching
    // the count the framework expects.
    if unsafe { names(handle, n, ptrs.as_mut_ptr()) } != 0 {
        return Vec::new();
    }
    ptrs.into_iter()
        .filter(|p| !p.is_null())
        .map(|p| {
            // SAFETY: `p` is a non-null library-owned NUL-terminated C string;
            // copied out immediately.
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        })
        .collect()
}

/// A zero-copy E5RT buffer object.
///
/// Wraps an existing `IOSurface` / `MTLBuffer` / host region, or freshly
/// allocated ANE memory, as an `e5rt_buffer_object` for the runtime beneath
/// `MLE5Engine`. Construct via [`StateModel::wrap_buffer`],
/// [`StateModel::alloc_buffer`], or the raw `wrap_*` escape hatches. Drop
/// releases the wrapper — not your surface.
///
/// The `'src` lifetime ties the wrapper to whatever owns the backing memory
/// (the source [`Buffer`] and the [`StateModel`] whose warm runtime created
/// it), so the handle cannot dangle.
pub struct E5rtBuffer<'src> {
    /// Owning `e5rt_buffer_object` handle.
    raw: sys::espresso::BufferObject,
    /// Borrows the backing memory's owner(s) for `'src`.
    _src: core::marker::PhantomData<&'src ()>,
}

// SAFETY: an `e5rt_buffer_object` is a retained provider handle with no thread
// affinity; like [`Buffer`] it can move between threads. Not `Sync` — the C
// side makes no interior-mutation guarantees.
unsafe impl Send for E5rtBuffer<'_> {}

impl core::fmt::Debug for E5rtBuffer<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("E5rtBuffer")
            .field("size", &self.size())
            .field("type", &self.raw_type())
            .finish_non_exhaustive()
    }
}

impl E5rtBuffer<'_> {
    /// Validate a `create_*` / `alloc` result (`0` code + non-null handle) and
    /// wrap it, or surface the failure.
    fn from_create(
        rc: sys::espresso::E5rtErrorCode,
        raw: sys::espresso::BufferObject,
    ) -> Result<Self> {
        if rc != 0 {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: format!(
                    "e5rt buffer creation failed: {} (code {rc})",
                    e5rt_error_string(rc)
                ),
            });
        }
        if raw.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: "e5rt buffer creation returned OK but null handle".into(),
            });
        }
        Ok(Self {
            raw,
            _src: core::marker::PhantomData,
        })
    }

    /// Byte size of the buffer (the backing surface's allocation size, which
    /// may exceed the logical request); `0` if the query fails.
    #[must_use]
    pub fn size(&self) -> usize {
        let mut out: usize = 0;
        // SAFETY: `self.raw` is a live buffer object; `out` is a writable usize.
        let rc = unsafe { sys::espresso::e5rt_buffer_object_get_size(self.raw, &raw mut out) };
        if rc == 0 { out } else { 0 }
    }

    /// Raw 32-bit type tag (enum not reversed; `0` for the verified backings).
    #[must_use]
    pub fn raw_type(&self) -> u32 {
        let mut out: u32 = 0;
        // SAFETY: live handle; `out` is a writable u32 (the call writes 4 bytes).
        let rc = unsafe { sys::espresso::e5rt_buffer_object_get_type(self.raw, &raw mut out) };
        if rc == 0 { out } else { 0 }
    }

    /// Borrowed backing `IOSurfaceRef` as a raw pointer, or null if this buffer
    /// is not `IOSurface`-backed. Do not release.
    #[must_use]
    pub fn iosurface_ref(&self) -> *mut c_void {
        let mut out: *mut c_void = ptr::null_mut();
        // SAFETY: live handle; `out` is a writable pointer slot.
        let rc = unsafe { sys::espresso::e5rt_buffer_object_get_iosurface(self.raw, &raw mut out) };
        if rc == 0 { out } else { ptr::null_mut() }
    }

    /// Borrowed backing `id<MTLBuffer>` as a raw pointer, or null if this
    /// buffer is not `MTLBuffer`-backed. Do not release.
    #[must_use]
    pub fn mtlbuffer(&self) -> *mut c_void {
        let mut out: *mut c_void = ptr::null_mut();
        // SAFETY: live handle; `out` is a writable pointer slot.
        let rc = unsafe { sys::espresso::e5rt_buffer_object_get_mtlbuffer(self.raw, &raw mut out) };
        if rc == 0 { out } else { ptr::null_mut() }
    }

    /// Host data pointer for the buffer, or null if unavailable.
    #[must_use]
    pub fn data_ptr(&self) -> *mut c_void {
        let mut out: *mut c_void = ptr::null_mut();
        // SAFETY: live handle; `out` is a writable pointer slot.
        let rc = unsafe { sys::espresso::e5rt_buffer_object_get_data_ptr(self.raw, &raw mut out) };
        if rc == 0 { out } else { ptr::null_mut() }
    }

    /// Copy `src` into the start of this buffer's host memory, returning the
    /// bytes written, or `None` if the buffer is not host-visible
    /// ([`Self::data_ptr`] null) or smaller than `src`.
    ///
    /// Safe for host-backed buffers (e.g. [`StateModel::alloc_buffer`]); the
    /// bytes are copied verbatim, so the caller picks the on-device layout.
    #[must_use]
    pub fn write_bytes(&self, src: &[u8]) -> Option<usize> {
        let dst = self.data_ptr();
        if dst.is_null() || self.size() < src.len() {
            return None;
        }
        // SAFETY: `dst` is host-visible for `size() >= src.len()` bytes (just
        // checked); `src` is valid for its length; regions do not overlap.
        unsafe { ptr::copy_nonoverlapping(src.as_ptr(), dst.cast::<u8>(), src.len()) };
        Some(src.len())
    }

    /// Copy the first `dst.len()` bytes of this buffer's host memory into `dst`,
    /// returning the bytes read, or `None` if not host-visible / too small.
    #[must_use]
    pub fn read_bytes(&self, dst: &mut [u8]) -> Option<usize> {
        let src = self.data_ptr();
        if src.is_null() || self.size() < dst.len() {
            return None;
        }
        // SAFETY: `src` is host-visible for `size() >= dst.len()` bytes; `dst`
        // is valid for its length; regions do not overlap.
        unsafe { ptr::copy_nonoverlapping(src.cast::<u8>(), dst.as_mut_ptr(), dst.len()) };
        Some(dst.len())
    }

    /// Write an `f32` slice into the buffer (native byte layout — matches an
    /// fp32 ANE tensor). `None` if not host-visible / too small. Convenience
    /// over [`Self::write_bytes`].
    #[must_use]
    pub fn write_f32(&self, src: &[f32]) -> Option<usize> {
        let len = src.len().checked_mul(core::mem::size_of::<f32>())?;
        // SAFETY: `f32` has no padding/invalid bit patterns; reading its bytes
        // is always valid, and the slice covers exactly `len` bytes of `src`.
        let bytes = unsafe { core::slice::from_raw_parts(src.as_ptr().cast::<u8>(), len) };
        self.write_bytes(bytes)
    }

    /// Read into an `f32` slice from the buffer (native byte layout). `None` if
    /// not host-visible / too small. Convenience over [`Self::read_bytes`].
    #[must_use]
    pub fn read_f32(&self, dst: &mut [f32]) -> Option<usize> {
        let len = dst.len().checked_mul(core::mem::size_of::<f32>())?;
        // SAFETY: `f32` accepts any bit pattern; the slice covers exactly `len`
        // bytes of `dst`, which we then fill from the buffer.
        let bytes = unsafe { core::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), len) };
        self.read_bytes(bytes)
    }

    /// The raw `e5rt_buffer_object` handle, for driving the rest of the
    /// `e5rt_*` C-ABI (e.g. binding it as a stream I/O port). Borrowed — do
    /// not release; the [`E5rtBuffer`] owns it.
    #[must_use]
    pub const fn as_raw(&self) -> sys::espresso::BufferObject {
        self.raw
    }
}

impl Drop for E5rtBuffer<'_> {
    fn drop(&mut self) {
        // SAFETY: `self.raw` came from a verified `create_*` / `alloc` and is
        // released exactly once here.
        unsafe { sys::espresso::e5rt_buffer_object_release(self.raw) };
    }
}

// =============================================================
// E5RT async events (timeline synchronization)
// =============================================================

/// A timeline-semaphore event for ordering work on the E5RT runtime.
///
/// Signal it with a monotonically increasing `u64` value and wait for a target
/// value (with a timeout). It is the synchronization primitive for the
/// `e5rt_execution_stream` submit path. Construct via
/// [`StateModel::create_event`]; drop releases it.
///
/// The `'src` lifetime ties the event to the [`StateModel`] whose warm runtime
/// created it. Although a timeline event is conceptually a cross-thread fence,
/// this wrapper is conservatively `Send` but not `Sync` (the C side's
/// concurrency guarantees for these entry points are unverified); use
/// [`Self::as_raw`] if you have established it is safe to drive concurrently.
pub struct E5rtEvent<'src> {
    /// Owning `e5rt_async_event` handle.
    raw: sys::espresso::AsyncEvent,
    /// Borrows the owning runtime for `'src`.
    _src: core::marker::PhantomData<&'src ()>,
}

// SAFETY: the handle is a retained event object with no thread affinity; it can
// move between threads. Deliberately not `Sync` — see the type docs.
unsafe impl Send for E5rtEvent<'_> {}

impl core::fmt::Debug for E5rtEvent<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("E5rtEvent")
            .field("last_signaled", &self.last_signaled_value())
            .finish_non_exhaustive()
    }
}

impl E5rtEvent<'_> {
    /// Signal the event, raising its last-signaled value to `value`.
    ///
    /// # Errors
    /// [`sys::AneStatus::Internal`] (with the framework message) if the signal
    /// fails.
    pub fn signal(&self, value: u64) -> Result<()> {
        // SAFETY: `self.raw` is a live event handle.
        let rc = unsafe { sys::espresso::e5rt_async_event_signal(self.raw, value) };
        e5rt_check(rc, "e5rt_async_event_signal")
    }

    /// Block until the event's value reaches `value`, or `timeout` elapses.
    ///
    /// # Errors
    /// [`sys::AneStatus::Internal`] on timeout or failure (the framework
    /// message distinguishes them).
    pub fn wait(&self, value: u64, timeout: core::time::Duration) -> Result<()> {
        let ns = u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX);
        // SAFETY: `self.raw` is a live event handle.
        let rc = unsafe { sys::espresso::e5rt_async_event_sync_wait(self.raw, value, ns) };
        e5rt_check(rc, "e5rt_async_event_sync_wait")
    }

    /// The event's last-signaled value (`0` if the query fails).
    #[must_use]
    pub fn last_signaled_value(&self) -> u64 {
        let mut out: u64 = 0;
        // SAFETY: live handle; `out` is a writable u64.
        let rc = unsafe {
            sys::espresso::e5rt_async_event_get_last_signaled_value(self.raw, &raw mut out)
        };
        if rc == 0 { out } else { 0 }
    }

    /// The event's active future (pending target) value (`0` if unset / on
    /// failure).
    #[must_use]
    pub fn active_future_value(&self) -> u64 {
        let mut out: u64 = 0;
        // SAFETY: live handle; `out` is a writable u64.
        let rc = unsafe {
            sys::espresso::e5rt_async_event_get_active_future_value(self.raw, &raw mut out)
        };
        if rc == 0 { out } else { 0 }
    }

    /// Set the event's active future (pending target) value.
    ///
    /// # Errors
    /// [`sys::AneStatus::Internal`] on failure.
    pub fn set_active_future_value(&self, value: u64) -> Result<()> {
        // SAFETY: `self.raw` is a live event handle.
        let rc =
            unsafe { sys::espresso::e5rt_async_event_set_active_future_value(self.raw, value) };
        e5rt_check(rc, "e5rt_async_event_set_active_future_value")
    }

    /// The event's name (the one it was created with), or `None` if
    /// unavailable.
    #[must_use]
    pub fn name(&self) -> Option<String> {
        let mut out: *const c_char = ptr::null();
        // SAFETY: live handle; `out` is a writable pointer slot. On success the
        // C side writes a borrowed, library-owned NUL-terminated string.
        let rc = unsafe { sys::espresso::e5rt_async_event_get_name(self.raw, &raw mut out) };
        if rc != 0 || out.is_null() {
            return None;
        }
        // SAFETY: `out` is a non-null borrowed C string owned by the event; we
        // copy it out immediately.
        Some(
            unsafe { CStr::from_ptr(out) }
                .to_string_lossy()
                .into_owned(),
        )
    }

    /// The raw `e5rt_async_event` handle, for driving the rest of the `e5rt_*`
    /// C-ABI (e.g. binding it as a stream operation's completion / dependent
    /// event). Borrowed — do not release; the [`E5rtEvent`] owns it.
    #[must_use]
    pub const fn as_raw(&self) -> sys::espresso::AsyncEvent {
        self.raw
    }
}

impl Drop for E5rtEvent<'_> {
    fn drop(&mut self) {
        // SAFETY: `self.raw` came from a verified `e5rt_async_event_create` and
        // is released exactly once here.
        unsafe { sys::espresso::e5rt_async_event_release(self.raw) };
    }
}

// =============================================================
// E5RT compute GPU device (Metal interop)
// =============================================================

/// An E5RT compute GPU device — a wrapper around an `id<MTLDevice>`.
///
/// Targets the GPU an E5RT operation runs on. Unlike buffers / events it needs
/// no warm runtime, so it is constructed directly from a device. Drop releases
/// the wrapper, not your `MTLDevice`.
pub struct E5rtGpuDevice {
    /// Owning `e5rt_compute_gpu_device` handle.
    raw: sys::espresso::GpuDevice,
}

// SAFETY: a retained device-wrapper handle with no thread affinity; it can move
// between threads. Not `Sync` — the C side makes no interior-mutation
// guarantees.
unsafe impl Send for E5rtGpuDevice {}

impl core::fmt::Debug for E5rtGpuDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("E5rtGpuDevice").finish_non_exhaustive()
    }
}

impl E5rtGpuDevice {
    /// Wrap an `id<MTLDevice>` (as a raw pointer) as an E5RT compute device. The
    /// device is retained, so you may release your own reference after this
    /// returns `Ok`.
    ///
    /// # Safety
    /// `mtl_device` must be a live `id<MTLDevice>`.
    ///
    /// # Errors
    /// [`sys::AneStatus::Internal`] (with the framework message) on failure.
    pub unsafe fn from_mtl_device(mtl_device: *mut c_void) -> Result<Self> {
        let mut raw: sys::espresso::GpuDevice = ptr::null_mut();
        // SAFETY: `mtl_device` is a live `id<MTLDevice>` per the contract; `raw`
        // is written on success.
        let rc = unsafe {
            sys::espresso::e5rt_compute_gpu_device_retain_from_mtl_device(&raw mut raw, mtl_device)
        };
        e5rt_check(rc, "e5rt_compute_gpu_device_retain_from_mtl_device")?;
        if raw.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message:
                    "e5rt_compute_gpu_device_retain_from_mtl_device returned OK but null handle"
                        .into(),
            });
        }
        Ok(Self { raw })
    }

    /// The backing `id<MTLDevice>` as a raw pointer (borrowed — do not release),
    /// or null if unavailable.
    #[must_use]
    pub fn mtl_device(&self) -> *mut c_void {
        let mut out: *mut c_void = ptr::null_mut();
        // SAFETY: `self.raw` is a live device handle; `out` is a writable
        // pointer slot.
        let rc = unsafe {
            sys::espresso::e5rt_compute_gpu_device_get_mtl_device(self.raw, &raw mut out)
        };
        if rc == 0 { out } else { ptr::null_mut() }
    }

    /// The raw `e5rt_compute_gpu_device` handle, for driving the rest of the
    /// `e5rt_*` C-ABI (e.g. overriding an operation's compute device). Borrowed
    /// — do not release; the [`E5rtGpuDevice`] owns it.
    #[must_use]
    pub const fn as_raw(&self) -> sys::espresso::GpuDevice {
        self.raw
    }
}

impl Drop for E5rtGpuDevice {
    fn drop(&mut self) {
        // SAFETY: `self.raw` came from a verified retain and is released once.
        unsafe { sys::espresso::e5rt_compute_gpu_device_release(self.raw) };
    }
}

// =============================================================
// E5RT graph introspection (read-only)
// =============================================================

/// A read-only view of the live `e5rt_program_library` an `MLE5Engine` loaded
/// for a [`StateModel`] — its functions and on-disk e5 bundle.
///
/// Borrowed from the model (the handle is engine-owned, so nothing is
/// released); `'src` keeps it from outliving the model. Obtain via
/// [`StateModel::e5rt_program_library`].
pub struct E5rtProgramLibrary<'src> {
    /// Borrowed, engine-owned `e5rt_program_library` handle.
    raw: sys::espresso::ProgramLibrary,
    /// Ties the view to the owning model for `'src`.
    _src: core::marker::PhantomData<&'src ()>,
}

// SAFETY: the handle is an engine-owned pointer with no thread affinity; the
// view only reads through it. Not `Sync`.
unsafe impl Send for E5rtProgramLibrary<'_> {}

impl core::fmt::Debug for E5rtProgramLibrary<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("E5rtProgramLibrary")
            .field("functions", &self.function_names())
            .finish_non_exhaustive()
    }
}

impl E5rtProgramLibrary<'_> {
    /// Number of functions in the library.
    #[must_use]
    pub fn num_functions(&self) -> usize {
        let mut n: u64 = 0;
        // SAFETY: `self.raw` is a live, engine-owned library; `n` is writable.
        let rc =
            unsafe { sys::espresso::e5rt_program_library_get_num_functions(self.raw, &raw mut n) };
        if rc == 0 {
            usize::try_from(n).unwrap_or(0)
        } else {
            0
        }
    }

    /// The library's function names (e.g. `["main"]`).
    #[must_use]
    pub fn function_names(&self) -> Vec<String> {
        // SAFETY: matching count / names pair for a live library handle.
        unsafe {
            e5rt_name_list(
                self.raw,
                sys::espresso::e5rt_program_library_get_num_functions,
                sys::espresso::e5rt_program_library_get_function_names,
            )
        }
    }

    /// The on-disk e5 bundle-cache path the program was compiled to, or `None`.
    #[must_use]
    pub fn bundle_path(&self) -> Option<String> {
        let mut p: *const c_char = ptr::null();
        // SAFETY: `self.raw` live; `p` is a writable pointer slot. The returned
        // string is library-owned; copied out immediately.
        let rc =
            unsafe { sys::espresso::e5rt_program_library_get_e5_bundle_path(self.raw, &raw mut p) };
        if rc != 0 || p.is_null() {
            return None;
        }
        // SAFETY: `p` is a non-null library-owned NUL-terminated C string.
        Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
    }

    /// The raw `e5rt_program_library` handle (borrowed — do not release).
    #[must_use]
    pub const fn as_raw(&self) -> sys::espresso::ProgramLibrary {
        self.raw
    }
}

/// A read-only view of one live `e5rt_execution_stream_operation` an
/// `MLE5Engine` built — its op name and I/O port names.
///
/// Borrowed from the model (engine-owned handle, nothing released); `'src`
/// keeps it from outliving the model. Obtain via [`StateModel::e5rt_operations`].
pub struct E5rtOperation<'src> {
    /// Borrowed, engine-owned `e5rt_execution_stream_operation` handle.
    raw: sys::espresso::Operation,
    /// Ties the view to the owning model for `'src`.
    _src: core::marker::PhantomData<&'src ()>,
}

// SAFETY: as [`E5rtProgramLibrary`] — engine-owned read-only handle.
unsafe impl Send for E5rtOperation<'_> {}

impl core::fmt::Debug for E5rtOperation<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("E5rtOperation")
            .field("name", &self.name())
            .field("inputs", &self.input_names())
            .field("outputs", &self.output_names())
            .finish_non_exhaustive()
    }
}

impl E5rtOperation<'_> {
    /// The operation's name (e.g. the model name), or `None`.
    #[must_use]
    pub fn name(&self) -> Option<String> {
        let mut p: *const c_char = ptr::null();
        // SAFETY: `self.raw` live; `p` is a writable pointer slot.
        let rc = unsafe {
            sys::espresso::e5rt_execution_stream_operation_get_opname(self.raw, &raw mut p)
        };
        if rc != 0 || p.is_null() {
            return None;
        }
        // SAFETY: `p` is a non-null op-owned NUL-terminated C string.
        Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
    }

    /// The operation's input feature names.
    #[must_use]
    pub fn input_names(&self) -> Vec<String> {
        // SAFETY: matching count / names pair for a live operation handle.
        unsafe {
            e5rt_name_list(
                self.raw,
                sys::espresso::e5rt_execution_stream_operation_get_num_inputs,
                sys::espresso::e5rt_execution_stream_operation_get_input_names,
            )
        }
    }

    /// The operation's output feature names.
    #[must_use]
    pub fn output_names(&self) -> Vec<String> {
        // SAFETY: matching count / names pair for a live operation handle.
        unsafe {
            e5rt_name_list(
                self.raw,
                sys::espresso::e5rt_execution_stream_operation_get_num_outputs,
                sys::espresso::e5rt_execution_stream_operation_get_output_names,
            )
        }
    }

    /// The operation's in-out (e.g. resident state) names.
    #[must_use]
    pub fn inout_names(&self) -> Vec<String> {
        // SAFETY: matching count / names pair for a live operation handle.
        unsafe {
            e5rt_name_list(
                self.raw,
                sys::espresso::e5rt_execution_stream_operation_get_num_inouts,
                sys::espresso::e5rt_execution_stream_operation_get_inout_names,
            )
        }
    }

    /// The raw `e5rt_execution_stream_operation` handle (borrowed — do not
    /// release).
    #[must_use]
    pub const fn as_raw(&self) -> sys::espresso::Operation {
        self.raw
    }
}

// =============================================================
// E5RT inference drive (reuse the engine's loaded operation)
// =============================================================

/// Drives ANE inference by re-running the engine's loaded operation.
///
/// Re-runs the operation an `MLE5Engine` already compiled + loaded, on the
/// borrowed stream, with caller-supplied [`E5rtBuffer`] I/O — entitlement-free,
/// with the resident state (KV cache) kept on-device across calls.
///
/// Build one with [`StateModel::e5rt_runner`] after at least one
/// [`StateModel::predict`] (which compiles + loads the operation). Each
/// [`Self::execute`] resets the stream, binds your input/output buffers to the
/// operation's ports by name, encodes, and runs synchronously. Inout (state)
/// ports are left bound to the engine's resident buffers, so state persists and
/// is shared with ordinary `predict` calls.
///
/// Ports are retained from the engine once per name and reused (a small,
/// bounded borrow — the engine co-owns them, and releasing an io port aborts,
/// so they are never released).
pub struct E5rtRunner<'model> {
    /// The model whose borrowed stream + operation this drives.
    model: &'model StateModel,
    /// Borrowed, engine-owned operation handle.
    op: sys::espresso::Operation,
    /// Cached retained ports: `(name, is_input, port)`.
    ports: Vec<(String, bool, sys::espresso::IoPort)>,
}

impl core::fmt::Debug for E5rtRunner<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("E5rtRunner")
            .field("ports", &self.ports.len())
            .finish_non_exhaustive()
    }
}

impl E5rtRunner<'_> {
    /// Retain (once) and cache the named input/output port.
    fn cached_port(&mut self, name: &str, is_input: bool) -> Result<sys::espresso::IoPort> {
        // Field access (`.0`/`.1`/`.2`) rather than a tuple pattern keeps the
        // `pattern_type_mismatch` restriction lint happy on the borrowed entry.
        if let Some(entry) = self
            .ports
            .iter()
            .find(|entry| entry.0 == name && entry.1 == is_input)
        {
            return Ok(entry.2);
        }
        let c = CString::new(name).map_err(|_nul| Error {
            status: sys::AneStatus::InvalidArg,
            message: "port name contains a NUL byte".into(),
        })?;
        let mut port: sys::espresso::IoPort = ptr::null_mut();
        // Pick the entry point first (safe), so the `unsafe` block holds exactly
        // one operation (the call).
        let retain = if is_input {
            sys::espresso::e5rt_execution_stream_operation_retain_input_port
        } else {
            sys::espresso::e5rt_execution_stream_operation_retain_output_port
        };
        // SAFETY: `self.op` is a live engine-owned operation; `c` lives for the
        // call; `port` is written on success. The port is engine-co-owned and
        // never released (cached and reused).
        let rc = unsafe { retain(self.op, c.as_ptr(), &raw mut port) };
        e5rt_check(rc, "e5rt_execution_stream_operation_retain_port")?;
        if port.is_null() {
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: format!("retain port '{name}' returned OK but null handle"),
            });
        }
        self.ports.push((name.to_owned(), is_input, port));
        Ok(port)
    }

    /// Run one inference step: bind `inputs` / `outputs` (by port name) to the
    /// operation's ports and execute synchronously on the ANE. The resident
    /// state is read + updated in place on-device; only the bound I/O crosses
    /// the host boundary.
    ///
    /// Write inputs into their [`E5rtBuffer`]s before calling, and read outputs
    /// after; the buffers must be sized for the operation's tensors.
    ///
    /// # Errors
    /// [`sys::AneStatus::Unsupported`] if the stream is gone (predict first);
    /// [`sys::AneStatus::InvalidArg`] for a NUL in a name;
    /// [`sys::AneStatus::Internal`] (with the framework message) if any e5rt
    /// step fails (e.g. a buffer that does not fit its port).
    pub fn execute(
        &mut self,
        inputs: &[(&str, &E5rtBuffer<'_>)],
        outputs: &[(&str, &E5rtBuffer<'_>)],
    ) -> Result<()> {
        let stream = self.bind_and_encode(inputs, outputs)?;
        // SAFETY: `stream` is live and now has encoded work.
        let exec_rc = unsafe { sys::espresso::e5rt_execution_stream_execute_sync(stream) };
        e5rt_check(exec_rc, "e5rt_execution_stream_execute_sync")
    }

    /// Submit one inference step **asynchronously**: bind `inputs` / `outputs`
    /// and `submit_async` the encoded work, returning an [`E5rtInflight`]. The
    /// host can do other work while the ANE runs; [`E5rtInflight::wait`] blocks
    /// for the result (dropping the handle also waits, so the ANE never writes
    /// into freed buffers). The runner and bound buffers stay borrowed until the
    /// handle resolves.
    ///
    /// # Errors
    /// As [`Self::execute`], plus [`sys::AneStatus::Internal`] if the async
    /// submit fails.
    pub fn execute_async<'io>(
        &'io mut self,
        inputs: &'io [(&str, &E5rtBuffer<'_>)],
        outputs: &'io [(&str, &E5rtBuffer<'_>)],
    ) -> Result<E5rtInflight<'io>> {
        let stream = self.bind_and_encode(inputs, outputs)?;
        let (tx, rx) = std::sync::mpsc::sync_channel::<bool>(1);
        let ctx: *mut c_void = Box::into_raw(Box::new(tx)).cast();
        // SAFETY: `stream` is live with encoded work; `e5rt_async_trampoline` is
        // a valid callback; `ctx` is the boxed sender it expects. The bound
        // buffers stay alive (borrowed for `'io`) until the completion fires.
        let rc =
            unsafe { sys::ane_state_e5rt_submit_async(stream, Some(e5rt_async_trampoline), ctx) };
        if rc != 0 {
            // The block never fired; reclaim the leaked sender box.
            // SAFETY: `ctx` came from `Box::into_raw` just above and was not
            // consumed (no completion ran).
            drop(unsafe { Box::from_raw(ctx.cast::<std::sync::mpsc::SyncSender<bool>>()) });
            return Err(Error {
                status: sys::AneStatus::Internal,
                message: format!("ane_state_e5rt_submit_async failed (rc {rc})"),
            });
        }
        Ok(E5rtInflight {
            rx,
            done: false,
            _io: core::marker::PhantomData,
        })
    }

    /// Reset the stream, prepare the op, bind I/O by name, and encode — the
    /// shared setup before a sync or async run. Returns the live stream handle.
    fn bind_and_encode(
        &mut self,
        inputs: &[(&str, &E5rtBuffer<'_>)],
        outputs: &[(&str, &E5rtBuffer<'_>)],
    ) -> Result<*mut c_void> {
        let stream = self.model.e5rt_stream_handle();
        if stream.is_null() {
            return Err(Error {
                status: sys::AneStatus::Unsupported,
                message: "E5RT stream not available: call predict() at least once first".into(),
            });
        }
        // SAFETY: `stream` is a live engine-owned execution stream. Reset then
        // prepare must precede binding/encoding (verified ordering).
        let reset_rc = unsafe { sys::espresso::e5rt_execution_stream_reset(stream) };
        e5rt_check(reset_rc, "e5rt_execution_stream_reset")?;
        // SAFETY: `self.op` is a live engine-owned operation.
        let prep_rc = unsafe {
            sys::espresso::e5rt_execution_stream_operation_prepare_op_for_encode(self.op)
        };
        e5rt_check(
            prep_rc,
            "e5rt_execution_stream_operation_prepare_op_for_encode",
        )?;
        // Field access (`.0`/`.1`) avoids the `pattern_type_mismatch` lint on
        // the borrowed `(name, buffer)` pairs.
        for pair in inputs {
            let port = self.cached_port(pair.0, true)?;
            // SAFETY: `port` is a live io port; `pair.1` is a live buffer object.
            // Binding after reset is the verified order.
            let bind_rc =
                unsafe { sys::espresso::e5rt_io_port_bind_buffer_object(port, pair.1.as_raw()) };
            e5rt_check(bind_rc, "e5rt_io_port_bind_buffer_object (input)")?;
        }
        for pair in outputs {
            let port = self.cached_port(pair.0, false)?;
            // SAFETY: `port` is a live io port; `pair.1` is a live buffer object.
            let bind_rc =
                unsafe { sys::espresso::e5rt_io_port_bind_buffer_object(port, pair.1.as_raw()) };
            e5rt_check(bind_rc, "e5rt_io_port_bind_buffer_object (output)")?;
        }
        // SAFETY: `stream` and `self.op` are live; the ports are bound.
        let encode_rc =
            unsafe { sys::espresso::e5rt_execution_stream_encode_operation(stream, self.op) };
        e5rt_check(encode_rc, "e5rt_execution_stream_encode_operation")?;
        Ok(stream)
    }
}

/// Completion trampoline for async E5RT submits: recover the boxed sender and
/// report success (`err` null) exactly once.
///
/// # Safety
/// `ctx` must be the `Box<SyncSender<bool>>` leaked by [`E5rtRunner::execute_async`];
/// the framework fires this exactly once, so it is reclaimed here.
unsafe extern "C" fn e5rt_async_trampoline(
    ctx: *mut c_void,
    _arg0: u64,
    _arg1: u64,
    err: *const c_void,
) {
    // SAFETY: per the contract, `ctx` is the boxed sender; reclaim and drop it.
    let tx: Box<std::sync::mpsc::SyncSender<bool>> = unsafe { Box::from_raw(ctx.cast()) };
    // Best-effort: a closed channel just means the handle was already waited on.
    tx.send(err.is_null()).unwrap_or(());
}

/// A handle to an in-flight async E5RT submission ([`E5rtRunner::execute_async`]).
///
/// The runner and bound buffers stay borrowed (`'io`) until this resolves.
/// [`Self::wait`] blocks for the result; dropping the handle also blocks, so the
/// ANE never writes into freed buffers.
pub struct E5rtInflight<'io> {
    /// Receives `true` on success (`err` null) when the completion fires.
    rx: std::sync::mpsc::Receiver<bool>,
    /// Whether the completion has already been consumed.
    done: bool,
    /// Borrows the runner + bound buffers for `'io`.
    _io: core::marker::PhantomData<&'io ()>,
}

impl E5rtInflight<'_> {
    /// Block until the ANE finishes the submitted work.
    ///
    /// # Errors
    /// [`sys::AneStatus::Internal`] if the completion reported an error or the
    /// channel closed before firing.
    pub fn wait(mut self) -> Result<()> {
        self.wait_inner()
    }

    /// Block once for the completion; idempotent (so `Drop` can also wait).
    fn wait_inner(&mut self) -> Result<()> {
        if self.done {
            return Ok(());
        }
        self.done = true;
        match self.rx.recv() {
            Ok(true) => Ok(()),
            Ok(false) => Err(Error {
                status: sys::AneStatus::Internal,
                message: "e5rt async completion reported an error".into(),
            }),
            Err(_recv) => Err(Error {
                status: sys::AneStatus::Internal,
                message: "e5rt async completion channel closed before firing".into(),
            }),
        }
    }
}

impl Drop for E5rtInflight<'_> {
    fn drop(&mut self) {
        // Block until completion so the ANE is not still writing into the
        // about-to-be-freed borrowed buffers.
        drop(self.wait_inner());
    }
}
