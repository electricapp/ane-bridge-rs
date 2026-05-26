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
// FFI-heavy code legitimately needs lossy/widening casts at the C boundary.
// The library's pointer/owner discipline is documented per-callsite via
// `// SAFETY:` comments; the broader pedantic group remains a warning.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::ptr_as_ptr
)]

use core::ffi::{CStr, c_void};
use core::ptr;
use std::ffi::CString;
use std::path::Path;
use std::sync::Arc;

/// Convert a `&Path` to a `CString`, going through the OS-string bytes
/// directly (so non-UTF-8 paths still work on Unix). On macOS this is
/// effectively a `as_os_str().as_bytes()` round-trip.
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

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.message.is_empty() {
            f.write_str(self.status.short_name())
        } else {
            write!(f, "{}: {}", self.status.short_name(), self.message)
        }
    }
}
impl std::error::Error for Error {}

/// Crate-wide `Result` alias.
pub type Result<T> = std::result::Result<T, Error>;

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
        match self {
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
        match self {
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
        match self {
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
pub fn display<T: Copy + ShortName>(v: T) -> Display<T> {
    Display(v)
}

/// Bridge version string.
pub fn version() -> &'static str {
    // SAFETY:
    // `ane_bridge_version` is a no-arg C function returning a pointer
    // to a `static const char[]` string literal in the library
    // (`#define ANE_BRIDGE_VERSION "..."`). It is never null.
    let p = unsafe { sys::ane_bridge_version() };
    debug_assert!(!p.is_null());
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
    name: String,
    dtype: Dtype,
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
    pub fn new(name: &str, dtype: Dtype, shape: &[i64]) -> Self {
        Self {
            name: name.to_string(),
            dtype,
            shape: shape.to_vec(),
        }
    }
    /// Element type.
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
    /// Borrowed shape as `[i64]`.
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }
    /// Informational tensor name.
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
    mil_path: CString,
    weights_path: CString,
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
    pub fn qos(mut self, qos: QoS) -> Self {
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
    raw: *mut sys::AneModel,
}

// SAFETY:
// The C-side `AneModel` is read-only after `ane_model_open` returns: it
// owns immutable `NSData` MIL + weights blobs, an immutable schema, and
// the compiled+loaded `_ANEInMemoryModel` + `_ANEClient`. We never
// mutate any model field after open. Apple's framework serializes
// hardware access internally when distinct threads submit through
// distinct `_ANERequest`s, which is the discipline we enforce. The
// destructor (`ane_model_close`) runs exactly once because
// `Arc::drop` only fires on the last reference.
unsafe impl Send for ModelInner {}
// SAFETY: see the `Send` impl just above — `ModelInner` is read-only
// after construction and the C side handles concurrent eval safely
// against the same model.
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
            unsafe { sys::ane_model_close(self.raw) };
        }
    }
}

/// A compiled, loaded model. Cheap to clone (`Arc`); share across threads.
#[derive(Clone)]
pub struct Model {
    inner: Arc<ModelInner>,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("num_inputs", &self.num_inputs())
            .field("num_outputs", &self.num_outputs())
            .finish()
    }
}

impl Model {
    /// Compile and load a model.
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
    pub fn num_inputs(&self) -> i32 {
        // SAFETY:
        // `inner.raw` is a valid model handle for the lifetime of
        // `self.inner` (released only in `ModelInner::drop`).
        // `ane_model_num_inputs` takes a `const`-qualified pointer
        // and performs no mutation; the function is reentrant.
        unsafe { sys::ane_model_num_inputs(self.inner.raw) }
    }
    /// Number of declared outputs.
    pub fn num_outputs(&self) -> i32 {
        // SAFETY: see `num_inputs`.
        unsafe { sys::ane_model_num_outputs(self.inner.raw) }
    }
    /// Total bytes of input `idx`. 0 if `idx` is out of range.
    pub fn input_nbytes(&self, idx: i32) -> usize {
        // SAFETY: see `num_inputs`; the C side range-checks `idx`.
        unsafe { sys::ane_model_input_nbytes(self.inner.raw, idx) }
    }
    /// Total bytes of output `idx`. 0 if `idx` is out of range.
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
    pub fn was_cached(&self) -> bool {
        // SAFETY: see `num_inputs`; the C accessor is a pure read of a
        // bool set at open time and never mutated afterwards.
        unsafe { sys::ane_model_was_cached(self.inner.raw) }
    }

    /// Framework-derived schema for input `idx`. Returns `None` if
    /// the index is out of range.
    pub fn input(&self, idx: i32) -> Option<TensorSpec> {
        // SAFETY: const pointer to a read-only model; C does the range check.
        let p = unsafe { sys::ane_model_input_spec(self.inner.raw, idx) };
        tensor_spec_from_raw(p)
    }
    /// Framework-derived schema for output `idx`.
    pub fn output(&self, idx: i32) -> Option<TensorSpec> {
        // SAFETY: see `input`.
        let p = unsafe { sys::ane_model_output_spec(self.inner.raw, idx) };
        tensor_spec_from_raw(p)
    }

    /// Allocate a fresh request bound to this model. The model is kept
    /// alive (via `Arc`) for the lifetime of the request.
    pub fn request(&self) -> Result<Request> {
        let n_in = self.num_inputs().max(0) as usize;
        let n_out = self.num_outputs().max(0) as usize;
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
        Ok(Request {
            raw,
            _model: self.clone(),
            callback: None,
            bound_inputs,
            bound_outputs,
        })
    }

    /// Allocate a fresh `IOSurface`-backed buffer of `nbytes`.
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
}

// =============================================================
// Buffer
// =============================================================

/// `IOSurface`-backed buffer. Drop releases the underlying surface.
pub struct Buffer {
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
    /// Buffer size in bytes.
    pub fn nbytes(&self) -> usize {
        // SAFETY: `self.raw` is non-null and owned by `self`.
        unsafe { sys::ane_buffer_nbytes(self.raw) }
    }
    /// Underlying `IOSurfaceID` for advanced inter-process use.
    pub fn iosurface_id(&self) -> u32 {
        // SAFETY: see `nbytes`.
        unsafe { sys::ane_buffer_iosurface_id(self.raw) }
    }

    /// Run `f` while the buffer is locked for host access. The slice
    /// passed to `f` is the entire mapped `IOSurface`; do not let it
    /// escape the closure — it becomes invalid after unlock.
    pub fn with_locked<R>(
        &mut self,
        access: BufferAccess,
        body: impl FnOnce(&mut [u8]) -> R,
    ) -> Result<R> {
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
        debug_assert!(!base_ptr.is_null());
        Ok(LockGuard {
            buf: self,
            base: base_ptr.cast::<u8>(),
            len: size,
        })
    }

    /// Internal: raw handle for binding into a request.
    pub(crate) fn raw(&self) -> *mut sys::AneBuffer {
        self.raw
    }
}

/// RAII guard returned by [`Buffer::lock`]. Derefs to `&mut [u8]`;
/// drops release the underlying `IOSurface` lock.
///
/// The guard borrows the buffer mutably, so the borrow checker
/// prevents concurrent locking or rebinding for its lifetime.
pub struct LockGuard<'a> {
    buf: &'a mut Buffer,
    base: *mut u8,
    len: usize,
}

impl LockGuard<'_> {
    /// Length of the locked mapping in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }
    /// Returns `true` if the mapping is zero-sized (would only happen
    /// for a buffer allocated with `nbytes == 0`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl core::ops::Deref for LockGuard<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        // SAFETY:
        // `self.base` was produced by `ane_buffer_lock` on
        // `self.buf.raw`, which we hold via `&'a mut Buffer`. The
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
        let _ = unsafe { sys::ane_buffer_unlock(self.buf.raw) };
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
            unsafe { sys::ane_buffer_release(self.raw) };
        }
    }
}

// =============================================================
// Request
// =============================================================

/// Type-erased completion closure. Boxed and pointed to by the C-side
/// `user` arg; freed when the `Request` is dropped or a new callback
/// replaces it.
type Callback = Box<dyn FnMut(std::result::Result<(), Error>) + Send + 'static>;

/// One in-flight (or idle) inference unit. Each request owns a serial
/// dispatch queue on the C side; submit dispatches an eval onto it.
pub struct Request {
    raw: *mut sys::AneRequest,
    // Keep the parent model alive for at least as long as the request.
    _model: Model,
    // Owning box for the registered completion closure (if any). We
    // keep a *mut so the address handed to C is stable across moves;
    // freed in `Drop` or when replaced.
    callback: Option<*mut Callback>,
    // Bindings: Rust must keep the `Buffer` alive for the entire
    // lifetime of the request, since the C side stores raw pointers
    // into them in `r->input_bound`/`r->output_bound` and dereferences
    // those on every `submit`. Without this, a caller could drop the
    // buffer immediately after `bind_*` returned and trigger a UAF.
    //
    // The vectors are sized once at request creation and are
    // append-replaced (each `bind_*(idx, buf)` overwrites slot `idx`).
    bound_inputs: Vec<Option<Buffer>>,
    bound_outputs: Vec<Option<Buffer>>,
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
    pub fn bind_input(&mut self, idx: i32, buf: Buffer) -> Result<()> {
        // SAFETY:
        // - `self.raw` is a valid request handle.
        // - `buf.raw()` is the buffer's stable handle; we move `buf`
        //   into `self.bound_inputs[..]` on success so the pointer
        //   stays valid for the Request's lifetime.
        // - On error we drop `buf` here and the C side never recorded
        //   any pointer to it (the C check happens before the store).
        unsafe { check(sys::ane_request_bind_input(self.raw, idx, buf.raw()))? };
        // The C check just confirmed `0 <= idx < num_inputs`, and
        // `bound_inputs.len() == num_inputs`, so the cast + index is
        // in-bounds. We use `debug_assert` to keep a non-panic
        // tripwire if those invariants ever drift.
        let slot = idx as usize;
        debug_assert!(slot < self.bound_inputs.len());
        // Replace last so the previous Buffer's Drop fires only after
        // the C side has rebound to the new pointer.
        self.bound_inputs[slot] = Some(buf);
        Ok(())
    }

    /// Bind a buffer for output `idx`. Same ownership rules as
    /// [`Self::bind_input`].
    pub fn bind_output(&mut self, idx: i32, buf: Buffer) -> Result<()> {
        // SAFETY: see `bind_input`.
        unsafe { check(sys::ane_request_bind_output(self.raw, idx, buf.raw()))? };
        let slot = idx as usize;
        debug_assert!(slot < self.bound_outputs.len());
        self.bound_outputs[slot] = Some(buf);
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
    pub fn submit(&mut self, qos: QoS) -> Result<()> {
        // SAFETY:
        // `self.raw` is valid. The C-side atomic `in_flight` rejects
        // concurrent submits with `Busy`. The dispatched block keeps
        // the request alive until its own internal flag is set, and
        // `Drop` flushes the queue before releasing.
        unsafe { check(sys::ane_request_submit(self.raw, qos)) }
    }

    /// Block until the in-flight submission completes.
    pub fn wait(&mut self, timeout_ms: i32) -> Result<()> {
        // SAFETY: `self.raw` valid; semaphore wait is internally safe.
        unsafe { check(sys::ane_request_wait(self.raw, timeout_ms)) }
    }

    /// Non-blocking completion check.
    pub fn is_done(&self) -> bool {
        // SAFETY: const pointer to a request we own; atomic load only.
        unsafe { sys::ane_request_is_done(self.raw) }
    }

    /// Submit + wait. Returns when the eval is done (or has failed).
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
        let shared = std::sync::Arc::new(EvalShared::new());
        let shared_cb = shared.clone();
        self.on_complete(move |result| {
            // Store the result first, then publish `done`. The future
            // uses Acquire on `done` to synchronize-with this Release,
            // so it always observes a fully-written `result`.
            *shared_cb.result.lock().expect("eval result mutex poisoned") = Some(result);
            shared_cb
                .done
                .store(true, core::sync::atomic::Ordering::Release);
            // Take the waker (if any) outside the lock, then wake.
            let waker = shared_cb
                .waker
                .lock()
                .expect("eval waker mutex poisoned")
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
    pub fn on_complete<F>(&mut self, f: F) -> Result<()>
    where
        F: FnMut(std::result::Result<(), Error>) + Send + 'static,
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
    done: core::sync::atomic::AtomicBool,
    waker: std::sync::Mutex<Option<core::task::Waker>>,
    result: std::sync::Mutex<Option<std::result::Result<(), Error>>>,
}

impl EvalShared {
    fn new() -> Self {
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
pub struct EvalFuture<'r> {
    _request: &'r mut Request,
    shared: std::sync::Arc<EvalShared>,
}

impl core::future::Future for EvalFuture<'_> {
    type Output = std::result::Result<(), Error>;

    fn poll(
        self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Self::Output> {
        // Fast path: the callback already ran before this poll.
        if self.shared.done.load(core::sync::atomic::Ordering::Acquire) {
            let r = self
                .shared
                .result
                .lock()
                .expect("eval result mutex poisoned")
                .take()
                .unwrap_or(Ok(()));
            return core::task::Poll::Ready(r);
        }
        // Register the current waker so the callback can wake us when
        // it runs. We re-check `done` after the store to close the
        // race where the callback fires between the first check and
        // the waker store.
        *self.shared.waker.lock().expect("eval waker mutex poisoned") = Some(cx.waker().clone());
        if self.shared.done.load(core::sync::atomic::Ordering::Acquire) {
            let r = self
                .shared
                .result
                .lock()
                .expect("eval result mutex poisoned")
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
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (cb_ref)(result)));
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
            unsafe { sys::ane_request_release(self.raw) };
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
