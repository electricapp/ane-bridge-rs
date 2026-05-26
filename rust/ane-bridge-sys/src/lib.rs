//! Raw FFI bindings to `libane_bridge`. See `ane_bridge.h` for the contract.
//!
//! Everything here is `extern "C"` and unsafe to call by definition: the
//! bindings reflect the C API verbatim. For a safe wrapper (`Result`-based,
//! RAII handle types, mapped buffer access) use the `ane-bridge` crate.
//!
//! # Pointer / lifetime rules at this layer
//!
//! - `*const T` arguments to the C API are read once during the call. Their
//!   memory must be valid for the duration of that call only — the C side
//!   copies any state it intends to retain.
//! - Returned `*mut AneModel`, `*mut AneBuffer`, `*mut AneRequest` are
//!   library-allocated. Each has a matching `..._release` / `..._close`
//!   function and must be freed exactly once.
//! - The `Drop` glue for any of these handles lives in `ane-bridge`, not
//!   here. Callers of this crate are responsible for releasing.
//! - String pointers returned by `ane_last_error` / `ane_bridge_version`
//!   are owned by the library and remain valid until the next call on
//!   the same thread (errors are thread-local) or program exit (version).
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
// FFI `repr(C)` enums and pointer types make many pedantic lints noisy.
// These specific allows are scoped to this crate and justified inline.
#![allow(
    clippy::missing_safety_doc,        // every `extern "C"` is unsafe by definition; docs live on the C header
    clippy::upper_case_acronyms,       // AneQoS / AneIO follow Apple naming
)]

use core::ffi::{c_char, c_int, c_void};

/// Boolean type returned by C functions. `bool` is ABI-compatible with C's
/// `_Bool` on the platforms we target (macOS / clang).
pub type c_bool = bool;

/// Status codes returned by every fallible C entry point.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AneStatus {
    /// Successful completion.
    Ok = 0,
    /// One of the caller's arguments was malformed or out of range.
    InvalidArg = 1,
    /// Filesystem read/write failure (MIL / weights / temp dir).
    Io = 2,
    /// `_ANEInMemoryModel compileWithQoS:` rejected the program.
    Compile = 3,
    /// `_ANEInMemoryModel loadWithQoS:` could not place the model on ANE.
    Load = 4,
    /// `_ANEClient doEvaluateDirectWithModel:` failed at runtime.
    Eval = 5,
    /// Allocation failure (`calloc` / `IOSurfaceCreate`).
    Oom = 6,
    /// Framework not loadable / required private symbol missing.
    Unsupported = 7,
    /// `ane_request_wait` exceeded its timeout.
    Timeout = 8,
    /// `ane_request_submit` called while a prior submission is in flight.
    Busy = 9,
    /// `ane_request_wait(0)` polled a request that has not yet completed.
    NotDone = 10,
    /// Catch-all for unreachable invariants violated by the runtime.
    Internal = 99,
}

/// Tensor element type. Sizes match the `ane_dtype_size` helper.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AneDtype {
    /// 32-bit IEEE float.
    Fp32 = 1,
    /// 16-bit half (ANE's native dtype).
    Fp16 = 2,
    /// 32-bit signed integer.
    Int32 = 3,
    /// 64-bit signed integer.
    Int64 = 4,
    /// Unsigned byte.
    UInt8 = 5,
    /// Signed byte.
    Int8 = 6,
}

/// Quality-of-service hint passed through to `_ANEClient`'s eval call.
/// Numeric values match the private framework's expected encoding; do not
/// reinterpret as POSIX `qos_class_t`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AneQoS {
    /// Library default — what existing direct-MIL benchmarks use.
    Default = 21,
    /// Latency-sensitive foreground work.
    UserInteractive = 33,
    /// User-initiated work.
    UserInitiated = 25,
    /// Background utility work.
    Utility = 17,
    /// Lowest priority.
    Background = 9,
}

/// Lock mode for `ane_buffer_lock`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AneBufferAccess {
    /// Read-only host access; allows ANE to keep ownership.
    Read = 1,
    /// Write-only host access.
    Write = 2,
    /// Read-write access.
    ReadWrite = 3,
}

/// One row of the tensor schema. The pointed-to buffers do not need to
/// outlive the `ane_model_open` call — the library copies them out.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AneTensorSpec {
    /// Informational name. May be `NULL`.
    pub name: *const c_char,
    /// Element dtype.
    pub dtype: AneDtype,
    /// Number of axes in `shape`.
    pub rank: i32,
    /// Pointer to `rank` `i64` elements giving the extent of each axis.
    pub shape: *const i64,
}

/// Arguments to `ane_model_open`. Input/output specs are derived from
/// the framework after `loadWithQoS:`; they are not user-supplied.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AneModelOpenOptions {
    /// Path to a MIL text file.
    pub mil_path: *const c_char,
    /// Path to the single weights blob referenced from MIL.
    pub weights_path: *const c_char,
    /// `QoS` for compile + load.
    pub compile_qos: AneQoS,
}

/// Opaque model handle. Use `*mut AneModel` / `*const AneModel`.
pub enum AneModel {}
/// Opaque IOSurface-backed buffer.
pub enum AneBuffer {}
/// Opaque request (one set of bindings + worker queue).
pub enum AneRequest {}

/// Completion callback signature: `(request, status, user_ptr)`. Invoked
/// from a library-owned worker thread; do not re-enter the library with
/// the same request from within the callback.
pub type AneCompletionFn =
    unsafe extern "C" fn(req: *mut AneRequest, status: AneStatus, user: *mut c_void);

unsafe extern "C" {
    /// Returns a static UTF-8 version string ("0.1.0" etc.). Never null.
    pub fn ane_bridge_version() -> *const c_char;

    /// Returns the thread-local last error message, or "" if none. The
    /// returned pointer is only valid until the next library call on
    /// this thread.
    pub fn ane_last_error() -> *const c_char;

    /// Returns the size in bytes of one element of `dt`, or 0 if unknown.
    pub fn ane_dtype_size(dt: AneDtype) -> usize;

    /// Compile + load a model. On success writes a non-null model handle to `*out`.
    pub fn ane_model_open(opts: *const AneModelOpenOptions, out: *mut *mut AneModel) -> AneStatus;
    /// Unload and free the model. Safe on null.
    pub fn ane_model_close(model: *mut AneModel);

    /// Number of declared input tensors.
    pub fn ane_model_num_inputs(model: *const AneModel) -> i32;
    /// Number of declared output tensors.
    pub fn ane_model_num_outputs(model: *const AneModel) -> i32;
    /// Returns a pointer to the input spec at `idx`, or null if out of range.
    /// Lifetime: valid until `ane_model_close(model)`.
    pub fn ane_model_input_spec(model: *const AneModel, idx: i32) -> *const AneTensorSpec;
    /// Returns a pointer to the output spec at `idx`. Same lifetime rules
    /// as `ane_model_input_spec`.
    pub fn ane_model_output_spec(model: *const AneModel, idx: i32) -> *const AneTensorSpec;
    /// Total byte size of input `idx`. 0 if out of range.
    pub fn ane_model_input_nbytes(model: *const AneModel, idx: i32) -> usize;
    /// Total byte size of output `idx`. 0 if out of range.
    pub fn ane_model_output_nbytes(model: *const AneModel, idx: i32) -> usize;

    /// True if `ane_model_open` reused a cached lowered artifact instead
    /// of running a fresh compile. The cache lives in the aned daemon
    /// keyed by content hash; it survives across processes but not aned
    /// restarts or its opaque eviction policy.
    pub fn ane_model_was_cached(model: *const AneModel) -> bool;

    /// Allocate a fresh `nbytes`-sized IOSurface-backed buffer.
    pub fn ane_buffer_create(nbytes: usize, out: *mut *mut AneBuffer) -> AneStatus;
    /// Convenience: allocate a buffer sized for input `idx` of `m`.
    pub fn ane_buffer_create_for_input(
        m: *const AneModel,
        idx: i32,
        out: *mut *mut AneBuffer,
    ) -> AneStatus;
    /// Convenience: allocate a buffer sized for output `idx` of `m`.
    pub fn ane_buffer_create_for_output(
        m: *const AneModel,
        idx: i32,
        out: *mut *mut AneBuffer,
    ) -> AneStatus;
    /// Release a buffer. Safe on null.
    pub fn ane_buffer_release(buf: *mut AneBuffer);
    /// Lock the buffer for host access; writes the mapped base pointer to `*out_ptr`.
    pub fn ane_buffer_lock(
        buf: *mut AneBuffer,
        access: AneBufferAccess,
        out_ptr: *mut *mut c_void,
    ) -> AneStatus;
    /// Pair to the most recent successful `ane_buffer_lock` on the same buffer.
    pub fn ane_buffer_unlock(buf: *mut AneBuffer) -> AneStatus;
    /// Buffer size in bytes.
    pub fn ane_buffer_nbytes(buf: *const AneBuffer) -> usize;
    /// Returns the `IOSurface` ID for cross-process sharing, or 0 if none.
    pub fn ane_buffer_iosurface_id(buf: *const AneBuffer) -> u32;

    /// Allocate a request bound to `model`.
    pub fn ane_request_create(model: *mut AneModel, out: *mut *mut AneRequest) -> AneStatus;
    /// Drain any in-flight eval, then release the request.
    pub fn ane_request_release(req: *mut AneRequest);
    /// Zero-copy bind: associate `buf` with input slot `idx`.
    pub fn ane_request_bind_input(req: *mut AneRequest, idx: i32, buf: *mut AneBuffer)
    -> AneStatus;
    /// Zero-copy bind: associate `buf` with output slot `idx`.
    pub fn ane_request_bind_output(
        req: *mut AneRequest,
        idx: i32,
        buf: *mut AneBuffer,
    ) -> AneStatus;
    /// Fast-path: memcpy `nbytes` of `data` into the internal input buffer for `idx`.
    pub fn ane_request_set_input_bytes(
        req: *mut AneRequest,
        idx: i32,
        data: *const c_void,
        nbytes: usize,
    ) -> AneStatus;
    /// Fast-path: memcpy `nbytes` of the completed output `idx` into `data`.
    pub fn ane_request_get_output_bytes(
        req: *mut AneRequest,
        idx: i32,
        data: *mut c_void,
        nbytes: usize,
    ) -> AneStatus;
    /// Submit asynchronously. Returns `Busy` if a prior submission is still in flight.
    pub fn ane_request_submit(req: *mut AneRequest, qos: AneQoS) -> AneStatus;
    /// Wait up to `timeout_ms` for the in-flight submission to complete.
    /// `< 0` = forever, `0` = poll.
    pub fn ane_request_wait(req: *mut AneRequest, timeout_ms: c_int) -> AneStatus;
    /// Non-blocking completion check.
    pub fn ane_request_is_done(req: *const AneRequest) -> c_bool;
    /// Convenience: submit + wait(-1).
    pub fn ane_request_run(req: *mut AneRequest, qos: AneQoS) -> AneStatus;
    /// Install (or clear, with `fn_ = None`) a completion callback.
    pub fn ane_request_set_completion(
        req: *mut AneRequest,
        fn_: Option<AneCompletionFn>,
        user: *mut c_void,
    ) -> AneStatus;
    /// Per-request error message captured by the worker thread on the
    /// most recent eval. Returns `""` if no error or no submission yet.
    pub fn ane_request_last_error(req: *const AneRequest) -> *const c_char;

    /// Internal: run a single adversarial case through the C-side
    /// `LiveInputList` parser. Not stable API.
    pub fn _ane_internal_fuzz_parse_one(fc: *const AneFuzzCase) -> AneStatus;
    /// Internal: drive the outer `modelAttributes` parser. Not stable API.
    pub fn _ane_internal_fuzz_parse_attrs(fc: *const AneFuzzAttrsCase) -> AneStatus;
    /// Internal: feed the leaf parser a `Batches` that wraps an
    /// `NSNumber(double)` to test `longLongValue` edge cases.
    pub fn _ane_internal_fuzz_parse_one_with_double_batches(dbl: f64) -> AneStatus;
    /// Internal: replace dim `which_dim` (0..=4) with `NSNumber(double)`.
    pub fn _ane_internal_fuzz_dim_as_double(which_dim: i32, dbl: f64) -> AneStatus;
    /// Internal: replace dim with `NSNumber(uint64_t)` — tests signed/unsigned
    /// boundary in `longLongValue`.
    pub fn _ane_internal_fuzz_dim_as_uint64(which_dim: i32, value: u64) -> AneStatus;
    /// Internal: replace dim with an `NSDecimalNumber`.
    pub fn _ane_internal_fuzz_dim_as_decimal(
        which_dim: i32,
        decimal_str: *const c_char,
    ) -> AneStatus;
    /// Internal: replace one dict value with `NSNull`. `which_key`:
    /// 0=Name, 1=Type, 2..=6=dims in Batches/Channels/Depth/Height/Width order.
    pub fn _ane_internal_fuzz_value_is_nsnull(which_key: i32) -> AneStatus;
    /// Internal: name has an embedded NUL byte.
    pub fn _ane_internal_fuzz_name_with_embedded_nul() -> AneStatus;
    /// Internal: name of length `length` (capped internally).
    pub fn _ane_internal_fuzz_huge_name(length: usize) -> AneStatus;
    /// Internal: two-entry list, one valid + one invalid. Whole list
    /// must reject, not partially accept.
    pub fn _ane_internal_fuzz_mixed_validity_two_entries() -> AneStatus;
}

/// Mutation bitmask for [`AneFuzzAttrsCase`]. Mirrors `AneFuzzAttrsMutation`.
pub mod fuzz_attrs {
    /// Replace `NetworkStatusList` with a non-array.
    pub const NSL_NOT_ARRAY: u32 = 1 << 0;
    /// Omit `NetworkStatusList` entirely.
    pub const NSL_MISSING: u32 = 1 << 1;
    /// `NetworkStatusList` is empty.
    pub const NSL_EMPTY: u32 = 1 << 2;
    /// `NetworkStatusList[0]` is not a dictionary.
    pub const PROC_NOT_DICT: u32 = 1 << 3;
    /// Omit `LiveInputList`.
    pub const LIVEIN_MISSING: u32 = 1 << 4;
    /// Omit `LiveOutputList`.
    pub const LIVEOUT_MISSING: u32 = 1 << 5;
    /// `LiveInputList` is not an array.
    pub const LIVEIN_NOT_ARRAY: u32 = 1 << 6;
    /// `LiveOutputList` is not an array.
    pub const LIVEOUT_NOT_ARRAY: u32 = 1 << 7;
}

/// Adversarial input for [`_ane_internal_fuzz_parse_attrs`]. Layout
/// must match the C struct.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AneFuzzAttrsCase {
    /// Bitmask, see `fuzz_attrs`.
    pub mutations: u32,
    /// Number of well-formed input entries to synthesize.
    pub n_inputs: i32,
    /// Number of well-formed output entries to synthesize.
    pub n_outputs: i32,
}

const _: () = {
    assert!(core::mem::size_of::<AneFuzzAttrsCase>() == 12);
    assert!(core::mem::offset_of!(AneFuzzAttrsCase, mutations) == 0);
    assert!(core::mem::offset_of!(AneFuzzAttrsCase, n_inputs) == 4);
    assert!(core::mem::offset_of!(AneFuzzAttrsCase, n_outputs) == 8);
};

/// Field-presence bitmask used by [`AneFuzzCase`].
pub mod fuzz_field {
    /// `Name` key present.
    pub const NAME: u32 = 1 << 0;
    /// `Type` key present.
    pub const TYPE: u32 = 1 << 1;
    /// `Batches` key present.
    pub const BATCHES: u32 = 1 << 2;
    /// `Channels` key present.
    pub const CHANNELS: u32 = 1 << 3;
    /// `Depth` key present.
    pub const DEPTH: u32 = 1 << 4;
    /// `Height` key present.
    pub const HEIGHT: u32 = 1 << 5;
    /// `Width` key present.
    pub const WIDTH: u32 = 1 << 6;
    /// All dimension + name/type keys present.
    pub const ALL: u32 = 0x7F;
}

/// Adversarial input to [`_ane_internal_fuzz_parse_one`].
///
/// The `present_mask` controls which keys appear in the synthesized
/// dict; `flags` bits swap a value's type (`NSString` for dims,
/// `NSNumber` for name/type) so we can test type-confusion paths.
///
/// Field order **must** match the C definition in `ane_bridge.h`.
/// Verified at compile time below.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AneFuzzCase {
    /// Which keys to include (see `fuzz_field`).
    pub present_mask: u32,
    /// Bitfield: when set, the corresponding dim/name/type value is
    /// replaced by a wrong-type `NSObject` (`NSString` for dims,
    /// `NSNumber` for name/type) so we can test type-confusion
    /// robustness.
    pub flags: u32,
    /// Name value (UTF-8, ignored unless `NAME` bit is set).
    pub name: *const c_char,
    /// Type value (e.g., "Float32"), ignored unless `TYPE` bit set.
    pub type_string: *const c_char,
    /// Dimension values used when their bit is set in `present_mask`.
    pub batches: i64,
    /// See `batches`.
    pub channels: i64,
    /// See `batches`.
    pub depth: i64,
    /// See `batches`.
    pub height: i64,
    /// See `batches`.
    pub width: i64,
}

/// Compile-time guard against drift between this Rust mirror and the
/// C struct layout. If anything changes on either side, this check
/// breaks the build instead of silently misinterpreting fields.
const _: () = {
    assert!(core::mem::size_of::<AneFuzzCase>() == 64);
    assert!(core::mem::offset_of!(AneFuzzCase, present_mask) == 0);
    assert!(core::mem::offset_of!(AneFuzzCase, flags) == 4);
    assert!(core::mem::offset_of!(AneFuzzCase, name) == 8);
    assert!(core::mem::offset_of!(AneFuzzCase, type_string) == 16);
    assert!(core::mem::offset_of!(AneFuzzCase, batches) == 24);
    assert!(core::mem::offset_of!(AneFuzzCase, channels) == 32);
    assert!(core::mem::offset_of!(AneFuzzCase, depth) == 40);
    assert!(core::mem::offset_of!(AneFuzzCase, height) == 48);
    assert!(core::mem::offset_of!(AneFuzzCase, width) == 56);
};

impl AneFuzzCase {
    /// Flag: `batches` slot is an `NSString` instead of `NSNumber`.
    pub const FLAG_BATCHES_AS_STRING: u32 = 1 << 0;
    /// Flag: `channels` slot is an `NSString` instead of `NSNumber`.
    pub const FLAG_CHANNELS_AS_STRING: u32 = 1 << 1;
    /// Flag: `depth` slot is an `NSString` instead of `NSNumber`.
    pub const FLAG_DEPTH_AS_STRING: u32 = 1 << 2;
    /// Flag: `height` slot is an `NSString` instead of `NSNumber`.
    pub const FLAG_HEIGHT_AS_STRING: u32 = 1 << 3;
    /// Flag: `width` slot is an `NSString` instead of `NSNumber`.
    pub const FLAG_WIDTH_AS_STRING: u32 = 1 << 4;
    /// Flag: `Name` slot is an `NSNumber` instead of `NSString`.
    pub const FLAG_NAME_AS_NUMBER: u32 = 1 << 5;
    /// Flag: `Type` slot is an `NSNumber` instead of `NSString`.
    pub const FLAG_TYPE_AS_NUMBER: u32 = 1 << 6;
}
