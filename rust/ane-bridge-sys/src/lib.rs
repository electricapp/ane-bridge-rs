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
/// Opaque hardware-counters object populated by `_ANEPerformanceStats`.
/// Pass to a [`AneRequest`] before submit; read counters back after wait.
pub enum AnePerfStats {}
/// Opaque container of signal + wait events for GPU↔ANE synchronization.
/// Wraps `_ANESharedEvents`.
pub enum AneSharedEvents {}
/// Opaque pipeline that submits multiple linked models as one ANE dispatch.
/// Wraps an `NSArray<_ANEChainingRequest*>`.
pub enum AneChain {}

/// Device-level facts about the ANE on this machine, returned by
/// [`ane_device_info`]. Backed by `_ANEVirtualClient`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AneDeviceInfo {
    /// Number of ANE compute cores (e.g. 16 on M4 H16G).
    pub num_cores: i32,
    /// Number of ANE units (typically 1).
    pub num_anes: i32,
    /// True if the framework reports an ANE present on this device.
    pub has_ane: bool,
    /// Board type identifier (opaque).
    pub board_type: i64,
    /// Null-terminated UTF-8 architecture string (e.g. `"H16G"`).
    pub arch_type: [c_char; 32],
}

/// One named weight blob for the multi-blob open path.
/// Exactly one of (`path`) or (`bytes`, `nbytes`) must be set.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AneWeightEntry {
    /// Dictionary key the MIL references this blob by. Required.
    pub name: *const c_char,
    /// File path to read the blob from, or null if `bytes` is set.
    pub path: *const c_char,
    /// In-memory blob, or null if `path` is set.
    pub bytes: *const c_void,
    /// Length of `bytes`. Ignored when `path` is set.
    pub nbytes: usize,
}

/// Extended open options: in-memory MIL text + `NSDictionary` weights.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AneModelOpenOptionsEx {
    /// Path to a MIL text file, or null when `mil_bytes` is supplied.
    pub mil_path: *const c_char,
    /// In-memory MIL text, or null when `mil_path` is supplied.
    pub mil_bytes: *const c_void,
    /// Byte count for `mil_bytes`; ignored if `mil_path` is set.
    pub mil_nbytes: usize,
    /// Pointer to an array of `n_weights` named blob descriptors.
    pub weights: *const AneWeightEntry,
    /// Length of the `weights` array (0 means weight-less).
    pub n_weights: i32,
    /// `QoS` for compile + load.
    pub compile_qos: AneQoS,
    /// When false the bytes are treated as a `NetworkDescription`
    /// (non-MIL legacy input format).
    pub is_mil_model: bool,
    /// Optional options-plist JSON; null for none.
    pub options_plist_json: *const c_char,
}

/// Selects how `_ANEModel` derives the cache identity for a file-based open.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AneIdentifierSource {
    /// Framework default.
    Default = 0,
    /// Derive identity from the model URL.
    Url = 1,
    /// Derive identity from an explicit UUID.
    Uuid = 2,
    /// Derive identity from content hash.
    Content = 3,
}

/// Options for the file-based open path (`_ANEModel modelAtURL:key:`).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AneModelFileOpenOptions {
    /// Compiled `.mlmodelc` or `.bundle` URL or path. Required.
    pub model_url: *const c_char,
    /// Explicit cache key. Null derives one from the URL.
    pub cache_key: *const c_char,
    /// Optional `cacheURLIdentifier` for cache placement.
    pub cache_url_identifier: *const c_char,
    /// See [`AneIdentifierSource`].
    pub identifier_source: AneIdentifierSource,
    /// `QoS` for compile + load.
    pub compile_qos: AneQoS,
}

/// Type tag carried by each `_ANESharedSignalEvent` / `_ANESharedWaitEvent`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AneEventType {
    /// Framework default.
    Default = 0,
    /// Fires on inference start.
    Inference = 1,
    /// Fires on inference completion.
    Completion = 2,
}

/// Opaque per-process session hint handed to `_ANEVirtualClient`.
pub enum AneSessionHint {}

/// Tag selecting which load-tuning hint to apply.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AneSessionHintKind {
    /// Prefetch the compiled program into ANE SRAM.
    Prefetch = 1,
    /// Bias the loader toward low-latency placement.
    LowLatency = 2,
    /// Bias toward high-throughput placement.
    HighThroughput = 3,
}

/// Parameters for [`ane_model_new_instance`].
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AneModelInstanceParams {
    /// Optional cache-key override for the new instance.
    pub key: *const c_char,
    /// Reserved; pass 0.
    pub flags: u64,
}

/// Extended file-open options carrying Metal Performance Shaders
/// constants and optional `modelAttributes` overrides.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AneModelFileOpenOptionsEx {
    /// Base options (URL, cache key, QoS, etc.).
    pub base: AneModelFileOpenOptions,
    /// `id` pointer to an `NSDictionary*` of MPS constants, or null.
    pub mps_constants_id: *mut c_void,
    /// `id` pointer to a `modelAttributes` `NSDictionary*` override, or null.
    pub model_attributes_id: *mut c_void,
}

/// One stage of a chained dispatch.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AneChainStep {
    /// Configured [`AneRequest`] supplying the bindings, procedure index,
    /// shared events, and transaction handle for this stage.
    pub request: *mut AneRequest,
    /// Loopback input symbol id wiring this stage to the previous stage.
    pub lb_input_symbol_id: i64,
    /// Loopback output symbol id wiring this stage to the next stage.
    pub lb_output_symbol_id: i64,
    /// Firmware enqueue delay in nanoseconds (0 = no delay).
    pub fw_enqueue_delay: u64,
    /// Shared intermediate memory pool id (0 = no pool).
    pub memory_pool_id: u64,
}

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

    /// Populate `*out` with the ANE device facts (core count, arch string).
    pub fn ane_device_info(out: *mut AneDeviceInfo) -> AneStatus;

    /// Number of procedures (entry points) in the loaded model. Always >= 1.
    pub fn ane_model_num_procedures(model: *const AneModel) -> i32;
    /// Number of inputs for procedure `p`.
    pub fn ane_model_num_inputs_for_procedure (m: *const AneModel, p: i32) -> i32;
    /// Number of outputs for procedure `p`.
    pub fn ane_model_num_outputs_for_procedure(m: *const AneModel, p: i32) -> i32;
    /// Spec pointer for input `idx` of procedure `p`. Valid until model close.
    pub fn ane_model_input_spec_for_procedure (m: *const AneModel, p: i32, idx: i32) -> *const AneTensorSpec;
    /// Spec pointer for output `idx` of procedure `p`. Valid until model close.
    pub fn ane_model_output_spec_for_procedure(m: *const AneModel, p: i32, idx: i32) -> *const AneTensorSpec;
    /// Byte size of input `idx` of procedure `p`. 0 if out of range.
    pub fn ane_model_input_nbytes_for_procedure (m: *const AneModel, p: i32, idx: i32) -> usize;
    /// Byte size of output `idx` of procedure `p`. 0 if out of range.
    pub fn ane_model_output_nbytes_for_procedure(m: *const AneModel, p: i32, idx: i32) -> usize;

    /// Hardware queue depth reported for this model (max in-flight requests).
    pub fn ane_model_queue_depth(m: *const AneModel) -> i32;
    /// Number of evaluations currently in flight against this model.
    pub fn ane_model_in_flight  (m: *const AneModel) -> i64;

    /// `hexStringIdentifier` of the loaded program. Library-owned; empty if absent.
    pub fn ane_model_program_id   (m: *const AneModel) -> *const c_char;
    /// `weightsHash` of the descriptor. Library-owned; empty if absent.
    pub fn ane_model_weights_hash (m: *const AneModel) -> *const c_char;
    /// Writes the opaque driver `programHandle`. Returns true on success.
    pub fn ane_model_program_handle            (m: *const AneModel, out: *mut u64) -> bool;
    /// Writes the opaque `intermediateBufferHandle`. Returns true on success.
    pub fn ane_model_intermediate_buffer_handle(m: *const AneModel, out: *mut u64) -> bool;

    /// True if `aned` already has a compiled artifact for this hex hash.
    pub fn ane_cache_exists_for_hash(hex: *const c_char) -> bool;
    /// Evict the compiled artifact for this hex hash from `aned`.
    pub fn ane_cache_purge_for_hash (hex: *const c_char) -> AneStatus;

    /// Extended open: in-memory MIL bytes + multi-blob `NSDictionary` weights.
    pub fn ane_model_open_ex  (opts: *const AneModelOpenOptionsEx, out: *mut *mut AneModel) -> AneStatus;
    /// File-based open via `_ANEModel modelAtURL:key:`. Accepts compiled bundles.
    pub fn ane_model_open_file(opts: *const AneModelFileOpenOptions, out: *mut *mut AneModel) -> AneStatus;

    /// Open with the real-time priority class. Pair with [`ane_realtime_task_begin`].
    pub fn ane_model_open_realtime   (opts: *const AneModelOpenOptions,   out: *mut *mut AneModel) -> AneStatus;
    /// Real-time variant of [`ane_model_open_ex`].
    pub fn ane_model_open_realtime_ex(opts: *const AneModelOpenOptionsEx, out: *mut *mut AneModel) -> AneStatus;
    /// Enter the ANE real-time scheduling class for the current thread.
    pub fn ane_realtime_task_begin() -> AneStatus;
    /// Leave the real-time scheduling class.
    pub fn ane_realtime_task_end  () -> AneStatus;

    /// Allocate a fresh `_ANEPerformanceStats` wrapper for use in a request.
    pub fn ane_perf_stats_create (out: *mut *mut AnePerfStats) -> AneStatus;
    /// Release the stats wrapper.
    pub fn ane_perf_stats_release(ps: *mut AnePerfStats);
    /// Hardware execution time of the most recent eval, in nanoseconds.
    pub fn ane_perf_stats_hw_execution_ns(ps: *const AnePerfStats) -> u64;
    /// Byte size of the raw counter blob; use to size a buffer for `_counters_copy`.
    pub fn ane_perf_stats_counters_nbytes(ps: *const AnePerfStats) -> usize;
    /// Copy up to `cap` bytes of the raw counter blob into `out`; returns copied byte count.
    pub fn ane_perf_stats_counters_copy  (ps: *const AnePerfStats, out: *mut c_void, cap: usize) -> usize;

    /// Bitmask telling the driver which hardware counters to populate; set before eval.
    pub fn ane_model_set_perf_stats_mask(m: *mut AneModel, mask: u32) -> AneStatus;
    /// Read back the current `perfStatsMask`.
    pub fn ane_model_get_perf_stats_mask(m: *const AneModel) -> u32;

    /// Allocate a fresh `_ANESharedEvents` container.
    pub fn ane_shared_events_create (out: *mut *mut AneSharedEvents) -> AneStatus;
    /// Release the container and drop its event references.
    pub fn ane_shared_events_release(ev: *mut AneSharedEvents);
    /// Append a signal event. `mtl_shared_event` is a borrowed `id`
    /// pointing at a Metal `MTLSharedEvent` (retained by the container).
    pub fn ane_shared_events_add_signal(
        ev: *mut AneSharedEvents, value: u64, symbol_index: u32,
        event_type: AneEventType, mtl_shared_event: *mut c_void, agent_mask: u64
    ) -> AneStatus;
    /// Append a wait event. Same `mtl_shared_event` contract as `_add_signal`.
    pub fn ane_shared_events_add_wait(
        ev: *mut AneSharedEvents, value: u64,
        mtl_shared_event: *mut c_void, event_type: AneEventType
    ) -> AneStatus;
    /// Number of signal events queued.
    pub fn ane_shared_events_num_signals(ev: *const AneSharedEvents) -> i32;
    /// Number of wait events queued.
    pub fn ane_shared_events_num_waits  (ev: *const AneSharedEvents) -> i32;

    /// Per-request weight override. Pass null to clear. Buffer must outlive the request.
    pub fn ane_request_set_weights        (req: *mut AneRequest, weights: *mut AneBuffer)     -> AneStatus;
    /// Select which procedure (entry point) this request targets. Default 0.
    pub fn ane_request_set_procedure_index(req: *mut AneRequest, proc_idx: i32)               -> AneStatus;
    /// Attach a perf-stats sink for the next eval. Null clears.
    pub fn ane_request_set_perf_stats     (req: *mut AneRequest, ps: *mut AnePerfStats)       -> AneStatus;
    /// Attach a shared-events container for GPU↔ANE synchronization.
    pub fn ane_request_set_shared_events  (req: *mut AneRequest, ev: *mut AneSharedEvents)    -> AneStatus;
    /// Set the transaction handle used to group this request with others.
    pub fn ane_request_set_transaction    (req: *mut AneRequest, handle: u64)                 -> AneStatus;
    /// Read the currently-set procedure index.
    pub fn ane_request_procedure_index    (req: *const AneRequest) -> i32;
    /// Read the currently-set transaction handle.
    pub fn ane_request_transaction        (req: *const AneRequest) -> u64;

    /// Returns the borrowed `IOSurfaceRef` backing this buffer; do not release.
    pub fn ane_buffer_iosurface_ref  (buf: *const AneBuffer) -> *mut c_void;
    /// Wrap a caller-supplied `IOSurfaceRef` in a fresh `AneBuffer` (CFRetains the surface).
    pub fn ane_buffer_adopt_iosurface(surface: *mut c_void, nbytes: usize, out: *mut *mut AneBuffer) -> AneStatus;

    /// Build a chained dispatch from `n_steps` configured requests.
    pub fn ane_chain_create (steps: *const AneChainStep, n_steps: i32, out: *mut *mut AneChain) -> AneStatus;
    /// One-time preparation: maps intermediate buffers and resolves loopback symbols.
    pub fn ane_chain_prepare(chain: *mut AneChain, qos: AneQoS) -> AneStatus;
    /// Submit the chain. May be called repeatedly after a successful prepare.
    pub fn ane_chain_enqueue(chain: *mut AneChain, qos: AneQoS) -> AneStatus;
    /// Block until the most recent enqueue completes; same timeout semantics as `ane_request_wait`.
    pub fn ane_chain_wait   (chain: *mut AneChain, timeout_ms: c_int) -> AneStatus;
    /// Release the chain. Safe on null.
    pub fn ane_chain_release(chain: *mut AneChain);

    /// Route a compressed weight blob through the framework's parallel decompressor.
    /// Allocates `*out_bytes` via `malloc`; caller frees with `libc::free`.
    pub fn ane_decompress_weights(
        compressed: *const c_void, nbytes: usize,
        out_bytes: *mut *mut c_void, out_nbytes: *mut usize,
    ) -> AneStatus;

    /// `_ANEModel.UUID` as a UTF-8 string; library-owned; empty if absent.
    pub fn ane_model_uuid                (m: *const AneModel) -> *const c_char;
    /// `_ANEModel.sourceURL.absoluteString`; library-owned; empty if absent.
    pub fn ane_model_source_url          (m: *const AneModel) -> *const c_char;
    /// `_ANEModel.modelURL.absoluteString`; library-owned; empty if absent.
    pub fn ane_model_model_url           (m: *const AneModel) -> *const c_char;
    /// `_ANEModel.key`; library-owned; empty if absent.
    pub fn ane_model_key                 (m: *const AneModel) -> *const c_char;
    /// `_ANEModel.cacheURLIdentifier`; library-owned; empty if absent.
    pub fn ane_model_cache_url_identifier(m: *const AneModel) -> *const c_char;
    /// `_ANEModel.identifierSource` raw enum value.
    pub fn ane_model_identifier_source   (m: *const AneModel) -> i64;

    /// Reset transient model state on the next unload.
    pub fn ane_model_reset_on_unload(m: *mut AneModel) -> AneStatus;
    /// Explicit unload (without freeing the handle). Pairs with re-load patterns.
    pub fn ane_model_unload         (m: *mut AneModel) -> AneStatus;

    /// Load a fresh runtime instance sharing the same compiled program.
    pub fn ane_model_new_instance(
        src: *mut AneModel, params: *const AneModelInstanceParams,
        qos: AneQoS, out: *mut *mut AneModel,
    ) -> AneStatus;

    /// Number of aned connections currently open for loading models.
    pub fn ane_client_num_connections() -> i32;
    /// True if this model's `_ANEClient` is a virtual (per-process) client.
    pub fn ane_model_is_virtual_client(m: *const AneModel) -> bool;

    /// Allocate a session-hint object of the given kind.
    pub fn ane_session_hint_create (kind: AneSessionHintKind, out: *mut *mut AneSessionHint) -> AneStatus;
    /// Release the session-hint object.
    pub fn ane_session_hint_release(hint: *mut AneSessionHint);

    /// Apply `hint` to `model`; if `out_report_json` is non-null, the
    /// framework's per-hint report is `malloc`'d as a UTF-8 JSON string.
    pub fn ane_model_apply_session_hint(
        m: *mut AneModel, hint: *const AneSessionHint, out_report_json: *mut *mut c_char,
    ) -> AneStatus;

    /// Extended file-open: includes `mpsConstants` and `modelAttributes`.
    pub fn ane_model_open_file_ex(
        opts: *const AneModelFileOpenOptionsEx, out: *mut *mut AneModel,
    ) -> AneStatus;

    /// Human-readable name of perf counter `counter_idx`. Library-owned.
    pub fn ane_perf_counter_name(counter_idx: i32) -> *const c_char;
    /// Emit os_signpost events for the counters captured by `ps`.
    pub fn ane_perf_stats_emit_signpost(ps: *const AnePerfStats, model_string_id: u64) -> AneStatus;
    /// Number of per-stage perf-stats objects attached to `req` (1 if single, N for chained).
    pub fn ane_request_num_perf_stats(req: *const AneRequest) -> i32;
    /// Borrowed stats handle for stage `idx`; null if out of range.
    pub fn ane_request_perf_stats_at (req: *const AneRequest, idx: i32) -> *mut AnePerfStats;

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
