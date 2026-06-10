//! Raw bindings to Apple's private Espresso framework.
//!
//! See `c/include/espresso.h` for the contract. These mirror that header
//! one-to-one and link directly against the framework (see the build
//! script). They are not part of `libane_bridge`.
//!
//! They extend the bridge to Apple's `E5RT` execution engine — plan,
//! network, and buffer control, including ANE-resident state buffers
//! (`read_state` / `write_state`) for key/value-cache workloads. This is the
//! raw binding layer; the safe wrapper, its tests, and fuzz coverage are
//! still to come.
//!
//! # Safety
//!
//! Every item reflects a private, unstable symbol recovered from runtime
//! symbol tables. Two blanket hazards apply to all of them, on top of each
//! function's own pointer contract:
//!
//! - **Unstable ABI.** Apple may rename, remove, or re-sign any of these in
//!   any OS update. All 73 were verified present on `macOS` 26.2; nothing
//!   guarantees the next release (the `symbols_link` test is the canary).
//! - **Unverified signatures.** Parameter lists were inferred from `lldb`,
//!   not headers. Anything noted `unverified` below has a best-effort
//!   signature past its leading arguments — disassemble and confirm before
//!   calling it. A wrong signature is undefined behavior.
//!
//! Build a safe, `Result`-based wrapper over this in the `ane-bridge` crate;
//! do not call these from application code.

use core::ffi::{c_char, c_int, c_void};

/// Opaque `EspressoLight::espresso_context*`.
pub type Context = *mut c_void;
/// Opaque `EspressoLight::abstract_espresso_plan*`.
pub type Plan = *mut c_void;
/// Opaque per-network handle on a loaded plan.
pub type Network = *mut c_void;
/// `dispatch_queue_t`, opaque at this layer.
pub type DispatchQueue = *mut c_void;
/// Weight storage-class enum (concrete values TBD).
pub type StorageType = c_int;
/// Plan-phase enum: build / load / execute (concrete values TBD).
pub type PlanPhase = c_int;
/// Objective-C completion block `void (^)(espresso_error_info_t *)`. Opaque here;
/// construct via the `block2` crate in the safe wrapper.
pub type CompletionBlock = *mut c_void;

/// Opaque Espresso buffer (raw pointer + shape + dtype). Touch only through
/// the `espresso_buffer_*` accessors.
#[repr(C)]
pub struct Buffer {
    /// Zero-sized marker; this type is only ever used behind a pointer.
    _private: [u8; 0],
}

/// Opaque error record filled by `espresso_plan_get_error_info` and by
/// completion blocks. Layout private until reversed.
#[repr(C)]
pub struct ErrorInfo {
    /// Zero-sized marker; this type is only ever used behind a pointer.
    _private: [u8; 0],
}

/// Opaque `e5rt_buffer_object` handle.
///
/// A retained ANE memory provider wrapping an `IOSurface`, `MTLBuffer`, host
/// pointer, or freshly allocated ANE memory. Touch only through the
/// `e5rt_buffer_object_*` accessors.
pub type BufferObject = *mut c_void;

/// `e5rt_error_code_t`: `0` is success, non-zero a failure code.
///
/// Partially reversed via [`e5rt_error_code_get_string`]: `0` = `OK`,
/// `1` = `INVALID FUNCTION ARGUMENT`, `2` = `INVALID OPERATION`,
/// `3` = `MEM ALLOC FAILURE`, `4` = `OUT OF BOUNDS ACCESS`,
/// `5` = `INVALID TYPE PARAMETER`, `6` = `INVALID DATA TYPE`.
pub type E5rtErrorCode = c_int;

/// Opaque `e5rt_async_event` handle.
///
/// A timeline-semaphore-style synchronization primitive: signal it with a
/// monotonically increasing `u64` value and wait for a target value. Used to
/// order work on an `e5rt_execution_stream`. Touch only through the
/// `e5rt_async_event_*` accessors.
pub type AsyncEvent = *mut c_void;

/// Opaque `e5rt_execution_stream_config_options` handle — a builder of stream
/// behavior flags. Touch only through the
/// `e5rt_execution_stream_config_options_*` accessors.
pub type StreamConfigOptions = *mut c_void;

unsafe extern "C" {
    // ----- Context -----

    /// Create a CPU test-vector context (unverified arguments).
    pub fn espresso_context_create_for_cpu_test_vectors() -> Context;
    /// Destroy a context.
    pub fn espresso_context_destroy(ctx: Context);
    /// Set an integer context option (unverified).
    pub fn espresso_context_set_int_option(ctx: Context, option: c_int, value: c_int);
    /// Toggle low-precision accumulation on the context.
    pub fn espresso_context_set_low_precision_accumulation(ctx: Context, enable: bool);
    /// Emit a benchmark report for the context (unverified).
    pub fn espresso_context_report_bench(ctx: Context, out: *mut c_void);

    // ----- Plan: lifecycle / build / execute / submit -----

    /// Destroy a plan.
    pub fn espresso_plan_destroy(plan: Plan);
    /// Non-zero if the plan supports the async `submit` path.
    pub fn espresso_plan_can_use_submit(plan: Plan) -> c_int;
    /// Current plan phase (see [`PlanPhase`]).
    pub fn espresso_plan_get_phase(plan: Plan) -> c_int;
    /// Set the dispatch queue the plan executes on.
    pub fn espresso_plan_set_execution_queue(plan: Plan, queue: DispatchQueue) -> c_int;
    /// Set the plan's scheduling priority.
    pub fn espresso_plan_set_priority(plan: Plan, priority: c_int) -> c_int;
    /// Build the plan (compile/prepare for execution).
    pub fn espresso_plan_build(plan: Plan) -> c_int;
    /// Build the plan with options (unverified).
    pub fn espresso_plan_build_with_options(plan: Plan, options: *mut c_void) -> c_int;
    /// Build the plan, clearing any prior build state.
    pub fn espresso_plan_build_clean(plan: Plan) -> c_int;
    /// Execute the plan synchronously on the calling thread.
    pub fn espresso_plan_execute_sync(plan: Plan) -> c_int;
    /// Submit the plan for async execution; `cb` fires on completion.
    pub fn espresso_plan_submit(plan: Plan, queue: DispatchQueue, cb: CompletionBlock) -> c_int;
    /// Submit with an argument bundle (unverified).
    pub fn espresso_plan_submit_with_args(
        plan: Plan,
        args: *mut c_void,
        cb: CompletionBlock,
    ) -> c_int;
    /// Submit on the camera path; `cb` is the per-network completion block.
    pub fn espresso_plan_submit_camera(plan: Plan, cb: *mut c_void) -> c_int;
    /// Configure multiple-buffering depth for submit (unverified).
    pub fn espresso_plan_submit_set_multiple_buffering(plan: Plan, count: c_int) -> c_int;
    /// Share an intermediate buffer with another plan (unverified).
    pub fn espresso_plan_share_intermediate_buffer(plan: Plan, other: *mut c_void);
    /// Add a network to the plan from a file URL (unverified tail args).
    pub fn espresso_plan_add_network(
        plan: Plan,
        file_url: *const c_char,
        storage: StorageType,
    ) -> c_int;
    /// Add a network to the plan from an in-memory blob (unverified tail args).
    pub fn espresso_plan_add_network_from_memory(
        plan: Plan,
        data: *const c_void,
        size: usize,
        storage: StorageType,
    ) -> c_int;

    // ----- Plan: profiling -----

    /// Turn on the debug firehose for the plan.
    pub fn espresso_plan_activate_debug_firehose(plan: Plan);
    /// Enable automatic profiling.
    pub fn espresso_plan_auto_profile(plan: Plan);
    /// Begin profiling.
    pub fn espresso_plan_start_profiling(plan: Plan);
    /// Begin profiling with options (unverified).
    pub fn espresso_plan_start_profiling_with_options(plan: Plan, options: *mut c_void);
    /// Finish profiling.
    pub fn espresso_plan_finish_profiling(plan: Plan);
    /// Run the plan's perf benchmark.
    pub fn espresso_plan_perfbench(plan: Plan);
    /// Run the plan's perf benchmark without output copies.
    pub fn espresso_plan_perfbench_nocopy(plan: Plan);
    /// Dump static profiling info for the plan.
    pub fn espresso_plan_static_profiling_info(plan: Plan);

    // ----- Plan: errors -----

    /// Fill `out` with the plan's last error record.
    pub fn espresso_plan_get_error_info(plan: Plan, out: *mut ErrorInfo);

    // ----- Network: buffer binding, shapes, and state -----

    /// Bind a buffer to a named blob.
    pub fn espresso_network_bind_buffer(
        net: Network,
        blob_name: *const c_char,
        buf: *mut Buffer,
    ) -> c_int;
    /// Bind a buffer to a named global blob (unverified).
    pub fn espresso_network_bind_buffer_to_global(
        net: Network,
        blob_name: *const c_char,
        buf: *mut Buffer,
    ) -> c_int;
    /// Unbind the buffer from a named blob.
    pub fn espresso_network_unbind_buffer(net: Network, blob_name: *const c_char) -> c_int;
    /// Unbind the buffer from a named global blob (unverified).
    pub fn espresso_network_unbind_buffer_to_global(
        net: Network,
        blob_name: *const c_char,
    ) -> c_int;
    /// Swap a named global blob (unverified).
    pub fn espresso_network_swap_global(net: Network, blob_name: *const c_char) -> c_int;
    /// Synchronously copy a named global blob (unverified).
    pub fn espresso_network_sync_copy_global(net: Network, blob_name: *const c_char) -> c_int;
    /// Declare a named input (unverified tail args).
    pub fn espresso_network_declare_input(net: Network, name: *const c_char) -> c_int;
    /// Declare a named output (unverified tail args).
    pub fn espresso_network_declare_output(net: Network, name: *const c_char) -> c_int;
    /// Change a blob's shape (unverified).
    pub fn espresso_network_change_blob_shape(
        net: Network,
        blob_name: *const c_char,
        shape: *const i64,
        rank: usize,
    ) -> c_int;
    /// Change all input blob shapes (unverified).
    pub fn espresso_network_change_input_blob_shapes(net: Network, shapes: *mut c_void) -> c_int;
    /// Change all input blob shapes, sequence form (unverified).
    pub fn espresso_network_change_input_blob_shapes_seq(
        net: Network,
        shapes: *mut c_void,
    ) -> c_int;
    /// Change all input blob shapes, sequence + rank form (unverified).
    pub fn espresso_network_change_input_blob_shapes_seq_rank(
        net: Network,
        shapes: *mut c_void,
    ) -> c_int;

    /// The state-specific entry point: zero all temporal (KV) state buffers.
    pub fn espresso_network_temporal_state_reset(net: Network) -> c_int;

    /// Query a blob's dimension count (unverified return).
    pub fn espresso_network_query_blob_dimensions(net: Network, blob_name: *const c_char) -> c_int;
    /// Query a blob's shape (unverified).
    pub fn espresso_network_query_blob_shape(net: Network, blob_name: *const c_char) -> c_int;
    /// Query a blob's quantization info (unverified).
    pub fn espresso_network_query_quantization_info(
        net: Network,
        blob_name: *const c_char,
    ) -> c_int;
    /// Return the network's version string (library-owned).
    pub fn espresso_network_get_version(net: Network) -> *const c_char;
    /// Select a named configuration on the network.
    pub fn espresso_network_select_configuration(net: Network, config: *const c_char) -> c_int;
    /// Set the network's active function name (e.g. `"main"`).
    pub fn espresso_network_set_function_name(net: Network, name: *const c_char) -> c_int;
    /// Set the network's inference weights (unverified).
    pub fn espresso_network_set_inference_weights(net: Network, weights: *mut c_void) -> c_int;
    /// Set the network's memory-pool id (unverified).
    pub fn espresso_network_set_memory_pool_id(net: Network, pool_id: u64) -> c_int;
    /// Set the network's tracing name.
    pub fn espresso_network_set_tracing_name(net: Network, name: *const c_char) -> c_int;
    /// Set a compiler metadata key/value pair on the network.
    pub fn espresso_network_compiler_set_metadata_key(
        net: Network,
        key: *const c_char,
        value: *const c_char,
    ) -> c_int;
    /// Pin the network's weights-blob storage.
    pub fn espresso_network_pin_weights_blob_storage(net: Network) -> c_int;
    /// Unpin the network's weights-blob storage.
    pub fn espresso_network_unpin_weights_blob_storage(net: Network) -> c_int;
    /// Dump a test vector for the network (unverified).
    pub fn espresso_network_dump_test_vector(net: Network, path: *const c_char) -> c_int;

    /// Bind a `CVPixelBuffer` input (legacy; signature unverified).
    pub fn espresso_network_bind_cvpixelbuffer(
        net: Network,
        blob_name: *const c_char,
        src: *mut c_void,
    ) -> c_int;
    /// Bind a `CVPixelBuffer` input without channel swap (legacy; unverified).
    pub fn espresso_network_bind_cvpixelbuffer_no_channel_swap(
        net: Network,
        blob_name: *const c_char,
        src: *mut c_void,
    ) -> c_int;
    /// Bind a `CVPixelBuffer` input directly (legacy; unverified).
    pub fn espresso_network_bind_direct_cvpixelbuffer(
        net: Network,
        blob_name: *const c_char,
        src: *mut c_void,
    ) -> c_int;
    /// Bind a `CVPixelBuffer` as a declared input (legacy; unverified).
    pub fn espresso_network_bind_input_cvpixelbuffer(
        net: Network,
        blob_name: *const c_char,
        src: *mut c_void,
    ) -> c_int;
    /// Bind a Metal texture as a declared input (legacy; unverified).
    pub fn espresso_network_bind_input_metaltexture(
        net: Network,
        blob_name: *const c_char,
        src: *mut c_void,
    ) -> c_int;
    /// Bind an ARGB8 vImage buffer input (legacy; unverified).
    pub fn espresso_network_bind_input_vimagebuffer_argb8(
        net: Network,
        blob_name: *const c_char,
        src: *mut c_void,
    ) -> c_int;
    /// Bind a BGRA8 vImage buffer input (legacy; unverified).
    pub fn espresso_network_bind_input_vimagebuffer_bgra8(
        net: Network,
        blob_name: *const c_char,
        src: *mut c_void,
    ) -> c_int;
    /// Bind a planar8 vImage buffer input (legacy; unverified).
    pub fn espresso_network_bind_input_vimagebuffer_planar8(
        net: Network,
        blob_name: *const c_char,
        src: *mut c_void,
    ) -> c_int;
    /// Bind an RGBA8 vImage buffer input (legacy; unverified).
    pub fn espresso_network_bind_input_vimagebuffer_rgba8(
        net: Network,
        blob_name: *const c_char,
        src: *mut c_void,
    ) -> c_int;

    // ----- ANE program cache (keyed by model path/URL) -----

    /// Query an ANE network cache for `model_path` (via `_ANEClient`'s shared
    /// connection and `model_path_to_model_url`).
    ///
    /// Writes the presence flag to `*out_exists` and returns an int status
    /// (`0` on success; a path with no cached network yields `0` with
    /// `*out_exists = false`). Which cache this reads is unconfirmed — it is
    /// not the in-memory hash cache behind [`crate::ane_cache_exists_for_hash`],
    /// and reports not-cached even for models the OS has ANE-run; it is
    /// probably the E5RT compiler bundle cache. See `c/include/espresso.h`.
    ///
    /// # Safety
    /// `model_path` must be a valid NUL-terminated C string and `out_exists`
    /// must point to a writable `bool`; both are accessed once during the
    /// call. See the module-level unstable-ABI / unverified-signature notes.
    pub fn espresso_ane_cache_has_network(
        model_path: *const c_char,
        out_exists: *mut bool,
    ) -> c_int;

    /// Evict the compiled network cached for `model_path`. Returns an int
    /// status (`0` on success). Path-keyed counterpart to
    /// [`crate::ane_cache_purge_for_hash`].
    ///
    /// # Safety
    /// `model_path` must be a valid NUL-terminated C string, read once.
    pub fn espresso_ane_cache_purge_network(model_path: *const c_char) -> c_int;

    // ----- Buffer -----

    /// Element count of the buffer.
    pub fn espresso_buffer_get_count(buf: *const Buffer) -> usize;
    /// Rank (number of dimensions) of the buffer.
    pub fn espresso_buffer_get_rank(buf: *const Buffer) -> usize;
    /// Byte size of the buffer.
    pub fn espresso_buffer_get_size(buf: *const Buffer) -> usize;
    /// Set the buffer's rank.
    pub fn espresso_buffer_set_rank(buf: *mut Buffer, rank: usize);
    /// Pack a tensor shape (`rank` dims from `dims`) into the buffer.
    pub fn espresso_buffer_pack_tensor_shape(buf: *mut Buffer, rank: usize, dims: *const usize);
    /// Unpack the buffer's tensor shape into `rank` / `dims`.
    pub fn espresso_buffer_unpack_tensor_shape(
        buf: *const Buffer,
        rank: *mut usize,
        dims: *mut usize,
    );

    // ----- E5RT execution stream (opaque handle) -----
    //
    // The E5RT runtime beneath MLE5Engine. Handles (`e5rt_execution_stream*`,
    // etc.) are opaque pointers; functions return an `e5rt_error_code_t` and
    // use out-params. Borrow a live stream from a loaded model via
    // `crate::ane_state_e5rt_stream`. Only the verified-signature subset is
    // bound so far; the broader `e5rt_*` surface (292 symbols, all present per
    // the `espresso_symbols` canary) is future work.

    /// Stream id of an `e5rt_execution_stream` (verified: returns the live id).
    ///
    /// # Safety
    /// `stream` must be a live `e5rt_execution_stream*` (e.g. from
    /// [`crate::ane_state_e5rt_stream`]); it is borrowed, not consumed.
    pub fn e5rt_execution_stream_get_stream_id(stream: *mut c_void) -> u64;

    // ----- E5RT buffer objects (zero-copy ANE I/O) -----
    //
    // Wrap an existing `IOSurface`, `MTLBuffer`, or host pointer as an
    // `e5rt_buffer_object` (no copy), or allocate fresh ANE memory; the
    // getters round-trip the backing handle and size. All return an
    // `E5rtErrorCode` (`0` == success); the `create_*` / `alloc` entry points
    // take the out-handle **first**, then their inputs (the getters take the
    // handle first, then the out-param — the opposite order). Signatures were
    // recovered by `lldb` AND verified by calling them: every create/alloc
    // produced a live handle whose `get_*` round-tripped the original surface /
    // `MTLBuffer` / pointer and reported the right size (see `tools/` probes).
    //
    // PRECONDITION (verified): the process-global E5RT runtime must already be
    // up, or `create_from_iosurface` and friends dereference an uninitialized
    // provider and crash. An `MLE5Engine` prediction brings it up; the safe
    // wrapper gates on a live `e5rt_execution_stream` before calling these.
    //
    // The 11th symbol, `e5rt_buffer_object_create_as_alias`, is present (see
    // the `espresso_symbols` canary) but unverified, so it is not bound here.

    /// Wrap an `IOSurfaceRef` as a buffer object (no copy). `out` is written on
    /// success. Verified: `get_iosurface` round-trips the same surface.
    ///
    /// # Safety
    /// `out` must be a writable `*mut BufferObject`; `surface` a live
    /// `IOSurfaceRef`. The E5RT runtime must be warm (see the section note).
    /// Release the result with [`e5rt_buffer_object_release`].
    pub fn e5rt_buffer_object_create_from_iosurface(
        out: *mut BufferObject,
        surface: *mut c_void,
    ) -> E5rtErrorCode;

    /// Wrap a host pointer + byte length as a buffer object (no copy).
    ///
    /// # Safety
    /// `out` must be writable; `data` must be valid for `size` bytes and
    /// outlive the buffer object. Runtime must be warm.
    pub fn e5rt_buffer_object_create_from_data_pointer(
        out: *mut BufferObject,
        data: *mut c_void,
        size: usize,
    ) -> E5rtErrorCode;

    /// Wrap an `id<MTLBuffer>` as a buffer object (no copy). Verified:
    /// `get_mtlbuffer` round-trips the same `MTLBuffer`.
    ///
    /// # Safety
    /// `out` must be writable; `mtlbuffer` a live `id<MTLBuffer>`. Runtime
    /// must be warm.
    pub fn e5rt_buffer_object_create_from_mtlbuffer(
        out: *mut BufferObject,
        mtlbuffer: *mut c_void,
    ) -> E5rtErrorCode;

    /// Allocate a fresh ANE buffer of `size` bytes. `buffer_type` selects the
    /// backing class; only `0` is verified (its `get_data_ptr` returns a live,
    /// page-aligned pointer). The concrete type enum is not reversed.
    ///
    /// # Safety
    /// `out` must be writable. Runtime must be warm.
    pub fn e5rt_buffer_object_alloc(
        out: *mut BufferObject,
        size: usize,
        buffer_type: u32,
    ) -> E5rtErrorCode;

    /// Write the buffer's host data pointer to `*out`.
    ///
    /// # Safety
    /// `obj` must be a live buffer object; `out` a writable `*mut *mut c_void`.
    pub fn e5rt_buffer_object_get_data_ptr(
        obj: BufferObject,
        out: *mut *mut c_void,
    ) -> E5rtErrorCode;

    /// Write the buffer's backing `IOSurfaceRef` to `*out` (errors if the
    /// buffer is not `IOSurface`-backed).
    ///
    /// # Safety
    /// `obj` must be live; `out` a writable `*mut *mut c_void` (an
    /// `IOSurfaceRef*`).
    pub fn e5rt_buffer_object_get_iosurface(
        obj: BufferObject,
        out: *mut *mut c_void,
    ) -> E5rtErrorCode;

    /// Write the buffer's backing `id<MTLBuffer>` to `*out` (errors if the
    /// buffer is not `MTLBuffer`-backed).
    ///
    /// # Safety
    /// `obj` must be live; `out` a writable `*mut *mut c_void`.
    pub fn e5rt_buffer_object_get_mtlbuffer(
        obj: BufferObject,
        out: *mut *mut c_void,
    ) -> E5rtErrorCode;

    /// Write the buffer's byte size to `*out`.
    ///
    /// # Safety
    /// `obj` must be live; `out` a writable `*mut usize`.
    pub fn e5rt_buffer_object_get_size(obj: BufferObject, out: *mut usize) -> E5rtErrorCode;

    /// Write the buffer's type tag to `*out` (a 32-bit value; the enum is not
    /// reversed — `0` observed for `IOSurface` / pointer / `MTLBuffer` backings).
    ///
    /// # Safety
    /// `obj` must be live; `out` a writable `*mut u32` (the call writes exactly
    /// four bytes — verified with a 64-bit sentinel).
    pub fn e5rt_buffer_object_get_type(obj: BufferObject, out: *mut u32) -> E5rtErrorCode;

    /// Release a buffer object (drops the provider's reference to the backing
    /// surface / buffer / allocation).
    ///
    /// # Safety
    /// `obj` must be a live buffer object, released exactly once.
    pub fn e5rt_buffer_object_release(obj: BufferObject) -> E5rtErrorCode;

    // ----- E5RT async events (timeline synchronization) -----
    //
    // A timeline-semaphore primitive for ordering stream work: `signal` raises
    // the event's value, `sync_wait` blocks until it reaches a target (or a
    // timeout elapses). All return `E5rtErrorCode` (`0` == success); `create`
    // is out-first like the buffer factory. Signatures recovered by `lldb` AND
    // verified by calling: create returned a live handle whose `get_name`
    // round-tripped the supplied name; `signal(v)` then `get_last_signaled_value`
    // returned `v` and `sync_wait(v, ...)` returned immediately; the active
    // future value round-tripped a set/get. Like buffer objects, these need a
    // warm E5RT runtime (the safe wrapper gates on a live execution stream).
    //
    // The remaining trio — `e5rt_async_event_async_notify` (takes a callback),
    // `_create_from_iosurface_shared_event`, `_get_iosurface_shared_event`
    // (Metal shared-event interop) — is present in the canary but unverified,
    // so it is left unbound.

    /// Create a named timeline event. `out` is written on success; `flags` is a
    /// 32-bit option word (`0` verified).
    ///
    /// # Safety
    /// `out` must be writable; `name` a valid NUL-terminated C string (read
    /// once). The E5RT runtime must be warm. Release with
    /// [`e5rt_async_event_release`].
    pub fn e5rt_async_event_create(
        out: *mut AsyncEvent,
        name: *const c_char,
        flags: u32,
    ) -> E5rtErrorCode;

    /// Signal the event, raising its last-signaled value to `value`.
    ///
    /// # Safety
    /// `evt` must be a live event handle.
    pub fn e5rt_async_event_signal(evt: AsyncEvent, value: u64) -> E5rtErrorCode;

    /// Block until the event's value reaches `value`, or `timeout_ns`
    /// nanoseconds elapse. Returns a non-zero code on timeout.
    ///
    /// # Safety
    /// `evt` must be a live event handle.
    pub fn e5rt_async_event_sync_wait(
        evt: AsyncEvent,
        value: u64,
        timeout_ns: u64,
    ) -> E5rtErrorCode;

    /// Write the event's last-signaled value to `*out`.
    ///
    /// # Safety
    /// `evt` must be live; `out` a writable `*mut u64`.
    pub fn e5rt_async_event_get_last_signaled_value(
        evt: AsyncEvent,
        out: *mut u64,
    ) -> E5rtErrorCode;

    /// Write the event's active future (pending target) value to `*out`.
    ///
    /// # Safety
    /// `evt` must be live; `out` a writable `*mut u64`.
    pub fn e5rt_async_event_get_active_future_value(
        evt: AsyncEvent,
        out: *mut u64,
    ) -> E5rtErrorCode;

    /// Set the event's active future (pending target) value.
    ///
    /// # Safety
    /// `evt` must be a live event handle.
    pub fn e5rt_async_event_set_active_future_value(evt: AsyncEvent, value: u64) -> E5rtErrorCode;

    /// Write the event's borrowed name (library-owned C string) to `*out`.
    ///
    /// # Safety
    /// `evt` must be live; `out` a writable `*mut *const c_char`. The returned
    /// string is owned by the event; do not free it.
    pub fn e5rt_async_event_get_name(evt: AsyncEvent, out: *mut *const c_char) -> E5rtErrorCode;

    /// Release a timeline event.
    ///
    /// # Safety
    /// `evt` must be a live event handle, released exactly once.
    pub fn e5rt_async_event_release(evt: AsyncEvent) -> E5rtErrorCode;

    // ----- E5RT diagnostics -----

    /// Static human-readable string for an `E5rtErrorCode` (e.g. `0` -> `"OK"`).
    /// Returns a pointer to a static string literal owned by the framework
    /// (never freed); verified for codes 0..=6.
    ///
    /// # Safety
    /// Always safe to call; the result is a `'static` C string or null.
    pub fn e5rt_error_code_get_string(code: E5rtErrorCode) -> *const c_char;

    /// The calling thread's last E5RT error message (a thread-local buffer;
    /// empty string when there is none).
    ///
    /// # Safety
    /// Always safe to call; the result points to a thread-local buffer valid
    /// until the next E5RT call on this thread.
    pub fn e5rt_get_last_error_message() -> *const c_char;

    // ----- E5RT execution-stream tuning -----
    //
    // Adjust the scheduling of an `e5rt_execution_stream`. All return an
    // `E5rtErrorCode` (`0` == success). Signatures recovered by `lldb` AND
    // verified by calling on a live (borrowed) stream: `set_quality_of_service`
    // (with a `qos_class_t`, e.g. `0x19` = user-initiated) and
    // `set_ane_execution_priority` both returned `0`.
    //
    // The config-options object round-trips cleanly (create / set / get /
    // release verified by calling — each flag set then read back). But
    // `set_config_options` is VERIFIED to crash (`SIGBUS`) when applied to a
    // borrowed, in-use stream: reconfiguring CoreML's live stream wholesale
    // tears down state the engine still holds. Use it only on a stream you
    // created yourself (`e5rt_execution_stream_create`, not yet bound) — never
    // on a borrowed one.

    /// Set the borrowed stream's scheduling quality-of-service. `qos_class` is a
    /// Darwin `qos_class_t` (`0x21` user-interactive, `0x19` user-initiated,
    /// `0x15` default, `0x11` utility, `0x09` background). Verified safe on a
    /// borrowed stream.
    ///
    /// # Safety
    /// `stream` must be a live `e5rt_execution_stream*` (borrowed, not consumed).
    pub fn e5rt_execution_stream_set_quality_of_service(
        stream: *mut c_void,
        qos_class: u32,
    ) -> E5rtErrorCode;

    /// Set the borrowed stream's ANE execution priority. The priority enum is
    /// not reversed (`0` verified). Verified safe on a borrowed stream.
    ///
    /// # Safety
    /// `stream` must be a live `e5rt_execution_stream*` (borrowed, not consumed).
    pub fn e5rt_execution_stream_set_ane_execution_priority(
        stream: *mut c_void,
        priority: u32,
    ) -> E5rtErrorCode;

    /// Apply a config-options object to a stream.
    ///
    /// # Safety
    /// `stream` must be a live `e5rt_execution_stream*` and `options` a live
    /// config-options object. **Verified to crash on a borrowed/in-use stream**
    /// — only call on a stream you created yourself.
    pub fn e5rt_execution_stream_set_config_options(
        stream: *mut c_void,
        options: StreamConfigOptions,
    ) -> E5rtErrorCode;

    /// Create a stream config-options object (out-first). Round-trip verified.
    ///
    /// # Safety
    /// `out` must be a writable `*mut StreamConfigOptions`. Release with
    /// [`e5rt_execution_stream_config_options_release`].
    pub fn e5rt_execution_stream_config_options_create(
        out: *mut StreamConfigOptions,
    ) -> E5rtErrorCode;

    /// Enable/disable concurrent synchronous execution on the options.
    ///
    /// # Safety
    /// `opts` must be a live config-options object.
    pub fn e5rt_execution_stream_config_options_set_enable_concurrent_sync_execution(
        opts: StreamConfigOptions,
        value: bool,
    ) -> E5rtErrorCode;

    /// Read the concurrent-sync-execution flag into `*out`.
    ///
    /// # Safety
    /// `opts` must be live; `out` a writable `*mut bool`.
    pub fn e5rt_execution_stream_config_options_get_enable_concurrent_sync_execution(
        opts: StreamConfigOptions,
        out: *mut bool,
    ) -> E5rtErrorCode;

    /// Enable/disable low-latency async events on the options.
    ///
    /// # Safety
    /// `opts` must be a live config-options object.
    pub fn e5rt_execution_stream_config_options_set_enable_low_latency_async_events(
        opts: StreamConfigOptions,
        value: bool,
    ) -> E5rtErrorCode;

    /// Read the low-latency-async-events flag into `*out`.
    ///
    /// # Safety
    /// `opts` must be live; `out` a writable `*mut bool`.
    pub fn e5rt_execution_stream_config_options_get_enable_low_latency_async_events(
        opts: StreamConfigOptions,
        out: *mut bool,
    ) -> E5rtErrorCode;

    /// Enable/disable skipping I/O fences on the options.
    ///
    /// # Safety
    /// `opts` must be a live config-options object.
    pub fn e5rt_execution_stream_config_options_set_skip_io_fences(
        opts: StreamConfigOptions,
        value: bool,
    ) -> E5rtErrorCode;

    /// Read the skip-I/O-fences flag into `*out`.
    ///
    /// # Safety
    /// `opts` must be live; `out` a writable `*mut bool`.
    pub fn e5rt_execution_stream_config_options_get_skip_io_fences(
        opts: StreamConfigOptions,
        out: *mut bool,
    ) -> E5rtErrorCode;

    /// Release a config-options object.
    ///
    /// # Safety
    /// `opts` must be a live config-options object, released exactly once.
    pub fn e5rt_execution_stream_config_options_release(opts: StreamConfigOptions)
    -> E5rtErrorCode;
}
