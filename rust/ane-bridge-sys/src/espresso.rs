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
}
