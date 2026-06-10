/* espresso.h — bindings to Apple's private Espresso.framework C ABI.
 *
 * Part of growing ane-bridge into a tested reference surface over the Apple
 * Neural Engine. The base bridge covers the stateless compile / eval path;
 * these symbols reach the E5RT execution engine beneath CoreML, which exposes
 * plan / network / buffer control — and, among other capabilities, an
 * ANE-resident state path (read_state / write_state) so a KV cache can stay
 * on the ANE instead of crossing the bus each call.
 *
 * These are Apple's symbols, not ours. They link directly via
 * `-framework Espresso` with a `-F/System/Library/PrivateFrameworks` search
 * path (all 73 verified present on macOS 26.2); they can also be
 * dlopen/dlsym'd for graceful degradation across OS versions.
 *
 * Signatures were recovered from symbol tables, not real headers. Anything
 * tagged UNVERIFIED has a best-effort parameter list past its leading args —
 * confirm against a disassembly before calling it. Declaring a wrong
 * signature is harmless; calling through one is undefined behavior.
 *
 * Omitted: the C++ entry points (EspressoLight::, E5RT::, Espresso::, the
 * espresso_plan_add_cpp_net* overloads) — C++ ABI, not expressible here.
 *
 * Status: raw binding layer only. The safe Rust wrapper, its functional
 * tests, and parser-fuzz-style adversarial coverage are the next milestone —
 * held to the bar the rest of the bridge sets: every entry point tested, the
 * input-parsing boundary fuzzed as a property.
 */
#ifndef ANE_BRIDGE_ESPRESSO_H
#define ANE_BRIDGE_ESPRESSO_H

#include <dispatch/dispatch.h> /* dispatch_queue_t */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =====================================================================
 * Opaque handles
 * ===================================================================== */

typedef void* espresso_context_t; /* EspressoLight::espresso_context*      */
typedef void* espresso_plan_t;    /* EspressoLight::abstract_espresso_plan* */
typedef void* espresso_network_t; /* per-network handle on a loaded plan    */

/* Small struct (raw ptr + shape + dtype). Layout is private; manipulate
 * only through the espresso_buffer_* accessors below. */
typedef struct espresso_buffer espresso_buffer_t;

/* Filled by espresso_plan_get_error_info and by completion blocks. Layout
 * private until reversed. */
typedef struct espresso_error_info espresso_error_info_t;

typedef int espresso_storage_type_t; /* enum: weight storage class (TBD)     */
typedef int espresso_plan_phase_t;   /* enum: build / load / execute (TBD)   */

/* Completion block handed to the async submit entry points. */
typedef void (^espresso_completion_block_t)(espresso_error_info_t*);

/* =====================================================================
 * Context
 * ===================================================================== */

espresso_context_t espresso_context_create_for_cpu_test_vectors(void); /* UNVERIFIED args */
void espresso_context_destroy(espresso_context_t ctx);
void espresso_context_set_int_option(espresso_context_t ctx, int option,
                                     int value); /* UNVERIFIED */
void espresso_context_set_low_precision_accumulation(espresso_context_t ctx, bool enable);
void espresso_context_report_bench(espresso_context_t ctx, void* out); /* UNVERIFIED */

/* =====================================================================
 * Plan — lifecycle / build / execute / submit
 * ===================================================================== */

void espresso_plan_destroy(espresso_plan_t plan);
int espresso_plan_can_use_submit(espresso_plan_t plan);
int espresso_plan_get_phase(espresso_plan_t plan);
int espresso_plan_set_execution_queue(espresso_plan_t plan, dispatch_queue_t queue);
int espresso_plan_set_priority(espresso_plan_t plan, int priority);
int espresso_plan_build(espresso_plan_t plan);
int espresso_plan_build_with_options(espresso_plan_t plan, void* options); /* UNVERIFIED */
int espresso_plan_build_clean(espresso_plan_t plan);
int espresso_plan_execute_sync(espresso_plan_t plan);
int espresso_plan_submit(espresso_plan_t plan, dispatch_queue_t queue,
                         espresso_completion_block_t cb);
int espresso_plan_submit_with_args(espresso_plan_t plan, void* args,
                                   espresso_completion_block_t cb); /* UNVERIFIED */
int espresso_plan_submit_camera(espresso_plan_t plan,
                                void (^cb)(espresso_network_t, espresso_error_info_t*));
int espresso_plan_submit_set_multiple_buffering(espresso_plan_t plan, int count); /* UNVERIFIED */
void espresso_plan_share_intermediate_buffer(espresso_plan_t plan, void* other);  /* UNVERIFIED */
int espresso_plan_add_network(espresso_plan_t plan, const char* file_url,
                              espresso_storage_type_t storage); /* UNVERIFIED tail */
int espresso_plan_add_network_from_memory(espresso_plan_t plan, const void* data, size_t size,
                                          espresso_storage_type_t storage); /* UNVERIFIED tail */

/* =====================================================================
 * Plan — profiling
 * ===================================================================== */

void espresso_plan_activate_debug_firehose(espresso_plan_t plan);
void espresso_plan_auto_profile(espresso_plan_t plan);
void espresso_plan_start_profiling(espresso_plan_t plan);
void espresso_plan_start_profiling_with_options(espresso_plan_t plan,
                                                void* options); /* UNVERIFIED */
void espresso_plan_finish_profiling(espresso_plan_t plan);
void espresso_plan_perfbench(espresso_plan_t plan);
void espresso_plan_perfbench_nocopy(espresso_plan_t plan);
void espresso_plan_static_profiling_info(espresso_plan_t plan);

/* =====================================================================
 * Plan — errors
 * ===================================================================== */

void espresso_plan_get_error_info(espresso_plan_t plan, espresso_error_info_t* out);

/* =====================================================================
 * Network — buffer binding, shapes, and state
 * ===================================================================== */

int espresso_network_bind_buffer(espresso_network_t net, const char* blob_name,
                                 espresso_buffer_t* buf);
int espresso_network_bind_buffer_to_global(espresso_network_t net, const char* blob_name,
                                           espresso_buffer_t* buf); /* UNVERIFIED */
int espresso_network_unbind_buffer(espresso_network_t net, const char* blob_name);
int espresso_network_unbind_buffer_to_global(espresso_network_t net,
                                             const char* blob_name);             /* UNVERIFIED */
int espresso_network_swap_global(espresso_network_t net, const char* blob_name); /* UNVERIFIED */
int espresso_network_sync_copy_global(espresso_network_t net,
                                      const char* blob_name);                  /* UNVERIFIED */
int espresso_network_declare_input(espresso_network_t net, const char* name);  /* UNVERIFIED tail */
int espresso_network_declare_output(espresso_network_t net, const char* name); /* UNVERIFIED tail */
int espresso_network_change_blob_shape(espresso_network_t net, const char* blob_name,
                                       const int64_t* shape, size_t rank); /* UNVERIFIED */
int espresso_network_change_input_blob_shapes(espresso_network_t net,
                                              void* shapes); /* UNVERIFIED */
int espresso_network_change_input_blob_shapes_seq(espresso_network_t net,
                                                  void* shapes); /* UNVERIFIED */
int espresso_network_change_input_blob_shapes_seq_rank(espresso_network_t net,
                                                       void* shapes); /* UNVERIFIED */

/* THE state-specific entry point: zero all temporal (KV) state buffers. */
int espresso_network_temporal_state_reset(espresso_network_t net);

int espresso_network_query_blob_dimensions(espresso_network_t net,
                                           const char* blob_name); /* UNVERIFIED ret */
int espresso_network_query_blob_shape(espresso_network_t net,
                                      const char* blob_name); /* UNVERIFIED */
int espresso_network_query_quantization_info(espresso_network_t net,
                                             const char* blob_name); /* UNVERIFIED */
const char* espresso_network_get_version(espresso_network_t net);
int espresso_network_select_configuration(espresso_network_t net, const char* config);
int espresso_network_set_function_name(espresso_network_t net, const char* name);
int espresso_network_set_inference_weights(espresso_network_t net, void* weights); /* UNVERIFIED */
int espresso_network_set_memory_pool_id(espresso_network_t net, uint64_t pool_id); /* UNVERIFIED */
int espresso_network_set_tracing_name(espresso_network_t net, const char* name);
int espresso_network_compiler_set_metadata_key(espresso_network_t net, const char* key,
                                               const char* value);
int espresso_network_pin_weights_blob_storage(espresso_network_t net);
int espresso_network_unpin_weights_blob_storage(espresso_network_t net);
int espresso_network_dump_test_vector(espresso_network_t net, const char* path); /* UNVERIFIED */

/* Image-model input binders (legacy paths). All signatures UNVERIFIED. */
int espresso_network_bind_cvpixelbuffer(espresso_network_t net, const char* blob_name, void* src);
int espresso_network_bind_cvpixelbuffer_no_channel_swap(espresso_network_t net,
                                                        const char* blob_name, void* src);
int espresso_network_bind_direct_cvpixelbuffer(espresso_network_t net, const char* blob_name,
                                               void* src);
int espresso_network_bind_input_cvpixelbuffer(espresso_network_t net, const char* blob_name,
                                              void* src);
int espresso_network_bind_input_metaltexture(espresso_network_t net, const char* blob_name,
                                             void* src);
int espresso_network_bind_input_vimagebuffer_argb8(espresso_network_t net, const char* blob_name,
                                                   void* src);
int espresso_network_bind_input_vimagebuffer_bgra8(espresso_network_t net, const char* blob_name,
                                                   void* src);
int espresso_network_bind_input_vimagebuffer_planar8(espresso_network_t net, const char* blob_name,
                                                     void* src);
int espresso_network_bind_input_vimagebuffer_rgba8(espresso_network_t net, const char* blob_name,
                                                   void* src);

/* =====================================================================
 * ANE program cache (aned-resident, keyed by model path/URL)
 * ===================================================================== */

/* Query / evict an ANE network cache by model path. Both build a
 * `standard_ane_cache_client` over `_ANEClient`'s shared connection and run
 * the path through `model_path_to_model_url`.
 *
 * Which cache this is has NOT been pinned down: it reports not-cached for
 * every model tried so far (including Apple-shipped models the OS has
 * ANE-run), so it is not the in-memory `modelWithMILText:` hash cache
 * (`ane_cache_exists_for_hash` in ane_bridge.h) nor, apparently, aned's
 * CoreML program cache. The `e5rt_*_cache_bundle_location` /
 * `force_fetch_from_cache` symbols suggest the E5RT compiler bundle cache,
 * populated via the Espresso plan path. Treat a positive result as
 * unverified until that is confirmed.
 *
 * Signatures recovered from arm64 disassembly on macOS 26.2, not headers:
 *   has:   x0=model_path -> model_path_to_model_url; x1=out, a byte holding
 *          the presence flag is stored to *out_exists. Returns an int
 *          status (0 = success; verified: a path with no cached network
 *          yields status 0 and *out_exists = false).
 *   purge: x0=model_path -> model_path_to_model_url. Returns an int status. */
int espresso_ane_cache_has_network(const char* model_path, bool* out_exists);
int espresso_ane_cache_purge_network(const char* model_path);

/* =====================================================================
 * Buffer
 * ===================================================================== */

size_t espresso_buffer_get_count(const espresso_buffer_t* buf);
size_t espresso_buffer_get_rank(const espresso_buffer_t* buf);
size_t espresso_buffer_get_size(const espresso_buffer_t* buf);
void espresso_buffer_set_rank(espresso_buffer_t* buf, size_t rank);
void espresso_buffer_pack_tensor_shape(espresso_buffer_t* buf, size_t rank, const size_t* dims);
void espresso_buffer_unpack_tensor_shape(const espresso_buffer_t* buf, size_t* rank, size_t* dims);

/* =====================================================================
 * E5RT runtime (the engine beneath MLE5Engine)
 * ===================================================================== */

typedef void* e5rt_execution_stream_t; /* opaque execution stream         */
typedef void* e5rt_buffer_object_t;    /* opaque ANE memory provider      */
typedef int e5rt_error_code_t;         /* 0 == success; enum not reversed */

/* Stream id of a live execution stream. Borrow a stream from a loaded model
 * (the MLE5Engine path) rather than creating one here. VERIFIED. */
uint64_t e5rt_execution_stream_get_stream_id(e5rt_execution_stream_t stream);

/* Buffer objects — zero-copy ANE I/O.
 *
 * Wrap an existing IOSurface / MTLBuffer / host pointer as an
 * e5rt_buffer_object (no copy), or allocate fresh ANE memory; the getters
 * round-trip the backing handle and size. All return e5rt_error_code_t
 * (0 == success). NOTE the argument convention: create_* / alloc take the
 * OUT handle FIRST then their inputs; the get_* accessors take the handle
 * first then the out-param (the opposite order).
 *
 * Signatures were recovered from arm64 disassembly AND verified by calling:
 * each create/alloc produced a live handle whose get_* round-tripped the
 * original surface / MTLBuffer / pointer and reported the right size (see the
 * /tmp probe in the commit message / tools).
 *
 * PRECONDITION (verified): the process-global E5RT runtime must already be up
 * or these dereference an uninitialized provider and crash. One MLE5Engine
 * prediction brings it up; the Rust wrapper gates on a live execution stream.
 *
 * e5rt_buffer_object_create_as_alias exists but is UNVERIFIED and omitted. */
e5rt_error_code_t e5rt_buffer_object_create_from_iosurface(e5rt_buffer_object_t* out,
                                                           void* surface /* IOSurfaceRef */);
e5rt_error_code_t e5rt_buffer_object_create_from_data_pointer(e5rt_buffer_object_t* out, void* data,
                                                              size_t size);
e5rt_error_code_t e5rt_buffer_object_create_from_mtlbuffer(e5rt_buffer_object_t* out,
                                                           void* mtlbuffer /* id<MTLBuffer> */);
e5rt_error_code_t e5rt_buffer_object_alloc(e5rt_buffer_object_t* out, size_t size,
                                           uint32_t buffer_type); /* only type 0 verified */
e5rt_error_code_t e5rt_buffer_object_get_data_ptr(e5rt_buffer_object_t obj, void** out);
e5rt_error_code_t e5rt_buffer_object_get_iosurface(e5rt_buffer_object_t obj,
                                                   void** out /* IOSurfaceRef* */);
e5rt_error_code_t e5rt_buffer_object_get_mtlbuffer(e5rt_buffer_object_t obj,
                                                   void** out /* id<MTLBuffer>* */);
e5rt_error_code_t e5rt_buffer_object_get_size(e5rt_buffer_object_t obj, size_t* out);
e5rt_error_code_t e5rt_buffer_object_get_type(e5rt_buffer_object_t obj,
                                              uint32_t* out); /* writes 4 bytes */
e5rt_error_code_t e5rt_buffer_object_release(e5rt_buffer_object_t obj);

/* Async events — timeline synchronization.
 *
 * A timeline-semaphore primitive for ordering stream work: signal raises the
 * value, sync_wait blocks until it reaches a target (or a timeout elapses).
 * create is out-first; all return e5rt_error_code_t (0 == success). VERIFIED by
 * calling: get_name round-tripped the supplied name; signal(v) then
 * get_last_signaled_value returned v and sync_wait(v, ...) returned at once; the
 * active future value round-tripped a set/get. Needs a warm E5RT runtime.
 *
 * Unbound (present but unverified): async_notify (callback),
 * create_from_iosurface_shared_event / get_iosurface_shared_event (Metal). */
typedef void* e5rt_async_event_t;

e5rt_error_code_t e5rt_async_event_create(e5rt_async_event_t* out, const char* name,
                                          uint32_t flags); /* only flags 0 verified */
e5rt_error_code_t e5rt_async_event_signal(e5rt_async_event_t evt, uint64_t value);
e5rt_error_code_t e5rt_async_event_sync_wait(e5rt_async_event_t evt, uint64_t value,
                                             uint64_t timeout_ns);
e5rt_error_code_t e5rt_async_event_get_last_signaled_value(e5rt_async_event_t evt, uint64_t* out);
e5rt_error_code_t e5rt_async_event_get_active_future_value(e5rt_async_event_t evt, uint64_t* out);
e5rt_error_code_t e5rt_async_event_set_active_future_value(e5rt_async_event_t evt, uint64_t value);
e5rt_error_code_t e5rt_async_event_get_name(e5rt_async_event_t evt, const char** out);
e5rt_error_code_t e5rt_async_event_release(e5rt_async_event_t evt);

/* Diagnostics. error_code_get_string returns a static string literal (verified
 * for codes 0..=6: OK / INVALID FUNCTION ARGUMENT / INVALID OPERATION /
 * MEM ALLOC FAILURE / OUT OF BOUNDS ACCESS / INVALID TYPE PARAMETER /
 * INVALID DATA TYPE). get_last_error_message returns a thread-local buffer. */
const char* e5rt_error_code_get_string(e5rt_error_code_t code);
const char* e5rt_get_last_error_message(void);

/* Execution-stream tuning.
 *
 * Adjust the scheduling of an e5rt_execution_stream; all return
 * e5rt_error_code_t (0 == success). VERIFIED by calling on a borrowed
 * (CoreML-owned) stream: set_quality_of_service (a Darwin qos_class_t, e.g.
 * 0x19 user-initiated) and set_ane_execution_priority both returned 0. The
 * config-options object round-trips (create/set/get/release verified).
 *
 * WARNING: set_config_options is VERIFIED to crash (SIGBUS) on a borrowed,
 * in-use stream — reconfiguring CoreML's live stream wholesale tears down
 * engine state. Use it only on a stream you created yourself. */
typedef void* e5rt_execution_stream_config_options_t;

e5rt_error_code_t
e5rt_execution_stream_set_quality_of_service(void* stream /* e5rt_execution_stream */,
                                             uint32_t qos_class);
e5rt_error_code_t e5rt_execution_stream_set_ane_execution_priority(void* stream, uint32_t priority);
e5rt_error_code_t e5rt_execution_stream_set_config_options(
    void* stream,
    e5rt_execution_stream_config_options_t options); /* crashes on a borrowed stream */

e5rt_error_code_t
e5rt_execution_stream_config_options_create(e5rt_execution_stream_config_options_t* out);
e5rt_error_code_t e5rt_execution_stream_config_options_set_enable_concurrent_sync_execution(
    e5rt_execution_stream_config_options_t opts, bool value);
e5rt_error_code_t e5rt_execution_stream_config_options_get_enable_concurrent_sync_execution(
    e5rt_execution_stream_config_options_t opts, bool* out);
e5rt_error_code_t e5rt_execution_stream_config_options_set_enable_low_latency_async_events(
    e5rt_execution_stream_config_options_t opts, bool value);
e5rt_error_code_t e5rt_execution_stream_config_options_get_enable_low_latency_async_events(
    e5rt_execution_stream_config_options_t opts, bool* out);
e5rt_error_code_t
e5rt_execution_stream_config_options_set_skip_io_fences(e5rt_execution_stream_config_options_t opts,
                                                        bool value);
e5rt_error_code_t
e5rt_execution_stream_config_options_get_skip_io_fences(e5rt_execution_stream_config_options_t opts,
                                                        bool* out);
e5rt_error_code_t
e5rt_execution_stream_config_options_release(e5rt_execution_stream_config_options_t opts);

#ifdef __cplusplus
}
#endif

#endif /* ANE_BRIDGE_ESPRESSO_H */
