/* ane_bridge.h — Generic C interface to the Apple Neural Engine.
 *
 * This is a thin, opinionated wrapper around Apple's private
 * AppleNeuralEngine.framework that lets you compile a MIL text program with
 * an associated weights blob, then dispatch evaluations asynchronously with
 * IOSurface-backed zero-copy buffers.
 *
 * It is generic: the library has no knowledge of any particular model. The
 * caller supplies the MIL + weights and declares the input/output schema by
 * index. The library handles compile/load, IOSurface lifetimes, request
 * dispatch, and async wait/callback semantics.
 *
 * Threading model
 *   - AneModel is read-only after open; safe to share across threads.
 *   - AneRequest is single-owner: one submit at a time per request. Create
 *     multiple AneRequests to overlap in-flight evaluations.
 *   - AneBuffer is not thread-safe; the owning thread must serialize
 *     lock/unlock and binding into requests.
 *
 * Error reporting
 *   - Every fallible call returns AneStatus. A non-zero return implies a
 *     human-readable message is available via ane_last_error() on the same
 *     thread.
 */
#ifndef ANE_BRIDGE_H
#define ANE_BRIDGE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =====================================================================
 * Status / errors
 * ===================================================================== */

typedef enum AneStatus {
    ANE_OK = 0,
    ANE_ERR_INVALID_ARG = 1,
    ANE_ERR_IO = 2, /* file read/write failure */
    ANE_ERR_COMPILE = 3,
    ANE_ERR_LOAD = 4,
    ANE_ERR_EVAL = 5,
    ANE_ERR_OOM = 6,
    ANE_ERR_UNSUPPORTED = 7,
    ANE_ERR_TIMEOUT = 8,
    ANE_ERR_BUSY = 9,      /* request already in flight */
    ANE_ERR_NOT_DONE = 10, /* poll on still-running request */
    ANE_ERR_INTERNAL = 99,
} AneStatus;

/* Thread-local last error message. NULL or "" if no error.
 * String is owned by the library and valid until the next library call
 * on the same thread. */
const char* ane_last_error(void);

/* =====================================================================
 * Dtypes
 * ===================================================================== */

typedef enum AneDtype {
    ANE_DTYPE_FP32 = 1,
    ANE_DTYPE_FP16 = 2,
    ANE_DTYPE_INT32 = 3,
    ANE_DTYPE_INT64 = 4,
    ANE_DTYPE_UINT8 = 5,
    ANE_DTYPE_INT8 = 6,
} AneDtype;

/* Size in bytes of one element of dtype. 0 if unknown. */
size_t ane_dtype_size(AneDtype dt);

/* =====================================================================
 * Tensor spec
 * ===================================================================== */

/* Caller-supplied schema for one tensor. The library copies all fields
 * out at ane_model_open() time; the caller's memory does not need to
 * outlive the open call. Rank is the number of dimensions; shape[i] is
 * the extent along axis i. Static shapes only (no dynamic axes). */
typedef struct AneTensorSpec {
    const char* name; /* informational; may be NULL */
    AneDtype dtype;
    int32_t rank;
    const int64_t* shape;
} AneTensorSpec;

/* =====================================================================
 * QoS (passed through to the private API)
 * ===================================================================== */

typedef enum AneQoS {
    ANE_QOS_DEFAULT = 21, /* matches existing run_mil_io behavior */
    ANE_QOS_USER_INTERACTIVE = 33,
    ANE_QOS_USER_INITIATED = 25,
    ANE_QOS_UTILITY = 17,
    ANE_QOS_BACKGROUND = 9,
} AneQoS;

/* =====================================================================
 * Model
 * ===================================================================== */

typedef struct AneModel AneModel;

typedef struct AneModelOpenOptions {
    /* Path to a MIL text file. Required. */
    const char* mil_path;
    /* Path to a single weights blob referenced from MIL as
     * @model_path/weights/weight.bin. Required. */
    const char* weights_path;
    /* QoS used during compile + load. 0 → ANE_QOS_DEFAULT. */
    AneQoS compile_qos;
} AneModelOpenOptions;

AneStatus ane_model_open(const AneModelOpenOptions* opts, AneModel** out_model);
void ane_model_close(AneModel* model);

int32_t ane_model_num_inputs(const AneModel* model);
int32_t ane_model_num_outputs(const AneModel* model);

/* Returns a pointer into model-owned storage; valid until ane_model_close. */
const AneTensorSpec* ane_model_input_spec(const AneModel* model, int32_t idx);
const AneTensorSpec* ane_model_output_spec(const AneModel* model, int32_t idx);

/* Convenience: byte size of input/output idx (= product of shape * dtype_size). */
size_t ane_model_input_nbytes(const AneModel* model, int32_t idx);
size_t ane_model_output_nbytes(const AneModel* model, int32_t idx);

/* True if `ane_model_open` reused a cached lowering instead of running a
 * fresh compile. The cache lives in the aned daemon, keyed by content
 * hash; it survives across processes but not aned restarts or its
 * (opaque) eviction policy. Use this to confirm a fast warm start, or
 * to surface the cold-compile cost in your own metrics. */
bool ane_model_was_cached(const AneModel* model);

/* =====================================================================
 * Buffer (zero-copy, IOSurface-backed)
 * ===================================================================== */

typedef struct AneBuffer AneBuffer;

/* Create an IOSurface-backed buffer of nbytes. The buffer can be shared
 * across requests and reused; lock/unlock for CPU access. */
AneStatus ane_buffer_create(size_t nbytes, AneBuffer** out);

/* Convenience: create a buffer sized for input/output idx of model. */
AneStatus ane_buffer_create_for_input(const AneModel* model, int32_t idx, AneBuffer** out);
AneStatus ane_buffer_create_for_output(const AneModel* model, int32_t idx, AneBuffer** out);

void ane_buffer_release(AneBuffer* buf);

typedef enum AneBufferAccess {
    ANE_LOCK_READ = 1,
    ANE_LOCK_WRITE = 2,
    ANE_LOCK_READWRITE = 3,
} AneBufferAccess;

/* Lock for CPU access. *out_ptr receives a pointer valid until unlock. */
AneStatus ane_buffer_lock(AneBuffer* buf, AneBufferAccess access, void** out_ptr);
AneStatus ane_buffer_unlock(AneBuffer* buf);

size_t ane_buffer_nbytes(const AneBuffer* buf);
uint32_t ane_buffer_iosurface_id(const AneBuffer* buf); /* 0 if none */

/* =====================================================================
 * Request (one inference instance)
 * ===================================================================== */

typedef struct AneRequest AneRequest;

AneStatus ane_request_create(AneModel* model, AneRequest** out);
void ane_request_release(AneRequest* req);

/* Zero-copy: bind a caller-owned buffer for input/output idx. The buffer
 * must remain alive (i.e., the caller must not release it) until the
 * request itself is released, or until another buffer is bound to the
 * same index. Multiple requests may bind the same buffer.
 *
 * All inputs and outputs must be bound (by either bind_* or set_/get_bytes
 * fast-path APIs) before ane_request_submit. */
AneStatus ane_request_bind_input(AneRequest* req, int32_t idx, AneBuffer* buf);
AneStatus ane_request_bind_output(AneRequest* req, int32_t idx, AneBuffer* buf);

/* Fast path: library owns an internal IOSurface for input idx; the bytes
 * are memcpy'd in on each call. nbytes must equal the schema's byte size.
 * Convenient but adds a copy; use bind_* for hot loops. */
AneStatus ane_request_set_input_bytes(AneRequest* req, int32_t idx, const void* data,
                                      size_t nbytes);

/* Fast path output read. Valid only after the request has completed. */
AneStatus ane_request_get_output_bytes(AneRequest* req, int32_t idx, void* data, size_t nbytes);

/* Submit asynchronously. Non-blocking. Returns ANE_OK on enqueue, or
 * ANE_ERR_BUSY if a prior submit on this request hasn't completed yet. */
AneStatus ane_request_submit(AneRequest* req, AneQoS qos);

/* Block until the in-flight submission completes.
 *   timeout_ms < 0  → wait forever
 *   timeout_ms == 0 → poll (returns ANE_OK if done, ANE_ERR_NOT_DONE otherwise)
 *   timeout_ms > 0  → wait up to timeout, returns ANE_ERR_TIMEOUT on expiry
 * Returns the eval status (ANE_OK or ANE_ERR_EVAL) once done. */
AneStatus ane_request_wait(AneRequest* req, int32_t timeout_ms);

/* Non-blocking completion check. */
bool ane_request_is_done(const AneRequest* req);

/* Convenience: submit + wait(-1). */
AneStatus ane_request_run(AneRequest* req, AneQoS qos);

/* Optional completion callback fired from an internal worker thread when
 * an in-flight eval completes. Setting fn=NULL clears any prior callback.
 * Must be set before ane_request_submit; setting after submit is racy.
 *
 * The callback runs on a library-owned thread; do not call back into the
 * library with the same request from within the callback. */
typedef void (*AneCompletionFn)(AneRequest* req, AneStatus status, void* user);
AneStatus ane_request_set_completion(AneRequest* req, AneCompletionFn fn, void* user);

/* Returns the most recent per-request error message (or "" if none).
 *
 * Unlike ane_last_error() which is thread-local and reflects the most
 * recent library call on the *calling* thread, this accessor returns the
 * error captured by the worker thread that ran the eval. Useful from
 * completion callbacks. The returned pointer is valid until the next
 * submit on this request or until the request is released. */
const char* ane_request_last_error(const AneRequest* req);

/* =====================================================================
 * Resident state (MLState)
 *
 * A MIL program may declare state tensors that live on the ANE and
 * persist across evaluations. Unlike an input/output, a bound state
 * buffer is read and updated in place by the program's read_state /
 * coreml.update_state ops and never crosses the host boundary per call --
 * so a streaming cache stays engine-resident instead of being shipped in
 * and out every submit.
 *
 * Lifecycle: bind a persistent, caller-owned buffer to a state slot once.
 * Its contents survive every submit on the request until it is rebound or
 * the request is released. Initialize it (e.g. zero) before the first
 * submit. The buffer must outlive the request (same rule as bind_input).
 *
 * A state slot is NOT an input or output: it does not count against
 * num_inputs / num_outputs and is bound through bind_state, not
 * bind_input. All inputs, outputs, AND states must be bound before submit.
 * ===================================================================== */

int32_t ane_model_num_states(const AneModel* model);
const AneTensorSpec* ane_model_state_spec(const AneModel* model, int32_t idx);
size_t ane_model_state_nbytes(const AneModel* model, int32_t idx);

/* Convenience: IOSurface buffer sized for state slot idx. */
AneStatus ane_buffer_create_for_state(const AneModel* model, int32_t idx, AneBuffer** out);

/* Bind a persistent buffer to state slot idx. Persists + updates in place
 * across submits. */
AneStatus ane_request_bind_state(AneRequest* req, int32_t idx, AneBuffer* buf);

/* Per-procedure variants, mirroring the multi-procedure I/O accessors. */
int32_t ane_model_num_states_for_procedure(const AneModel* model, int32_t proc_idx);
const AneTensorSpec* ane_model_state_spec_for_procedure(const AneModel* model, int32_t proc_idx,
                                                        int32_t idx);
size_t ane_model_state_nbytes_for_procedure(const AneModel* model, int32_t proc_idx, int32_t idx);

/* =====================================================================
 * Version
 * ===================================================================== */

const char* ane_bridge_version(void);

/* =====================================================================
 * Device info
 * ===================================================================== */

typedef struct AneDeviceInfo {
    int32_t num_cores;
    int32_t num_anes;
    bool has_ane;
    int64_t board_type;
    char arch_type[32];
} AneDeviceInfo;

AneStatus ane_device_info(AneDeviceInfo* out);

/* =====================================================================
 * Multi-procedure schema
 * ===================================================================== */

int32_t ane_model_num_procedures(const AneModel* model);

int32_t ane_model_num_inputs_for_procedure(const AneModel* model, int32_t proc_idx);
int32_t ane_model_num_outputs_for_procedure(const AneModel* model, int32_t proc_idx);

const AneTensorSpec* ane_model_input_spec_for_procedure(const AneModel* model, int32_t proc_idx,
                                                        int32_t idx);
const AneTensorSpec* ane_model_output_spec_for_procedure(const AneModel* model, int32_t proc_idx,
                                                         int32_t idx);

size_t ane_model_input_nbytes_for_procedure(const AneModel* model, int32_t proc_idx, int32_t idx);
size_t ane_model_output_nbytes_for_procedure(const AneModel* model, int32_t proc_idx, int32_t idx);

/* =====================================================================
 * Queue depth / in-flight telemetry
 * ===================================================================== */

int32_t ane_model_queue_depth(const AneModel* model);
int64_t ane_model_in_flight(const AneModel* model);

/* =====================================================================
 * Identity / cache telemetry
 * ===================================================================== */

const char* ane_model_program_id(const AneModel* model);   /* hexStringIdentifier */
const char* ane_model_weights_hash(const AneModel* model); /* descriptor weightsHash */
bool ane_model_program_handle(const AneModel* model, uint64_t* out);
bool ane_model_intermediate_buffer_handle(const AneModel* model, uint64_t* out);

bool ane_cache_exists_for_hash(const char* hex_hash);
AneStatus ane_cache_purge_for_hash(const char* hex_hash);

/* =====================================================================
 * Extended open: in-memory MIL bytes + multi-blob NSDictionary weights
 * ===================================================================== */

typedef struct AneWeightEntry {
    /* Key name inside the weights NSDictionary. Required. */
    const char* name;
    /* Exactly one of (path) OR (bytes,nbytes) must be set. */
    const char* path;
    const void* bytes;
    size_t nbytes;
} AneWeightEntry;

typedef struct AneModelOpenOptionsEx {
    /* Provide MIL either by path or by in-memory bytes. */
    const char* mil_path;
    const void* mil_bytes;
    size_t mil_nbytes;

    /* Multiple named weight blobs. NULL+0 is permitted for weight-less
     * MIL programs. */
    const AneWeightEntry* weights;
    int32_t n_weights;

    AneQoS compile_qos;

    /* When false, MIL text is treated as a NetworkDescription
     * (non-MIL legacy input format). Default true. */
    bool is_mil_model;

    /* Optional options plist; pass NULL for none. JSON-encoded UTF-8. */
    const char* options_plist_json;
} AneModelOpenOptionsEx;

AneStatus ane_model_open_ex(const AneModelOpenOptionsEx* opts, AneModel** out_model);

/* =====================================================================
 * File-based open path (_ANEModel modelAtURL:key:)
 * ===================================================================== */

typedef enum AneIdentifierSource {
    ANE_IDENT_DEFAULT = 0,
    ANE_IDENT_URL = 1,
    ANE_IDENT_UUID = 2,
    ANE_IDENT_CONTENT = 3,
} AneIdentifierSource;

typedef struct AneModelFileOpenOptions {
    /* Compiled .mlmodelc / .bundle directory URL or file path. Required. */
    const char* model_url;
    /* Explicit cache key. NULL → derive from URL. */
    const char* cache_key;
    /* Optional cacheURLIdentifier. NULL → derive. */
    const char* cache_url_identifier;
    /* See `AneIdentifierSource`. */
    AneIdentifierSource identifier_source;
    AneQoS compile_qos;
} AneModelFileOpenOptions;

AneStatus ane_model_open_file(const AneModelFileOpenOptions* opts, AneModel** out_model);

/* =====================================================================
 * Real-time priority class
 * ===================================================================== */

AneStatus ane_model_open_realtime(const AneModelOpenOptions* opts, AneModel** out_model);
AneStatus ane_model_open_realtime_ex(const AneModelOpenOptionsEx* opts, AneModel** out_model);
AneStatus ane_realtime_task_begin(void);
AneStatus ane_realtime_task_end(void);

/* =====================================================================
 * Performance stats
 * ===================================================================== */

typedef struct AnePerfStats AnePerfStats;

AneStatus ane_perf_stats_create(AnePerfStats** out);
void ane_perf_stats_release(AnePerfStats* ps);

/* Hardware execution time of the most recent eval that referenced this
 * stats object, in nanoseconds. 0 if not yet populated. */
uint64_t ane_perf_stats_hw_execution_ns(const AnePerfStats* ps);

/* Raw counter blob; size first via *_nbytes then copy with *_copy. */
size_t ane_perf_stats_counters_nbytes(const AnePerfStats* ps);
size_t ane_perf_stats_counters_copy(const AnePerfStats* ps, void* out, size_t cap);

/* Per-model mask controlling which hardware counters are populated.
 * Set BEFORE `ane_request_run`/`submit`. */
AneStatus ane_model_set_perf_stats_mask(AneModel* model, uint32_t mask);
uint32_t ane_model_get_perf_stats_mask(const AneModel* model);

/* =====================================================================
 * GPU↔ANE shared event sync (Metal interop)
 * ===================================================================== */

typedef struct AneSharedEvents AneSharedEvents;

typedef enum AneEventType {
    ANE_EVT_DEFAULT = 0,
    ANE_EVT_INFERENCE = 1,
    ANE_EVT_COMPLETION = 2,
} AneEventType;

AneStatus ane_shared_events_create(AneSharedEvents** out);
void ane_shared_events_release(AneSharedEvents* ev);

/* `mtl_shared_event` is a (void*)id pointer to a Metal `MTLSharedEvent`
 * (or compatible). It is retained by the events object.
 *
 * `agent_mask` selects which ANE agent should signal. Pass 0 for default. */
AneStatus ane_shared_events_add_signal(AneSharedEvents* ev, uint64_t value, uint32_t symbol_index,
                                       AneEventType event_type, void* mtl_shared_event,
                                       uint64_t agent_mask);

AneStatus ane_shared_events_add_wait(AneSharedEvents* ev, uint64_t value, void* mtl_shared_event,
                                     AneEventType event_type);

int32_t ane_shared_events_num_signals(const AneSharedEvents* ev);
int32_t ane_shared_events_num_waits(const AneSharedEvents* ev);

/* =====================================================================
 * Extended request configuration
 * ===================================================================== */

/* Per-request weights override. The buffer remains caller-owned and
 * must outlive the request (or until cleared by passing NULL). */
AneStatus ane_request_set_weights(AneRequest* req, AneBuffer* weights);
AneStatus ane_request_set_procedure_index(AneRequest* req, int32_t proc_idx);
AneStatus ane_request_set_perf_stats(AneRequest* req, AnePerfStats* ps);
AneStatus ane_request_set_shared_events(AneRequest* req, AneSharedEvents* ev);
AneStatus ane_request_set_transaction(AneRequest* req, uint64_t handle);

int32_t ane_request_procedure_index(const AneRequest* req);
uint64_t ane_request_transaction(const AneRequest* req);

/* =====================================================================
 * IOSurface interop
 * ===================================================================== */

/* Returns a borrowed `IOSurfaceRef`. Caller must NOT release it.
 * Cast the returned `void*` to `IOSurfaceRef` after including
 * <IOSurface/IOSurfaceRef.h>. */
void* ane_buffer_iosurface_ref(const AneBuffer* buf);

/* Adopt a caller-owned `IOSurfaceRef` into a fresh `AneBuffer`.
 * The buffer retains the surface; you may release your own reference
 * after this call returns OK. `nbytes` is the logical payload size. */
AneStatus ane_buffer_adopt_iosurface(void* iosurface_ref, size_t nbytes, AneBuffer** out);

/* =====================================================================
 * Multi-model chaining (_ANEChainingRequest)
 * ===================================================================== */

typedef struct AneChain AneChain;

typedef struct AneChainStep {
    /* Request whose bindings define this stage's I/O. The request's
     * model + buffers + procedure index + signal events are used.
     * The request must already have its inputs/outputs bound. */
    AneRequest* request;

    /* Loopback symbol IDs wiring this stage's output to the next
     * stage's input inside ANE without a host round-trip. */
    int64_t lb_input_symbol_id;
    int64_t lb_output_symbol_id;

    /* Firmware enqueue delay in nanoseconds (0 = no delay). */
    uint64_t fw_enqueue_delay;

    /* Shared intermediate memory pool ID for this step (0 = no pool). */
    uint64_t memory_pool_id;
} AneChainStep;

AneStatus ane_chain_create(const AneChainStep* steps, int32_t n_steps, AneChain** out);
AneStatus ane_chain_prepare(AneChain* chain, AneQoS qos);
AneStatus ane_chain_enqueue(AneChain* chain, AneQoS qos);
AneStatus ane_chain_wait(AneChain* chain, int32_t timeout_ms);
void ane_chain_release(AneChain* chain);

/* =====================================================================
 * Model accessors (post-load identity + state)
 * ===================================================================== */

const char* ane_model_uuid(const AneModel* model);
const char* ane_model_source_url(const AneModel* model);
const char* ane_model_model_url(const AneModel* model);
const char* ane_model_key(const AneModel* model);
const char* ane_model_cache_url_identifier(const AneModel* model);
int64_t ane_model_identifier_source(const AneModel* model);

AneStatus ane_model_reset_on_unload(AneModel* model);
AneStatus ane_model_unload(AneModel* model);

/* =====================================================================
 * Load a fresh instance of an already-compiled model
 *
 * Returns a second `AneModel*` that shares the compiled program with
 * `src` but owns its own runtime state. Useful for the
 * train-from-checkpoint pattern (one instance per parameter snapshot)
 * without re-paying the compile.
 * ===================================================================== */

typedef struct AneModelInstanceParams {
    /* Optional explicit key override; NULL inherits from `src`. */
    const char* key;
    /* Reserved; pass 0. */
    uint64_t flags;
} AneModelInstanceParams;

AneStatus ane_model_new_instance(AneModel* src, const AneModelInstanceParams* params, AneQoS qos,
                                 AneModel** out);

/* =====================================================================
 * Connection management
 * ===================================================================== */

int32_t ane_client_num_connections(void);
bool ane_model_is_virtual_client(const AneModel* model);

/* =====================================================================
 * Session hints (load-tuning advisories)
 * ===================================================================== */

typedef struct AneSessionHint AneSessionHint;

typedef enum AneSessionHintKind {
    ANE_HINT_PREFETCH = 1,
    ANE_HINT_LOW_LATENCY = 2,
    ANE_HINT_HIGH_THROUGHPUT = 3,
} AneSessionHintKind;

AneStatus ane_session_hint_create(AneSessionHintKind kind, AneSessionHint** out);
void ane_session_hint_release(AneSessionHint* hint);

/* Apply `hint` to `model`. Writes the framework's per-hint report
 * (an opaque NSData/NSDictionary description) into a freshly-malloc'd
 * UTF-8 string at `*out_report_json` if non-NULL; caller frees with
 * `free()`. */
AneStatus ane_model_apply_session_hint(AneModel* model, const AneSessionHint* hint,
                                       char** out_report_json);

/* =====================================================================
 * Metal Performance Shaders constants (file-open interop)
 *
 * `mps_constants_id` is an Obj-C `id` pointer to an
 * `NSDictionary*` (or compatible) keyed by MPS constant names.
 * Pass NULL when not using MPS.
 * ===================================================================== */

typedef struct AneModelFileOpenOptionsEx {
    AneModelFileOpenOptions base;
    void* mps_constants_id;
    /* Optional dictionary of `modelAttributes` overrides;
     * NSDictionary id, may be NULL. */
    void* model_attributes_id;
} AneModelFileOpenOptionsEx;

AneStatus ane_model_open_file_ex(const AneModelFileOpenOptionsEx* opts, AneModel** out_model);

/* =====================================================================
 * Performance counter naming / signpost emission
 * ===================================================================== */

/* Name of perf counter at index `counter_idx`. Library-owned; empty on
 * out-of-range. */
const char* ane_perf_counter_name(int32_t counter_idx);

/* Emit os_signpost events for the perf counters captured by `ps`. */
AneStatus ane_perf_stats_emit_signpost(const AnePerfStats* ps, uint64_t model_string_id);

/* Number of per-stage stats objects on a chained request. Returns 0 if
 * the underlying `_ANERequest` has no `perfStatsArray`. */
int32_t ane_request_num_perf_stats(const AneRequest* req);

/* Copy out the perf-stats object at `idx`. Returns NULL on out-of-range. */
AnePerfStats* ane_request_perf_stats_at(const AneRequest* req, int32_t idx);

/* =====================================================================
 * Cached-bundle / weight decompression
 * ===================================================================== */

/* If the framework exposes a parallel decompressor for compressed weight
 * blobs, this routes through it. Returns OK and writes the decompressed
 * bytes to a freshly malloc()'d buffer in `*out_bytes`; caller frees
 * with `free()`. */
AneStatus ane_decompress_weights(const void* compressed, size_t cn, void** out_bytes,
                                 size_t* out_nbytes);

/* =====================================================================
 * Stateful inference (CoreML MLModel + MLState) — ANE-resident state
 *
 * A SEPARATE backend from everything above. Models that declare state
 * (read_state / coreml_update_state — e.g. a KV cache) cannot compile through
 * the private _ANE path (ANECCompile rejects state ops); they go through
 * CoreML, which builds the MLE5Engine / E5RT execution stream that keeps the
 * state resident on the ANE across calls. Only the (small) inputs/outputs
 * cross the host<->device boundary per step — the state never does — and the
 * path needs no ANE entitlement. Backed by ane_state.m (links CoreML).
 * ===================================================================== */

typedef struct AneStateModel AneStateModel; /* wraps MLModel */
typedef struct AneState AneState;           /* wraps MLState (resident KV cache) */

/* Open a compiled `.mlmodelc` (or `.mlpackage`, compiled on the fly) that
 * declares one or more state buffers. Pinned to CPU+ANE. */
AneStatus ane_state_model_open(const char* model_url, AneStateModel** out_model);
void ane_state_model_close(AneStateModel* m);

/* Schema. Names are stable, sorted, and owned by the model handle. */
int32_t ane_state_model_num_inputs(const AneStateModel* m);
int32_t ane_state_model_num_outputs(const AneStateModel* m);
int32_t ane_state_model_num_states(const AneStateModel* m);
const char* ane_state_model_input_name(const AneStateModel* m, int32_t i);  /* NULL if oob */
const char* ane_state_model_output_name(const AneStateModel* m, int32_t i); /* NULL if oob */
const char* ane_state_model_state_name(const AneStateModel* m, int32_t i);  /* NULL if oob */

/* Declared element count for a named input/output (product of dims; 0 if
 * unknown). Use to size the flat f32 buffers passed to predict. */
size_t ane_state_model_input_count(const AneStateModel* m, const char* name);
size_t ane_state_model_output_count(const AneStateModel* m, const char* name);

/* Allocate a fresh resident state (e.g. a zeroed KV cache) for this model. */
AneStatus ane_state_create(AneStateModel* m, AneState** out_state);
void ane_state_release(AneState* s);

/* One inference step using the resident state. Inputs/outputs are bound by
 * name to flat row-major f32 buffers (the library converts to/from the
 * model's declared dtype). Counts must equal the model's declared element
 * counts. The state is read and updated in place on the ANE and is never
 * copied across the host boundary. */
AneStatus ane_state_predict_f32(AneStateModel* m, AneState* s, const char* const* in_names,
                                const float* const* in_data, const size_t* in_counts, int32_t n_in,
                                const char* const* out_names, float* const* out_data,
                                const size_t* out_counts, int32_t n_out);

/* Most recent error on this thread from an ane_state_* call, or NULL. */
const char* ane_state_last_error(void);

/* =====================================================================
 * Internal test hooks (NOT part of the stable API)
 *
 * These functions exist purely so Rust integration tests can exercise
 * the C-side spec-derivation parser with adversarial inputs without
 * having to construct `NSArray`s of `NSDictionary`s by hand from Rust.
 * They are subject to removal without notice.
 * ===================================================================== */

/* Bitmask of which keys are present in the synthesized LiveInputList
 * entry that `_ane_internal_fuzz_parse_one` builds. */
typedef enum {
    ANE_FUZZ_FIELD_NAME = 1 << 0,
    ANE_FUZZ_FIELD_TYPE = 1 << 1,
    ANE_FUZZ_FIELD_BATCHES = 1 << 2,
    ANE_FUZZ_FIELD_CHANNELS = 1 << 3,
    ANE_FUZZ_FIELD_DEPTH = 1 << 4,
    ANE_FUZZ_FIELD_HEIGHT = 1 << 5,
    ANE_FUZZ_FIELD_WIDTH = 1 << 6,
    ANE_FUZZ_FIELD_ALL = 0x7F,
} AneFuzzFieldMask;

/* Bit positions used by AneFuzzCase.flags. Kept in this enum so the
 * Rust mirror stays in lockstep without relying on implementation-
 * defined C bitfield packing. */
typedef enum {
    ANE_FUZZ_FLAG_BATCHES_AS_STRING = 1 << 0,
    ANE_FUZZ_FLAG_CHANNELS_AS_STRING = 1 << 1,
    ANE_FUZZ_FLAG_DEPTH_AS_STRING = 1 << 2,
    ANE_FUZZ_FLAG_HEIGHT_AS_STRING = 1 << 3,
    ANE_FUZZ_FLAG_WIDTH_AS_STRING = 1 << 4,
    ANE_FUZZ_FLAG_NAME_AS_NUMBER = 1 << 5,
    ANE_FUZZ_FLAG_TYPE_AS_NUMBER = 1 << 6,
} AneFuzzFlag;

typedef struct AneFuzzCase {
    uint32_t present_mask; /* see AneFuzzFieldMask */
    uint32_t flags;        /* see AneFuzzFlag */
    const char* name;
    const char* type_string;
    int64_t batches, channels, depth, height, width;
} AneFuzzCase;

/* The fuzz hooks below intentionally take a leading underscore to mark them
 * internal / do-not-link. A leading underscore at file scope is reserved for
 * the implementation by the C standard, so -Wreserved-identifier is silenced
 * for this block only — the names are ours and the convention is deliberate.
 * (Renaming would ripple across the Rust FFI bindings that reference them.) */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-identifier"

/* Synthesize a one-entry LiveInputList from `fc`, run it through the
 * production spec parser, and return the resulting AneStatus. Used
 * only by adversarial tests; the function name is underscored to
 * mark it as internal. */
AneStatus _ane_internal_fuzz_parse_one(const AneFuzzCase* fc);

/* Bitmask controlling how `_ane_internal_fuzz_parse_attrs` synthesizes
 * an outer `modelAttributes` dictionary. Each ON bit either adds or
 * mutates a key to test the type-checks in `derive_specs_from_attrs`. */
typedef enum {
    /* Replace `NetworkStatusList` value with a non-array. */
    ANE_FUZZ_ATTRS_NSL_NOT_ARRAY = 1 << 0,
    /* Omit `NetworkStatusList` entirely. */
    ANE_FUZZ_ATTRS_NSL_MISSING = 1 << 1,
    /* `NetworkStatusList` is an empty array. */
    ANE_FUZZ_ATTRS_NSL_EMPTY = 1 << 2,
    /* `NetworkStatusList[0]` is not a dictionary. */
    ANE_FUZZ_ATTRS_PROC_NOT_DICT = 1 << 3,
    /* Omit `LiveInputList` from the procedure dict. */
    ANE_FUZZ_ATTRS_LIVEIN_MISSING = 1 << 4,
    /* Omit `LiveOutputList` from the procedure dict. */
    ANE_FUZZ_ATTRS_LIVEOUT_MISSING = 1 << 5,
    /* `LiveInputList` is not an array. */
    ANE_FUZZ_ATTRS_LIVEIN_NOT_ARRAY = 1 << 6,
    /* `LiveOutputList` is not an array. */
    ANE_FUZZ_ATTRS_LIVEOUT_NOT_ARRAY = 1 << 7,
} AneFuzzAttrsMutation;

typedef struct AneFuzzAttrsCase {
    /* Mutation bitmask. */
    uint32_t mutations;
    /* Number of well-formed entries to put in LiveInputList (unless
     * a mutation overrides). */
    int32_t n_inputs;
    /* Number of well-formed entries to put in LiveOutputList. */
    int32_t n_outputs;
} AneFuzzAttrsCase;

/* Synthesize a full `modelAttributes`-shaped dictionary controlled by
 * `fc` and run it through `derive_specs_from_attrs`. Returns the
 * AneStatus that the production code would have returned for this
 * input. Test-only. */
AneStatus _ane_internal_fuzz_parse_attrs(const AneFuzzAttrsCase* fc);

/* Wrap a non-integer NSNumber (a double) as the `Batches` value and
 * run the leaf parser. Exposes the "longLongValue on a double" UB
 * shape. `dbl_value` is the double we wrap. */
AneStatus _ane_internal_fuzz_parse_one_with_double_batches(double dbl_value);

/* Replace the dim at `which_dim` (0=Batches, 1=Channels, 2=Depth,
 * 3=Height, 4=Width) with an NSNumber wrapping the given double.
 * Other dims are the canonical valid values. */
AneStatus _ane_internal_fuzz_dim_as_double(int32_t which_dim, double dbl_value);

/* Replace the dim at `which_dim` with an NSNumber created from an
 * unsigned 64-bit value (`@(uint64_t)`). Forces `longLongValue` to
 * cross the signed-overflow boundary at UINT64_MAX/2. */
AneStatus _ane_internal_fuzz_dim_as_uint64(int32_t which_dim, uint64_t value);

/* Replace the dim at `which_dim` with an `NSDecimalNumber` constructed
 * from the given string. NSDecimalNumber is an NSNumber subclass; its
 * `longLongValue` has its own implementation worth exercising. */
AneStatus _ane_internal_fuzz_dim_as_decimal(int32_t which_dim, const char* decimal_str);

/* Replace one dict value with `NSNull`. `which_key` selects:
 *   0=Name, 1=Type, 2=Batches, 3=Channels, 4=Depth, 5=Height, 6=Width.
 * Tests that the `isKindOfClass:` guards correctly reject NSNull. */
AneStatus _ane_internal_fuzz_value_is_nsnull(int32_t which_key);

/* Use a `Name` with an embedded NUL byte: "valid\0continued". The
 * parser uses `strdup` after `UTF8String`, so we expect the name to
 * be silently truncated at the NUL — never crash. */
AneStatus _ane_internal_fuzz_name_with_embedded_nul(void);

/* Use a `Name` of arbitrary length (filled with 'a'). Tests strdup
 * + buffer-size handling on long names. */
AneStatus _ane_internal_fuzz_huge_name(size_t length);

/* Run a 2-entry LiveInputList where the FIRST entry is well-formed
 * and the SECOND has an invalid Type. The parser must reject the
 * whole list, not partial-accept just the first entry. */
AneStatus _ane_internal_fuzz_mixed_validity_two_entries(void);

#pragma clang diagnostic pop

#ifdef __cplusplus
}
#endif

#endif /* ANE_BRIDGE_H */
