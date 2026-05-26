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
    ANE_ERR_IO         = 2,   /* file read/write failure */
    ANE_ERR_COMPILE    = 3,
    ANE_ERR_LOAD       = 4,
    ANE_ERR_EVAL       = 5,
    ANE_ERR_OOM        = 6,
    ANE_ERR_UNSUPPORTED = 7,
    ANE_ERR_TIMEOUT    = 8,
    ANE_ERR_BUSY       = 9,   /* request already in flight */
    ANE_ERR_NOT_DONE   = 10,  /* poll on still-running request */
    ANE_ERR_INTERNAL   = 99,
} AneStatus;

/* Thread-local last error message. NULL or "" if no error.
 * String is owned by the library and valid until the next library call
 * on the same thread. */
const char* ane_last_error(void);

/* =====================================================================
 * Dtypes
 * ===================================================================== */

typedef enum AneDtype {
    ANE_DTYPE_FP32  = 1,
    ANE_DTYPE_FP16  = 2,
    ANE_DTYPE_INT32 = 3,
    ANE_DTYPE_INT64 = 4,
    ANE_DTYPE_UINT8 = 5,
    ANE_DTYPE_INT8  = 6,
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
    const char* name;       /* informational; may be NULL */
    AneDtype    dtype;
    int32_t     rank;
    const int64_t* shape;
} AneTensorSpec;

/* =====================================================================
 * QoS (passed through to the private API)
 * ===================================================================== */

typedef enum AneQoS {
    ANE_QOS_DEFAULT          = 21,  /* matches existing run_mil_io behavior */
    ANE_QOS_USER_INTERACTIVE = 33,
    ANE_QOS_USER_INITIATED   = 25,
    ANE_QOS_UTILITY          = 17,
    ANE_QOS_BACKGROUND       = 9,
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
    ANE_LOCK_READ      = 1,
    ANE_LOCK_WRITE     = 2,
    ANE_LOCK_READWRITE = 3,
} AneBufferAccess;

/* Lock for CPU access. *out_ptr receives a pointer valid until unlock. */
AneStatus ane_buffer_lock(AneBuffer* buf, AneBufferAccess access, void** out_ptr);
AneStatus ane_buffer_unlock(AneBuffer* buf);

size_t   ane_buffer_nbytes(const AneBuffer* buf);
uint32_t ane_buffer_iosurface_id(const AneBuffer* buf);  /* 0 if none */

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
AneStatus ane_request_bind_input (AneRequest* req, int32_t idx, AneBuffer* buf);
AneStatus ane_request_bind_output(AneRequest* req, int32_t idx, AneBuffer* buf);

/* Fast path: library owns an internal IOSurface for input idx; the bytes
 * are memcpy'd in on each call. nbytes must equal the schema's byte size.
 * Convenient but adds a copy; use bind_* for hot loops. */
AneStatus ane_request_set_input_bytes(AneRequest* req, int32_t idx,
                                      const void* data, size_t nbytes);

/* Fast path output read. Valid only after the request has completed. */
AneStatus ane_request_get_output_bytes(AneRequest* req, int32_t idx,
                                       void* data, size_t nbytes);

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
 * Version
 * ===================================================================== */

const char* ane_bridge_version(void);

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
    ANE_FUZZ_FIELD_NAME     = 1 << 0,
    ANE_FUZZ_FIELD_TYPE     = 1 << 1,
    ANE_FUZZ_FIELD_BATCHES  = 1 << 2,
    ANE_FUZZ_FIELD_CHANNELS = 1 << 3,
    ANE_FUZZ_FIELD_DEPTH    = 1 << 4,
    ANE_FUZZ_FIELD_HEIGHT   = 1 << 5,
    ANE_FUZZ_FIELD_WIDTH    = 1 << 6,
    ANE_FUZZ_FIELD_ALL      = 0x7F,
} AneFuzzFieldMask;

/* Bit positions used by AneFuzzCase.flags. Kept in this enum so the
 * Rust mirror stays in lockstep without relying on implementation-
 * defined C bitfield packing. */
typedef enum {
    ANE_FUZZ_FLAG_BATCHES_AS_STRING  = 1 << 0,
    ANE_FUZZ_FLAG_CHANNELS_AS_STRING = 1 << 1,
    ANE_FUZZ_FLAG_DEPTH_AS_STRING    = 1 << 2,
    ANE_FUZZ_FLAG_HEIGHT_AS_STRING   = 1 << 3,
    ANE_FUZZ_FLAG_WIDTH_AS_STRING    = 1 << 4,
    ANE_FUZZ_FLAG_NAME_AS_NUMBER     = 1 << 5,
    ANE_FUZZ_FLAG_TYPE_AS_NUMBER     = 1 << 6,
} AneFuzzFlag;

typedef struct AneFuzzCase {
    uint32_t    present_mask;   /* see AneFuzzFieldMask */
    uint32_t    flags;          /* see AneFuzzFlag */
    const char* name;
    const char* type_string;
    int64_t     batches, channels, depth, height, width;
} AneFuzzCase;

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
    ANE_FUZZ_ATTRS_NSL_NOT_ARRAY        = 1 << 0,
    /* Omit `NetworkStatusList` entirely. */
    ANE_FUZZ_ATTRS_NSL_MISSING          = 1 << 1,
    /* `NetworkStatusList` is an empty array. */
    ANE_FUZZ_ATTRS_NSL_EMPTY            = 1 << 2,
    /* `NetworkStatusList[0]` is not a dictionary. */
    ANE_FUZZ_ATTRS_PROC_NOT_DICT        = 1 << 3,
    /* Omit `LiveInputList` from the procedure dict. */
    ANE_FUZZ_ATTRS_LIVEIN_MISSING       = 1 << 4,
    /* Omit `LiveOutputList` from the procedure dict. */
    ANE_FUZZ_ATTRS_LIVEOUT_MISSING      = 1 << 5,
    /* `LiveInputList` is not an array. */
    ANE_FUZZ_ATTRS_LIVEIN_NOT_ARRAY     = 1 << 6,
    /* `LiveOutputList` is not an array. */
    ANE_FUZZ_ATTRS_LIVEOUT_NOT_ARRAY    = 1 << 7,
} AneFuzzAttrsMutation;

typedef struct AneFuzzAttrsCase {
    /* Mutation bitmask. */
    uint32_t mutations;
    /* Number of well-formed entries to put in LiveInputList (unless
     * a mutation overrides). */
    int32_t  n_inputs;
    /* Number of well-formed entries to put in LiveOutputList. */
    int32_t  n_outputs;
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

#ifdef __cplusplus
}
#endif

#endif /* ANE_BRIDGE_H */
