/* ane_state.m — Stateful inference backend via CoreML (MLModel + MLState).
 *
 * A SEPARATE backend from the _ANE* bridge in ane_bridge.m. Models that
 * declare state (read_state / coreml_update_state — e.g. a KV cache) cannot be
 * compiled through the private _ANE compile path (ANECCompile rejects state
 * ops). They must go through CoreML, which builds the MLE5Engine / E5RT
 * execution stream that keeps the state ANE-resident across calls. Only the
 * (small) inputs/outputs cross the host<->device boundary per call; the state
 * never does. This path needs no ANE entitlement — it is the sanctioned
 * CoreML route, just driven from C.
 *
 * Verified end to end: see tools/validate_stateful_ane.py and
 * rust/ane-bridge/tests/stateful.rs.
 */
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <objc/runtime.h>

#include <dlfcn.h>
#include <limits.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include "ane_bridge.h" /* AneStatus */

/* ------------------------------------------------------------------ */
/* Error reporting. Per-thread, mirroring ane_bridge.m's pthread-key   */
/* approach (its buffer is static, so we keep our own here).           */
/* ------------------------------------------------------------------ */

static pthread_key_t g_state_err_key;
static pthread_once_t g_state_err_once = PTHREAD_ONCE_INIT;

static void state_err_destructor(void* p) {
    free(p);
}
static void state_err_key_init(void) {
    pthread_key_create(&g_state_err_key, state_err_destructor);
}

__attribute__((format(printf, 1, 2))) static void state_err(const char* fmt, ...) {
    pthread_once(&g_state_err_once, state_err_key_init);
    char* buf = (char*)malloc(512);
    if (!buf) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
    vsnprintf(buf, 512, fmt, ap);
#pragma clang diagnostic pop
    va_end(ap);
    char* prev = (char*)pthread_getspecific(g_state_err_key);
    if (prev) {
        free(prev);
    }
    pthread_setspecific(g_state_err_key, buf);
}

static void state_err_clear(void) {
    pthread_once(&g_state_err_once, state_err_key_init);
    char* prev = (char*)pthread_getspecific(g_state_err_key);
    if (prev) {
        free(prev);
        pthread_setspecific(g_state_err_key, NULL);
    }
}

const char* ane_state_last_error(void) {
    pthread_once(&g_state_err_once, state_err_key_init);
    return (const char*)pthread_getspecific(g_state_err_key);
}

/* ------------------------------------------------------------------ */
/* Opaque handles.                                                     */
/* ------------------------------------------------------------------ */

struct AneStateModel {
    MLModel* model;                  /* retained */
    NSArray<NSString*>* in_names;    /* retained, sorted for stable index */
    NSArray<NSString*>* out_names;   /* retained, sorted */
    NSArray<NSString*>* state_names; /* retained, sorted */
    char** in_c;                     /* strdup'd C strings, NULL-terminated members */
    char** out_c;
    char** state_c;
};

struct AneState {
    MLState* state; /* owned (+1 from -newState) */
};

/* ------------------------------------------------------------------ */
/* Helpers.                                                            */
/* ------------------------------------------------------------------ */

static char** dup_names(NSArray<NSString*>* names) {
    char** out = (char**)calloc((size_t)names.count, sizeof(char*));
    if (!out) {
        return NULL;
    }
    for (NSUInteger i = 0; i < names.count; i++) {
        out[i] = strdup([names[i] UTF8String]);
    }
    return out;
}

static void free_names(char** c, NSUInteger n) {
    if (!c) {
        return;
    }
    for (NSUInteger i = 0; i < n; i++) {
        free(c[i]);
    }
    free(c);
}

static NSArray<NSString*>* sorted_keys(NSDictionary* d) {
    if (!d) {
        return @[];
    }
    return [[d allKeys] sortedArrayUsingSelector:@selector(compare:)];
}

/* Element count a feature description declares (product of its dims). */
static size_t fd_count(MLFeatureDescription* fd) {
    MLMultiArrayConstraint* c = fd.multiArrayConstraint;
    if (!c) {
        return 0;
    }
    size_t n = 1;
    for (NSNumber* dim in c.shape) {
        n *= (size_t)dim.unsignedLongLongValue;
    }
    return n;
}

/* Build an MLMultiArray for a model input from a flat f32 buffer, converting
 * to the input's declared dtype (fp32 / fp16). Returns autoreleased, or nil. */
static MLMultiArray* array_from_f32(MLFeatureDescription* fd, const float* src, size_t count,
                                    NSError** e) {
    MLMultiArrayConstraint* c = fd.multiArrayConstraint;
    if (!c) {
        return nil;
    }
    MLMultiArray* arr = [[[MLMultiArray alloc] initWithShape:c.shape dataType:c.dataType
                                                       error:e] autorelease];
    if (!arr) {
        return nil;
    }
    __block BOOL ok = YES;
    [arr getMutableBytesWithHandler:^(void* ptr, NSInteger size, NSArray<NSNumber*>* strides) {
        (void)strides;
        if ((size_t)arr.count != count) {
            ok = NO;
            return;
        }
        /* Exhaustive over MLMultiArrayDataType (no default → -Wcovered-switch).
       * Unsupported-for-input dtypes fall to the trailing `ok = NO`. */
        switch (c.dataType) {
        case MLMultiArrayDataTypeFloat32:
            if ((size_t)size >= count * sizeof(float)) {
                memcpy(ptr, src, count * sizeof(float));
            } else {
                ok = NO;
            }
            break;
        case MLMultiArrayDataTypeFloat16: {
            __fp16* d = (__fp16*)ptr;
            for (size_t i = 0; i < count; i++) {
                d[i] = (__fp16)src[i];
            }
            break;
        }
        case MLMultiArrayDataTypeDouble:
        case MLMultiArrayDataTypeInt32:
        case MLMultiArrayDataTypeInt8:
            ok = NO;
            break;
        }
    }];
    return ok ? arr : nil;
}

/* Copy an output MLMultiArray into a flat f32 buffer, converting from its
 * dtype. Returns YES on success. */
static BOOL array_to_f32(MLMultiArray* arr, float* dst, size_t count) {
    if ((size_t)arr.count != count) {
        return NO;
    }
    __block BOOL ok = YES;
    [arr getBytesWithHandler:^(const void* ptr, NSInteger size) {
        /* Exhaustive over MLMultiArrayDataType (no default → -Wcovered-switch). */
        switch (arr.dataType) {
        case MLMultiArrayDataTypeFloat32:
            if ((size_t)size >= count * sizeof(float)) {
                memcpy(dst, ptr, count * sizeof(float));
            } else {
                ok = NO;
            }
            break;
        case MLMultiArrayDataTypeFloat16: {
            const __fp16* s = (const __fp16*)ptr;
            for (size_t i = 0; i < count; i++) {
                dst[i] = (float)s[i];
            }
            break;
        }
        case MLMultiArrayDataTypeInt32: {
            const int32_t* s = (const int32_t*)ptr;
            for (size_t i = 0; i < count; i++) {
                dst[i] = (float)s[i];
            }
            break;
        }
        case MLMultiArrayDataTypeDouble:
        case MLMultiArrayDataTypeInt8:
            ok = NO;
            break;
        }
    }];
    return ok;
}

/* ------------------------------------------------------------------ */
/* Model lifecycle.                                                    */
/* ------------------------------------------------------------------ */

AneStatus ane_state_model_open(const char* model_url, AneStateModel** out_model) {
    state_err_clear();
    if (!model_url || !out_model) {
        state_err("ane_state_model_open: null arg");
        return ANE_ERR_INVALID_ARG;
    }
    @autoreleasepool {
        NSString* p = [NSString stringWithUTF8String:model_url];
        NSURL* url = [p hasPrefix:@"file:"] ? [NSURL URLWithString:p] : [NSURL fileURLWithPath:p];
        NSError* e = nil;
        if ([[url path] hasSuffix:@".mlpackage"]) {
            NSURL* compiled = [MLModel compileModelAtURL:url error:&e];
            if (!compiled) {
                state_err("compileModelAtURL: %s",
                          e ? [[e localizedDescription] UTF8String] : "<no error>");
                return ANE_ERR_COMPILE;
            }
            url = compiled;
        }
        MLModelConfiguration* cfg = [[[MLModelConfiguration alloc] init] autorelease];
        cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        MLModel* model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&e];
        if (!model) {
            state_err("modelWithContentsOfURL: %s",
                      e ? [[e localizedDescription] UTF8String] : "<no error>");
            return ANE_ERR_LOAD;
        }
        AneStateModel* m = (AneStateModel*)calloc(1, sizeof(AneStateModel));
        if (!m) {
            return ANE_ERR_OOM;
        }
        m->model = [model retain];
        MLModelDescription* d = model.modelDescription;
        m->in_names = [sorted_keys(d.inputDescriptionsByName) retain];
        m->out_names = [sorted_keys(d.outputDescriptionsByName) retain];
        m->state_names = [sorted_keys(d.stateDescriptionsByName) retain];
        m->in_c = dup_names(m->in_names);
        m->out_c = dup_names(m->out_names);
        m->state_c = dup_names(m->state_names);
        *out_model = m;
        return ANE_OK;
    }
}

void ane_state_model_close(AneStateModel* m) {
    if (!m) {
        return;
    }
    free_names(m->in_c, m->in_names.count);
    free_names(m->out_c, m->out_names.count);
    free_names(m->state_c, m->state_names.count);
    [m->in_names release];
    [m->out_names release];
    [m->state_names release];
    [m->model release];
    free(m);
}

int32_t ane_state_model_num_inputs(const AneStateModel* m) {
    return m ? (int32_t)m->in_names.count : 0;
}
int32_t ane_state_model_num_outputs(const AneStateModel* m) {
    return m ? (int32_t)m->out_names.count : 0;
}
int32_t ane_state_model_num_states(const AneStateModel* m) {
    return m ? (int32_t)m->state_names.count : 0;
}

const char* ane_state_model_input_name(const AneStateModel* m, int32_t i) {
    return (m && i >= 0 && i < (int32_t)m->in_names.count) ? m->in_c[i] : NULL;
}
const char* ane_state_model_output_name(const AneStateModel* m, int32_t i) {
    return (m && i >= 0 && i < (int32_t)m->out_names.count) ? m->out_c[i] : NULL;
}
const char* ane_state_model_state_name(const AneStateModel* m, int32_t i) {
    return (m && i >= 0 && i < (int32_t)m->state_names.count) ? m->state_c[i] : NULL;
}

size_t ane_state_model_input_count(const AneStateModel* m, const char* name) {
    if (!m || !name) {
        return 0;
    }
    @autoreleasepool {
        NSString* key = [NSString stringWithUTF8String:name];
        if (!key) {
            return 0;
        }
        MLFeatureDescription* fd = m->model.modelDescription.inputDescriptionsByName[key];
        return fd ? fd_count(fd) : 0;
    }
}
size_t ane_state_model_output_count(const AneStateModel* m, const char* name) {
    if (!m || !name) {
        return 0;
    }
    @autoreleasepool {
        NSString* key = [NSString stringWithUTF8String:name];
        if (!key) {
            return 0;
        }
        MLFeatureDescription* fd = m->model.modelDescription.outputDescriptionsByName[key];
        return fd ? fd_count(fd) : 0;
    }
}

/* ------------------------------------------------------------------ */
/* State lifecycle.                                                    */
/* ------------------------------------------------------------------ */

AneStatus ane_state_create(AneStateModel* m, AneState** out_state) {
    state_err_clear();
    if (!m || !out_state) {
        state_err("ane_state_create: null arg");
        return ANE_ERR_INVALID_ARG;
    }
    @autoreleasepool {
        MLState* st = [m->model newState]; /* +1 owned (NS_RETURNS_RETAINED) */
        if (!st) {
            state_err("newState returned nil");
            return ANE_ERR_INTERNAL;
        }
        AneState* s = (AneState*)calloc(1, sizeof(AneState));
        if (!s) {
            [st release];
            return ANE_ERR_OOM;
        }
        s->state = st;
        *out_state = s;
        return ANE_OK;
    }
}

void ane_state_release(AneState* s) {
    if (!s) {
        return;
    }
    [s->state release];
    free(s);
}

/* ------------------------------------------------------------------ */
/* Step.                                                               */
/* ------------------------------------------------------------------ */

AneStatus ane_state_predict_f32(AneStateModel* m, AneState* s, const char* const* in_names,
                                const float* const* in_data, const size_t* in_counts, int32_t n_in,
                                const char* const* out_names, float* const* out_data,
                                const size_t* out_counts, int32_t n_out) {
    state_err_clear();
    if (!m || !s || (n_in > 0 && (!in_names || !in_data || !in_counts)) ||
        (n_out > 0 && (!out_names || !out_data || !out_counts))) {
        state_err("ane_state_predict_f32: null arg");
        return ANE_ERR_INVALID_ARG;
    }
    @autoreleasepool {
        NSError* e = nil;
        NSMutableDictionary<NSString*, MLFeatureValue*>* feats = [NSMutableDictionary dictionary];
        for (int32_t i = 0; i < n_in; i++) {
            NSString* name = [NSString stringWithUTF8String:in_names[i]];
            if (!name) {
                state_err("input name %d is not valid UTF-8", i);
                return ANE_ERR_INVALID_ARG;
            }
            MLFeatureDescription* fd = m->model.modelDescription.inputDescriptionsByName[name];
            if (!fd) {
                state_err("unknown input '%s'", in_names[i]);
                return ANE_ERR_INVALID_ARG;
            }
            MLMultiArray* arr = array_from_f32(fd, in_data[i], in_counts[i], &e);
            if (!arr) {
                state_err("input '%s': bad shape/dtype or count (%s)", in_names[i],
                          e ? [[e localizedDescription] UTF8String] : "count mismatch");
                return ANE_ERR_INVALID_ARG;
            }
            feats[name] = [MLFeatureValue featureValueWithMultiArray:arr];
        }
        MLDictionaryFeatureProvider* in =
            [[[MLDictionaryFeatureProvider alloc] initWithDictionary:feats error:&e] autorelease];
        if (!in) {
            state_err("feature provider: %s",
                      e ? [[e localizedDescription] UTF8String] : "<no error>");
            return ANE_ERR_INVALID_ARG;
        }
        id<MLFeatureProvider> out = [m->model predictionFromFeatures:in
                                                          usingState:s->state
                                                               error:&e];
        if (!out) {
            state_err("predictionFromFeatures:usingState: %s",
                      e ? [[e localizedDescription] UTF8String] : "<no error>");
            return ANE_ERR_EVAL;
        }
        for (int32_t i = 0; i < n_out; i++) {
            NSString* name = [NSString stringWithUTF8String:out_names[i]];
            if (!name) {
                state_err("output name %d is not valid UTF-8", i);
                return ANE_ERR_INVALID_ARG;
            }
            MLFeatureValue* fv = [out featureValueForName:name];
            MLMultiArray* arr = fv.multiArrayValue;
            if (!arr) {
                state_err("output '%s' missing or not a multiarray", out_names[i]);
                return ANE_ERR_EVAL;
            }
            if (!array_to_f32(arr, out_data[i], out_counts[i])) {
                state_err("output '%s': count mismatch or unsupported dtype", out_names[i]);
                return ANE_ERR_INVALID_ARG;
            }
        }
        return ANE_OK;
    }
}

/* ------------------------------------------------------------------ */
/* E5RT escape hatch: borrow the live execution stream.               */
/* ------------------------------------------------------------------ */

/* Read an ivar's raw value (object or pointer) via the Obj-C runtime. */
static void* ivar_value(id obj, const char* name) {
    if (!obj) {
        return NULL;
    }
    void* value = NULL;
    object_getInstanceVariable(obj, name, &value);
    return value;
}

void* ane_state_e5rt_stream(const AneStateModel* m) {
    if (!m) {
        return NULL;
    }
    @autoreleasepool {
        /* MLModel._internalEngine (MLE5Engine) -> _streamPool
         * (MLE5ExecutionStreamPool) -> _allStreams / _pool
         * (NSSet<MLE5ExecutionStream>) -> any ._streamHandle
         * (e5rt_execution_stream*). Private ivars; best-effort across OS
         * versions, and nil until the engine has built a stream (i.e. before
         * the first predict) or for non-MLE5 (non-stateful) models. */
        id engine = (id)ivar_value(m->model, "_internalEngine");
        id pool = (id)ivar_value(engine, "_streamPool");
        if (!pool) {
            return NULL;
        }
        id streams = (id)ivar_value(pool, "_allStreams");
        if (![streams isKindOfClass:[NSSet class]] || [(NSSet*)streams count] == 0) {
            streams = (id)ivar_value(pool, "_pool");
        }
        if (![streams isKindOfClass:[NSSet class]]) {
            return NULL;
        }
        id stream = [(NSSet*)streams anyObject];
        if (!stream) {
            return NULL;
        }
        return ivar_value(stream, "_streamHandle");
    }
}

void* ane_state_e5rt_program_library(const AneStateModel* m) {
    if (!m) {
        return NULL;
    }
    @autoreleasepool {
        /* MLE5Engine._programLibrary (MLE5ProgramLibrary) ->
         * _programLibraryHandle (e5rt_program_library*). Nil before the engine
         * loads its program, or for non-MLE5 models. Borrowed; do not release. */
        id engine = (id)ivar_value(m->model, "_internalEngine");
        id lib = (id)ivar_value(engine, "_programLibrary");
        return ivar_value(lib, "_programLibraryHandle");
    }
}

/* The per-stream operations array (MLE5ExecutionStream._operations), or nil.
 * Each element is an MLE5ExecutionStreamOperation whose _operationHandle ivar
 * is the e5rt_execution_stream_operation*. Walks the same borrowed stream as
 * ane_state_e5rt_stream. */
static id state_stream_operations(const AneStateModel* m) {
    if (!m) {
        return nil;
    }
    id engine = (id)ivar_value(m->model, "_internalEngine");
    id pool = (id)ivar_value(engine, "_streamPool");
    if (!pool) {
        return nil;
    }
    id streams = (id)ivar_value(pool, "_allStreams");
    if (![streams isKindOfClass:[NSSet class]] || [(NSSet*)streams count] == 0) {
        streams = (id)ivar_value(pool, "_pool");
    }
    if (![streams isKindOfClass:[NSSet class]]) {
        return nil;
    }
    id stream = [(NSSet*)streams anyObject];
    if (!stream) {
        return nil;
    }
    id ops = (id)ivar_value(stream, "_operations");
    if (![ops isKindOfClass:[NSArray class]]) {
        return nil;
    }
    return ops;
}

int ane_state_e5rt_operation_count(const AneStateModel* m) {
    @autoreleasepool {
        id ops = state_stream_operations(m);
        if (!ops) {
            return 0;
        }
        NSUInteger count = [(NSArray*)ops count];
        if (count > (NSUInteger)INT_MAX) {
            return INT_MAX;
        }
        return (int)count;
    }
}

void* ane_state_e5rt_operation_at(const AneStateModel* m, int idx) {
    if (idx < 0) {
        return NULL;
    }
    @autoreleasepool {
        id ops = state_stream_operations(m);
        if (!ops) {
            return NULL;
        }
        if ((NSUInteger)idx >= [(NSArray*)ops count]) {
            return NULL;
        }
        id op = [(NSArray*)ops objectAtIndex:(NSUInteger)idx];
        return ivar_value(op, "_operationHandle");
    }
}

/* e5rt_execution_stream_submit_async, resolved lazily. Espresso is not linked
 * into this dylib (only the Rust sys crate links it), so we dlsym it instead of
 * referencing it directly — which also keeps the standalone `make` build (no
 * -framework Espresso) linking. The completion block type is
 * void(^)(uint64_t, uint64_t, e5rt_error*). */
typedef int (*ane_e5rt_submit_async_fn)(void*, void (^)(uint64_t, uint64_t, void*));
static ane_e5rt_submit_async_fn g_submit_async = NULL;
static pthread_once_t g_submit_once = PTHREAD_ONCE_INIT;

static void load_submit_async(void) {
    void* handle =
        dlopen("/System/Library/PrivateFrameworks/Espresso.framework/Espresso", RTLD_NOW);
    if (handle) {
        g_submit_async =
            (ane_e5rt_submit_async_fn)dlsym(handle, "e5rt_execution_stream_submit_async");
    }
}

int ane_state_e5rt_submit_async(void* stream, AneE5rtCompletion cb, void* ctx) {
    if (!stream || !cb) {
        return -1;
    }
    pthread_once(&g_submit_once, load_submit_async);
    if (!g_submit_async) {
        return -1;
    }
    /* The framework copies (objc_retainBlock) this block, so the stack block is
     * fine; it fires `cb(ctx, ...)` once on completion. */
    return g_submit_async(stream, ^(uint64_t arg0, uint64_t arg1, void* err) {
        cb(ctx, arg0, arg1, err);
    });
}
