/* ane_bridge.m — Core implementation. See ane_bridge.h for the contract. */
#import "ane_bridge.h"
#import "ane_private.h"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <pthread.h>
#import <stdatomic.h>
#import <string.h>
#import <stdlib.h>

#define ANE_BRIDGE_VERSION "0.1.0"

/* ============================================================
 * Thread-local last error
 * ============================================================ */

static pthread_key_t g_err_key;
static pthread_once_t g_err_once = PTHREAD_ONCE_INIT;
static void err_destructor(void* p) { free(p); }
static void err_key_init(void) { pthread_key_create(&g_err_key, err_destructor); }

static void set_last_error(const char* fmt, ...) {
    pthread_once(&g_err_once, err_key_init);
    char* buf = (char*)malloc(1024);
    if (!buf) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, 1024, fmt, ap);
    va_end(ap);
    char* prev = (char*)pthread_getspecific(g_err_key);
    if (prev) free(prev);
    pthread_setspecific(g_err_key, buf);
}

static void clear_last_error(void) {
    pthread_once(&g_err_once, err_key_init);
    char* prev = (char*)pthread_getspecific(g_err_key);
    if (prev) { free(prev); pthread_setspecific(g_err_key, NULL); }
}

const char* ane_last_error(void) {
    pthread_once(&g_err_once, err_key_init);
    const char* s = (const char*)pthread_getspecific(g_err_key);
    return s ? s : "";
}

const char* ane_bridge_version(void) { return ANE_BRIDGE_VERSION; }

/* ============================================================
 * Dtype
 * ============================================================ */

size_t ane_dtype_size(AneDtype dt) {
    switch (dt) {
        case ANE_DTYPE_FP32:  return 4;
        case ANE_DTYPE_FP16:  return 2;
        case ANE_DTYPE_INT32: return 4;
        case ANE_DTYPE_INT64: return 8;
        case ANE_DTYPE_UINT8: return 1;
        case ANE_DTYPE_INT8:  return 1;
    }
    return 0;
}

/* ============================================================
 * IOSurface helpers
 * ============================================================ */

static IOSurfaceRef make_surface(size_t bytes) {
    size_t W, H;
    if (bytes == 0) { W = 1; H = 1; }
    else if (bytes <= 16384) { W = bytes; H = 1; }
    else {
        W = 4096;
        while (bytes % W) W /= 2;
        H = bytes / W;
    }
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:           @(W),
        (id)kIOSurfaceHeight:          @(H),
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow:     @(W),
        (id)kIOSurfaceAllocSize:       @(bytes ? bytes : 1),
        (id)kIOSurfacePixelFormat:     @0,
    });
}

/* ============================================================
 * Buffer
 * ============================================================ */

struct AneBuffer {
    IOSurfaceRef surface;
    id           wrapper;       /* _ANEIOSurfaceObject */
    size_t       nbytes;
    /* The IOSurface flags from the most recent `ane_buffer_lock`; replayed
     * in `ane_buffer_unlock` because IOSurfaceUnlock needs the same mask
     * that was passed to IOSurfaceLock. Single-slot by design — callers
     * must not nest two locks of differing access modes on the same
     * buffer. The Rust wrapper enforces this via `&mut Buffer` on
     * `Buffer::lock`. */
    int          last_lock_flags;
};

AneStatus ane_buffer_create(size_t nbytes, AneBuffer** out) {
    clear_last_error();
    if (!out) { set_last_error("ane_buffer_create: null out"); return ANE_ERR_INVALID_ARG; }
    if (!ane_private_load()) {
        set_last_error("AppleNeuralEngine framework not loadable");
        return ANE_ERR_UNSUPPORTED;
    }
    AneBuffer* b = (AneBuffer*)calloc(1, sizeof(AneBuffer));
    if (!b) { set_last_error("oom"); return ANE_ERR_OOM; }

    /* `make_surface` builds an NSDictionary and several NSNumber boxed
     * literals; `objectWithIOSurface:` returns an autoreleased wrapper.
     * Without an enclosing pool — common on Rust worker threads that
     * don't run a Cocoa runloop — those objects accumulate in some
     * outer pool that may never drain. Wrap the whole body so we
     * deterministically drop them before returning. The pieces we
     * intend to keep (`b->surface` via CFRetain, `b->wrapper` via
     * explicit retain) survive the drain. */
    @autoreleasepool {
        b->surface = make_surface(nbytes);
        if (!b->surface) {
            free(b);
            set_last_error("IOSurfaceCreate failed");
            return ANE_ERR_OOM;
        }
        b->wrapper = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            g_AneIOSurfaceCls, sel_getUid("objectWithIOSurface:"), b->surface);
        if (!b->wrapper) {
            CFRelease(b->surface); free(b);
            set_last_error("_ANEIOSurfaceObject wrap failed");
            return ANE_ERR_INTERNAL;
        }
        [b->wrapper retain];
        b->nbytes = nbytes;
        *out = b;
    }
    return ANE_OK;
}

AneStatus ane_buffer_create_for_input(const AneModel* m, int32_t idx, AneBuffer** out) {
    size_t n = ane_model_input_nbytes(m, idx);
    if (n == 0) return ANE_ERR_INVALID_ARG;
    return ane_buffer_create(n, out);
}

AneStatus ane_buffer_create_for_output(const AneModel* m, int32_t idx, AneBuffer** out) {
    size_t n = ane_model_output_nbytes(m, idx);
    if (n == 0) return ANE_ERR_INVALID_ARG;
    return ane_buffer_create(n, out);
}

void ane_buffer_release(AneBuffer* b) {
    if (!b) return;
    if (b->wrapper) [b->wrapper release];
    if (b->surface) CFRelease(b->surface);
    free(b);
}

AneStatus ane_buffer_lock(AneBuffer* b, AneBufferAccess access, void** out_ptr) {
    clear_last_error();
    if (!b || !out_ptr) { set_last_error("null arg"); return ANE_ERR_INVALID_ARG; }
    uint32_t flags = (access == ANE_LOCK_READ) ? kIOSurfaceLockReadOnly : 0;
    if (IOSurfaceLock(b->surface, flags, NULL) != kIOReturnSuccess) {
        set_last_error("IOSurfaceLock failed");
        return ANE_ERR_INTERNAL;
    }
    b->last_lock_flags = (int)flags;
    *out_ptr = IOSurfaceGetBaseAddress(b->surface);
    return ANE_OK;
}

AneStatus ane_buffer_unlock(AneBuffer* b) {
    clear_last_error();
    if (!b) return ANE_ERR_INVALID_ARG;
    uint32_t flags = (uint32_t)b->last_lock_flags;
    if (IOSurfaceUnlock(b->surface, flags, NULL) != kIOReturnSuccess) {
        set_last_error("IOSurfaceUnlock failed");
        return ANE_ERR_INTERNAL;
    }
    return ANE_OK;
}

size_t   ane_buffer_nbytes(const AneBuffer* b)        { return b ? b->nbytes : 0; }
uint32_t ane_buffer_iosurface_id(const AneBuffer* b)  { return (b && b->surface) ? IOSurfaceGetID(b->surface) : 0; }

/* ============================================================
 * Model
 * ============================================================ */

typedef struct {
    char*   name;
    AneDtype dtype;
    int32_t rank;
    int64_t* shape;
    size_t  nbytes;
} OwnedSpec;

struct AneModel {
    NSData*    mil_data;
    NSData*    weights_data;
    id         descriptor;
    id         in_memory_model;
    id         client;
    /* `temp_dir` is `NSTemporaryDirectory()/<hex_id>/` populated with
     * `model.mil` + `weights/weight.bin`. `_ANECompiler` reads those
     * files (its `ANECCompile()` log shows the path it consults), so
     * the materialization is required on the cold-compile path. On a
     * cache hit we skip both fields, since `loadWithQoS:` does not
     * touch the directory. */
    NSString*  hex_id;
    NSString*  temp_dir;
    AneQoS     compile_qos;
    bool       cache_hit;       /* set when `compiledModelExists` short-circuited compile */
    bool       loaded;          /* set after a successful loadWithQoS:; gates unloadWithQoS: */

    int32_t    num_inputs;
    int32_t    num_outputs;
    OwnedSpec* input_specs;
    OwnedSpec* output_specs;

    /* Public-facing view (filled with name/shape pointers into OwnedSpec). */
    AneTensorSpec* input_view;
    AneTensorSpec* output_view;
};

static void free_owned_specs(OwnedSpec* specs, int32_t n) {
    if (!specs) return;
    for (int32_t i = 0; i < n; i++) {
        free(specs[i].name);
        free(specs[i].shape);
    }
    free(specs);
}

/* Map framework dtype strings to our AneDtype. Returns 0 on unknown or
 * non-string input. Type-checks `s` before sending NSString selectors;
 * a future framework version could legally store something else under
 * the `Type` key, and we must not crash on it. */
static AneDtype parse_dtype_string(id s) {
    if (![s isKindOfClass:[NSString class]]) return 0;
    NSString* ns = (NSString*)s;
    if ([ns isEqualToString:@"Float32"]) return ANE_DTYPE_FP32;
    if ([ns isEqualToString:@"Float16"]) return ANE_DTYPE_FP16;
    if ([ns isEqualToString:@"Int32"])   return ANE_DTYPE_INT32;
    if ([ns isEqualToString:@"Int64"])   return ANE_DTYPE_INT64;
    if ([ns isEqualToString:@"UInt8"])   return ANE_DTYPE_UINT8;
    if ([ns isEqualToString:@"Int8"])    return ANE_DTYPE_INT8;
    return 0;
}

/* Strip the framework's "@output" suffix from output tensor names so
 * users see the same name the MIL declared. Returns NULL if the input
 * isn't an NSString — caller must check. */
static char* dup_name_clean(id nsname) {
    if (![nsname isKindOfClass:[NSString class]]) return NULL;
    NSString* clean = (NSString*)nsname;
    if ([clean hasSuffix:@"@output"]) {
        clean = [clean substringToIndex:clean.length - 7];
    }
    const char* u = [clean UTF8String];
    return u ? strdup(u) : NULL;
}

/* Build OwnedSpec[] + AneTensorSpec[] from a NetworkStatusList entry's
 * LiveInputList/LiveOutputList NSArray of dicts. Each dict has keys
 * Name, Type, Batches, Channels, Depth, Height, Width. */
static AneStatus build_specs_from_live_list(NSArray* live, int32_t n,
                                            OwnedSpec** out_owned,
                                            AneTensorSpec** out_view) {
    OwnedSpec* o = (OwnedSpec*)calloc((size_t)(n > 0 ? n : 1), sizeof(OwnedSpec));
    AneTensorSpec* v = (AneTensorSpec*)calloc((size_t)(n > 0 ? n : 1), sizeof(AneTensorSpec));
    if (!o || !v) { free(o); free(v); return ANE_ERR_OOM; }
    for (int32_t i = 0; i < n; i++) {
        id d_id = live[i];
        if (![d_id isKindOfClass:[NSDictionary class]]) {
            free_owned_specs(o, i); free(v);
            set_last_error("LiveList[%d] not a dict", i);
            return ANE_ERR_INTERNAL;
        }
        NSDictionary* d = (NSDictionary*)d_id;
        id name  = d[@"Name"];
        id type  = d[@"Type"];
        id batch = d[@"Batches"];
        id chan  = d[@"Channels"];
        id depth = d[@"Depth"];
        id h     = d[@"Height"];
        id w     = d[@"Width"];
        if (!name || !type || !batch || !chan || !depth || !h || !w) {
            free_owned_specs(o, i); free(v);
            set_last_error("LiveList[%d] missing Name/Type/dim key", i);
            return ANE_ERR_INTERNAL;
        }
        /* Type-check every value before sending selectors. A future
         * framework version could put non-string Names or non-number
         * dims under the same keys; the parser must reject cleanly
         * rather than throw NSInvalidArgumentException. */
        if (![name isKindOfClass:[NSString class]]) {
            free_owned_specs(o, i); free(v);
            set_last_error("LiveList[%d] Name not a string", i);
            return ANE_ERR_INTERNAL;
        }
        if (![batch isKindOfClass:[NSNumber class]] ||
            ![chan  isKindOfClass:[NSNumber class]] ||
            ![depth isKindOfClass:[NSNumber class]] ||
            ![h     isKindOfClass:[NSNumber class]] ||
            ![w     isKindOfClass:[NSNumber class]]) {
            free_owned_specs(o, i); free(v);
            set_last_error("LiveList[%d] non-numeric dim", i);
            return ANE_ERR_INTERNAL;
        }
        AneDtype dt = parse_dtype_string(type);
        if (dt == 0) {
            free_owned_specs(o, i); free(v);
            set_last_error("LiveList[%d] unsupported/non-string dtype", i);
            return ANE_ERR_UNSUPPORTED;
        }
        /* Compose a 4-D NCHW shape, expanding to 5-D NCDHW when depth>1.
         * This mirrors how the framework canonicalizes ANE tensor
         * layout — we don't recover the user's original MIL rank, but
         * the bytes are exact. */
        int64_t bz = [(NSNumber*)batch longLongValue];
        int64_t cz = [(NSNumber*)chan  longLongValue];
        int64_t dz = [(NSNumber*)depth longLongValue];
        int64_t hz = [(NSNumber*)h     longLongValue];
        int64_t wz = [(NSNumber*)w     longLongValue];
        /* Every reported dim must be positive — depth=1 is the
         * canonical "no third spatial axis" value, but anything <1
         * is malformed framework output. Reject up front rather
         * than letting the rank-selection branch silently drop a
         * negative depth from the shape. */
        if (bz < 1 || cz < 1 || dz < 1 || hz < 1 || wz < 1) {
            free_owned_specs(o, i); free(v);
            set_last_error("LiveList[%d] non-positive dimension", i);
            return ANE_ERR_INTERNAL;
        }
        int32_t rank;
        int64_t shape[5];
        if (dz > 1) {
            rank = 5;
            shape[0] = bz; shape[1] = cz; shape[2] = dz; shape[3] = hz; shape[4] = wz;
        } else {
            rank = 4;
            shape[0] = bz; shape[1] = cz; shape[2] = hz; shape[3] = wz;
        }
        /* Byte count via checked mul. */
        size_t elt = ane_dtype_size(dt);
        size_t nbytes = elt;
        for (int32_t k = 0; k < rank; k++) {
            if (shape[k] <= 0) {
                free_owned_specs(o, i); free(v);
                set_last_error("LiveList[%d] non-positive dim", i);
                return ANE_ERR_INTERNAL;
            }
            size_t dim = (size_t)shape[k];
            if (dim != 0 && nbytes > SIZE_MAX / dim) {
                free_owned_specs(o, i); free(v);
                set_last_error("LiveList[%d] nbytes overflows", i);
                return ANE_ERR_INTERNAL;
            }
            nbytes *= dim;
        }
        o[i].name  = dup_name_clean(name);
        o[i].dtype = dt;
        o[i].rank  = rank;
        o[i].shape = (int64_t*)calloc((size_t)rank, sizeof(int64_t));
        if (!o[i].shape) {
            free_owned_specs(o, i+1); free(v); return ANE_ERR_OOM;
        }
        memcpy(o[i].shape, shape, (size_t)rank * sizeof(int64_t));
        o[i].nbytes = nbytes;
        v[i].name  = o[i].name;
        v[i].dtype = o[i].dtype;
        v[i].rank  = o[i].rank;
        v[i].shape = o[i].shape;
    }
    *out_owned = o;
    *out_view  = v;
    return ANE_OK;
}

/* Lower half of the schema-derivation path: given the framework's
 * `modelAttributes` dict already in hand, walk the
 * NetworkStatusList → procedure → LiveInputList / LiveOutputList
 * chain and populate `m`'s spec arrays. Split out so the fuzz
 * harness can construct adversarial dicts and call this directly. */
static AneStatus derive_specs_from_attrs_dict(id attrs_id, AneModel* m) {
    if (!attrs_id || ![attrs_id isKindOfClass:[NSDictionary class]]) {
        set_last_error("modelAttributes missing or wrong type");
        return ANE_ERR_INTERNAL;
    }
    NSDictionary* attrs = (NSDictionary*)attrs_id;
    id nsl_id = attrs[@"NetworkStatusList"];
    if (!nsl_id || ![nsl_id isKindOfClass:[NSArray class]]) {
        set_last_error("NetworkStatusList missing or wrong type");
        return ANE_ERR_INTERNAL;
    }
    NSArray* nsl = (NSArray*)nsl_id;
    if (nsl.count == 0) {
        set_last_error("NetworkStatusList empty");
        return ANE_ERR_INTERNAL;
    }
    id proc_id = nsl[0];   /* procedure 0 = "main" */
    if (![proc_id isKindOfClass:[NSDictionary class]]) {
        set_last_error("NetworkStatusList[0] not a dict");
        return ANE_ERR_INTERNAL;
    }
    NSDictionary* proc = (NSDictionary*)proc_id;
    id lin_id  = proc[@"LiveInputList"];
    id lout_id = proc[@"LiveOutputList"];
    if (!lin_id || !lout_id
        || ![lin_id  isKindOfClass:[NSArray class]]
        || ![lout_id isKindOfClass:[NSArray class]]) {
        set_last_error("LiveInputList/LiveOutputList missing or wrong type");
        return ANE_ERR_INTERNAL;
    }
    NSArray* lin  = (NSArray*)lin_id;
    NSArray* lout = (NSArray*)lout_id;
    int32_t nin  = (int32_t)lin.count;
    int32_t nout = (int32_t)lout.count;
    AneStatus s = build_specs_from_live_list(lin, nin, &m->input_specs, &m->input_view);
    if (s != ANE_OK) return s;
    s = build_specs_from_live_list(lout, nout, &m->output_specs, &m->output_view);
    if (s != ANE_OK) return s;
    m->num_inputs  = nin;
    m->num_outputs = nout;
    return ANE_OK;
}

/* After loadWithQoS, ask the loaded model for its true schema and
 * populate the AneModel's spec arrays from it. This is the single
 * source of truth — `modelAttributes` is the framework's own view
 * of the model, so we cannot disagree with it. */
static AneStatus derive_specs_from_attrs(id mdl, AneModel* m) {
    id attrs_id = ((id(*)(id, SEL))objc_msgSend)(mdl, sel_getUid("modelAttributes"));
    return derive_specs_from_attrs_dict(attrs_id, m);
}

AneStatus ane_model_open(const AneModelOpenOptions* opts, AneModel** out_model) {
    clear_last_error();
    if (!opts || !out_model || !opts->mil_path || !opts->weights_path) {
        set_last_error("ane_model_open: invalid args");
        return ANE_ERR_INVALID_ARG;
    }
    if (!ane_private_load()) {
        set_last_error("AppleNeuralEngine framework not loadable");
        return ANE_ERR_UNSUPPORTED;
    }

    @autoreleasepool {
        NSData* mil = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:opts->mil_path]];
        NSData* wts = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:opts->weights_path]];
        if (!mil) { set_last_error("failed to read MIL: %s", opts->mil_path); return ANE_ERR_IO; }
        if (!wts) { set_last_error("failed to read weights: %s", opts->weights_path); return ANE_ERR_IO; }

        AneModel* m = (AneModel*)calloc(1, sizeof(AneModel));
        if (!m) { set_last_error("oom"); return ANE_ERR_OOM; }
        m->mil_data = [mil retain];
        m->weights_data = [wts retain];
        m->compile_qos = opts->compile_qos ? opts->compile_qos : ANE_QOS_DEFAULT;
        /* num_inputs/num_outputs and the spec arrays are derived from
         * the framework after loadWithQoS:; see derive_specs_from_attrs. */

        NSDictionary* wdict = @{
            @"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wts}
        };
        m->descriptor = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            g_AneDescriptorCls, sel_getUid("modelWithMILText:weights:optionsPlist:"),
            mil, wdict, nil);
        if (!m->descriptor) {
            ane_model_close(m);
            set_last_error("_ANEInMemoryModelDescriptor creation failed");
            return ANE_ERR_COMPILE;
        }
        [m->descriptor retain];

        m->in_memory_model = ((id(*)(Class, SEL, id))objc_msgSend)(
            g_AneInMemoryCls, sel_getUid("inMemoryModelWithDescriptor:"), m->descriptor);
        if (!m->in_memory_model) {
            ane_model_close(m);
            set_last_error("_ANEInMemoryModel creation failed");
            return ANE_ERR_COMPILE;
        }
        [m->in_memory_model retain];

        NSError* e = nil;

        /* aned caches the lowered ANE program keyed by the descriptor's
         * content hash (`hexStringIdentifier`). When `compiledModelExists`
         * returns YES, calling `compileWithQoS:` again still pays the full
         * lowering cost (~tens of seconds on large graphs) but produces no
         * new artifact — `loadWithQoS:` on its own reuses the cached
         * lowering and returns in milliseconds. Skip compile on cache hit.
         *
         * The cache lives in aned daemon state, so it survives across
         * processes but not aned restarts or its (opaque) eviction policy.
         * On miss we fall through to the normal compile path.
         *
         * `compiledModelExists` is a private-framework selector. Guard the
         * call with `respondsToSelector:` so that a future macOS that
         * renames or removes it falls back to a normal compile rather
         * than crashing in `objc_msgSend` with `doesNotRecognizeSelector:`. */
        SEL exists_sel = sel_getUid("compiledModelExists");
        BOOL cache_hit = NO;
        if ([m->in_memory_model respondsToSelector:exists_sel]) {
            cache_hit = ((BOOL(*)(id, SEL))objc_msgSend)(
                m->in_memory_model, exists_sel);
        }
        m->cache_hit = cache_hit ? true : false;

        if (!cache_hit) {
            /* `_ANECompiler` reads the program text + weights from
             * `NSTemporaryDirectory()/<hex_id>/{model.mil,
             * weights/weight.bin}` (its `ANECCompile()` failure log
             * shows the path it consults). Materialize only on the
             * compile path — `loadWithQoS:` does not read from disk,
             * so a warm open skips this I/O entirely. */
            id hxid = ((id(*)(id, SEL))objc_msgSend)(
                m->in_memory_model, sel_getUid("hexStringIdentifier"));
            m->hex_id = [hxid retain];
            m->temp_dir = [[NSTemporaryDirectory()
                stringByAppendingPathComponent:m->hex_id] retain];
            NSFileManager* fm = [NSFileManager defaultManager];
            NSError* fserr = nil;
            if (![fm createDirectoryAtPath:[m->temp_dir stringByAppendingPathComponent:@"weights"]
                    withIntermediateDirectories:YES attributes:nil error:&fserr]) {
                ane_model_close(m);
                set_last_error("failed to create temp dir: %s",
                    fserr ? [[fserr localizedDescription] UTF8String] : "?");
                return ANE_ERR_IO;
            }
            if (![mil writeToFile:[m->temp_dir stringByAppendingPathComponent:@"model.mil"]
                       atomically:YES]) {
                ane_model_close(m);
                set_last_error("failed to write model.mil to temp dir");
                return ANE_ERR_IO;
            }
            if (![wts writeToFile:[m->temp_dir stringByAppendingPathComponent:@"weights/weight.bin"]
                       atomically:YES]) {
                ane_model_close(m);
                set_last_error("failed to write weights to temp dir");
                return ANE_ERR_IO;
            }

            BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
                m->in_memory_model, sel_getUid("compileWithQoS:options:error:"),
                (unsigned int)m->compile_qos, @{}, &e);
            if (!ok) {
                /* The framework's top-level message is usually
                 * `_ANECompiler : ANECCompile() FAILED`. The underlying
                 * error in `NSUnderlyingErrorKey` carries the *kind* of
                 * failure (e.g. `CompilationFailure` when the MIL graph
                 * cannot be lowered to ANE bytecode — typically a non-ANE-
                 * targetable model). Surface both so a "compile failed"
                 * report tells the caller whether the model is broken or
                 * just not ANE-compatible. */
                NSError* under = e ? e.userInfo[NSUnderlyingErrorKey] : nil;
                const char* desc  = e     ? [[e     localizedDescription] UTF8String] : "<no NSError>";
                const char* udesc = under ? [[under localizedDescription] UTF8String] : NULL;
                if (udesc) {
                    set_last_error("compile failed: %s | underlying: %s", desc, udesc);
                } else {
                    set_last_error("compile failed: %s", desc);
                }
                ane_model_close(m);
                return ANE_ERR_COMPILE;
            }
        }

        BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
            m->in_memory_model, sel_getUid("loadWithQoS:options:error:"),
            (unsigned int)m->compile_qos, @{}, &e);
        if (!ok) {
            NSError* under = e ? e.userInfo[NSUnderlyingErrorKey] : nil;
            const char* desc  = e     ? [[e     localizedDescription] UTF8String] : "<no NSError>";
            const char* udesc = under ? [[under localizedDescription] UTF8String] : NULL;
            if (udesc) {
                set_last_error("load failed: %s | underlying: %s", desc, udesc);
            } else {
                set_last_error("load failed: %s", desc);
            }
            ane_model_close(m);
            return ANE_ERR_LOAD;
        }
        m->loaded = true;

        /* The loaded model knows its own schema; derive it from the
         * framework's `modelAttributes` rather than trusting whatever
         * the caller might have declared. This is the single source
         * of truth for input/output shapes, names, and dtypes. */
        AneStatus dst = derive_specs_from_attrs(m->in_memory_model, m);
        if (dst != ANE_OK) {
            ane_model_close(m);
            return dst;
        }

        m->client = ((id(*)(id, SEL, BOOL))objc_msgSend)(
            [g_AneClientCls alloc], sel_getUid("initWithRestrictedAccessAllowed:"), YES);
        if (!m->client) {
            ane_model_close(m);
            set_last_error("_ANEClient init failed");
            return ANE_ERR_INTERNAL;
        }

        *out_model = m;
        return ANE_OK;
    }
}

void ane_model_close(AneModel* m) {
    if (!m) return;
    @autoreleasepool {
        if (m->in_memory_model) {
            /* Only `unloadWithQoS:` if `loadWithQoS:` actually succeeded.
             * Calling unload on a never-loaded model returns NO with an
             * NSError that we'd otherwise log as a misleading "model
             * unload failed" — exactly the kind of noise that masks a
             * real failure during partial-open cleanup. */
            if (m->loaded) {
                NSError* e = nil;
                BOOL ok = ((BOOL(*)(id, SEL, unsigned int, NSError**))objc_msgSend)(
                    m->in_memory_model, sel_getUid("unloadWithQoS:error:"),
                    (unsigned int)(m->compile_qos ? m->compile_qos : ANE_QOS_DEFAULT), &e);
                if (!ok) {
                    fprintf(stderr, "ane-bridge: model unload failed: %s\n",
                            e ? [[e localizedDescription] UTF8String] : "<no NSError>");
                }
            }
            [m->in_memory_model release];
        }
        if (m->descriptor) [m->descriptor release];
        if (m->client) [m->client release];
        /* temp_dir/hex_id are only set on the cache-miss path; safe to
         * skip cleanly on warm starts. */
        if (m->temp_dir) {
            [[NSFileManager defaultManager] removeItemAtPath:m->temp_dir error:nil];
            [m->temp_dir release];
        }
        if (m->hex_id) [m->hex_id release];
        if (m->mil_data) [m->mil_data release];
        if (m->weights_data) [m->weights_data release];
        free_owned_specs(m->input_specs, m->num_inputs);
        free_owned_specs(m->output_specs, m->num_outputs);
        free(m->input_view);
        free(m->output_view);
        free(m);
    }
}

int32_t ane_model_num_inputs(const AneModel* m)  { return m ? m->num_inputs : 0; }
int32_t ane_model_num_outputs(const AneModel* m) { return m ? m->num_outputs : 0; }

const AneTensorSpec* ane_model_input_spec(const AneModel* m, int32_t i) {
    if (!m || i < 0 || i >= m->num_inputs) return NULL;
    return &m->input_view[i];
}
const AneTensorSpec* ane_model_output_spec(const AneModel* m, int32_t i) {
    if (!m || i < 0 || i >= m->num_outputs) return NULL;
    return &m->output_view[i];
}
size_t ane_model_input_nbytes(const AneModel* m, int32_t i) {
    if (!m || i < 0 || i >= m->num_inputs) return 0;
    return m->input_specs[i].nbytes;
}
size_t ane_model_output_nbytes(const AneModel* m, int32_t i) {
    if (!m || i < 0 || i >= m->num_outputs) return 0;
    return m->output_specs[i].nbytes;
}

bool ane_model_was_cached(const AneModel* m) { return m ? m->cache_hit : false; }

/* ============================================================
 * Request
 * ============================================================ */

struct AneRequest {
    AneModel*       model;

    /* Binding state — one entry per input/output. Either bound (external
     * buffer ref) or owned (internal fast-path IOSurface). */
    AneBuffer**     input_bound;   /* num_inputs */
    AneBuffer**     output_bound;  /* num_outputs */
    AneBuffer**     input_owned;
    AneBuffer**     output_owned;

    /* Async machinery.
     *
     * Race-free read pattern: writers complete `last_status` and
     * `last_err_msg` BEFORE atomic_store(has_result, 1). Both the
     * store and the matching reader loads default to seq_cst (no
     * memory_order argument passed), which is stronger than the
     * acquire/release pair needed here; readers that see
     * `has_result == 1` are guaranteed to see the prior field writes.
     *
     * `last_err_msg` is an inline fixed-size buffer (not a heap
     * pointer) specifically to avoid a use-after-free if a reader
     * with a stale pointer races with a new submission. A reader can
     * still observe a torn byte sequence mid-write, but worst case
     * is a slightly garbled message — not undefined behaviour. */
    dispatch_queue_t queue;
    dispatch_semaphore_t done_sem;
    atomic_int       in_flight;
    atomic_int       has_result;
    AneStatus        last_status;
    char             last_err_msg[256];

    AneCompletionFn  completion_fn;
    void*            completion_user;
};

AneStatus ane_request_create(AneModel* m, AneRequest** out) {
    clear_last_error();
    if (!m || !out) { set_last_error("invalid arg"); return ANE_ERR_INVALID_ARG; }
    AneRequest* r = (AneRequest*)calloc(1, sizeof(AneRequest));
    if (!r) { set_last_error("oom"); return ANE_ERR_OOM; }
    r->model = m;
    r->input_bound  = (AneBuffer**)calloc((size_t)(m->num_inputs  > 0 ? m->num_inputs  : 1), sizeof(AneBuffer*));
    r->output_bound = (AneBuffer**)calloc((size_t)(m->num_outputs > 0 ? m->num_outputs : 1), sizeof(AneBuffer*));
    r->input_owned  = (AneBuffer**)calloc((size_t)(m->num_inputs  > 0 ? m->num_inputs  : 1), sizeof(AneBuffer*));
    r->output_owned = (AneBuffer**)calloc((size_t)(m->num_outputs > 0 ? m->num_outputs : 1), sizeof(AneBuffer*));
    if (!r->input_bound || !r->output_bound || !r->input_owned || !r->output_owned) {
        ane_request_release(r); set_last_error("oom"); return ANE_ERR_OOM;
    }
    r->queue = dispatch_queue_create("ane.bridge.request", DISPATCH_QUEUE_SERIAL);
    r->done_sem = dispatch_semaphore_create(0);
    atomic_init(&r->in_flight, 0);
    atomic_init(&r->has_result, 1);   /* a fresh request has "no submission, but is not pending" */
    *out = r;
    return ANE_OK;
}

void ane_request_release(AneRequest* r) {
    if (!r) return;
    /* Drain any in-flight evaluation so we don't free state under it. */
    if (r->queue) {
        dispatch_sync(r->queue, ^{});
        dispatch_release(r->queue);
    }
    if (r->done_sem) dispatch_release(r->done_sem);
    if (r->input_owned) {
        for (int32_t i = 0; i < r->model->num_inputs; i++)
            if (r->input_owned[i]) ane_buffer_release(r->input_owned[i]);
        free(r->input_owned);
    }
    if (r->output_owned) {
        for (int32_t i = 0; i < r->model->num_outputs; i++)
            if (r->output_owned[i]) ane_buffer_release(r->output_owned[i]);
        free(r->output_owned);
    }
    free(r->input_bound);
    free(r->output_bound);
    /* last_err_msg is inline, nothing to free. */
    free(r);
}

AneStatus ane_request_bind_input(AneRequest* r, int32_t i, AneBuffer* b) {
    clear_last_error();
    if (!r || !b) { set_last_error("invalid arg"); return ANE_ERR_INVALID_ARG; }
    if (i < 0 || i >= r->model->num_inputs) { set_last_error("input idx out of range"); return ANE_ERR_INVALID_ARG; }
    if (b->nbytes < r->model->input_specs[i].nbytes) {
        set_last_error("input %d: buffer too small (%zu < %zu)",
                       i, b->nbytes, r->model->input_specs[i].nbytes);
        return ANE_ERR_INVALID_ARG;
    }
    r->input_bound[i] = b;
    return ANE_OK;
}

AneStatus ane_request_bind_output(AneRequest* r, int32_t i, AneBuffer* b) {
    clear_last_error();
    if (!r || !b) { set_last_error("invalid arg"); return ANE_ERR_INVALID_ARG; }
    if (i < 0 || i >= r->model->num_outputs) { set_last_error("output idx out of range"); return ANE_ERR_INVALID_ARG; }
    if (b->nbytes < r->model->output_specs[i].nbytes) {
        set_last_error("output %d: buffer too small (%zu < %zu)",
                       i, b->nbytes, r->model->output_specs[i].nbytes);
        return ANE_ERR_INVALID_ARG;
    }
    r->output_bound[i] = b;
    return ANE_OK;
}

static AneBuffer* ensure_owned_input(AneRequest* r, int32_t i) {
    if (r->input_owned[i]) return r->input_owned[i];
    AneBuffer* b = NULL;
    if (ane_buffer_create(r->model->input_specs[i].nbytes, &b) != ANE_OK) return NULL;
    r->input_owned[i] = b;
    return b;
}

static AneBuffer* ensure_owned_output(AneRequest* r, int32_t i) {
    if (r->output_owned[i]) return r->output_owned[i];
    AneBuffer* b = NULL;
    if (ane_buffer_create(r->model->output_specs[i].nbytes, &b) != ANE_OK) return NULL;
    r->output_owned[i] = b;
    return b;
}

AneStatus ane_request_set_input_bytes(AneRequest* r, int32_t i, const void* data, size_t nbytes) {
    clear_last_error();
    if (!r || !data) { set_last_error("invalid arg"); return ANE_ERR_INVALID_ARG; }
    if (i < 0 || i >= r->model->num_inputs) { set_last_error("input idx out of range"); return ANE_ERR_INVALID_ARG; }
    if (nbytes != r->model->input_specs[i].nbytes) {
        set_last_error("input %d: byte count mismatch (%zu != %zu)",
                       i, nbytes, r->model->input_specs[i].nbytes);
        return ANE_ERR_INVALID_ARG;
    }
    /* If a submission is in flight, the worker thread is reading from
     * this very buffer's IOSurface via DMA. Writing into it now would
     * race on the payload bytes — `IOSurfaceLock` synchronizes CPU
     * accesses but does not necessarily synchronize with ANE/DMA.
     * Reject with Busy; the caller must wait for the previous eval
     * to complete first. */
    if (atomic_load(&r->in_flight) != 0) {
        set_last_error("cannot set_input_bytes while a submission is in flight");
        return ANE_ERR_BUSY;
    }
    AneBuffer* b = ensure_owned_input(r, i);
    if (!b) return ANE_ERR_OOM;
    void* p = NULL;
    AneStatus st = ane_buffer_lock(b, ANE_LOCK_WRITE, &p);
    if (st != ANE_OK) return st;
    memcpy(p, data, nbytes);
    ane_buffer_unlock(b);
    /* Use the owned buffer as the effective binding. */
    r->input_bound[i] = b;
    return ANE_OK;
}

AneStatus ane_request_get_output_bytes(AneRequest* r, int32_t i, void* data, size_t nbytes) {
    clear_last_error();
    if (!r || !data) { set_last_error("invalid arg"); return ANE_ERR_INVALID_ARG; }
    if (i < 0 || i >= r->model->num_outputs) { set_last_error("output idx out of range"); return ANE_ERR_INVALID_ARG; }
    if (nbytes != r->model->output_specs[i].nbytes) {
        set_last_error("output %d: byte count mismatch (%zu != %zu)",
                       i, nbytes, r->model->output_specs[i].nbytes);
        return ANE_ERR_INVALID_ARG;
    }
    if (!atomic_load(&r->has_result)) {
        set_last_error("output %d: no result available yet", i);
        return ANE_ERR_NOT_DONE;
    }
    AneBuffer* b = r->output_bound[i];
    if (!b) {
        set_last_error("output %d: no buffer bound", i);
        return ANE_ERR_INVALID_ARG;
    }
    void* p = NULL;
    AneStatus st = ane_buffer_lock(b, ANE_LOCK_READ, &p);
    if (st != ANE_OK) return st;
    memcpy(data, p, nbytes);
    ane_buffer_unlock(b);
    return ANE_OK;
}

/* Build a fresh _ANERequest from current bindings; falls back to owned
 * IOSurfaces for any unset output slot so the eval has somewhere to
 * write. (Inputs MUST be supplied by the caller.) */
static id build_ane_request(AneRequest* r, AneStatus* status_out) {
    NSMutableArray* wIn  = [NSMutableArray arrayWithCapacity:(NSUInteger)r->model->num_inputs];
    NSMutableArray* iIn  = [NSMutableArray arrayWithCapacity:(NSUInteger)r->model->num_inputs];
    NSMutableArray* wOut = [NSMutableArray arrayWithCapacity:(NSUInteger)r->model->num_outputs];
    NSMutableArray* iOut = [NSMutableArray arrayWithCapacity:(NSUInteger)r->model->num_outputs];

    for (int32_t i = 0; i < r->model->num_inputs; i++) {
        AneBuffer* b = r->input_bound[i];
        if (!b) {
            set_last_error("input %d not bound", i);
            *status_out = ANE_ERR_INVALID_ARG;
            return nil;
        }
        [wIn addObject:b->wrapper];
        [iIn addObject:@(i)];
    }
    for (int32_t i = 0; i < r->model->num_outputs; i++) {
        AneBuffer* b = r->output_bound[i];
        if (!b) {
            b = ensure_owned_output(r, i);
            if (!b) { *status_out = ANE_ERR_OOM; return nil; }
            r->output_bound[i] = b;
        }
        [wOut addObject:b->wrapper];
        [iOut addObject:@(i)];
    }

    id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        g_AneRequestCls,
        sel_getUid("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"),
        wIn, iIn, wOut, iOut, nil, nil, @0);
    if (!req) {
        set_last_error("_ANERequest creation returned nil");
        *status_out = ANE_ERR_INTERNAL;
    }
    return req;
}

static void store_err(AneRequest* r, const char* msg) {
    if (!msg) { r->last_err_msg[0] = '\0'; return; }
    /* `strncpy` zero-fills any remaining bytes if `msg` is shorter than
     * the buffer; the explicit terminator below covers the case where
     * `msg` is longer than the buffer minus one. */
    strncpy(r->last_err_msg, msg, sizeof(r->last_err_msg) - 1);
    r->last_err_msg[sizeof(r->last_err_msg) - 1] = '\0';
}

AneStatus ane_request_submit(AneRequest* r, AneQoS qos) {
    clear_last_error();
    if (!r) { set_last_error("null request"); return ANE_ERR_INVALID_ARG; }
    int expected = 0;
    if (!atomic_compare_exchange_strong(&r->in_flight, &expected, 1)) {
        set_last_error("request already in flight");
        return ANE_ERR_BUSY;
    }
    atomic_store(&r->has_result, 0);

    AneStatus build_status = ANE_OK;
    id req_obj = nil;
    @autoreleasepool {
        req_obj = build_ane_request(r, &build_status);
        if (req_obj) [req_obj retain];
    }
    if (!req_obj) {
        /* Write last_status BEFORE publishing has_result=1 so that
         * any future reader gated on `has_result` (via acquire load)
         * is guaranteed to observe the updated status. Same
         * happens-before discipline as the worker block below. */
        r->last_status = build_status;
        atomic_store(&r->has_result, 1);
        atomic_store(&r->in_flight, 0);
        return build_status;
    }

    unsigned int qos_v = (unsigned int)(qos ? qos : ANE_QOS_DEFAULT);
    AneModel* model = r->model;
    AneCompletionFn cb = r->completion_fn;
    void* cb_user = r->completion_user;

    dispatch_async(r->queue, ^{
        @autoreleasepool {
            NSError* e = nil;
            BOOL ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError**))objc_msgSend)(
                model->client,
                sel_getUid("doEvaluateDirectWithModel:options:request:qos:error:"),
                model->in_memory_model, @{}, req_obj, qos_v, &e);
            AneStatus s = ok ? ANE_OK : ANE_ERR_EVAL;
            if (!ok) store_err(r, e ? [[e localizedDescription] UTF8String] : "<no NSError>");
            else     store_err(r, NULL);
            r->last_status = s;
            [req_obj release];
            atomic_store(&r->has_result, 1);
            atomic_store(&r->in_flight, 0);
            dispatch_semaphore_signal(r->done_sem);
            if (cb) cb(r, s, cb_user);
        }
    });
    return ANE_OK;
}

AneStatus ane_request_wait(AneRequest* r, int32_t timeout_ms) {
    clear_last_error();
    if (!r) { set_last_error("null request"); return ANE_ERR_INVALID_ARG; }
    /* If nothing in flight and we already have a result, the semaphore
     * was already signaled by the worker block.  Consume the signal so
     * the count stays balanced — skipping this was a bug that caused
     * stale semaphore counts to make later waits return prematurely. */
    if (atomic_load(&r->in_flight) == 0 && atomic_load(&r->has_result) == 1) {
        dispatch_semaphore_wait(r->done_sem, DISPATCH_TIME_NOW);
        if (r->last_status != ANE_OK && r->last_err_msg[0] != '\0') {
            set_last_error("%s", r->last_err_msg);
        }
        return r->last_status;
    }
    dispatch_time_t t;
    if (timeout_ms < 0)      t = DISPATCH_TIME_FOREVER;
    else if (timeout_ms == 0) t = DISPATCH_TIME_NOW;
    else                      t = dispatch_time(DISPATCH_TIME_NOW, (int64_t)timeout_ms * NSEC_PER_MSEC);
    long rc = dispatch_semaphore_wait(r->done_sem, t);
    if (rc != 0) {
        if (timeout_ms == 0) return ANE_ERR_NOT_DONE;
        set_last_error("wait timed out after %d ms", timeout_ms);
        return ANE_ERR_TIMEOUT;
    }
    if (r->last_status != ANE_OK && r->last_err_msg[0] != '\0') {
        set_last_error("%s", r->last_err_msg);
    }
    return r->last_status;
}

bool ane_request_is_done(const AneRequest* r) {
    if (!r) return false;
    return atomic_load((atomic_int*)&((AneRequest*)r)->has_result) == 1
        && atomic_load((atomic_int*)&((AneRequest*)r)->in_flight) == 0;
}

AneStatus ane_request_run(AneRequest* r, AneQoS qos) {
    clear_last_error();
    if (!r) { set_last_error("null request"); return ANE_ERR_INVALID_ARG; }
    int expected = 0;
    if (!atomic_compare_exchange_strong(&r->in_flight, &expected, 1)) {
        set_last_error("request already in flight");
        return ANE_ERR_BUSY;
    }
    atomic_store(&r->has_result, 0);

    /* Synchronous fast path. The async submit/wait flow pays for two
     * thread switches per call (caller -> serial queue worker -> wait
     * semaphore -> caller); the worker queue is only useful when the
     * caller wants fire-and-forget (`ane_request_submit`). For
     * `run()`, the framework's `doEvaluateDirectWithModel:` is itself
     * synchronous, so we can stay on the caller's thread and skip
     * both the dispatch_async and the dispatch_semaphore. This cuts
     * ~25 us of dispatch overhead per call and, more importantly,
     * eliminates the 1-3 ms tail caused by GCD scheduling under load.
     *
     * We still go through the same `@autoreleasepool` discipline so
     * NSError objects + the request descriptor get released
     * promptly, and we still update `in_flight` / `has_result` so a
     * concurrent `is_done()` sees the correct state. We intentionally
     * do NOT signal `done_sem` — run() is synchronous and no one is
     * waiting; signaling would leak a semaphore count that breaks a
     * subsequent submit()+wait() sequence. */
    AneStatus status;
    @autoreleasepool {
        AneStatus build_status = ANE_OK;
        id req_obj = build_ane_request(r, &build_status);
        if (!req_obj) {
            r->last_status = build_status;
            atomic_store(&r->has_result, 1);
            atomic_store(&r->in_flight, 0);
            /* Do NOT signal done_sem — run() is synchronous, no one is
             * waiting on the semaphore. Signaling would leak a count that
             * causes a subsequent submit()+wait() to consume a stale
             * signal and return before the eval actually completes. */
            return build_status;
        }
        unsigned int qos_v = (unsigned int)(qos ? qos : ANE_QOS_DEFAULT);
        NSError* e = nil;
        BOOL ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError**))objc_msgSend)(
            r->model->client,
            sel_getUid("doEvaluateDirectWithModel:options:request:qos:error:"),
            r->model->in_memory_model, @{}, req_obj, qos_v, &e);
        status = ok ? ANE_OK : ANE_ERR_EVAL;
        if (!ok) {
            store_err(r, e ? [[e localizedDescription] UTF8String] : "<no NSError>");
            set_last_error("%s", r->last_err_msg);
        } else {
            store_err(r, NULL);
        }
        r->last_status = status;
        atomic_store(&r->has_result, 1);
        atomic_store(&r->in_flight, 0);
        AneCompletionFn cb = r->completion_fn;
        void* cb_user = r->completion_user;
        if (cb) cb(r, status, cb_user);
    }
    return status;
}

/* ============================================================
 * Internal fuzz harness for the spec parser. Not part of the
 * stable API; declared in ane_bridge.h purely so Rust tests can
 * link against it. See AneFuzzCase for the input model.
 * ============================================================ */
AneStatus _ane_internal_fuzz_parse_one(const AneFuzzCase* fc) {
    if (!fc) return ANE_ERR_INVALID_ARG;
    @autoreleasepool {
        NSMutableDictionary* entry = [NSMutableDictionary dictionary];
        if (fc->present_mask & ANE_FUZZ_FIELD_NAME) {
            id name_obj;
            if (fc->flags & ANE_FUZZ_FLAG_NAME_AS_NUMBER) {
                name_obj = @42;
            } else if (fc->name) {
                name_obj = [NSString stringWithUTF8String:fc->name];
                if (!name_obj) name_obj = @"";
            } else {
                name_obj = @"";
            }
            entry[@"Name"] = name_obj;
            entry[@"Symbol"] = name_obj;
        }
        if (fc->present_mask & ANE_FUZZ_FIELD_TYPE) {
            id t;
            if (fc->flags & ANE_FUZZ_FLAG_TYPE_AS_NUMBER) {
                t = @42;
            } else if (fc->type_string) {
                t = [NSString stringWithUTF8String:fc->type_string];
                if (!t) t = @"";
            } else {
                t = @"Float32";
            }
            entry[@"Type"] = t;
        }
        #define PUT_DIM(MASK, KEY, FIELD, FLAG) \
            if (fc->present_mask & MASK) { \
                if (fc->flags & FLAG) entry[KEY] = @"not_a_number"; \
                else                  entry[KEY] = @(fc->FIELD); \
            }
        PUT_DIM(ANE_FUZZ_FIELD_BATCHES,  @"Batches",  batches,  ANE_FUZZ_FLAG_BATCHES_AS_STRING)
        PUT_DIM(ANE_FUZZ_FIELD_CHANNELS, @"Channels", channels, ANE_FUZZ_FLAG_CHANNELS_AS_STRING)
        PUT_DIM(ANE_FUZZ_FIELD_DEPTH,    @"Depth",    depth,    ANE_FUZZ_FLAG_DEPTH_AS_STRING)
        PUT_DIM(ANE_FUZZ_FIELD_HEIGHT,   @"Height",   height,   ANE_FUZZ_FLAG_HEIGHT_AS_STRING)
        PUT_DIM(ANE_FUZZ_FIELD_WIDTH,    @"Width",    width,    ANE_FUZZ_FLAG_WIDTH_AS_STRING)
        #undef PUT_DIM

        NSArray* live = @[entry];
        OwnedSpec* o = NULL;
        AneTensorSpec* v = NULL;
        AneStatus s = build_specs_from_live_list(live, 1, &o, &v);
        if (s == ANE_OK) {
            free_owned_specs(o, 1);
            free(v);
        }
        return s;
    }
}

/* Build one valid LiveInputList/LiveOutputList entry dictionary for
 * `_ane_internal_fuzz_parse_attrs`. The entry uses the same canonical
 * fp32 [1, 64, 1, 16] shape as the identity model. */
static NSDictionary* fuzz_attrs_make_entry(NSString* name) {
    return @{
        @"Name":     name,
        @"Symbol":   name,
        @"Type":     @"Float32",
        @"Batches":  @1,
        @"Channels": @64,
        @"Depth":    @1,
        @"Height":   @1,
        @"Width":    @16,
    };
}

AneStatus _ane_internal_fuzz_parse_attrs(const AneFuzzAttrsCase* fc) {
    if (!fc) return ANE_ERR_INVALID_ARG;
    @autoreleasepool {
        /* Build a baseline procedure dict, then apply mutations. */
        NSMutableArray* lin  = [NSMutableArray array];
        NSMutableArray* lout = [NSMutableArray array];
        int32_t nin  = fc->n_inputs  < 0 ? 0 : fc->n_inputs;
        int32_t nout = fc->n_outputs < 0 ? 0 : fc->n_outputs;
        for (int32_t i = 0; i < nin; i++) {
            NSString* nm = [NSString stringWithFormat:@"in_%d", i];
            [lin addObject:fuzz_attrs_make_entry(nm)];
        }
        for (int32_t i = 0; i < nout; i++) {
            NSString* nm = [NSString stringWithFormat:@"out_%d", i];
            [lout addObject:fuzz_attrs_make_entry(nm)];
        }

        NSMutableDictionary* proc = [@{
            @"LiveInputList":  lin,
            @"LiveOutputList": lout,
            @"Name":           @"main",
        } mutableCopy];

        if (fc->mutations & ANE_FUZZ_ATTRS_LIVEIN_MISSING)    [proc removeObjectForKey:@"LiveInputList"];
        if (fc->mutations & ANE_FUZZ_ATTRS_LIVEOUT_MISSING)   [proc removeObjectForKey:@"LiveOutputList"];
        if (fc->mutations & ANE_FUZZ_ATTRS_LIVEIN_NOT_ARRAY)  proc[@"LiveInputList"]  = @"not_an_array";
        if (fc->mutations & ANE_FUZZ_ATTRS_LIVEOUT_NOT_ARRAY) proc[@"LiveOutputList"] = @42;

        /* Wrap into a NetworkStatusList. */
        id nsl;
        if (fc->mutations & ANE_FUZZ_ATTRS_NSL_EMPTY) {
            nsl = @[];
        } else if (fc->mutations & ANE_FUZZ_ATTRS_PROC_NOT_DICT) {
            nsl = @[ @"not_a_dict" ];
        } else {
            nsl = @[ proc ];
        }
        if (fc->mutations & ANE_FUZZ_ATTRS_NSL_NOT_ARRAY) {
            nsl = @"not_an_array";
        }

        NSMutableDictionary* attrs = [NSMutableDictionary dictionary];
        if (!(fc->mutations & ANE_FUZZ_ATTRS_NSL_MISSING)) {
            attrs[@"NetworkStatusList"] = nsl;
        }

        /* Call the dict-level helper directly. Avoids any
         * `objc_allocateClassPair` reuse problems and is also a more
         * faithful fuzz target — we're testing the production
         * dict-walking code, not a class-method indirection. */
        AneModel scratch = (AneModel){ 0 };
        AneStatus s = derive_specs_from_attrs_dict(attrs, &scratch);
        if (s == ANE_OK) {
            free_owned_specs(scratch.input_specs,  scratch.num_inputs);
            free_owned_specs(scratch.output_specs, scratch.num_outputs);
            free(scratch.input_view);
            free(scratch.output_view);
        }
        return s;
    }
}

/* Lookup table for dim-replacement helpers: maps integer index to the
 * canonical dictionary key. Returns NULL on out-of-range. */
static NSString* fuzz_dim_key(int32_t which_dim) {
    switch (which_dim) {
        case 0: return @"Batches";
        case 1: return @"Channels";
        case 2: return @"Depth";
        case 3: return @"Height";
        case 4: return @"Width";
        default: return nil;
    }
}

/* Build a minimal well-formed entry, then return it as a mutable
 * dict so callers can swap one value. */
static NSMutableDictionary* fuzz_canonical_entry(void) {
    return [@{
        @"Name":     @"x",
        @"Symbol":   @"x",
        @"Type":     @"Float32",
        @"Batches":  @1,
        @"Channels": @64,
        @"Depth":    @1,
        @"Height":   @1,
        @"Width":    @16,
    } mutableCopy];
}

/* Run the leaf parser on a single entry. Frees any returned specs. */
static AneStatus fuzz_run_leaf(NSDictionary* entry) {
    @autoreleasepool {
        NSArray* live = @[ entry ];
        OwnedSpec* o = NULL;
        AneTensorSpec* v = NULL;
        AneStatus s = build_specs_from_live_list(live, 1, &o, &v);
        if (s == ANE_OK) {
            free_owned_specs(o, 1);
            free(v);
        }
        return s;
    }
}

AneStatus _ane_internal_fuzz_dim_as_double(int32_t which_dim, double dbl_value) {
    @autoreleasepool {
        NSString* key = fuzz_dim_key(which_dim);
        if (!key) return ANE_ERR_INVALID_ARG;
        NSMutableDictionary* entry = fuzz_canonical_entry();
        entry[key] = @(dbl_value);
        return fuzz_run_leaf(entry);
    }
}

AneStatus _ane_internal_fuzz_dim_as_uint64(int32_t which_dim, uint64_t value) {
    @autoreleasepool {
        NSString* key = fuzz_dim_key(which_dim);
        if (!key) return ANE_ERR_INVALID_ARG;
        NSMutableDictionary* entry = fuzz_canonical_entry();
        entry[key] = [NSNumber numberWithUnsignedLongLong:value];
        return fuzz_run_leaf(entry);
    }
}

AneStatus _ane_internal_fuzz_dim_as_decimal(int32_t which_dim, const char* decimal_str) {
    @autoreleasepool {
        NSString* key = fuzz_dim_key(which_dim);
        if (!key || !decimal_str) return ANE_ERR_INVALID_ARG;
        NSMutableDictionary* entry = fuzz_canonical_entry();
        NSString* str = [NSString stringWithUTF8String:decimal_str];
        if (!str) return ANE_ERR_INVALID_ARG;
        NSDecimalNumber* d = [NSDecimalNumber decimalNumberWithString:str];
        entry[key] = d;
        return fuzz_run_leaf(entry);
    }
}

AneStatus _ane_internal_fuzz_value_is_nsnull(int32_t which_key) {
    @autoreleasepool {
        static NSString* const keys[7] = {
            @"Name", @"Type", @"Batches", @"Channels", @"Depth", @"Height", @"Width",
        };
        if (which_key < 0 || which_key >= 7) return ANE_ERR_INVALID_ARG;
        NSMutableDictionary* entry = fuzz_canonical_entry();
        entry[keys[which_key]] = [NSNull null];
        return fuzz_run_leaf(entry);
    }
}

AneStatus _ane_internal_fuzz_name_with_embedded_nul(void) {
    @autoreleasepool {
        /* NSString *will* accept embedded NULs (it's char-based, not
         * NUL-terminated). UTF8String returns a C-string that, when
         * passed to strdup, truncates at the first NUL. */
        NSString* name = [[NSString alloc]
            initWithBytes:"valid\0continued"
                   length:15
                 encoding:NSUTF8StringEncoding];
        if (!name) return ANE_ERR_INTERNAL;
        NSMutableDictionary* entry = fuzz_canonical_entry();
        entry[@"Name"]   = name;
        entry[@"Symbol"] = name;
        AneStatus s = fuzz_run_leaf(entry);
        [name release];
        return s;
    }
}

AneStatus _ane_internal_fuzz_huge_name(size_t length) {
    @autoreleasepool {
        /* Cap absurd sizes — proptest could send length=usize::MAX. */
        if (length > (1u << 24)) return ANE_ERR_INVALID_ARG;  /* 16 MB cap */
        char* buf = (char*)malloc(length + 1);
        if (!buf) return ANE_ERR_OOM;
        memset(buf, 'a', length);
        buf[length] = '\0';
        NSString* name = [NSString stringWithUTF8String:buf];
        free(buf);
        if (!name) return ANE_ERR_INTERNAL;
        NSMutableDictionary* entry = fuzz_canonical_entry();
        entry[@"Name"]   = name;
        entry[@"Symbol"] = name;
        return fuzz_run_leaf(entry);
    }
}

AneStatus _ane_internal_fuzz_mixed_validity_two_entries(void) {
    @autoreleasepool {
        NSMutableDictionary* good = fuzz_canonical_entry();
        NSMutableDictionary* bad  = fuzz_canonical_entry();
        bad[@"Type"] = @"NotARealDtype";
        NSArray* live = @[ good, bad ];
        OwnedSpec* o = NULL;
        AneTensorSpec* v = NULL;
        AneStatus s = build_specs_from_live_list(live, 2, &o, &v);
        if (s == ANE_OK) {
            free_owned_specs(o, 2);
            free(v);
        }
        return s;
    }
}

AneStatus _ane_internal_fuzz_parse_one_with_double_batches(double dbl_value) {
    @autoreleasepool {
        NSMutableDictionary* entry = [@{
            @"Name":     @"x",
            @"Symbol":   @"x",
            @"Type":     @"Float32",
            @"Channels": @64,
            @"Depth":    @1,
            @"Height":   @1,
            @"Width":    @16,
        } mutableCopy];
        entry[@"Batches"] = @(dbl_value);  /* NSNumber wrapping a double */
        NSArray* live = @[ entry ];
        OwnedSpec* o = NULL;
        AneTensorSpec* v = NULL;
        AneStatus s = build_specs_from_live_list(live, 1, &o, &v);
        if (s == ANE_OK) {
            free_owned_specs(o, 1);
            free(v);
        }
        return s;
    }
}

AneStatus ane_request_set_completion(AneRequest* r, AneCompletionFn fn, void* user) {
    clear_last_error();
    if (!r) { set_last_error("null request"); return ANE_ERR_INVALID_ARG; }
    /* Drain any pending callback before swapping `(fn, user)`.
     *
     * Without this, the dispatch_async block already sets has_result=1,
     * in_flight=0, and signals the semaphore *before* invoking the
     * user callback. A waiter (or async Future) returning from
     * `wait`/`.await` at the semaphore signal could call
     * `ane_request_set_completion` on this same request while the
     * worker thread is still inside the previous callback, with its
     * own captured `cb_user` pointing at the about-to-be-freed box.
     *
     * `dispatch_sync` on the request's serial queue serializes us
     * after any block currently executing on it, so by the time this
     * returns the previous callback has run to completion and the
     * caller can safely free the old user pointer. Calling this from
     * inside a completion callback would deadlock — that is documented
     * as a misuse on the Rust side. */
    dispatch_sync(r->queue, ^{});
    r->completion_fn = fn;
    r->completion_user = user;
    return ANE_OK;
}

const char* ane_request_last_error(const AneRequest* r) {
    if (!r) return "";
    /* The worker writes `last_err_msg` BEFORE `atomic_store(has_result,
     * 1)` (seq_cst). If `has_result` is 0 an eval is still in flight
     * and the buffer could be mid-write — return "" rather than risk
     * exposing a partial string. The buffer itself is inline, so even
     * a torn read cannot dereference freed memory. */
    if (atomic_load((atomic_int*)&((AneRequest*)r)->has_result) == 0) return "";
    return r->last_err_msg;
}
