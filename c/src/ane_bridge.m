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
static void err_destructor(void* p) {
    free(p);
}
static void err_key_init(void) {
    pthread_key_create(&g_err_key, err_destructor);
}

__attribute__((format(printf, 1, 2))) static void set_last_error(const char* fmt, ...) {
    pthread_once(&g_err_once, err_key_init);
    char* buf = (char*)malloc(1024);
    if (!buf) {
        return;
    }
    va_list ap; // NOLINT(cppcoreguidelines-init-variables): va_list is initialized by va_start
    va_start(ap, fmt);
    /* `fmt` is forwarded from a checked vararg wrapper, not attacker input;
     * the format attribute above validates call sites. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
    vsnprintf(buf, 1024, fmt, ap);
#pragma clang diagnostic pop
    va_end(ap);
    char* prev = (char*)pthread_getspecific(g_err_key);
    if (prev) {
        free(prev);
    }
    pthread_setspecific(g_err_key, buf);
}

static void clear_last_error(void) {
    pthread_once(&g_err_once, err_key_init);
    char* prev = (char*)pthread_getspecific(g_err_key);
    if (prev) {
        free(prev);
        pthread_setspecific(g_err_key, NULL);
    }
}

const char* ane_last_error(void) {
    pthread_once(&g_err_once, err_key_init);
    const char* s = (const char*)pthread_getspecific(g_err_key);
    return s ? s : "";
}

const char* ane_bridge_version(void) {
    return ANE_BRIDGE_VERSION;
}

/* Format an NSError (plus any `NSUnderlyingErrorKey`) into the
 * thread-local last-error string, prefixed by `phase` (e.g. "compile
 * failed"). The underlying error carries the failure *kind*, so
 * surfacing both tells the caller whether the model is broken or just
 * not ANE-compatible. */
static void set_phase_error(const char* phase, NSError* e) {
    NSError* under = e ? e.userInfo[NSUnderlyingErrorKey] : nil;
    const char* desc = e ? [[e localizedDescription] UTF8String] : "<no NSError>";
    const char* udesc = under ? [[under localizedDescription] UTF8String] : NULL;
    if (udesc) {
        set_last_error("%s: %s | underlying: %s", phase, desc, udesc);
    } else {
        set_last_error("%s: %s", phase, desc);
    }
}

/* ============================================================
 * Dtype
 * ============================================================ */

size_t ane_dtype_size(AneDtype dt) {
    switch (dt) {
    case ANE_DTYPE_FP32:
        return 4;
    case ANE_DTYPE_FP16:
        return 2;
    case ANE_DTYPE_INT32:
        return 4;
    case ANE_DTYPE_INT64:
        return 8;
    case ANE_DTYPE_UINT8:
    case ANE_DTYPE_INT8:
        return 1;
    }
    return 0;
}

/* ============================================================
 * IOSurface helpers
 * ============================================================ */

static IOSurfaceRef make_surface(size_t bytes) {
    size_t W = 0;
    size_t H = 0;
    if (bytes == 0) {
        W = 1;
        H = 1;
    } else if (bytes <= 16384) {
        /* Single row: BytesPerRow alignment is irrelevant with H == 1 (there
         * are no inter-row gaps), so an exact-width 1xN surface is contiguous. */
        W = bytes;
        H = 1;
    } else {
        /* Multi-row: pin the row width to 4096 (page-aligned, so IOSurface
         * never rounds BytesPerRow up and inserts inter-row padding that would
         * break the ANE's linear read of the first b->nbytes bytes) and pad the
         * height. The earlier "largest power-of-two divisor" width could
         * collapse below 64 bytes for sizes with few factors of two (e.g. a
         * large prime) — risking exactly that row-alignment padding, plus a
         * pathologically tall surface. Trailing bytes of the last row are
         * unused; b->nbytes tracks the logical size. */
        W = 4096;
        H = (bytes + W - 1) / W;
    }
    /* W*H == bytes for an exact tiling, and >= bytes when the last row is
     * padded; either way it covers the logical payload. */
    size_t alloc = W * H;
    return IOSurfaceCreate((__bridge CFDictionaryRef) @{
        (id)kIOSurfaceWidth:
            @(W), // NOLINT(readability-redundant-parentheses): @() boxing requires parens
        (id)kIOSurfaceHeight:
            @(H), // NOLINT(readability-redundant-parentheses): @() boxing requires parens
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow:
            @(W), // NOLINT(readability-redundant-parentheses): @() boxing requires parens
        (id)kIOSurfaceAllocSize:
            @(alloc), // NOLINT(readability-redundant-parentheses): @() boxing requires parens
        (id)kIOSurfacePixelFormat: @0,
    });
}

/* `_ANEClient` is constructed identically everywhere: alloc +
 * `initWithRestrictedAccessAllowed:YES`, returning a +1 owned reference
 * the caller is responsible for releasing. Centralize the objc_msgSend
 * cast so the eight construction sites cannot drift. */
static id make_ane_client(void) NS_RETURNS_RETAINED {
    // NOLINTNEXTLINE(clang-analyzer-osx.cocoa.RetainCount): analyzer can't track the +1 through the objc_msgSend cast; function returns +1 (NS_RETURNS_RETAINED)
    return ((id(*)(id, SEL, BOOL))objc_msgSend)(
        [g_AneClientCls alloc], sel_getUid("initWithRestrictedAccessAllowed:"), YES);
}

/* ============================================================
 * Buffer
 * ============================================================ */

struct AneBuffer {
    IOSurfaceRef surface;
    id wrapper; /* _ANEIOSurfaceObject */
    size_t nbytes;
    /* The IOSurface flags from the most recent `ane_buffer_lock`; replayed
     * in `ane_buffer_unlock` because IOSurfaceUnlock needs the same mask
     * that was passed to IOSurfaceLock. Single-slot by design — callers
     * must not nest two locks of differing access modes on the same
     * buffer. The Rust wrapper enforces this via `&mut Buffer` on
     * `Buffer::lock`. */
    int last_lock_flags;
};

AneStatus ane_buffer_create(size_t nbytes, AneBuffer** out) {
    clear_last_error();
    if (!out) {
        set_last_error("ane_buffer_create: null out");
        return ANE_ERR_INVALID_ARG;
    }
    if (!ane_private_load()) {
        set_last_error("AppleNeuralEngine framework not loadable");
        return ANE_ERR_UNSUPPORTED;
    }
    AneBuffer* b = (AneBuffer*)calloc(1, sizeof(AneBuffer));
    if (!b) {
        set_last_error("oom");
        return ANE_ERR_OOM;
    }

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
            CFRelease(b->surface);
            free(b);
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
    clear_last_error();
    size_t n = ane_model_input_nbytes(m, idx);
    if (n == 0) {
        return ANE_ERR_INVALID_ARG;
    }
    return ane_buffer_create(n, out);
}

AneStatus ane_buffer_create_for_output(const AneModel* m, int32_t idx, AneBuffer** out) {
    clear_last_error();
    size_t n = ane_model_output_nbytes(m, idx);
    if (n == 0) {
        return ANE_ERR_INVALID_ARG;
    }
    return ane_buffer_create(n, out);
}

AneStatus ane_buffer_create_for_state(const AneModel* m, int32_t idx, AneBuffer** out) {
    clear_last_error();
    size_t n = ane_model_state_nbytes(m, idx);
    if (n == 0) {
        return ANE_ERR_INVALID_ARG;
    }
    return ane_buffer_create(n, out);
}

void ane_buffer_release(AneBuffer* b) {
    if (!b) {
        return;
    }
    if (b->wrapper) {
        [b->wrapper release];
    }
    if (b->surface) {
        CFRelease(b->surface);
    }
    free(b);
}

AneStatus ane_buffer_lock(AneBuffer* b, AneBufferAccess access, void** out_ptr) {
    clear_last_error();
    if (!b || !out_ptr) {
        set_last_error("null arg");
        return ANE_ERR_INVALID_ARG;
    }
    uint32_t flags = 0;
    if (access == ANE_LOCK_READ) {
        flags = kIOSurfaceLockReadOnly;
    } else if (access == ANE_LOCK_WRITE_NOSYNC) {
        flags = kIOSurfaceLockAvoidSync;
    }
    if (IOSurfaceLock(b->surface, (IOSurfaceLockOptions)flags, NULL) != kIOReturnSuccess) {
        set_last_error("IOSurfaceLock failed");
        return ANE_ERR_INTERNAL;
    }
    b->last_lock_flags = (int)flags;
    *out_ptr = IOSurfaceGetBaseAddress(b->surface);
    return ANE_OK;
}

AneStatus ane_buffer_unlock(AneBuffer* b) {
    clear_last_error();
    if (!b) {
        return ANE_ERR_INVALID_ARG;
    }
    /* Replay the lock flags, but strip kIOSurfaceLockAvoidSync: a write made
     * under ANE_LOCK_WRITE_NOSYNC skips the read-side cache maintenance on
     * *lock* (the dominant cost), yet its writes must still be published to
     * the device on *unlock*. Avoiding sync on unlock too would leave CPU
     * writes stranded in cache, invisible to an ANE that later DMA-reads the
     * buffer. The kIOSurfaceLockReadOnly bit (the only other flag we set) is
     * preserved so a read lock stays non-dirtying. */
    uint32_t flags = (uint32_t)b->last_lock_flags & ~(uint32_t)kIOSurfaceLockAvoidSync;
    if (IOSurfaceUnlock(b->surface, (IOSurfaceLockOptions)flags, NULL) != kIOReturnSuccess) {
        set_last_error("IOSurfaceUnlock failed");
        return ANE_ERR_INTERNAL;
    }
    return ANE_OK;
}

size_t ane_buffer_nbytes(const AneBuffer* b) {
    return b ? b->nbytes : 0;
}
uint32_t ane_buffer_iosurface_id(const AneBuffer* b) {
    return (b && b->surface) ? IOSurfaceGetID(b->surface) : 0;
}

/* ============================================================
 * Model
 * ============================================================ */

typedef struct {
    char* name;
    AneDtype dtype;
    int32_t rank;
    int64_t* shape;
    size_t nbytes;
} OwnedSpec;

typedef struct ProcSpecs {
    int32_t num_inputs;
    int32_t num_outputs;
    int32_t num_states;
    OwnedSpec* input_specs;
    OwnedSpec* output_specs;
    OwnedSpec* state_specs;
    AneTensorSpec* input_view;
    AneTensorSpec* output_view;
    AneTensorSpec* state_view;
} ProcSpecs;

struct AneModel {
    /* Source-artifact keepalives. The framework reads MIL + weights at
     * compile time (from the temp-dir files we materialize) and bakes the
     * lowered program into aned's cache, so these bytes are not referenced
     * at eval. We nonetheless retain them for the model's lifetime as a
     * deliberate belt-and-suspenders against any framework version that
     * holds an unretained reference into the source NSData. Never read
     * directly by the bridge — do not mistake them for dead fields. */
    NSData* mil_data;
    NSData* weights_data;
    NSMutableArray* extra_weights_data; /* multi-blob variant for open_ex */
    id descriptor;
    id in_memory_model;
    id xpc_model; /* `_ANEModel` from `+modelWithCacheURLIdentifier:` — NSSecureCoding-conformant;
                     used for XPC-bound paths like chain dispatch. */
    id client;
    bool is_secondary_instance; /* shares in_memory_model with primary; do not unload/release on
                                   close */
    /* `Model` is `Send + Sync` on the Rust side and shared via `Arc`, so
     * the two fields that change after open — `perf_stats_mask` (via
     * `ane_model_set_perf_stats_mask`) and `loaded` (via
     * `ane_model_unload`) — can be written from one thread while another
     * reads them. Make them atomic so those `&self` mutators are not a
     * data race; the underlying objc message is serialized by the
     * framework. Every other field is write-once before publication. */
    atomic_uint perf_stats_mask;
    /* `temp_dir` is `NSTemporaryDirectory()/<hex_id>/` populated with
     * `model.mil` + `weights/weight.bin`. `_ANECompiler` reads those
     * files (its `ANECCompile()` log shows the path it consults), so
     * the materialization is required on the cold-compile path. On a
     * cache hit we skip both fields, since `loadWithQoS:` does not
     * touch the directory. */
    NSString* hex_id;
    NSString* temp_dir;
    AneQoS compile_qos;
    bool cache_hit;     /* set when `compiledModelExists` short-circuited compile */
    atomic_bool loaded; /* set after a successful loadWithQoS:; gates unloadWithQoS: */

    int32_t num_inputs;
    int32_t num_outputs;
    OwnedSpec* input_specs;
    OwnedSpec* output_specs;

    /* Public-facing view (filled with name/shape pointers into OwnedSpec). */
    AneTensorSpec* input_view;
    AneTensorSpec* output_view;

    /* Resident state tensors (MLState). A third tensor category alongside
     * inputs/outputs; populated once the framework's state list is parsed.
     * A model with no states reports 0 cleanly —
     * `calloc` zeroes these. */
    int32_t num_states;
    OwnedSpec* state_specs;
    AneTensorSpec* state_view;

    int32_t num_procedures;
    ProcSpecs* extra_procs; /* length = num_procedures - 1 (procs 1..N) */
};

static void free_owned_specs(OwnedSpec* specs, int32_t n);

static void free_proc_specs(ProcSpecs* p) {
    if (!p) {
        return;
    }
    free_owned_specs(p->input_specs, p->num_inputs);
    free_owned_specs(p->output_specs, p->num_outputs);
    free_owned_specs(p->state_specs, p->num_states);
    free(p->input_view);
    free(p->output_view);
    free(p->state_view);
}

static void free_owned_specs(OwnedSpec* specs, int32_t n) {
    if (!specs) {
        return;
    }
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
    if (![s isKindOfClass:[NSString class]]) {
        // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange): 0 is the documented "unknown dtype" sentinel
        return (AneDtype)0;
    }
    NSString* ns = (NSString*)s;
    if ([ns isEqualToString:@"Float32"]) {
        return ANE_DTYPE_FP32;
    }
    if ([ns isEqualToString:@"Float16"]) {
        return ANE_DTYPE_FP16;
    }
    if ([ns isEqualToString:@"Int32"]) {
        return ANE_DTYPE_INT32;
    }
    if ([ns isEqualToString:@"Int64"]) {
        return ANE_DTYPE_INT64;
    }
    if ([ns isEqualToString:@"UInt8"]) {
        return ANE_DTYPE_UINT8;
    }
    if ([ns isEqualToString:@"Int8"]) {
        return ANE_DTYPE_INT8;
    }
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange): 0 is the documented "unknown dtype" sentinel
    return (AneDtype)0;
}

/* Strip the framework's "@output" suffix from output tensor names so
 * users see the same name the MIL declared. Returns NULL if the input
 * isn't an NSString — caller must check. */
static char* dup_name_clean(id nsname) {
    if (![nsname isKindOfClass:[NSString class]]) {
        return NULL;
    }
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
static AneStatus build_specs_from_live_list(NSArray* live, int32_t n, OwnedSpec** out_owned,
                                            AneTensorSpec** out_view) {
    OwnedSpec* o = (OwnedSpec*)calloc((size_t)(n > 0 ? n : 1), sizeof(OwnedSpec));
    AneTensorSpec* v = (AneTensorSpec*)calloc((size_t)(n > 0 ? n : 1), sizeof(AneTensorSpec));
    if (!o || !v) {
        free(o);
        free(v);
        return ANE_ERR_OOM;
    }
    for (int32_t i = 0; i < n; i++) {
        id d_id = live[(NSUInteger)i];
        if (![d_id isKindOfClass:[NSDictionary class]]) {
            free_owned_specs(o, i);
            free(v);
            set_last_error("LiveList[%d] not a dict", i);
            return ANE_ERR_INTERNAL;
        }
        NSDictionary* d = (NSDictionary*)d_id;
        id name = d[@"Name"];
        id type = d[@"Type"];
        id batch = d[@"Batches"];
        id chan = d[@"Channels"];
        id depth = d[@"Depth"];
        id h = d[@"Height"];
        id w = d[@"Width"];
        if (!name || !type || !batch || !chan || !depth || !h || !w) {
            free_owned_specs(o, i);
            free(v);
            set_last_error("LiveList[%d] missing Name/Type/dim key", i);
            return ANE_ERR_INTERNAL;
        }
        /* Type-check every value before sending selectors. A future
         * framework version could put non-string Names or non-number
         * dims under the same keys; the parser must reject cleanly
         * rather than throw NSInvalidArgumentException. */
        if (![name isKindOfClass:[NSString class]]) {
            free_owned_specs(o, i);
            free(v);
            set_last_error("LiveList[%d] Name not a string", i);
            return ANE_ERR_INTERNAL;
        }
        if (![batch isKindOfClass:[NSNumber class]] || ![chan isKindOfClass:[NSNumber class]] ||
            ![depth isKindOfClass:[NSNumber class]] || ![h isKindOfClass:[NSNumber class]] ||
            ![w isKindOfClass:[NSNumber class]]) {
            free_owned_specs(o, i);
            free(v);
            set_last_error("LiveList[%d] non-numeric dim", i);
            return ANE_ERR_INTERNAL;
        }
        AneDtype dt = parse_dtype_string(type);
        if (dt == 0) {
            free_owned_specs(o, i);
            free(v);
            set_last_error("LiveList[%d] unsupported/non-string dtype", i);
            return ANE_ERR_UNSUPPORTED;
        }
        /* Compose a 4-D NCHW shape, expanding to 5-D NCDHW when depth>1.
         * This mirrors how the framework canonicalizes ANE tensor
         * layout — we don't recover the user's original MIL rank, but
         * the bytes are exact. */
        int64_t bz = [(NSNumber*)batch longLongValue];
        int64_t cz = [(NSNumber*)chan longLongValue];
        int64_t dz = [(NSNumber*)depth longLongValue];
        int64_t hz = [(NSNumber*)h longLongValue];
        int64_t wz = [(NSNumber*)w longLongValue];
        /* Every reported dim must be positive — depth=1 is the
         * canonical "no third spatial axis" value, but anything <1
         * is malformed framework output. Reject up front rather
         * than letting the rank-selection branch silently drop a
         * negative depth from the shape. */
        if (bz < 1 || cz < 1 || dz < 1 || hz < 1 || wz < 1) {
            free_owned_specs(o, i);
            free(v);
            set_last_error("LiveList[%d] non-positive dimension", i);
            return ANE_ERR_INTERNAL;
        }
        int32_t rank = 0;
        int64_t shape[5];
        if (dz > 1) {
            rank = 5;
            shape[0] = bz;
            shape[1] = cz;
            shape[2] = dz;
            shape[3] = hz;
            shape[4] = wz;
        } else {
            rank = 4;
            shape[0] = bz;
            shape[1] = cz;
            shape[2] = hz;
            shape[3] = wz;
        }
        /* Byte count via checked mul. */
        size_t elt = ane_dtype_size(dt);
        size_t nbytes = elt;
        for (int32_t k = 0; k < rank; k++) {
            if (shape[k] <= 0) {
                free_owned_specs(o, i);
                free(v);
                set_last_error("LiveList[%d] non-positive dim", i);
                return ANE_ERR_INTERNAL;
            }
            size_t dim = (size_t)shape[k];
            if (dim != 0 && nbytes > SIZE_MAX / dim) {
                free_owned_specs(o, i);
                free(v);
                set_last_error("LiveList[%d] nbytes overflows", i);
                return ANE_ERR_INTERNAL;
            }
            nbytes *= dim;
        }
        o[i].name = dup_name_clean(name);
        o[i].dtype = dt;
        o[i].rank = rank;
        o[i].shape = (int64_t*)calloc((size_t)rank, sizeof(int64_t));
        if (!o[i].shape) {
            free_owned_specs(o, i + 1);
            free(v);
            return ANE_ERR_OOM;
        }
        memcpy(o[i].shape, shape, (size_t)rank * sizeof(int64_t));
        o[i].nbytes = nbytes;
        v[i].name = o[i].name;
        v[i].dtype = o[i].dtype;
        v[i].rank = o[i].rank;
        v[i].shape = o[i].shape;
    }
    *out_owned = o;
    *out_view = v;
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
    id proc_id = nsl[0]; /* procedure 0 = "main" */
    if (![proc_id isKindOfClass:[NSDictionary class]]) {
        set_last_error("NetworkStatusList[0] not a dict");
        return ANE_ERR_INTERNAL;
    }
    NSDictionary* proc = (NSDictionary*)proc_id;
    id lin_id = proc[@"LiveInputList"];
    id lout_id = proc[@"LiveOutputList"];
    if (!lin_id || !lout_id || ![lin_id isKindOfClass:[NSArray class]] ||
        ![lout_id isKindOfClass:[NSArray class]]) {
        set_last_error("LiveInputList/LiveOutputList missing or wrong type");
        return ANE_ERR_INTERNAL;
    }
    NSArray* lin = (NSArray*)lin_id;
    NSArray* lout = (NSArray*)lout_id;
    int32_t nin = (int32_t)lin.count;
    int32_t nout = (int32_t)lout.count;
    AneStatus s = build_specs_from_live_list(lin, nin, &m->input_specs, &m->input_view);
    if (s != ANE_OK) {
        return s;
    }
    /* Record nin before attempting output build so cleanup can free the
     * allocated input_specs if the output step fails. */
    m->num_inputs = nin;
    s = build_specs_from_live_list(lout, nout, &m->output_specs, &m->output_view);
    if (s != ANE_OK) {
        return s;
    }
    m->num_outputs = nout;

    int32_t n_extra = (int32_t)nsl.count - 1;
    m->num_procedures = (int32_t)nsl.count;
    if (n_extra > 0) {
        m->extra_procs = (ProcSpecs*)calloc((size_t)n_extra, sizeof(ProcSpecs));
        if (!m->extra_procs) {
            return ANE_ERR_OOM;
        }
        for (int32_t i = 0; i < n_extra; i++) {
            id extra_proc_id = nsl[(NSUInteger)(i + 1)];
            if (![extra_proc_id isKindOfClass:[NSDictionary class]]) {
                continue;
            }
            NSDictionary* pd = (NSDictionary*)extra_proc_id;
            id li = pd[@"LiveInputList"];
            id lo = pd[@"LiveOutputList"];
            if (![li isKindOfClass:[NSArray class]] || ![lo isKindOfClass:[NSArray class]]) {
                continue;
            }
            int32_t pnin = (int32_t)((NSArray*)li).count;
            int32_t pnout = (int32_t)((NSArray*)lo).count;
            AneStatus ps = build_specs_from_live_list(
                (NSArray*)li, pnin, &m->extra_procs[i].input_specs, &m->extra_procs[i].input_view);
            if (ps != ANE_OK) {
                return ps;
            }
            m->extra_procs[i].num_inputs = pnin;
            ps = build_specs_from_live_list((NSArray*)lo, pnout, &m->extra_procs[i].output_specs,
                                            &m->extra_procs[i].output_view);
            if (ps != ANE_OK) {
                return ps;
            }
            m->extra_procs[i].num_outputs = pnout;
        }
    }
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
        NSString* milPath = [NSString stringWithUTF8String:opts->mil_path];
        NSString* wtsPath = [NSString stringWithUTF8String:opts->weights_path];
        if (!milPath || !wtsPath) {
            set_last_error("model path is not valid UTF-8");
            return ANE_ERR_INVALID_ARG;
        }
        NSData* mil = [NSData dataWithContentsOfFile:milPath];
        NSData* wts = [NSData dataWithContentsOfFile:wtsPath];
        if (!mil) {
            set_last_error("failed to read MIL: %s", opts->mil_path);
            return ANE_ERR_IO;
        }
        if (!wts) {
            set_last_error("failed to read weights: %s", opts->weights_path);
            return ANE_ERR_IO;
        }

        AneModel* m = (AneModel*)calloc(1, sizeof(AneModel));
        if (!m) {
            set_last_error("oom");
            return ANE_ERR_OOM;
        }
        m->mil_data = [mil retain];
        m->weights_data = [wts retain];
        m->compile_qos = opts->compile_qos ? opts->compile_qos : ANE_QOS_DEFAULT;
        /* num_inputs/num_outputs and the spec arrays are derived from
         * the framework after loadWithQoS:; see derive_specs_from_attrs. */

        NSDictionary* wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wts}};
        m->descriptor = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            g_AneDescriptorCls, sel_getUid("modelWithMILText:weights:optionsPlist:"), mil, wdict,
            nil);
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
            cache_hit = ((BOOL(*)(id, SEL))objc_msgSend)(m->in_memory_model, exists_sel);
        }
        m->cache_hit = cache_hit ? true : false;

        if (!cache_hit) {
            /* `_ANECompiler` reads the program text + weights from
             * `NSTemporaryDirectory()/<hex_id>/{model.mil,
             * weights/weight.bin}` (its `ANECCompile()` failure log
             * shows the path it consults). Materialize only on the
             * compile path — `loadWithQoS:` does not read from disk,
             * so a warm open skips this I/O entirely. */
            id hxid = ((id(*)(id, SEL))objc_msgSend)(m->in_memory_model,
                                                     sel_getUid("hexStringIdentifier"));
            m->hex_id = [hxid retain];
            m->temp_dir =
                [[NSTemporaryDirectory() stringByAppendingPathComponent:m->hex_id] retain];
            NSFileManager* fm = [NSFileManager defaultManager];
            NSError* fserr = nil;
            if (![fm createDirectoryAtPath:[m->temp_dir stringByAppendingPathComponent:@"weights"]
                    withIntermediateDirectories:YES
                                     attributes:nil
                                          error:&fserr]) {
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
                /* Top-level message is usually `_ANECompiler :
                 * ANECCompile() FAILED`; the underlying error names the
                 * kind (e.g. `CompilationFailure` for a non-ANE-targetable
                 * graph). set_phase_error surfaces both. */
                set_phase_error("compile failed", e);
                ane_model_close(m);
                return ANE_ERR_COMPILE;
            }
        }

        BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
            m->in_memory_model, sel_getUid("loadWithQoS:options:error:"),
            (unsigned int)m->compile_qos, @{}, &e);
        if (!ok) {
            set_phase_error("load failed", e);
            ane_model_close(m);
            return ANE_ERR_LOAD;
        }
        atomic_store(&m->loaded, true);

        /* The loaded model knows its own schema; derive it from the
         * framework's `modelAttributes` rather than trusting whatever
         * the caller might have declared. This is the single source
         * of truth for input/output shapes, names, and dtypes. */
        AneStatus dst = derive_specs_from_attrs(m->in_memory_model, m);
        if (dst != ANE_OK) {
            ane_model_close(m);
            return dst;
        }

        m->client = make_ane_client();
        if (!m->client) {
            ane_model_close(m);
            set_last_error("_ANEClient init failed");
            return ANE_ERR_INTERNAL;
        }

        /* After a successful compile (or load on a cache hit) the
         * framework's `aned` cache has a bundle keyed by our
         * `hexStringIdentifier`. `_ANEModel
         * +modelWithCacheURLIdentifier:` returns an NSSecureCoding-
         * conformant handle to that bundle, which is the type the
         * chain-dispatch XPC encoder expects. The in-memory model is
         * kept for `doEvaluateDirectWithModel:`; the xpc_model is the
         * alternate handle for any path that crosses the XPC boundary. */
        if (!m->hex_id) {
            id hxid = ((id(*)(id, SEL))objc_msgSend)(m->in_memory_model,
                                                     sel_getUid("hexStringIdentifier"));
            if (hxid) {
                m->hex_id = [hxid retain];
            }
        }
        if (g_AneModelCls && m->hex_id) {
            SEL s = sel_getUid("modelWithCacheURLIdentifier:");
            if ([g_AneModelCls respondsToSelector:s]) {
                id xm = ((id(*)(Class, SEL, id))objc_msgSend)(g_AneModelCls, s, m->hex_id);
                if (xm) {
                    /* Load via `_ANEClient.loadModel:` so the daemon
                     * side associates the cache-derived handle with
                     * the program loaded by `in_memory_model`. */
                    NSError* lerr = nil;
                    BOOL lok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError**))objc_msgSend)(
                        m->client, sel_getUid("loadModel:options:qos:error:"), xm, @{},
                        (unsigned int)m->compile_qos, &lerr);
                    /* Silently skip when loadModel fails — this happens
                     * on every unsigned binary because aned demands a
                     * sandbox extension we can't mint. The xpc_model
                     * remains nil and chain/new_instance/etc. cleanly
                     * surface Unsupported when reached. */
                    if (lok) {
                        m->xpc_model = [xm retain];
                    }
                    (void)lerr;
                }
            }
        }

        *out_model = m;
        return ANE_OK;
    }
}

void ane_model_close(AneModel* m) {
    if (!m) {
        return;
    }
    @autoreleasepool {
        if (m->in_memory_model) {
            /* Only `unloadWithQoS:` if `loadWithQoS:` actually succeeded.
             * Calling unload on a never-loaded model returns NO with an
             * NSError that we'd otherwise log as a misleading "model
             * unload failed" — exactly the kind of noise that masks a
             * real failure during partial-open cleanup.
             *
             * Secondary instances share the `_ANEInMemoryModel*` with the
             * primary; only the primary owns the unload. Skip unload AND
             * the matching retain-balanced release here — the primary's
             * close drops the final retain. */
            if (atomic_load(&m->loaded) && !m->is_secondary_instance) {
                NSError* e = nil;
                BOOL ok = ((BOOL(*)(id, SEL, unsigned int, NSError**))objc_msgSend)(
                    m->in_memory_model, sel_getUid("unloadWithQoS:error:"),
                    (m->compile_qos ? m->compile_qos : ANE_QOS_DEFAULT), &e);
                if (!ok) {
                    (void)fprintf(stderr, "ane-bridge: model unload failed: %s\n",
                                  e ? [[e localizedDescription] UTF8String] : "<no NSError>");
                }
            }
            [m->in_memory_model release];
        }
        if (m->descriptor && !m->is_secondary_instance) {
            [m->descriptor release];
        }
        if (m->xpc_model) {
            [m->xpc_model release];
        }
        if (m->client) {
            [m->client release];
        }
        /* temp_dir/hex_id are only set on the cache-miss path; safe to
         * skip cleanly on warm starts. */
        if (m->temp_dir) {
            [[NSFileManager defaultManager] removeItemAtPath:m->temp_dir error:nil];
            [m->temp_dir release];
        }
        if (m->hex_id) {
            [m->hex_id release];
        }
        if (m->mil_data) {
            [m->mil_data release];
        }
        if (m->weights_data) {
            [m->weights_data release];
        }
        if (m->extra_weights_data) {
            [m->extra_weights_data release];
        }
        free_owned_specs(m->input_specs, m->num_inputs);
        free_owned_specs(m->output_specs, m->num_outputs);
        free_owned_specs(m->state_specs, m->num_states);
        free(m->input_view);
        free(m->output_view);
        free(m->state_view);
        if (m->extra_procs) {
            int32_t n_extra = m->num_procedures > 0 ? m->num_procedures - 1 : 0;
            for (int32_t i = 0; i < n_extra; i++) {
                free_proc_specs(&m->extra_procs[i]);
            }
            free(m->extra_procs);
        }
        free(m);
    }
}

int32_t ane_model_num_inputs(const AneModel* m) {
    return m ? m->num_inputs : 0;
}
int32_t ane_model_num_outputs(const AneModel* m) {
    return m ? m->num_outputs : 0;
}

const AneTensorSpec* ane_model_input_spec(const AneModel* m, int32_t i) {
    if (!m || i < 0 || i >= m->num_inputs) {
        return NULL;
    }
    return &m->input_view[i];
}
const AneTensorSpec* ane_model_output_spec(const AneModel* m, int32_t i) {
    if (!m || i < 0 || i >= m->num_outputs) {
        return NULL;
    }
    return &m->output_view[i];
}
size_t ane_model_input_nbytes(const AneModel* m, int32_t i) {
    if (!m || i < 0 || i >= m->num_inputs) {
        return 0;
    }
    return m->input_specs[i].nbytes;
}
size_t ane_model_output_nbytes(const AneModel* m, int32_t i) {
    if (!m || i < 0 || i >= m->num_outputs) {
        return 0;
    }
    return m->output_specs[i].nbytes;
}

int32_t ane_model_num_states(const AneModel* m) {
    return m ? m->num_states : 0;
}
const AneTensorSpec* ane_model_state_spec(const AneModel* m, int32_t i) {
    if (!m || i < 0 || i >= m->num_states) {
        return NULL;
    }
    return &m->state_view[i];
}
size_t ane_model_state_nbytes(const AneModel* m, int32_t i) {
    if (!m || i < 0 || i >= m->num_states) {
        return 0;
    }
    return m->state_specs[i].nbytes;
}

bool ane_model_was_cached(const AneModel* m) {
    return m ? m->cache_hit : false;
}

/* ============================================================
 * Request
 * ============================================================ */

struct AneRequest {
    AneModel* model;

    /* Binding state — one entry per input/output. Either bound (external
     * buffer ref) or owned (internal fast-path IOSurface). */
    AneBuffer** input_bound;  /* num_inputs */
    AneBuffer** output_bound; /* num_outputs */
    AneBuffer** input_owned;
    AneBuffer** output_owned;

    /* Resident state bindings — one entry per state slot. A state is
     * always externally bound (a resident cache is meaningless as a
     * per-call fast-path copy), so there is no `state_owned` array. The
     * bound buffer persists and is updated in place across submits. */
    AneBuffer** state_bound; /* num_states */

    /* Cached index arrays (`@[@0, @1, …]`) handed to `_ANERequest`'s
     * inputIndices:/outputIndices: slots. They depend only on the
     * input/output counts — which are fixed for the life of the request
     * — so they are built once on the first submit and reused, sparing
     * the hot path N NSNumber boxings + 2 array allocations per eval. */
    id idx_in_arr;
    id idx_out_arr;

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
    atomic_int in_flight;
    atomic_int has_result;
    AneStatus last_status;
    char last_err_msg[256];

    AneCompletionFn completion_fn;
    void* completion_user;

    int32_t procedure_index;
    AneBuffer* weights_buf;
    AnePerfStats* perf_stats;
    AneSharedEvents* shared_events;
    uint64_t transaction_handle;
};

AneStatus ane_request_create(AneModel* m, AneRequest** out) {
    clear_last_error();
    if (!m || !out) {
        set_last_error("invalid arg");
        return ANE_ERR_INVALID_ARG;
    }
    AneRequest* r = (AneRequest*)calloc(1, sizeof(AneRequest));
    if (!r) {
        set_last_error("oom");
        return ANE_ERR_OOM;
    }
    r->model = m;
    r->input_bound =
        (AneBuffer**)calloc((size_t)(m->num_inputs > 0 ? m->num_inputs : 1), sizeof(AneBuffer*));
    r->output_bound =
        (AneBuffer**)calloc((size_t)(m->num_outputs > 0 ? m->num_outputs : 1), sizeof(AneBuffer*));
    r->input_owned =
        (AneBuffer**)calloc((size_t)(m->num_inputs > 0 ? m->num_inputs : 1), sizeof(AneBuffer*));
    r->output_owned =
        (AneBuffer**)calloc((size_t)(m->num_outputs > 0 ? m->num_outputs : 1), sizeof(AneBuffer*));
    r->state_bound =
        (AneBuffer**)calloc((size_t)(m->num_states > 0 ? m->num_states : 1), sizeof(AneBuffer*));
    if (!r->input_bound || !r->output_bound || !r->input_owned || !r->output_owned ||
        !r->state_bound) {
        ane_request_release(r);
        set_last_error("oom");
        return ANE_ERR_OOM;
    }
    r->queue = dispatch_queue_create("ane.bridge.request", DISPATCH_QUEUE_SERIAL);
    r->done_sem = dispatch_semaphore_create(0);
    atomic_init(&r->in_flight, 0);
    atomic_init(&r->has_result, 1); /* a fresh request has "no submission, but is not pending" */
    *out = r;
    return ANE_OK;
}

void ane_request_release(AneRequest* r) {
    if (!r) {
        return;
    }
    /* Drain any in-flight evaluation so we don't free state under it. */
    if (r->queue) {
        dispatch_sync(r->queue, ^{
                      });
        dispatch_release(r->queue);
    }
    if (r->done_sem) {
        dispatch_release(r->done_sem);
    }
    if (r->input_owned) {
        for (int32_t i = 0; i < r->model->num_inputs; i++) {
            if (r->input_owned[i]) {
                ane_buffer_release(r->input_owned[i]);
            }
        }
        free((void*)r->input_owned);
    }
    if (r->output_owned) {
        for (int32_t i = 0; i < r->model->num_outputs; i++) {
            if (r->output_owned[i]) {
                ane_buffer_release(r->output_owned[i]);
            }
        }
        free((void*)r->output_owned);
    }
    free((void*)r->input_bound);
    free((void*)r->output_bound);
    /* state_bound holds external (unowned) refs, like input_bound. */
    free((void*)r->state_bound);
    if (r->idx_in_arr) {
        [r->idx_in_arr release];
    }
    if (r->idx_out_arr) {
        [r->idx_out_arr release];
    }
    /* last_err_msg is inline, nothing to free. */
    free(r);
}

AneStatus ane_request_bind_input(AneRequest* r, int32_t i, AneBuffer* b) {
    clear_last_error();
    if (!r || !b) {
        set_last_error("invalid arg");
        return ANE_ERR_INVALID_ARG;
    }
    /* An in-flight eval reads the currently bound surfaces via DMA;
     * swapping a binding under it races the payload pages and lets the
     * caller free the old Buffer while the engine still reads it. Same
     * guard as set_input_bytes. */
    if (atomic_load(&r->in_flight) != 0) {
        set_last_error("cannot bind_input while a submission is in flight");
        return ANE_ERR_BUSY;
    }
    if (i < 0 || i >= r->model->num_inputs) {
        set_last_error("input idx out of range");
        return ANE_ERR_INVALID_ARG;
    }
    if (b->nbytes < r->model->input_specs[i].nbytes) {
        set_last_error("input %d: buffer too small (%zu < %zu)", i, b->nbytes,
                       r->model->input_specs[i].nbytes);
        return ANE_ERR_INVALID_ARG;
    }
    r->input_bound[i] = b;
    return ANE_OK;
}

AneStatus ane_request_bind_output(AneRequest* r, int32_t i, AneBuffer* b) {
    clear_last_error();
    if (!r || !b) {
        set_last_error("invalid arg");
        return ANE_ERR_INVALID_ARG;
    }
    /* Same in-flight guard as bind_input: the engine writes the bound
     * output surfaces via DMA until the eval completes. */
    if (atomic_load(&r->in_flight) != 0) {
        set_last_error("cannot bind_output while a submission is in flight");
        return ANE_ERR_BUSY;
    }
    if (i < 0 || i >= r->model->num_outputs) {
        set_last_error("output idx out of range");
        return ANE_ERR_INVALID_ARG;
    }
    if (b->nbytes < r->model->output_specs[i].nbytes) {
        set_last_error("output %d: buffer too small (%zu < %zu)", i, b->nbytes,
                       r->model->output_specs[i].nbytes);
        return ANE_ERR_INVALID_ARG;
    }
    r->output_bound[i] = b;
    return ANE_OK;
}

AneStatus ane_request_bind_state(AneRequest* r, int32_t i, AneBuffer* b) {
    clear_last_error();
    /* The whole surface (struct fields, accessors, request slot) is in
     * place, but the two private-framework unknowns that actually place a
     * state IOSurface on `_ANERequest` are not yet reverse-engineered.
     * Fail loudly rather than silently no-op: a no-op would make the
     * resident cache silently wrong (stale state) instead of unsupported. */
    (void)r;
    (void)i;
    (void)b;
    set_last_error("state binding not yet wired");
    return ANE_ERR_UNSUPPORTED;
}

static AneBuffer* ensure_owned_input(AneRequest* r, int32_t i) {
    if (r->input_owned[i]) {
        return r->input_owned[i];
    }
    AneBuffer* b = NULL;
    if (ane_buffer_create(r->model->input_specs[i].nbytes, &b) != ANE_OK) {
        return NULL;
    }
    r->input_owned[i] = b;
    return b;
}

static AneBuffer* ensure_owned_output(AneRequest* r, int32_t i) {
    if (r->output_owned[i]) {
        return r->output_owned[i];
    }
    AneBuffer* b = NULL;
    if (ane_buffer_create(r->model->output_specs[i].nbytes, &b) != ANE_OK) {
        return NULL;
    }
    r->output_owned[i] = b;
    return b;
}

AneStatus ane_request_set_input_bytes(AneRequest* r, int32_t i, const void* data, size_t nbytes) {
    clear_last_error();
    if (!r || !data) {
        set_last_error("invalid arg");
        return ANE_ERR_INVALID_ARG;
    }
    if (i < 0 || i >= r->model->num_inputs) {
        set_last_error("input idx out of range");
        return ANE_ERR_INVALID_ARG;
    }
    if (nbytes != r->model->input_specs[i].nbytes) {
        set_last_error("input %d: byte count mismatch (%zu != %zu)", i, nbytes,
                       r->model->input_specs[i].nbytes);
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
    if (!b) {
        return ANE_ERR_OOM;
    }
    void* p = NULL;
    AneStatus st = ane_buffer_lock(b, ANE_LOCK_WRITE, &p);
    if (st != ANE_OK) {
        return st;
    }
    memcpy(p, data, nbytes);
    ane_buffer_unlock(b);
    /* Use the owned buffer as the effective binding. */
    r->input_bound[i] = b;
    return ANE_OK;
}

AneStatus ane_request_get_output_bytes(AneRequest* r, int32_t i, void* data, size_t nbytes) {
    clear_last_error();
    if (!r || !data) {
        set_last_error("invalid arg");
        return ANE_ERR_INVALID_ARG;
    }
    if (i < 0 || i >= r->model->num_outputs) {
        set_last_error("output idx out of range");
        return ANE_ERR_INVALID_ARG;
    }
    if (nbytes != r->model->output_specs[i].nbytes) {
        set_last_error("output %d: byte count mismatch (%zu != %zu)", i, nbytes,
                       r->model->output_specs[i].nbytes);
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
    if (st != ANE_OK) {
        return st;
    }
    memcpy(data, p, nbytes);
    ane_buffer_unlock(b);
    return ANE_OK;
}

/* Build a fresh _ANERequest from current bindings; falls back to owned
 * IOSurfaces for any unset output slot so the eval has somewhere to
 * write. (Inputs MUST be supplied by the caller.) */
static id build_ane_request(AneRequest* r, AneStatus* status_out) {
    NSMutableArray* wIn = [NSMutableArray arrayWithCapacity:(NSUInteger)r->model->num_inputs];
    NSMutableArray* wOut = [NSMutableArray arrayWithCapacity:(NSUInteger)r->model->num_outputs];

    for (int32_t i = 0; i < r->model->num_inputs; i++) {
        AneBuffer* b = r->input_bound[i];
        if (!b) {
            set_last_error("input %d not bound", i);
            *status_out = ANE_ERR_INVALID_ARG;
            return nil;
        }
        [wIn addObject:b->wrapper];
    }
    for (int32_t i = 0; i < r->model->num_outputs; i++) {
        AneBuffer* b = r->output_bound[i];
        if (!b) {
            b = ensure_owned_output(r, i);
            if (!b) {
                *status_out = ANE_ERR_OOM;
                return nil;
            }
            r->output_bound[i] = b;
        }
        [wOut addObject:b->wrapper];
    }

    /* Index arrays are `@[@0 … @(n-1)]` — every input is bound and every
     * output now has a buffer, so the indices are always the full dense
     * range. Build the immutable arrays once and reuse across evals. */
    if (!r->idx_in_arr) {
        NSMutableArray* a = [NSMutableArray arrayWithCapacity:(NSUInteger)r->model->num_inputs];
        for (int32_t i = 0; i < r->model->num_inputs; i++) {
            [a addObject:
                    @(i)]; // NOLINT(readability-redundant-parentheses): @() boxing requires parens
        }
        r->idx_in_arr = [a copy];
    }
    if (!r->idx_out_arr) {
        NSMutableArray* a = [NSMutableArray arrayWithCapacity:(NSUInteger)r->model->num_outputs];
        for (int32_t i = 0; i < r->model->num_outputs; i++) {
            [a addObject:
                    @(i)]; // NOLINT(readability-redundant-parentheses): @() boxing requires parens
        }
        r->idx_out_arr = [a copy];
    }
    id iIn = r->idx_in_arr;
    id iOut = r->idx_out_arr;

    id weights_obj = (r->weights_buf && r->weights_buf->wrapper) ? r->weights_buf->wrapper : nil;
    /* The framework's `_ANERequest validate` calls `-count` on the
     * `perfStats:` argument — it expects an NSArray of stats objects
     * (one per pipeline stage), not a bare `_ANEPerformanceStats`.
     * Wrap our single sink in a one-element array. */
    id perf_obj = r->perf_stats ? (id) @[*(id*)r->perf_stats] : nil;
    id events_obj = r->shared_events ? *(id*)r->shared_events : nil;
    NSNumber* proc_num = @(r->procedure_index);

    SEL full_sel = sel_getUid("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:"
                              "perfStats:procedureIndex:sharedEvents:transactionHandle:");
    id req = nil;
    if ([g_AneRequestCls respondsToSelector:full_sel]) {
        req = ((id(*)(Class, SEL, id, id, id, id, id, id, id, id, unsigned long))objc_msgSend)(
            g_AneRequestCls, full_sel, wIn, iIn, wOut, iOut, weights_obj, perf_obj, proc_num,
            events_obj, (unsigned long)r->transaction_handle);
    } else {
        req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            g_AneRequestCls,
            sel_getUid("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:"
                       "perfStats:procedureIndex:"),
            wIn, iIn, wOut, iOut, weights_obj, perf_obj, proc_num);
    }
    if (!req) {
        set_last_error("_ANERequest creation returned nil");
        *status_out = ANE_ERR_INTERNAL;
    }
    return req;
}

static void store_err(AneRequest* r, const char* msg) {
    if (!msg) {
        r->last_err_msg[0] = '\0';
        return;
    }
    /* `strncpy` zero-fills any remaining bytes if `msg` is shorter than
     * the buffer; the explicit terminator below covers the case where
     * `msg` is longer than the buffer minus one. */
    strncpy(r->last_err_msg, msg, sizeof(r->last_err_msg) - 1);
    r->last_err_msg[sizeof(r->last_err_msg) - 1] = '\0';
}

AneStatus ane_request_submit(AneRequest* r, AneQoS qos) {
    clear_last_error();
    if (!r) {
        set_last_error("null request");
        return ANE_ERR_INVALID_ARG;
    }
    /* Shared events attached to a direct-MIL eval abort the framework
     * worker thread on every (eventType, agentMask) combination. The
     * slot lives on `_ANERequest` but is only honored by the chained
     * dispatch path. Refuse here so a caller never hits the abort. */
    if (r->shared_events) {
        set_last_error("shared events on a direct request crash the framework worker; "
                       "the chained-dispatch alternative path is under investigation "
                       "(see ane_chain_prepare for current status)");
        return ANE_ERR_UNSUPPORTED;
    }
    int expected = 0;
    if (!atomic_compare_exchange_strong(&r->in_flight, &expected, 1)) {
        set_last_error("request already in flight");
        return ANE_ERR_BUSY;
    }
    /* Drain any signal left from a previous submission that was never
     * wait()ed. The worker signals `done_sem` once per completion, but
     * only wait() consumes it; submitting again (legal once the prior
     * eval finished) would otherwise leave a stale count that the next
     * timed wait() consumes immediately — returning before THIS eval
     * completes. Resetting to zero here keeps signal/consume balanced
     * 1:1 per submission. */
    while (dispatch_semaphore_wait(r->done_sem, DISPATCH_TIME_NOW) == 0) {
    }
    atomic_store(&r->has_result, 0);

    AneStatus build_status = ANE_OK;
    id req_obj = nil;
    @autoreleasepool {
        req_obj = build_ane_request(r, &build_status);
        if (req_obj) {
            [req_obj retain];
        }
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

    unsigned int qos_v = (qos ? qos : ANE_QOS_DEFAULT);
    AneModel* model = r->model;
    AneCompletionFn cb = r->completion_fn;
    void* cb_user = r->completion_user;

    dispatch_async(r->queue, ^{
        @autoreleasepool {
            NSError* e = nil;
            BOOL ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError**))objc_msgSend)(
                model->client, sel_getUid("doEvaluateDirectWithModel:options:request:qos:error:"),
                model->in_memory_model, @{}, req_obj, qos_v, &e);
            AneStatus s = ok ? ANE_OK : ANE_ERR_EVAL;
            if (!ok) {
                store_err(r, e ? [[e localizedDescription] UTF8String] : "<no NSError>");
            } else {
                store_err(r, NULL);
            }
            r->last_status = s;
            [req_obj release];
            atomic_store(&r->has_result, 1);
            atomic_store(&r->in_flight, 0);
            dispatch_semaphore_signal(r->done_sem);
            if (cb) {
                cb(r, s, cb_user);
            }
        }
    });
    return ANE_OK;
}

AneStatus ane_request_wait(AneRequest* r, int32_t timeout_ms) {
    clear_last_error();
    if (!r) {
        set_last_error("null request");
        return ANE_ERR_INVALID_ARG;
    }
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
    dispatch_time_t t = 0;
    if (timeout_ms < 0) {
        t = DISPATCH_TIME_FOREVER;
    } else if (timeout_ms == 0) {
        t = DISPATCH_TIME_NOW;
    } else {
        t = dispatch_time(DISPATCH_TIME_NOW, (int64_t)timeout_ms * (int64_t)NSEC_PER_MSEC);
    }
    long rc = dispatch_semaphore_wait(r->done_sem, t);
    if (rc != 0) {
        if (timeout_ms == 0) {
            return ANE_ERR_NOT_DONE;
        }
        set_last_error("wait timed out after %d ms", timeout_ms);
        return ANE_ERR_TIMEOUT;
    }
    if (r->last_status != ANE_OK && r->last_err_msg[0] != '\0') {
        set_last_error("%s", r->last_err_msg);
    }
    return r->last_status;
}

/* Atomic reads are logically const, but C's atomic_load wants a non-const
 * (volatile _Atomic) pointer, so the const accessors below must launder the
 * qualifier. Routing through uintptr_t drops it without tripping -Wcast-qual;
 * this never mutates the request. */
static AneRequest* ane_request_deconst(const AneRequest* r) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr): intentional const-laundering for atomic_load on a const accessor
    return (AneRequest*)(uintptr_t)r;
}

bool ane_request_is_done(const AneRequest* r) {
    if (!r) {
        return false;
    }
    return atomic_load((atomic_int*)&ane_request_deconst(r)->has_result) == 1 &&
           atomic_load((atomic_int*)&ane_request_deconst(r)->in_flight) == 0;
}

AneStatus ane_request_run(AneRequest* r, AneQoS qos) {
    clear_last_error();
    if (!r) {
        set_last_error("null request");
        return ANE_ERR_INVALID_ARG;
    }
    if (r->shared_events) {
        /* See ane_request_submit — same crash defense. */
        set_last_error("shared events on a direct request crash the framework worker; "
                       "the chained-dispatch alternative path is under investigation "
                       "(see ane_chain_prepare for current status)");
        return ANE_ERR_UNSUPPORTED;
    }
    int expected = 0;
    if (!atomic_compare_exchange_strong(&r->in_flight, &expected, 1)) {
        set_last_error("request already in flight");
        return ANE_ERR_BUSY;
    }
    /* Clear any stale signal from a prior un-wait()ed submission so a
     * later submit()+wait() stays balanced. run() itself never signals
     * `done_sem` (it is synchronous), so this only ever drains leftovers.
     * See the matching comment in ane_request_submit. */
    while (dispatch_semaphore_wait(r->done_sem, DISPATCH_TIME_NOW) == 0) {
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
    AneStatus status = ANE_OK;
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
        unsigned int qos_v = (qos ? qos : ANE_QOS_DEFAULT);
        NSError* e = nil;
        BOOL ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError**))objc_msgSend)(
            r->model->client, sel_getUid("doEvaluateDirectWithModel:options:request:qos:error:"),
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
        if (cb) {
            cb(r, status, cb_user);
        }
    }
    return status;
}

/* ============================================================
 * Internal fuzz harness for the spec parser. Not part of the
 * stable API; declared in ane_bridge.h purely so Rust tests can
 * link against it. See AneFuzzCase for the input model.
 * ============================================================ */
AneStatus _ane_internal_fuzz_parse_one(const AneFuzzCase* fc) {
    if (!fc) {
        return ANE_ERR_INVALID_ARG;
    }
    @autoreleasepool {
        NSMutableDictionary* entry = [NSMutableDictionary dictionary];
        if (fc->present_mask & ANE_FUZZ_FIELD_NAME) {
            id name_obj = nil;
            if (fc->flags & ANE_FUZZ_FLAG_NAME_AS_NUMBER) {
                name_obj = @42;
            } else if (fc->name) {
                name_obj = [NSString stringWithUTF8String:fc->name];
                if (!name_obj) {
                    name_obj = @"";
                }
            } else {
                name_obj = @"";
            }
            entry[@"Name"] = name_obj;
            entry[@"Symbol"] = name_obj;
        }
        if (fc->present_mask & ANE_FUZZ_FIELD_TYPE) {
            id t = nil;
            if (fc->flags & ANE_FUZZ_FLAG_TYPE_AS_NUMBER) {
                t = @42;
            } else if (fc->type_string) {
                t = [NSString stringWithUTF8String:fc->type_string];
                if (!t) {
                    t = @"";
                }
            } else {
                t = @"Float32";
            }
            entry[@"Type"] = t;
        }
#define PUT_DIM(MASK, KEY, FIELD, FLAG)   \
    if (fc->present_mask & (MASK)) {      \
        if (fc->flags & (FLAG)) {         \
            entry[KEY] = @"not_a_number"; \
        } else {                          \
            entry[KEY] = @(fc->FIELD);    \
        }                                 \
    }
        PUT_DIM(ANE_FUZZ_FIELD_BATCHES, @"Batches", batches, ANE_FUZZ_FLAG_BATCHES_AS_STRING)
        PUT_DIM(ANE_FUZZ_FIELD_CHANNELS, @"Channels", channels, ANE_FUZZ_FLAG_CHANNELS_AS_STRING)
        PUT_DIM(ANE_FUZZ_FIELD_DEPTH, @"Depth", depth, ANE_FUZZ_FLAG_DEPTH_AS_STRING)
        PUT_DIM(ANE_FUZZ_FIELD_HEIGHT, @"Height", height, ANE_FUZZ_FLAG_HEIGHT_AS_STRING)
        PUT_DIM(ANE_FUZZ_FIELD_WIDTH, @"Width", width, ANE_FUZZ_FLAG_WIDTH_AS_STRING)
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
        @"Name": name,
        @"Symbol": name,
        @"Type": @"Float32",
        @"Batches": @1,
        @"Channels": @64,
        @"Depth": @1,
        @"Height": @1,
        @"Width": @16,
    };
}

AneStatus _ane_internal_fuzz_parse_attrs(const AneFuzzAttrsCase* fc) {
    if (!fc) {
        return ANE_ERR_INVALID_ARG;
    }
    @autoreleasepool {
        /* Build a baseline procedure dict, then apply mutations. */
        NSMutableArray* lin = [NSMutableArray array];
        NSMutableArray* lout = [NSMutableArray array];
        int32_t nin = fc->n_inputs < 0 ? 0 : fc->n_inputs;
        int32_t nout = fc->n_outputs < 0 ? 0 : fc->n_outputs;
        for (int32_t i = 0; i < nin; i++) {
            NSString* nm = [NSString stringWithFormat:@"in_%d", i];
            [lin addObject:fuzz_attrs_make_entry(nm)];
        }
        for (int32_t i = 0; i < nout; i++) {
            NSString* nm = [NSString stringWithFormat:@"out_%d", i];
            [lout addObject:fuzz_attrs_make_entry(nm)];
        }

        NSMutableDictionary* proc = [[@{
            @"LiveInputList": lin,
            @"LiveOutputList": lout,
            @"Name": @"main",
        } mutableCopy] autorelease];

        if (fc->mutations & ANE_FUZZ_ATTRS_LIVEIN_MISSING) {
            [proc removeObjectForKey:@"LiveInputList"];
        }
        if (fc->mutations & ANE_FUZZ_ATTRS_LIVEOUT_MISSING) {
            [proc removeObjectForKey:@"LiveOutputList"];
        }
        if (fc->mutations & ANE_FUZZ_ATTRS_LIVEIN_NOT_ARRAY) {
            proc[@"LiveInputList"] = @"not_an_array";
        }
        if (fc->mutations & ANE_FUZZ_ATTRS_LIVEOUT_NOT_ARRAY) {
            proc[@"LiveOutputList"] = @42;
        }

        /* Wrap into a NetworkStatusList. */
        id nsl = nil;
        if (fc->mutations & ANE_FUZZ_ATTRS_NSL_EMPTY) {
            nsl = @[];
        } else if (fc->mutations & ANE_FUZZ_ATTRS_PROC_NOT_DICT) {
            nsl = @[@"not_a_dict"];
        } else {
            nsl = @[proc];
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
        AneModel scratch = (AneModel){0}; // NOLINT(bugprone-invalid-enum-default-initialization):
                                          // compile_qos 0 means "use ANE_QOS_DEFAULT"
        AneStatus s = derive_specs_from_attrs_dict(attrs, &scratch);
        if (s == ANE_OK) {
            free_owned_specs(scratch.input_specs, scratch.num_inputs);
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
    case 0:
        return @"Batches";
    case 1:
        return @"Channels";
    case 2:
        return @"Depth";
    case 3:
        return @"Height";
    case 4:
        return @"Width";
    default:
        return nil;
    }
}

/* Build a minimal well-formed entry, then return it as a mutable
 * dict so callers can swap one value. */
static NSMutableDictionary* fuzz_canonical_entry(void) {
    /* `mutableCopy` returns a +1 owned reference; autorelease it so callers
     * (which run inside an @autoreleasepool) need not release it. */
    return [[@{
        @"Name": @"x",
        @"Symbol": @"x",
        @"Type": @"Float32",
        @"Batches": @1,
        @"Channels": @64,
        @"Depth": @1,
        @"Height": @1,
        @"Width": @16,
    } mutableCopy] autorelease];
}

/* Run the leaf parser on a single entry. Frees any returned specs. */
static AneStatus fuzz_run_leaf(NSDictionary* entry) {
    @autoreleasepool {
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

AneStatus _ane_internal_fuzz_dim_as_double(int32_t which_dim, double dbl_value) {
    @autoreleasepool {
        NSString* key = fuzz_dim_key(which_dim);
        if (!key) {
            return ANE_ERR_INVALID_ARG;
        }
        NSMutableDictionary* entry = fuzz_canonical_entry();
        entry[key] =
            @(dbl_value); // NOLINT(readability-redundant-parentheses): @() boxing requires parens
        return fuzz_run_leaf(entry);
    }
}

AneStatus _ane_internal_fuzz_dim_as_uint64(int32_t which_dim, uint64_t value) {
    @autoreleasepool {
        NSString* key = fuzz_dim_key(which_dim);
        if (!key) {
            return ANE_ERR_INVALID_ARG;
        }
        NSMutableDictionary* entry = fuzz_canonical_entry();
        entry[key] = [NSNumber numberWithUnsignedLongLong:value];
        return fuzz_run_leaf(entry);
    }
}

AneStatus _ane_internal_fuzz_dim_as_decimal(int32_t which_dim, const char* decimal_str) {
    @autoreleasepool {
        NSString* key = fuzz_dim_key(which_dim);
        if (!key || !decimal_str) {
            return ANE_ERR_INVALID_ARG;
        }
        NSMutableDictionary* entry = fuzz_canonical_entry();
        NSString* str = [NSString stringWithUTF8String:decimal_str];
        if (!str) {
            return ANE_ERR_INVALID_ARG;
        }
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
        if (which_key < 0 || which_key >= 7) {
            return ANE_ERR_INVALID_ARG;
        }
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
        NSString* name = [[NSString alloc] initWithBytes:"valid\0continued"
                                                  length:15
                                                encoding:NSUTF8StringEncoding];
        if (!name) {
            return ANE_ERR_INTERNAL;
        }
        NSMutableDictionary* entry = fuzz_canonical_entry();
        entry[@"Name"] = name;
        entry[@"Symbol"] = name;
        AneStatus s = fuzz_run_leaf(entry);
        [name release];
        return s;
    }
}

AneStatus _ane_internal_fuzz_huge_name(size_t length) {
    @autoreleasepool {
        /* Cap absurd sizes — proptest could send length=usize::MAX. */
        if (length > (1U << 24)) {
            return ANE_ERR_INVALID_ARG; /* 16 MB cap */
        }
        char* buf = (char*)malloc(length + 1);
        if (!buf) {
            return ANE_ERR_OOM;
        }
        memset(buf, 'a', length);
        buf[length] = '\0';
        NSString* name = [NSString stringWithUTF8String:buf];
        free(buf);
        if (!name) {
            return ANE_ERR_INTERNAL;
        }
        NSMutableDictionary* entry = fuzz_canonical_entry();
        entry[@"Name"] = name;
        entry[@"Symbol"] = name;
        return fuzz_run_leaf(entry);
    }
}

AneStatus _ane_internal_fuzz_mixed_validity_two_entries(void) {
    @autoreleasepool {
        NSMutableDictionary* good = fuzz_canonical_entry();
        NSMutableDictionary* bad = fuzz_canonical_entry();
        bad[@"Type"] = @"NotARealDtype";
        NSArray* live = @[good, bad];
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
        NSMutableDictionary* entry = [[@{
            @"Name": @"x",
            @"Symbol": @"x",
            @"Type": @"Float32",
            @"Channels": @64,
            @"Depth": @1,
            @"Height": @1,
            @"Width": @16,
        } mutableCopy] autorelease];
        // NOLINTNEXTLINE(readability-redundant-parentheses): @() boxing requires parens (NSNumber wrapping a double)
        entry[@"Batches"] = @(dbl_value);
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

AneStatus ane_request_set_completion(AneRequest* r, AneCompletionFn fn, void* user) {
    clear_last_error();
    if (!r) {
        set_last_error("null request");
        return ANE_ERR_INVALID_ARG;
    }
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
    dispatch_sync(r->queue, ^{
                  });
    r->completion_fn = fn;
    r->completion_user = user;
    return ANE_OK;
}

const char* ane_request_last_error(const AneRequest* r) {
    if (!r) {
        return "";
    }
    /* The worker writes `last_err_msg` BEFORE `atomic_store(has_result,
     * 1)` (seq_cst). If `has_result` is 0 an eval is still in flight
     * and the buffer could be mid-write — return "" rather than risk
     * exposing a partial string. The buffer itself is inline, so even
     * a torn read cannot dereference freed memory. */
    if (atomic_load((atomic_int*)&ane_request_deconst(r)->has_result) == 0) {
        return "";
    }
    return r->last_err_msg;
}

/* ============================================================
 * Extended API surface
 * ============================================================ */

#include <IOSurface/IOSurfaceRef.h>

struct AnePerfStats {
    id obj; /* `_ANEPerformanceStatsIOSurface` — what goes into the perfStats: array. */
    IOSurfaceRef surface; /* backing IOSurface owned by us. */
};
struct AneSharedEvents {
    id obj;
    NSMutableArray* signals;
    NSMutableArray* waits;
};
struct AneChain {
    id obj;
    AneModel* model;
    id req_obj;
    atomic_int in_flight;
    AneStatus last_status;
};

AneStatus ane_device_info(AneDeviceInfo* out) {
    clear_last_error();
    if (!out) {
        return ANE_ERR_INVALID_ARG;
    }
    if (!ane_private_load() || !g_AneDeviceInfoCls) {
        return ANE_ERR_UNSUPPORTED;
    }
    memset(out, 0, sizeof(*out));
    @autoreleasepool {
        Class c = g_AneDeviceInfoCls;
        SEL has = sel_getUid("hasANE");
        SEL ncores = sel_getUid("numANECores");
        SEL nanes = sel_getUid("numANEs");
        SEL board = sel_getUid("aneBoardType");
        SEL arch = sel_getUid("aneArchitectureType");
        SEL pname = sel_getUid("productName");
        if ([c respondsToSelector:has]) {
            out->has_ane = ((BOOL(*)(Class, SEL))objc_msgSend)(c, has) ? true : false;
        }
        if ([c respondsToSelector:ncores]) {
            out->num_cores = (int32_t)((unsigned int (*)(Class, SEL))objc_msgSend)(c, ncores);
        }
        if ([c respondsToSelector:nanes]) {
            out->num_anes = (int32_t)((unsigned int (*)(Class, SEL))objc_msgSend)(c, nanes);
        }
        if ([c respondsToSelector:board]) {
            out->board_type = (int64_t)((long long (*)(Class, SEL))objc_msgSend)(c, board);
        }
        /* `+aneArchitectureType` may return either an NSString or an
         * integer-wrapped NSNumber depending on macOS version. Prefer
         * the string form; fall back to `productName` for a printable
         * fallback. */
        if ([c respondsToSelector:arch]) {
            id v = ((id(*)(Class, SEL))objc_msgSend)(c, arch);
            if ([v isKindOfClass:[NSString class]]) {
                const char* u = [(NSString*)v UTF8String];
                if (u) {
                    strncpy(out->arch_type, u, sizeof(out->arch_type) - 1);
                    out->arch_type[sizeof(out->arch_type) - 1] = '\0';
                }
            } else if ([v isKindOfClass:[NSNumber class]]) {
                snprintf(out->arch_type, sizeof(out->arch_type), "%lld",
                         (long long)[(NSNumber*)v longLongValue]);
            }
        }
        if (out->arch_type[0] == '\0' && [c respondsToSelector:pname]) {
            id v = ((id(*)(Class, SEL))objc_msgSend)(c, pname);
            if ([v isKindOfClass:[NSString class]]) {
                const char* u = [(NSString*)v UTF8String];
                if (u) {
                    strncpy(out->arch_type, u, sizeof(out->arch_type) - 1);
                    out->arch_type[sizeof(out->arch_type) - 1] = '\0';
                }
            }
        }
    }
    return ANE_OK;
}

int32_t ane_model_num_procedures(const AneModel* m) {
    if (!m) {
        return 0;
    }
    return m->num_procedures > 0 ? m->num_procedures : 1;
}

static const ProcSpecs* proc_at(const AneModel* m, int32_t p) {
    if (!m || p < 0) {
        return NULL;
    }
    if (p == 0) {
        static __thread ProcSpecs tls;
        tls.num_inputs = m->num_inputs;
        tls.num_outputs = m->num_outputs;
        tls.num_states = m->num_states;
        tls.input_specs = m->input_specs;
        tls.output_specs = m->output_specs;
        tls.state_specs = m->state_specs;
        tls.input_view = m->input_view;
        tls.output_view = m->output_view;
        tls.state_view = m->state_view;
        return &tls;
    }
    if (p - 1 >= (m->num_procedures > 0 ? m->num_procedures - 1 : 0)) {
        return NULL;
    }
    return &m->extra_procs[p - 1];
}

int32_t ane_model_num_inputs_for_procedure(const AneModel* m, int32_t p) {
    const ProcSpecs* ps = proc_at(m, p);
    return ps ? ps->num_inputs : 0;
}
int32_t ane_model_num_outputs_for_procedure(const AneModel* m, int32_t p) {
    const ProcSpecs* ps = proc_at(m, p);
    return ps ? ps->num_outputs : 0;
}
const AneTensorSpec* ane_model_input_spec_for_procedure(const AneModel* m, int32_t p, int32_t i) {
    const ProcSpecs* ps = proc_at(m, p);
    if (!ps || i < 0 || i >= ps->num_inputs) {
        return NULL;
    }
    return &ps->input_view[i];
}
const AneTensorSpec* ane_model_output_spec_for_procedure(const AneModel* m, int32_t p, int32_t i) {
    const ProcSpecs* ps = proc_at(m, p);
    if (!ps || i < 0 || i >= ps->num_outputs) {
        return NULL;
    }
    return &ps->output_view[i];
}
size_t ane_model_input_nbytes_for_procedure(const AneModel* m, int32_t p, int32_t i) {
    const ProcSpecs* ps = proc_at(m, p);
    if (!ps || i < 0 || i >= ps->num_inputs) {
        return 0;
    }
    return ps->input_specs[i].nbytes;
}
size_t ane_model_output_nbytes_for_procedure(const AneModel* m, int32_t p, int32_t i) {
    const ProcSpecs* ps = proc_at(m, p);
    if (!ps || i < 0 || i >= ps->num_outputs) {
        return 0;
    }
    return ps->output_specs[i].nbytes;
}

int32_t ane_model_num_states_for_procedure(const AneModel* m, int32_t p) {
    const ProcSpecs* ps = proc_at(m, p);
    return ps ? ps->num_states : 0;
}
const AneTensorSpec* ane_model_state_spec_for_procedure(const AneModel* m, int32_t p, int32_t i) {
    const ProcSpecs* ps = proc_at(m, p);
    if (!ps || i < 0 || i >= ps->num_states) {
        return NULL;
    }
    return &ps->state_view[i];
}
size_t ane_model_state_nbytes_for_procedure(const AneModel* m, int32_t p, int32_t i) {
    const ProcSpecs* ps = proc_at(m, p);
    if (!ps || i < 0 || i >= ps->num_states) {
        return 0;
    }
    return ps->state_specs[i].nbytes;
}

int32_t ane_model_queue_depth(const AneModel* m) {
    if (!m || !m->in_memory_model) {
        return 0;
    }
    SEL s = sel_getUid("queueDepth");
    if (![m->in_memory_model respondsToSelector:s]) {
        return 0;
    }
    return (int32_t)((char (*)(id, SEL))objc_msgSend)(m->in_memory_model, s);
}

int64_t ane_model_in_flight(const AneModel* m) {
    if (!m || !m->in_memory_model) {
        return 0;
    }
    SEL pgm = sel_getUid("program");
    if (![m->in_memory_model respondsToSelector:pgm]) {
        return 0;
    }
    id program = ((id(*)(id, SEL))objc_msgSend)(m->in_memory_model, pgm);
    if (!program) {
        return 0;
    }
    SEL inf = sel_getUid("currentAsyncRequestsInFlight");
    if (![program respondsToSelector:inf]) {
        return 0;
    }
    return (int64_t)((long long (*)(id, SEL))objc_msgSend)(program, inf);
}

const char* ane_model_program_id(const AneModel* m) {
    if (!m) {
        return "";
    }
    if (m->hex_id) {
        const char* s = [m->hex_id UTF8String];
        return s ? s : "";
    }
    if (m->in_memory_model) {
        SEL s = sel_getUid("hexStringIdentifier");
        if ([m->in_memory_model respondsToSelector:s]) {
            id h = ((id(*)(id, SEL))objc_msgSend)(m->in_memory_model, s);
            if ([h isKindOfClass:[NSString class]]) {
                const char* u = [(NSString*)h UTF8String];
                return u ? u : "";
            }
        }
    }
    return "";
}

const char* ane_model_weights_hash(const AneModel* m) {
    if (!m || !m->descriptor) {
        return "";
    }
    SEL s = sel_getUid("weightsHash");
    if (![m->descriptor respondsToSelector:s]) {
        return "";
    }
    id h = ((id(*)(id, SEL))objc_msgSend)(m->descriptor, s);
    if ([h isKindOfClass:[NSString class]]) {
        const char* u = [(NSString*)h UTF8String];
        return u ? u : "";
    }
    if ([h isKindOfClass:[NSData class]]) {
        static __thread char buf[129];
        const unsigned char* b = (const unsigned char*)[(NSData*)h bytes];
        size_t n = [(NSData*)h length];
        if (n > 64) {
            n = 64;
        }
        for (size_t i = 0; i < n; i++) {
            snprintf(buf + (i * 2), 3, "%02x", b[i]);
        }
        buf[n * 2] = '\0';
        return buf;
    }
    return "";
}

bool ane_model_program_handle(const AneModel* m, uint64_t* out) {
    if (!m || !out || !m->in_memory_model) {
        return false;
    }
    SEL s = sel_getUid("programHandle");
    if (![m->in_memory_model respondsToSelector:s]) {
        return false;
    }
    *out = (uint64_t)((unsigned long long (*)(id, SEL))objc_msgSend)(m->in_memory_model, s);
    return true;
}

bool ane_model_intermediate_buffer_handle(const AneModel* m, uint64_t* out) {
    if (!m || !out || !m->in_memory_model) {
        return false;
    }
    SEL s = sel_getUid("intermediateBufferHandle");
    if (![m->in_memory_model respondsToSelector:s]) {
        return false;
    }
    *out = (uint64_t)((unsigned long long (*)(id, SEL))objc_msgSend)(m->in_memory_model, s);
    return true;
}

bool ane_cache_exists_for_hash(const char* hex_hash) {
    if (!hex_hash || !ane_private_load() || !g_AneClientCls) {
        return false;
    }
    @autoreleasepool {
        id client = make_ane_client();
        if (!client) {
            return false;
        }
        SEL s = sel_getUid("compiledModelExistsMatchingHash:");
        BOOL r = NO;
        if ([client respondsToSelector:s]) {
            NSString* h = [NSString stringWithUTF8String:hex_hash];
            r = ((BOOL(*)(id, SEL, id))objc_msgSend)(client, s, h);
        }
        [client release];
        return r ? true : false;
    }
}

AneStatus ane_cache_purge_for_hash(const char* hex_hash) {
    clear_last_error();
    if (!hex_hash || !ane_private_load() || !g_AneClientCls) {
        return ANE_ERR_UNSUPPORTED;
    }
    @autoreleasepool {
        id client = make_ane_client();
        if (!client) {
            return ANE_ERR_INTERNAL;
        }
        SEL s = sel_getUid("purgeCompiledModelMatchingHash:");
        if ([client respondsToSelector:s]) {
            NSString* h = [NSString stringWithUTF8String:hex_hash];
            ((void (*)(id, SEL, id))objc_msgSend)(client, s, h);
            [client release];
            return ANE_OK;
        }
        [client release];
        return ANE_ERR_UNSUPPORTED;
    }
}

AneStatus ane_perf_stats_create(AnePerfStats** out) {
    clear_last_error();
    if (!out) {
        return ANE_ERR_INVALID_ARG;
    }
    if (!ane_private_load()) {
        return ANE_ERR_UNSUPPORTED;
    }
    Class iosCls = NSClassFromString(@"_ANEPerformanceStatsIOSurface");
    if (!iosCls) {
        return ANE_ERR_UNSUPPORTED;
    }
    AnePerfStats* ps = (AnePerfStats*)calloc(1, sizeof(AnePerfStats));
    if (!ps) {
        return ANE_ERR_OOM;
    }
    /* The `perfStats:` argument of `_ANERequest` is an NSArray of
     * `_ANEPerformanceStatsIOSurface` objects — the framework writes
     * counter data into the backing IOSurface during eval, and the
     * caller reads it via `[ios stats]` (returns the IOSurface for
     * downstream `+statsWithRequestPerformanceBuffer:statsBufferSize:`
     * parsing). A bare `_ANEPerformanceStats` fails `_ANERequest
     * validate` because it has no `-statType`.
     *
     * `doEvaluateDirectWithModel:` does NOT write to the stats
     * IOSurface — stats are populated only via the daemon-routed
     * chain dispatch which requires the appleneuralengine
     * entitlement. The wiring here is correct for entitled callers;
     * unsigned binaries running direct eval will read zero counters. */
    ps->surface = make_surface(4096);
    if (!ps->surface) {
        free(ps);
        return ANE_ERR_OOM;
    }
    @autoreleasepool {
        SEL s = sel_getUid("objectWithIOSurface:statType:");
        id o = ((id(*)(Class, SEL, IOSurfaceRef, unsigned int))objc_msgSend)(iosCls, s, ps->surface,
                                                                             (unsigned int)0);
        if (!o) {
            CFRelease(ps->surface);
            free(ps);
            return ANE_ERR_INTERNAL;
        }
        ps->obj = [o retain];
    }
    *out = ps;
    return ANE_OK;
}

void ane_perf_stats_release(AnePerfStats* ps) {
    if (!ps) {
        return;
    }
    if (ps->obj) {
        [ps->obj release];
    }
    if (ps->surface) {
        CFRelease(ps->surface);
    }
    free(ps);
}

/* Parse the populated `_ANEPerformanceStats` from raw IOSurface bytes
 * via `+statsWithRequestPerformanceBuffer:statsBufferSize:`. The
 * `_ANEPerformanceStatsIOSurface.-stats` accessor returns the raw
 * IOSurface, not a parsed stats object; parsing is a separate
 * class-method call on `_ANEPerformanceStats`. Returns autoreleased.
 *
 * The framework reads a header at the start of the buffer; if the
 * IOSurface was never populated (direct eval path on a non-entitled
 * binary, or pre-eval), the all-zero header crashes the parser.
 * Skip the call when the leading 16 bytes are all zero. */
static id perf_stats_parse(const AnePerfStats* ps) {
    if (!ps || !ps->surface) {
        return nil;
    }
    Class psCls = NSClassFromString(@"_ANEPerformanceStats");
    if (!psCls) {
        return nil;
    }
    SEL ctor = sel_getUid("statsWithRequestPerformanceBuffer:statsBufferSize:");
    if (![psCls respondsToSelector:ctor]) {
        return nil;
    }
    void* base = IOSurfaceGetBaseAddress(ps->surface);
    if (!base) {
        return nil;
    }
    /* The all-zero-header probe below reads two uint64_t (16 bytes). Every
     * in-tree perf-stats surface is allocated at 4096 bytes, but guard the
     * read so a smaller surface can never make `hdr[1]` run past the mapping. */
    if (IOSurfaceGetAllocSize(ps->surface) < 2 * sizeof(uint64_t)) {
        return nil;
    }
    const uint64_t* hdr = (const uint64_t*)base;
    if (hdr[0] == 0 && hdr[1] == 0) {
        return nil;
    }
    return ((id(*)(Class, SEL, void*, unsigned int))objc_msgSend)(
        psCls, ctor, base, (unsigned int)IOSurfaceGetAllocSize(ps->surface));
}

uint64_t ane_perf_stats_hw_execution_ns(const AnePerfStats* ps) {
    @autoreleasepool {
        id inner = perf_stats_parse(ps);
        if (!inner) {
            return 0;
        }
        SEL s = sel_getUid("hwExecutionTime");
        if (![inner respondsToSelector:s]) {
            return 0;
        }
        return (uint64_t)((unsigned long long (*)(id, SEL))objc_msgSend)(inner, s);
    }
}

size_t ane_perf_stats_counters_nbytes(const AnePerfStats* ps) {
    @autoreleasepool {
        id inner = perf_stats_parse(ps);
        if (!inner) {
            return 0;
        }
        SEL s = sel_getUid("perfCounterData");
        if (![inner respondsToSelector:s]) {
            return 0;
        }
        id d = ((id(*)(id, SEL))objc_msgSend)(inner, s);
        if (![d isKindOfClass:[NSData class]]) {
            return 0;
        }
        return [(NSData*)d length];
    }
}

size_t ane_perf_stats_counters_copy(const AnePerfStats* ps, void* out, size_t cap) {
    if (!out) {
        return 0;
    }
    @autoreleasepool {
        id inner = perf_stats_parse(ps);
        if (!inner) {
            return 0;
        }
        SEL s = sel_getUid("perfCounterData");
        if (![inner respondsToSelector:s]) {
            return 0;
        }
        id d = ((id(*)(id, SEL))objc_msgSend)(inner, s);
        if (![d isKindOfClass:[NSData class]]) {
            return 0;
        }
        NSData* nd = (NSData*)d;
        size_t n = [nd length];
        if (n > cap) {
            n = cap;
        }
        memcpy(out, [nd bytes], n);
        return n;
    }
}

AneStatus ane_model_set_perf_stats_mask(AneModel* m, uint32_t mask) {
    clear_last_error();
    if (!m || !m->in_memory_model) {
        return ANE_ERR_INVALID_ARG;
    }
    atomic_store(&m->perf_stats_mask, mask);
    SEL s = sel_getUid("setPerfStatsMask:");
    if ([m->in_memory_model respondsToSelector:s]) {
        ((void (*)(id, SEL, unsigned int))objc_msgSend)(m->in_memory_model, s, mask);
    }
    return ANE_OK;
}

uint32_t ane_model_get_perf_stats_mask(const AneModel* m) {
    if (!m) {
        return 0;
    }
    if (m->in_memory_model) {
        SEL s = sel_getUid("perfStatsMask");
        if ([m->in_memory_model respondsToSelector:s]) {
            return (uint32_t)((unsigned int (*)(id, SEL))objc_msgSend)(m->in_memory_model, s);
        }
    }
    return atomic_load(&m->perf_stats_mask);
}

/* `_ANESharedEvents` refuses construction from `+new` / empty-array
 * factories — its factory requires at least one real signal or wait
 * event. Defer building the framework object until the first event is
 * added and rebuild on every subsequent add. The cost is N small
 * rebuilds per chain setup; events are typically tiny in count. */
static void rebuild_shared_events_obj(AneSharedEvents* ev) {
    if (!ev) {
        return;
    }
    if ([ev->signals count] == 0 && [ev->waits count] == 0) {
        if (ev->obj) {
            [ev->obj release];
            ev->obj = nil;
        }
        return;
    }
    SEL s = sel_getUid("sharedEventsWithSignalEvents:waitEvents:");
    id o =
        ((id(*)(Class, SEL, id, id))objc_msgSend)(g_AneSharedEventsCls, s, ev->signals, ev->waits);
    if (!o) {
        return;
    }
    if (ev->obj) {
        [ev->obj release];
    }
    ev->obj = [o retain];
}

AneStatus ane_shared_events_create(AneSharedEvents** out) {
    clear_last_error();
    if (!out) {
        return ANE_ERR_INVALID_ARG;
    }
    if (!ane_private_load() || !g_AneSharedEventsCls) {
        return ANE_ERR_UNSUPPORTED;
    }
    AneSharedEvents* ev = (AneSharedEvents*)calloc(1, sizeof(AneSharedEvents));
    if (!ev) {
        return ANE_ERR_OOM;
    }
    ev->signals = [[NSMutableArray alloc] init];
    ev->waits = [[NSMutableArray alloc] init];
    if (!ev->signals || !ev->waits) {
        if (ev->signals) {
            [ev->signals release];
        }
        if (ev->waits) {
            [ev->waits release];
        }
        free(ev);
        return ANE_ERR_OOM;
    }
    *out = ev;
    return ANE_OK;
}

void ane_shared_events_release(AneSharedEvents* ev) {
    if (!ev) {
        return;
    }
    if (ev->obj) {
        [ev->obj release];
    }
    if (ev->signals) {
        [ev->signals release];
    }
    if (ev->waits) {
        [ev->waits release];
    }
    free(ev);
}

AneStatus ane_shared_events_add_signal(AneSharedEvents* ev, uint64_t value, uint32_t symbol_index,
                                       AneEventType event_type, void* mtl_shared_event,
                                       uint64_t agent_mask) {
    clear_last_error();
    if (!ev || !g_AneSharedSignalEventCls) {
        return ANE_ERR_INVALID_ARG;
    }
    @autoreleasepool {
        SEL s = sel_getUid("signalEventWithValue:symbolIndex:eventType:sharedEvent:");
        id obj = ((id(*)(Class, SEL, unsigned long long, unsigned int, long long, id))objc_msgSend)(
            g_AneSharedSignalEventCls, s, (unsigned long long)value, (unsigned int)symbol_index,
            (long long)event_type, (id)mtl_shared_event);
        if (!obj) {
            return ANE_ERR_INTERNAL;
        }
        if (agent_mask) {
            SEL am = sel_getUid("setAgentMask:");
            if ([obj respondsToSelector:am]) {
                ((void (*)(id, SEL, unsigned long long))objc_msgSend)(
                    obj, am, (unsigned long long)agent_mask);
            }
        }
        [ev->signals addObject:obj];
        rebuild_shared_events_obj(ev);
    }
    return ANE_OK;
}

AneStatus ane_shared_events_add_wait(AneSharedEvents* ev, uint64_t value, void* mtl_shared_event,
                                     AneEventType event_type) {
    clear_last_error();
    if (!ev || !g_AneSharedWaitEventCls) {
        return ANE_ERR_INVALID_ARG;
    }
    @autoreleasepool {
        SEL s = sel_getUid("waitEventWithValue:sharedEvent:eventType:");
        id obj = nil;
        if ([g_AneSharedWaitEventCls respondsToSelector:s]) {
            obj = ((id(*)(Class, SEL, unsigned long long, id, unsigned long long))objc_msgSend)(
                g_AneSharedWaitEventCls, s, (unsigned long long)value, (id)mtl_shared_event,
                (unsigned long long)event_type);
        } else {
            SEL s2 = sel_getUid("waitEventWithValue:sharedEvent:");
            obj = ((id(*)(Class, SEL, unsigned long long, id))objc_msgSend)(
                g_AneSharedWaitEventCls, s2, (unsigned long long)value, (id)mtl_shared_event);
        }
        if (!obj) {
            return ANE_ERR_INTERNAL;
        }
        [ev->waits addObject:obj];
        rebuild_shared_events_obj(ev);
    }
    return ANE_OK;
}

int32_t ane_shared_events_num_signals(const AneSharedEvents* ev) {
    return (ev && ev->signals) ? (int32_t)[ev->signals count] : 0;
}
int32_t ane_shared_events_num_waits(const AneSharedEvents* ev) {
    return (ev && ev->waits) ? (int32_t)[ev->waits count] : 0;
}

/* The setters below mutate `AneRequest` fields that `build_ane_request`
 * reads on the submit/run path. Rejecting writes while in_flight is what
 * stops a C-side caller from racing with the worker thread (the Rust
 * wrapper takes `&mut self`, so this is purely defense for C callers). */
#define ANE_REQUEST_SETTER_GUARD(req)            \
    do {                                         \
        clear_last_error();                      \
        if (!(req))                              \
            return ANE_ERR_INVALID_ARG;          \
        if (atomic_load(&(req)->in_flight) != 0) \
            return ANE_ERR_BUSY;                 \
    } while (0)

AneStatus ane_request_set_weights(AneRequest* req, AneBuffer* weights) {
    ANE_REQUEST_SETTER_GUARD(req);
    req->weights_buf = weights;
    return ANE_OK;
}
AneStatus ane_request_set_procedure_index(AneRequest* req, int32_t proc_idx) {
    if (proc_idx < 0) {
        return ANE_ERR_INVALID_ARG;
    }
    ANE_REQUEST_SETTER_GUARD(req);
    req->procedure_index = proc_idx;
    return ANE_OK;
}
AneStatus ane_request_set_perf_stats(AneRequest* req, AnePerfStats* ps) {
    ANE_REQUEST_SETTER_GUARD(req);
    req->perf_stats = ps;
    return ANE_OK;
}
AneStatus ane_request_set_shared_events(AneRequest* req, AneSharedEvents* ev) {
    ANE_REQUEST_SETTER_GUARD(req);
    req->shared_events = ev;
    return ANE_OK;
}
AneStatus ane_request_set_transaction(AneRequest* req, uint64_t handle) {
    ANE_REQUEST_SETTER_GUARD(req);
    req->transaction_handle = handle;
    return ANE_OK;
}
int32_t ane_request_procedure_index(const AneRequest* req) {
    return req ? req->procedure_index : 0;
}
uint64_t ane_request_transaction(const AneRequest* req) {
    return req ? req->transaction_handle : 0;
}

void* ane_buffer_iosurface_ref(const AneBuffer* buf) {
    return buf ? (void*)buf->surface : NULL;
}

AneStatus ane_buffer_adopt_iosurface(void* iosurface_ref, size_t nbytes, AneBuffer** out) {
    clear_last_error();
    if (!iosurface_ref || !out) {
        return ANE_ERR_INVALID_ARG;
    }
    if (!ane_private_load() || !g_AneIOSurfaceCls) {
        return ANE_ERR_UNSUPPORTED;
    }
    AneBuffer* b = (AneBuffer*)calloc(1, sizeof(AneBuffer));
    if (!b) {
        return ANE_ERR_OOM;
    }
    IOSurfaceRef s = (IOSurfaceRef)iosurface_ref;
    CFRetain(s);
    b->surface = s;
    b->nbytes = nbytes;
    @autoreleasepool {
        id wrap = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            g_AneIOSurfaceCls, sel_getUid("objectWithIOSurface:"), s);
        if (!wrap) {
            CFRelease(s);
            free(b);
            return ANE_ERR_INTERNAL;
        }
        b->wrapper = [wrap retain];
    }
    *out = b;
    return ANE_OK;
}

static NSDictionary* build_weights_dict(const AneWeightEntry* w, int32_t n,
                                        NSMutableArray* keepalive) {
    NSMutableDictionary* d = [NSMutableDictionary dictionary];
    for (int32_t i = 0; i < n; i++) {
        if (!w[i].name) {
            continue;
        }
        NSData* blob = nil;
        if (w[i].path) {
            NSString* wpath = [NSString stringWithUTF8String:w[i].path];
            if (wpath) {
                blob = [NSData dataWithContentsOfFile:wpath];
            }
        } else if (w[i].bytes && w[i].nbytes > 0) {
            blob = [NSData dataWithBytes:w[i].bytes length:w[i].nbytes];
        }
        if (!blob) {
            return nil;
        }
        [keepalive addObject:blob];
        NSString* key = [NSString stringWithUTF8String:w[i].name];
        if (!key) {
            return nil;
        }
        d[key] = @{@"offset": @0, @"data": blob};
    }
    return d;
}

AneStatus ane_model_open_ex(const AneModelOpenOptionsEx* opts, AneModel** out_model) {
    clear_last_error();
    if (!opts || !out_model) {
        return ANE_ERR_INVALID_ARG;
    }
    if (!opts->mil_path && (!opts->mil_bytes || opts->mil_nbytes == 0)) {
        return ANE_ERR_INVALID_ARG;
    }
    if (!ane_private_load()) {
        return ANE_ERR_UNSUPPORTED;
    }

    @autoreleasepool {
        NSData* mil = nil;
        if (opts->mil_path) {
            NSString* milPath = [NSString stringWithUTF8String:opts->mil_path];
            if (!milPath) {
                return ANE_ERR_INVALID_ARG;
            }
            mil = [NSData dataWithContentsOfFile:milPath];
        } else {
            mil = [NSData dataWithBytes:opts->mil_bytes length:opts->mil_nbytes];
        }
        if (!mil) {
            return ANE_ERR_IO;
        }

        AneModel* m = (AneModel*)calloc(1, sizeof(AneModel));
        if (!m) {
            return ANE_ERR_OOM;
        }
        m->mil_data = [mil retain];
        m->compile_qos = opts->compile_qos ? opts->compile_qos : ANE_QOS_DEFAULT;
        m->extra_weights_data = [[NSMutableArray alloc] init];

        NSDictionary* wdict = nil;
        if (opts->n_weights > 0 && opts->weights) {
            wdict = build_weights_dict(opts->weights, opts->n_weights, m->extra_weights_data);
            if (!wdict) {
                ane_model_close(m);
                return ANE_ERR_IO;
            }
        } else {
            wdict = @{};
        }
        if (opts->n_weights > 0 && wdict.count > 0) {
            id first_key = wdict.allKeys.firstObject;
            NSDictionary* first = first_key ? wdict[first_key] : nil;
            id data = first ? first[@"data"] : nil;
            if ([data isKindOfClass:[NSData class]]) {
                m->weights_data = [data retain];
            }
        }

        id descriptor = nil;
        SEL ctor = sel_getUid("modelWithMILText:weights:optionsPlist:");
        descriptor = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(g_AneDescriptorCls, ctor, mil,
                                                                   wdict, nil);
        if (!descriptor && !opts->is_mil_model) {
            SEL alt = sel_getUid("modelWithNetworkDescription:weights:optionsPlist:");
            if ([g_AneDescriptorCls respondsToSelector:alt]) {
                descriptor = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(g_AneDescriptorCls, alt,
                                                                           mil, wdict, nil);
            }
        }
        if (!descriptor) {
            ane_model_close(m);
            return ANE_ERR_COMPILE;
        }
        m->descriptor = [descriptor retain];

        m->in_memory_model = ((id(*)(Class, SEL, id))objc_msgSend)(
            g_AneInMemoryCls, sel_getUid("inMemoryModelWithDescriptor:"), m->descriptor);
        if (!m->in_memory_model) {
            ane_model_close(m);
            return ANE_ERR_COMPILE;
        }
        [m->in_memory_model retain];

        NSError* e = nil;
        BOOL cache_hit = NO;
        SEL exists_sel = sel_getUid("compiledModelExists");
        if ([m->in_memory_model respondsToSelector:exists_sel]) {
            cache_hit = ((BOOL(*)(id, SEL))objc_msgSend)(m->in_memory_model, exists_sel);
        }
        m->cache_hit = cache_hit ? true : false;

        if (!cache_hit) {
            id hxid = ((id(*)(id, SEL))objc_msgSend)(m->in_memory_model,
                                                     sel_getUid("hexStringIdentifier"));
            m->hex_id = [hxid retain];
            m->temp_dir =
                [[NSTemporaryDirectory() stringByAppendingPathComponent:m->hex_id] retain];
            NSFileManager* fm = [NSFileManager defaultManager];
            [fm createDirectoryAtPath:[m->temp_dir stringByAppendingPathComponent:@"weights"]
                withIntermediateDirectories:YES
                                 attributes:nil
                                      error:nil];
            [mil writeToFile:[m->temp_dir stringByAppendingPathComponent:@"model.mil"]
                  atomically:YES];
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables): fast-enumeration loop variable is bound by the `in` clause
            for (NSString* key in [wdict allKeys]) {
                NSDictionary* sub = wdict[key];
                NSData* data = sub[@"data"];
                NSString* leaf = [[key componentsSeparatedByString:@"/"] lastObject];
                if (!leaf || ![data isKindOfClass:[NSData class]]) {
                    continue;
                }
                NSString* path = [m->temp_dir
                    stringByAppendingPathComponent:[@"weights"
                                                       stringByAppendingPathComponent:leaf]];
                [data writeToFile:path atomically:YES];
            }
            BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
                m->in_memory_model, sel_getUid("compileWithQoS:options:error:"),
                (unsigned int)m->compile_qos, @{}, &e);
            if (!ok) {
                ane_model_close(m);
                return ANE_ERR_COMPILE;
            }
        }

        BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
            m->in_memory_model, sel_getUid("loadWithQoS:options:error:"),
            (unsigned int)m->compile_qos, @{}, &e);
        if (!ok) {
            ane_model_close(m);
            return ANE_ERR_LOAD;
        }
        atomic_store(&m->loaded, true);

        AneStatus dst = derive_specs_from_attrs(m->in_memory_model, m);
        if (dst != ANE_OK) {
            ane_model_close(m);
            return dst;
        }

        m->client = make_ane_client();
        if (!m->client) {
            ane_model_close(m);
            return ANE_ERR_INTERNAL;
        }

        *out_model = m;
        return ANE_OK;
    }
}

AneStatus ane_model_open_file(const AneModelFileOpenOptions* opts, AneModel** out_model) {
    clear_last_error();
    if (!opts || !out_model || !opts->model_url) {
        return ANE_ERR_INVALID_ARG;
    }
    if (!ane_private_load() || !g_AneModelCls) {
        return ANE_ERR_UNSUPPORTED;
    }

    @autoreleasepool {
        NSString* p = [NSString stringWithUTF8String:opts->model_url];
        NSURL* url = [p hasPrefix:@"file:"] ? [NSURL URLWithString:p] : [NSURL fileURLWithPath:p];
        NSString* key =
            opts->cache_key ? [NSString stringWithUTF8String:opts->cache_key] : @"default";
        id mdl = nil;
        if (opts->cache_url_identifier) {
            NSString* ci = [NSString stringWithUTF8String:opts->cache_url_identifier];
            SEL s = sel_getUid("modelAtURLWithCacheURLIdentifier:key:cacheURLIdentifier:");
            if ([g_AneModelCls respondsToSelector:s]) {
                mdl = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(g_AneModelCls, s, url, key, ci);
            }
        }
        if (!mdl) {
            mdl = ((id(*)(Class, SEL, id, id))objc_msgSend)(
                g_AneModelCls, sel_getUid("modelAtURL:key:"), url, key);
        }
        if (!mdl) {
            return ANE_ERR_LOAD;
        }

        AneModel* m = (AneModel*)calloc(1, sizeof(AneModel));
        if (!m) {
            return ANE_ERR_OOM;
        }
        m->in_memory_model = [mdl retain];
        m->compile_qos = opts->compile_qos ? opts->compile_qos : ANE_QOS_DEFAULT;

        NSError* e = nil;
        id client = make_ane_client();
        if (!client) {
            ane_model_close(m);
            return ANE_ERR_INTERNAL;
        }
        BOOL ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError**))objc_msgSend)(
            client, sel_getUid("loadModel:options:qos:error:"), m->in_memory_model, @{},
            (unsigned int)m->compile_qos, &e);
        if (!ok) {
            set_last_error("open_file loadModel: %s",
                           e ? [[e localizedDescription] UTF8String] : "<no error>");
            [client release];
            ane_model_close(m);
            return ANE_ERR_LOAD;
        }
        atomic_store(&m->loaded, true);
        m->client = client;

        AneStatus dst = derive_specs_from_attrs(m->in_memory_model, m);
        if (dst != ANE_OK) {
            ane_model_close(m);
            return dst;
        }

        *out_model = m;
        return ANE_OK;
    }
}

/* The real-time scheduling class is entered per-thread via
 * `ane_realtime_task_begin`/`_end`, not at open time, so these are
 * currently thin pass-throughs kept as distinct entry points for the
 * QoS wiring that will eventually differ. */
AneStatus ane_model_open_realtime(const AneModelOpenOptions* opts, AneModel** out_model) {
    return ane_model_open(opts, out_model);
}
AneStatus ane_model_open_realtime_ex(const AneModelOpenOptionsEx* opts, AneModel** out_model) {
    return ane_model_open_ex(opts, out_model);
}

/* `_ANEVirtualClient` is unreachable from unsigned binaries
 * (`+sharedConnection` returns nil). `_ANEClient.beginRealTimeTask` /
 * `endRealTimeTask` are exposed but call into aned over XPC and
 * require the same entitlement chain-dispatch does. Route through
 * `_ANEClient` so the call reaches the framework; on failure surface
 * the entitlement message clearly. */
static AneStatus realtime_task(const char* sel_name) {
    clear_last_error();
    if (!ane_private_load() || !g_AneClientCls) {
        return ANE_ERR_UNSUPPORTED;
    }
    @autoreleasepool {
        id client = make_ane_client();
        if (!client) {
            return ANE_ERR_INTERNAL;
        }
        SEL s = sel_getUid(sel_name);
        AneStatus rc = ANE_OK;
        if (![client respondsToSelector:s]) {
            rc = ANE_ERR_UNSUPPORTED;
        } else {
            BOOL ok = ((BOOL(*)(id, SEL))objc_msgSend)(client, s);
            if (!ok) {
                set_last_error("%s returned NO; requires the appleneuralengine "
                               "realtime entitlement that unsigned binaries lack",
                               sel_name);
            }
            rc = ok ? ANE_OK : ANE_ERR_UNSUPPORTED;
        }
        [client release];
        return rc;
    }
}
AneStatus ane_realtime_task_begin(void) {
    return realtime_task("beginRealTimeTask");
}
AneStatus ane_realtime_task_end(void) {
    return realtime_task("endRealTimeTask");
}

AneStatus ane_chain_create(const AneChainStep* steps, int32_t n_steps, AneChain** out) {
    clear_last_error();
    if (!steps || n_steps <= 0 || !out) {
        return ANE_ERR_INVALID_ARG;
    }
    if (!ane_private_load() || !g_AneChainingRequestCls) {
        return ANE_ERR_UNSUPPORTED;
    }
    /* Every step must share the same model — the chain dispatch goes
     * through a single `_ANEClient.doEnqueueSetsWithModel:`, so mixing
     * models would silently drop all but the first model's stages. */
    if (!steps[0].request) {
        return ANE_ERR_INVALID_ARG;
    }
    AneModel* anchor = steps[0].request->model;
    for (int32_t i = 1; i < n_steps; i++) {
        if (!steps[i].request) {
            return ANE_ERR_INVALID_ARG;
        }
        if (steps[i].request->model != anchor) {
            return ANE_ERR_INVALID_ARG;
        }
    }
    AneChain* c = (AneChain*)calloc(1, sizeof(AneChain));
    if (!c) {
        return ANE_ERR_OOM;
    }
    c->model = anchor;
    atomic_init(&c->in_flight, 0);

    @autoreleasepool {
        NSMutableArray* arr = [NSMutableArray array];
        Class outSetsCls = NSClassFromString(@"_ANEIOSurfaceOutputSets");
        for (int32_t i = 0; i < n_steps; i++) {
            AneRequest* rq = steps[i].request;

            /* `inputs:` — array of "input sets". Each set is itself an
             * NSArray of `_ANEIOSurfaceObject` wrappers. Both the set
             * and its members carry a -symbolIndex (stub installed at
             * load time). */
            NSMutableArray* inSets = [NSMutableArray array];
            for (int32_t k = 0; k < rq->model->num_inputs; k++) {
                AneBuffer* b = rq->input_bound[k];
                if (!b) {
                    continue;
                }
                objc_setAssociatedObject(b->wrapper, &g_ane_symbol_index_key, @((unsigned int)k),
                                         OBJC_ASSOCIATION_RETAIN);
                NSArray* set = @[b->wrapper];
                objc_setAssociatedObject(set, &g_ane_symbol_index_key, @((unsigned int)k),
                                         OBJC_ASSOCIATION_RETAIN);
                [inSets addObject:set];
            }

            /* `outputSets:` — array of `_ANEIOSurfaceOutputSets`. Each
             * wraps an NSArray of `_ANEIOSurfaceObject` as its
             * `outputBuffer`; `statsSurRef` may reuse the same
             * IOSurface wrapper (per probing). */
            NSMutableArray* outSets = [NSMutableArray array];
            for (int32_t k = 0; k < rq->model->num_outputs; k++) {
                AneBuffer* b = rq->output_bound[k];
                if (!b) {
                    continue;
                }
                objc_setAssociatedObject(b->wrapper, &g_ane_symbol_index_key, @((unsigned int)k),
                                         OBJC_ASSOCIATION_RETAIN);
                NSArray* inner = @[b->wrapper];
                SEL ctorO = sel_getUid("objectWithstatsSurRef:outputBuffer:");
                id setWrap =
                    ((id(*)(Class, SEL, id, id))objc_msgSend)(outSetsCls, ctorO, b->wrapper, inner);
                if (!setWrap) {
                    free(c);
                    return ANE_ERR_INTERNAL;
                }
                objc_setAssociatedObject(setWrap, &g_ane_symbol_index_key, @((unsigned int)k),
                                         OBJC_ASSOCIATION_RETAIN);
                [outSets addObject:setWrap];
            }

            /* `signalEvents:` takes an NSArray of `_ANESharedSignalEvent`
             * (not the `_ANESharedEvents` container). Pass empty array
             * when no events are configured. */
            NSArray* signals = (rq->shared_events && [rq->shared_events->signals count] > 0)
                                   ? (NSArray*)rq->shared_events->signals
                                   : @[];

            SEL s = sel_getUid(
                "chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:"
                "procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:");
            id step = ((id(*)(Class, SEL, id, id, long long, long long, id, id, unsigned long long,
                              unsigned long long, unsigned long long))objc_msgSend)(
                g_AneChainingRequestCls, s, inSets, outSets, (long long)steps[i].lb_input_symbol_id,
                (long long)steps[i].lb_output_symbol_id, @(rq->procedure_index), signals,
                (unsigned long long)rq->transaction_handle,
                (unsigned long long)steps[i].fw_enqueue_delay,
                (unsigned long long)steps[i].memory_pool_id);
            if (!step) {
                free(c);
                return ANE_ERR_INTERNAL;
            }
            [arr addObject:step];
        }
        c->req_obj = [arr retain];
    }
    *out = c;
    return ANE_OK;
}

AneStatus ane_chain_prepare(AneChain* chain, AneQoS qos) {
    clear_last_error();
    if (!chain || !chain->model || !chain->model->client) {
        return ANE_ERR_INVALID_ARG;
    }
    /* Chain dispatch always crosses the XPC boundary to `aned`. The
     * model must be NSSecureCoding-conformant (`_ANEModel`, not
     * `_ANEInMemoryModel`). The bridge populates `xpc_model` at open
     * time via `+modelWithCacheURLIdentifier:` when possible.
     *
     * NOTE on entitlements: even when both the model object and the
     * request structure are well-formed, `aned` requires a sandbox
     * extension for the model's underlying bundle path. Standard
     * user binaries without the requisite Apple entitlements receive
     * a silent `loadModel`/`prepareChaining` failure. The chain API
     * is exposed for clients that run in an appropriately entitled
     * process. */
    id model_for_xpc =
        chain->model->xpc_model ? chain->model->xpc_model : chain->model->in_memory_model;
    if (![model_for_xpc conformsToProtocol:@protocol(NSSecureCoding)]) {
        set_last_error("ane_chain_prepare: the loaded model is not NSSecureCoding-"
                       "conformant and the cache-derived _ANEModel handle could not "
                       "be obtained.");
        return ANE_ERR_UNSUPPORTED;
    }
    @autoreleasepool {
        NSError* e = nil;
        id first = [(NSArray*)chain->req_obj firstObject];
        if (!first) {
            return ANE_ERR_INVALID_ARG;
        }
        SEL s = sel_getUid("doPrepareChainingWithModel:options:chainingReq:qos:error:");
        if (![chain->model->client respondsToSelector:s]) {
            return ANE_ERR_UNSUPPORTED;
        }
        BOOL ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError**))objc_msgSend)(
            chain->model->client, s, model_for_xpc, @{}, first, (qos ? qos : ANE_QOS_DEFAULT), &e);
        if (!ok) {
            if (e) {
                set_last_error("chain prepare: %s", [[e localizedDescription] UTF8String]);
            }
            return ANE_ERR_EVAL;
        }
        return ANE_OK;
    }
}

AneStatus ane_chain_enqueue(AneChain* chain, AneQoS qos) {
    clear_last_error();
    if (!chain || !chain->model || !chain->model->client) {
        return ANE_ERR_INVALID_ARG;
    }
    int expected = 0;
    if (!atomic_compare_exchange_strong(&chain->in_flight, &expected, 1)) {
        return ANE_ERR_BUSY;
    }
    @autoreleasepool {
        NSError* e = nil;
        SEL s = sel_getUid("doEnqueueSetsWithModel:outputSet:options:qos:error:");
        if (![chain->model->client respondsToSelector:s]) {
            atomic_store(&chain->in_flight, 0);
            return ANE_ERR_UNSUPPORTED;
        }
        id model_for_xpc =
            chain->model->xpc_model ? chain->model->xpc_model : chain->model->in_memory_model;
        BOOL ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError**))objc_msgSend)(
            chain->model->client, s, model_for_xpc, chain->req_obj, @{},
            (qos ? qos : ANE_QOS_DEFAULT), &e);
        chain->last_status = ok ? ANE_OK : ANE_ERR_EVAL;
        atomic_store(&chain->in_flight, 0);
        return chain->last_status;
    }
}

/* `doEnqueueSetsWithModel:` is synchronous, so enqueue already blocks
 * until the chain completes. `wait` exists for API symmetry with
 * `ane_request_wait` and just returns the last enqueue's status.
 * `timeout_ms` is ignored. */
AneStatus ane_chain_wait(AneChain* chain, int32_t timeout_ms) {
    (void)timeout_ms;
    if (!chain) {
        return ANE_ERR_INVALID_ARG;
    }
    return chain->last_status;
}

void ane_chain_release(AneChain* chain) {
    if (!chain) {
        return;
    }
    if (chain->req_obj) {
        [chain->req_obj release];
    }
    free(chain);
}

struct AneSessionHint {
    AneSessionHintKind kind;
};

static const char* ns_string_or_empty(NSString* s) {
    if (![s isKindOfClass:[NSString class]]) {
        return "";
    }
    const char* u = [s UTF8String];
    return u ? u : "";
}

/* `_ANEInMemoryModel` doesn't expose UUID / sourceURL / modelURL /
 * key / cacheURLIdentifier / identifierSource — those live on
 * `_ANEModel`. When the bridge has a cache-derived `_ANEModel`
 * (populated by `+modelWithCacheURLIdentifier:` at open time),
 * route accessors there; otherwise return the empty/zero default. */
static id model_with_metadata(const AneModel* m) {
    if (!m) {
        return nil;
    }
    if (m->xpc_model) {
        return m->xpc_model;
    }
    return m->in_memory_model;
}

const char* ane_model_uuid(const AneModel* m) {
    id mdl = model_with_metadata(m);
    if (!mdl) {
        return "";
    }
    SEL s = sel_getUid("UUID");
    if (![mdl respondsToSelector:s]) {
        return "";
    }
    id u = ((id(*)(id, SEL))objc_msgSend)(mdl, s);
    if ([u isKindOfClass:[NSUUID class]]) {
        return ns_string_or_empty([(NSUUID*)u UUIDString]);
    }
    return ns_string_or_empty(u);
}

const char* ane_model_source_url(const AneModel* m) {
    id mdl = model_with_metadata(m);
    if (!mdl) {
        return "";
    }
    SEL s = sel_getUid("sourceURL");
    if (![mdl respondsToSelector:s]) {
        return "";
    }
    id u = ((id(*)(id, SEL))objc_msgSend)(mdl, s);
    if ([u isKindOfClass:[NSURL class]]) {
        return ns_string_or_empty([(NSURL*)u absoluteString]);
    }
    return ns_string_or_empty(u);
}

const char* ane_model_model_url(const AneModel* m) {
    id mdl = model_with_metadata(m);
    if (!mdl) {
        return "";
    }
    SEL s = sel_getUid("modelURL");
    if (![mdl respondsToSelector:s]) {
        return "";
    }
    id u = ((id(*)(id, SEL))objc_msgSend)(mdl, s);
    if ([u isKindOfClass:[NSURL class]]) {
        return ns_string_or_empty([(NSURL*)u absoluteString]);
    }
    return ns_string_or_empty(u);
}

const char* ane_model_key(const AneModel* m) {
    id mdl = model_with_metadata(m);
    if (!mdl) {
        return "";
    }
    SEL s = sel_getUid("key");
    if (![mdl respondsToSelector:s]) {
        return "";
    }
    return ns_string_or_empty(((id(*)(id, SEL))objc_msgSend)(mdl, s));
}

const char* ane_model_cache_url_identifier(const AneModel* m) {
    id mdl = model_with_metadata(m);
    if (!mdl) {
        return "";
    }
    SEL s = sel_getUid("cacheURLIdentifier");
    if (![mdl respondsToSelector:s]) {
        return "";
    }
    return ns_string_or_empty(((id(*)(id, SEL))objc_msgSend)(mdl, s));
}

int64_t ane_model_identifier_source(const AneModel* m) {
    id mdl = model_with_metadata(m);
    if (!mdl) {
        return 0;
    }
    SEL s = sel_getUid("identifierSource");
    if (![mdl respondsToSelector:s]) {
        return 0;
    }
    return (int64_t)((long long (*)(id, SEL))objc_msgSend)(mdl, s);
}

AneStatus ane_model_reset_on_unload(AneModel* m) {
    clear_last_error();
    if (!m || !m->in_memory_model) {
        return ANE_ERR_INVALID_ARG;
    }
    SEL s = sel_getUid("resetOnUnload");
    if (![m->in_memory_model respondsToSelector:s]) {
        return ANE_ERR_UNSUPPORTED;
    }
    ((void (*)(id, SEL))objc_msgSend)(m->in_memory_model, s);
    return ANE_OK;
}

AneStatus ane_model_unload(AneModel* m) {
    clear_last_error();
    if (!m || !m->in_memory_model || !atomic_load(&m->loaded)) {
        return ANE_ERR_INVALID_ARG;
    }
    @autoreleasepool {
        NSError* e = nil;
        BOOL ok = ((BOOL(*)(id, SEL, unsigned int, NSError**))objc_msgSend)(
            m->in_memory_model, sel_getUid("unloadWithQoS:error:"),
            (m->compile_qos ? m->compile_qos : ANE_QOS_DEFAULT), &e);
        if (!ok) {
            return ANE_ERR_INTERNAL;
        }
        atomic_store(&m->loaded, false);
        return ANE_OK;
    }
}

AneStatus ane_model_new_instance(AneModel* src, const AneModelInstanceParams* params, AneQoS qos,
                                 AneModel** out) {
    clear_last_error();
    (void)params;
    if (!src || !src->client || !src->in_memory_model || !out) {
        return ANE_ERR_INVALID_ARG;
    }
    Class mipCls = NSClassFromString(@"_ANEModelInstanceParameters");
    Class pdCls = NSClassFromString(@"_ANEProcedureData");
    if (!mipCls || !pdCls) {
        return ANE_ERR_UNSUPPORTED;
    }
    /* `loadModelNewInstance:` goes through XPC to aned. The model
     * argument must be NSSecureCoding-conformant; `_ANEInMemoryModel`
     * isn't. Refuse cleanly. The full-fidelity path requires an
     * NSSecureCoding `_ANEModel` whose underlying bundle can be
     * resolved by the daemon — entitlements that standard user
     * binaries don't carry. */
    id model_for_xpc = src->xpc_model ? src->xpc_model : src->in_memory_model;
    if (![model_for_xpc conformsToProtocol:@protocol(NSSecureCoding)]) {
        set_last_error("ane_model_new_instance: requires an NSSecureCoding-conformant "
                       "model (file-opened _ANEModel with daemon-trusted bundle path); "
                       "the in-memory descriptor path cannot cross the XPC boundary.");
        return ANE_ERR_UNSUPPORTED;
    }
    @autoreleasepool {
        NSError* e = nil;
        SEL s = sel_getUid("loadModelNewInstance:options:modelInstParams:qos:error:");
        if (![src->client respondsToSelector:s]) {
            return ANE_ERR_UNSUPPORTED;
        }

        /* The `modelInstParams:` slot expects an
         * `_ANEModelInstanceParameters` whose `procedureArray` is an
         * NSArray of `_ANEProcedureData` (one per procedure of the
         * source model). The framework reads `procedureArray` on the
         * params during load. For a plain "clone" instance with no
         * per-procedure overrides, build one empty entry per
         * procedure. */
        int32_t nproc = src->num_procedures > 0 ? src->num_procedures : 1;
        NSMutableArray* parr = [NSMutableArray arrayWithCapacity:(NSUInteger)nproc];
        for (int32_t i = 0; i < nproc; i++) {
            NSNumber* idx_obj =
                @(i); // NOLINT(readability-redundant-parentheses): @() boxing requires parens
            id pd = ((id(*)(Class, SEL, id, id))objc_msgSend)(
                pdCls, sel_getUid("procedureDataWithSymbol:weightArray:"), idx_obj, @[]);
            if (pd) {
                [parr addObject:pd];
            }
        }
        id inst_params = ((id(*)(Class, SEL, id, id))objc_msgSend)(
            mipCls, sel_getUid("withProcedureData:procedureArray:"), (id)nil, parr);
        if (!inst_params) {
            set_last_error(
                "ane_model_new_instance: _ANEModelInstanceParameters factory returned nil");
            return ANE_ERR_INTERNAL;
        }

        BOOL ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError**))objc_msgSend)(
            src->client, s, model_for_xpc, @{}, inst_params, (qos ? qos : ANE_QOS_DEFAULT), &e);
        if (!ok) {
            if (e) {
                set_last_error("new_instance loadModelNewInstance: %s",
                               [[e localizedDescription] UTF8String]);
            }
            return ANE_ERR_LOAD;
        }

        AneModel* m = (AneModel*)calloc(1, sizeof(AneModel));
        if (!m) {
            return ANE_ERR_OOM;
        }
        m->in_memory_model = [src->in_memory_model retain];
        m->client = [src->client retain];
        m->compile_qos = qos ? qos : ANE_QOS_DEFAULT;
        atomic_store(&m->loaded, true);
        m->is_secondary_instance = true;
        AneStatus dst = derive_specs_from_attrs(m->in_memory_model, m);
        if (dst != ANE_OK) {
            ane_model_close(m);
            return dst;
        }
        *out = m;
        return ANE_OK;
    }
}

int32_t ane_client_num_connections(void) {
    if (!ane_private_load() || !g_AneClientCls) {
        return 0;
    }
    @autoreleasepool {
        id client = make_ane_client();
        if (!client) {
            return 0;
        }
        SEL s = sel_getUid("connectionsUsedForLoadingModels");
        int32_t n = 0;
        if ([client respondsToSelector:s]) {
            id arr = ((id(*)(id, SEL))objc_msgSend)(client, s);
            if ([arr isKindOfClass:[NSArray class]]) {
                n = (int32_t)[(NSArray*)arr count];
            }
        }
        [client release];
        return n;
    }
}

bool ane_model_is_virtual_client(const AneModel* m) {
    if (!m || !m->client) {
        return false;
    }
    SEL s = sel_getUid("isVirtualClient");
    if (![m->client respondsToSelector:s]) {
        return false;
    }
    return ((BOOL(*)(id, SEL))objc_msgSend)(m->client, s) ? true : false;
}

AneStatus ane_session_hint_create(AneSessionHintKind kind, AneSessionHint** out) {
    clear_last_error();
    if (!out) {
        return ANE_ERR_INVALID_ARG;
    }
    AneSessionHint* h = (AneSessionHint*)calloc(1, sizeof(AneSessionHint));
    if (!h) {
        return ANE_ERR_OOM;
    }
    h->kind = kind;
    *out = h;
    return ANE_OK;
}

void ane_session_hint_release(AneSessionHint* h) {
    if (h) {
        free(h);
    }
}

AneStatus ane_model_apply_session_hint(AneModel* m, const AneSessionHint* h,
                                       char** out_report_json) {
    clear_last_error();
    if (!m || !m->client || !m->in_memory_model || !h) {
        return ANE_ERR_INVALID_ARG;
    }
    if (out_report_json) {
        *out_report_json = NULL;
    }
    /* Like `loadModel:`, `sessionHintWithModel:` XPC-encodes the model
     * argument; in-memory models are rejected at the encoder. */
    id model_for_xpc = m->xpc_model ? m->xpc_model : m->in_memory_model;
    if (![model_for_xpc conformsToProtocol:@protocol(NSSecureCoding)]) {
        set_last_error("ane_model_apply_session_hint: requires an NSSecureCoding-"
                       "conformant model; in-memory descriptor cannot cross XPC.");
        return ANE_ERR_UNSUPPORTED;
    }
    @autoreleasepool {
        SEL s = sel_getUid("sessionHintWithModel:hint:options:report:error:");
        if (![m->client respondsToSelector:s]) {
            return ANE_ERR_UNSUPPORTED;
        }
        NSError* e = nil;
        NSNumber* hint_num = @((int)h->kind);
        /* `report:` is a plain object slot — caller supplies a mutable
         * dictionary the framework populates. */
        NSMutableDictionary* report = [NSMutableDictionary dictionary];
        BOOL ok = ((BOOL(*)(id, SEL, id, id, id, id, NSError**))objc_msgSend)(
            m->client, s, model_for_xpc, hint_num, @{}, report, &e);
        if (!ok) {
            if (e) {
                set_last_error("sessionHintWithModel: %s", [[e localizedDescription] UTF8String]);
            }
            return ANE_ERR_INTERNAL;
        }
        if (out_report_json && report.count > 0) {
            NSError* je = nil;
            NSData* json = [NSJSONSerialization dataWithJSONObject:report
                                                           options:(NSJSONWritingOptions)0
                                                             error:&je];
            if (json) {
                size_t n = [json length];
                char* buf = (char*)malloc(n + 1);
                if (buf) {
                    memcpy(buf, [json bytes], n);
                    buf[n] = '\0';
                    *out_report_json = buf;
                }
            }
        }
        return ANE_OK;
    }
}

AneStatus ane_model_open_file_ex(const AneModelFileOpenOptionsEx* opts, AneModel** out_model) {
    clear_last_error();
    if (!opts || !out_model || !opts->base.model_url) {
        return ANE_ERR_INVALID_ARG;
    }
    if (!ane_private_load() || !g_AneModelCls) {
        return ANE_ERR_UNSUPPORTED;
    }

    @autoreleasepool {
        NSString* p = [NSString stringWithUTF8String:opts->base.model_url];
        NSURL* url = [p hasPrefix:@"file:"] ? [NSURL URLWithString:p] : [NSURL fileURLWithPath:p];
        NSString* key = opts->base.cache_key ? [NSString stringWithUTF8String:opts->base.cache_key]
                                             : @"default";

        id mdl = nil;
        if (opts->mps_constants_id) {
            SEL s = sel_getUid("modelAtURL:key:mpsConstants:");
            if ([g_AneModelCls respondsToSelector:s]) {
                mdl = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(g_AneModelCls, s, url, key,
                                                                    (id)opts->mps_constants_id);
            }
        }
        if (!mdl && opts->model_attributes_id) {
            SEL s = sel_getUid("modelAtURL:key:modelAttributes:");
            if ([g_AneModelCls respondsToSelector:s]) {
                mdl = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(g_AneModelCls, s, url, key,
                                                                    (id)opts->model_attributes_id);
            }
        }
        if (!mdl) {
            mdl = ((id(*)(Class, SEL, id, id))objc_msgSend)(
                g_AneModelCls, sel_getUid("modelAtURL:key:"), url, key);
        }
        if (!mdl) {
            return ANE_ERR_LOAD;
        }

        AneModel* m = (AneModel*)calloc(1, sizeof(AneModel));
        if (!m) {
            return ANE_ERR_OOM;
        }
        m->in_memory_model = [mdl retain];
        m->compile_qos = opts->base.compile_qos ? opts->base.compile_qos : ANE_QOS_DEFAULT;

        NSError* e = nil;
        id client = make_ane_client();
        if (!client) {
            ane_model_close(m);
            return ANE_ERR_INTERNAL;
        }
        BOOL ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError**))objc_msgSend)(
            client, sel_getUid("loadModel:options:qos:error:"), m->in_memory_model, @{},
            (unsigned int)m->compile_qos, &e);
        if (!ok) {
            set_last_error("open_file loadModel: %s",
                           e ? [[e localizedDescription] UTF8String] : "<no error>");
            [client release];
            ane_model_close(m);
            return ANE_ERR_LOAD;
        }
        atomic_store(&m->loaded, true);
        m->client = client;

        AneStatus dst = derive_specs_from_attrs(m->in_memory_model, m);
        if (dst != ANE_OK) {
            ane_model_close(m);
            return dst;
        }

        *out_model = m;
        return ANE_OK;
    }
}

const char* ane_perf_counter_name(int32_t counter_idx) {
    if (!ane_private_load() || !g_AnePerfStatsCls) {
        return "";
    }
    static __thread char buf[256];
    buf[0] = '\0';
    @autoreleasepool {
        id ps = ((id(*)(Class, SEL, unsigned long long))objc_msgSend)(
            g_AnePerfStatsCls, sel_getUid("statsWithHardwareExecutionNS:"), (unsigned long long)0);
        if (!ps) {
            return "";
        }
        /* `+statsWith…` returns an autoreleased object — do NOT release. */
        SEL s = sel_getUid("stringForPerfCounter:");
        if ([ps respondsToSelector:s]) {
            id str = ((id(*)(id, SEL, int))objc_msgSend)(ps, s, (int)counter_idx);
            if ([str isKindOfClass:[NSString class]]) {
                const char* u = [(NSString*)str UTF8String];
                if (u) {
                    strncpy(buf, u, sizeof(buf) - 1);
                    buf[sizeof(buf) - 1] = '\0';
                }
            }
        }
    }
    return buf;
}

AneStatus ane_perf_stats_emit_signpost(const AnePerfStats* ps, uint64_t model_string_id) {
    clear_last_error();
    if (!ps || !ps->obj) {
        return ANE_ERR_INVALID_ARG;
    }
    SEL s = sel_getUid("emitPerfcounterSignpostsWithModelStringID:");
    if (![ps->obj respondsToSelector:s]) {
        return ANE_ERR_UNSUPPORTED;
    }
    ((void (*)(id, SEL, unsigned long long))objc_msgSend)(ps->obj, s,
                                                          (unsigned long long)model_string_id);
    return ANE_OK;
}

int32_t ane_request_num_perf_stats(const AneRequest* req) {
    if (!req) {
        return 0;
    }
    if (!req->perf_stats || !req->perf_stats->obj) {
        return 0;
    }
    return 1;
}

AnePerfStats* ane_request_perf_stats_at(const AneRequest* req, int32_t idx) {
    if (!req || idx != 0) {
        return NULL;
    }
    return req->perf_stats;
}

/* Try every documented construction path for `_ANEVirtualClient`.
 * Entitled binaries get a real instance via `+sharedConnection`;
 * unsigned binaries get nil from every path and fall through to
 * Unsupported. Returns retained reference; caller releases. */
static id ane_acquire_virtual_client(void) {
    if (!g_AneVirtualClientCls) {
        return nil;
    }
    Class c = g_AneVirtualClientCls;
    SEL shared = sel_getUid("sharedConnection");
    if ([c respondsToSelector:shared]) {
        id v = ((id(*)(Class, SEL))objc_msgSend)(c, shared);
        if (v) {
            return [v retain];
        }
    }
    SEL initSingleton = sel_getUid("initWithSingletonAccess");
    id alloced = ((id(*)(Class, SEL))objc_msgSend)(c, sel_getUid("alloc"));
    if (alloced) {
        id v = ((id(*)(id, SEL))objc_msgSend)(alloced, initSingleton);
        if (v) {
            return v;
        }
    }
    id n = ((id(*)(Class, SEL))objc_msgSend)(c, sel_getUid("new"));
    if (n) {
        return n;
    }
    return nil;
}

AneStatus ane_decompress_weights(const void* compressed, size_t cn, void** out_bytes,
                                 size_t* out_nbytes) {
    clear_last_error();
    if (!compressed || !out_bytes || !out_nbytes) {
        return ANE_ERR_INVALID_ARG;
    }
    if (!ane_private_load() || !g_AneVirtualClientCls) {
        return ANE_ERR_UNSUPPORTED;
    }
    @autoreleasepool {
        id vc = ane_acquire_virtual_client();
        if (!vc) {
            set_last_error("ane_decompress_weights: could not acquire "
                           "_ANEVirtualClient (sharedConnection returns nil "
                           "without the appleneuralengine entitlement)");
            return ANE_ERR_UNSUPPORTED;
        }
        SEL s = sel_getUid("parallelDecompressedData:");
        if (![vc respondsToSelector:s]) {
            [vc release];
            return ANE_ERR_UNSUPPORTED;
        }
        NSData* in = [NSData dataWithBytes:compressed length:cn];
        id out = ((id(*)(id, SEL, id))objc_msgSend)(vc, s, in);
        [vc release];
        if (![out isKindOfClass:[NSData class]]) {
            return ANE_ERR_INTERNAL;
        }
        NSData* od = (NSData*)out;
        size_t n = [od length];
        void* buf = malloc(n);
        if (!buf) {
            return ANE_ERR_OOM;
        }
        memcpy(buf, [od bytes], n);
        *out_bytes = buf;
        *out_nbytes = n;
        return ANE_OK;
    }
}
