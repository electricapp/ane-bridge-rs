/* probe_chain_entitlements.m — Hunt for an XPC-load path that aned accepts
 * from an unsigned user binary, so that chained dispatch
 * (_ANEChainingRequest + doPrepareChainingWithModel: + doEnqueueSetsWithModel:)
 * actually fires.
 *
 * The probe assumes the bridge already loads the model in-memory (which it
 * does — and that path succeeds in production). What the probe attacks is
 * the *separate* `_ANEClient.loadModel:` step that the chain path takes on
 * top, which previously came back NO with no NSError.
 *
 * Variants (each in a forked child so a framework abort doesn't kill the
 * parent's matrix walk):
 *   1. loadRealTimeModel:options:qos:error: in place of loadModel:
 *   2. _ANEClient initWithRestrictedAccessAllowed:NO (vs YES)
 *   3. _ANEVirtualClient sessionHintWithModel:hint:options:report:error:
 *      as a "does this even reach aned?" check
 *   4. +modelWithCacheURLIdentifier: then load on _ANEVirtualClient
 *   5. +modelAtURL:key: pointing at NSTemporaryDirectory()/<hex_id>/
 *      (where the bridge already materialized model.mil + weights/weight.bin
 *      on the cache-miss compile path; if we hit cache we'll re-materialize)
 *   6. Cache existence checks: compiledModelExistsMatchingHash: / +Exists:
 *
 * Build:
 *   xcrun clang -O2 -fobjc-arc -Ic/include -framework Foundation \
 *     -framework IOSurface -framework Metal \
 *     -Wl,-rpath,@executable_path/../lib -Lbuild/lib -lane_bridge \
 *     -o build/bin/probe_chain_entitlements tools/probe_chain_entitlements.m
 *
 * Run:
 *   ./build/bin/probe_chain_entitlements build/identity/model.mil \
 *       build/identity/weights.bin
 */
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <IOSurface/IOSurfaceRef.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>

#include "ane_bridge.h"
#include <sys/wait.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ----- exit codes (RX_* to avoid colliding with unistd.h R_OK) ----- */
enum {
    RX_OK             = 0,   /* fully succeeded (load YES + prepare YES) */
    RX_LOAD_YES       = 1,   /* loadModel returned YES, prepare still failed */
    RX_LOAD_NO        = 2,   /* loadModel returned NO */
    RX_LOAD_EXC       = 3,   /* loadModel threw NSException */
    RX_OPEN_FAIL      = 10,
    RX_NO_FRAMEWORK   = 11,
    RX_NO_HEX_ID      = 12,
    RX_NO_CLIENT      = 13,
    RX_NO_MODEL_CLS   = 14,
    RX_NOT_SUPPORTED  = 15,
    RX_CRASH          = 64,
};

static const char* describe(int rc) {
    switch (rc) {
        case RX_OK:            return "OK (load YES + prepare YES)";
        case RX_LOAD_YES:      return "load YES, prepare NO";
        case RX_LOAD_NO:       return "load NO";
        case RX_LOAD_EXC:      return "load EXCEPTION";
        case RX_OPEN_FAIL:     return "ane_model_open failed";
        case RX_NO_FRAMEWORK:  return "framework dlopen failed";
        case RX_NO_HEX_ID:     return "no hex id from in-memory model";
        case RX_NO_CLIENT:     return "could not construct client";
        case RX_NO_MODEL_CLS:  return "_ANEModel class missing";
        case RX_NOT_SUPPORTED: return "selector not supported";
        case RX_CRASH:         return "CRASH (signal)";
        default:               return "?";
    }
}

/* ----- shared framework load ----- */
static Class g_modelCls         = nil;
static Class g_clientCls        = nil;
static Class g_virtualClientCls = nil;

static BOOL load_framework(void) {
    static dispatch_once_t once;
    static BOOL ok = NO;
    dispatch_once(&once, ^{
        if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
                    "AppleNeuralEngine", RTLD_NOW)) return;
        g_modelCls         = NSClassFromString(@"_ANEModel");
        g_clientCls        = NSClassFromString(@"_ANEClient");
        g_virtualClientCls = NSClassFromString(@"_ANEVirtualClient");
        ok = (g_modelCls && g_clientCls);
    });
    return ok;
}

/* ----- helpers ----- */

/* Try to read `hexStringIdentifier` from the bridge's loaded model. The
 * bridge exposes the program id as `ane_model_program_id`. We trust that. */
static NSString* hex_id_for(AneModel* m) {
    const char* h = ane_model_program_id(m);
    if (!h || !*h) return nil;
    return [NSString stringWithUTF8String:h];
}

/* Materialize model.mil + weights/weight.bin under NSTemporaryDirectory()/
 * <hex_id>/ — matches the bridge's compile-path layout. Returns the dir or
 * nil on failure. */
static NSString* materialize_dir(NSString* hex,
                                 const char* mil_path,
                                 const char* wts_path) {
    NSString* dir = [NSTemporaryDirectory() stringByAppendingPathComponent:hex];
    NSString* wdir = [dir stringByAppendingPathComponent:@"weights"];
    NSFileManager* fm = [NSFileManager defaultManager];
    NSError* e = nil;
    if (![fm createDirectoryAtPath:wdir withIntermediateDirectories:YES
                        attributes:nil error:&e]) {
        fprintf(stderr, "    [tmp] mkdir failed: %s\n",
                e ? [[e localizedDescription] UTF8String] : "?");
        return nil;
    }
    NSData* md = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:mil_path]];
    NSData* wd = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:wts_path]];
    if (!md || !wd) {
        fprintf(stderr, "    [tmp] read mil/wts failed\n");
        return nil;
    }
    if (![md writeToFile:[dir stringByAppendingPathComponent:@"model.mil"]
              atomically:YES]) {
        fprintf(stderr, "    [tmp] write mil failed\n"); return nil;
    }
    if (![wd writeToFile:[wdir stringByAppendingPathComponent:@"weight.bin"]
              atomically:YES]) {
        fprintf(stderr, "    [tmp] write weights failed\n"); return nil;
    }
    return dir;
}

/* Make an `_ANEClient` with the requested access flag. */
static id make_client(BOOL restricted_allowed) {
    if (!g_clientCls) return nil;
    id alloced = [g_clientCls alloc];
    SEL s = sel_getUid("initWithRestrictedAccessAllowed:");
    if (![alloced respondsToSelector:s]) {
        [alloced release]; return nil;
    }
    return ((id(*)(id, SEL, BOOL))objc_msgSend)(alloced, s, restricted_allowed);
}

/* Try to attach a chain prepare on top of (client, _ANEModel obj). Just
 * builds the minimum: 1 step, 1 input/output bound to the bridge's
 * existing AneModel. */
static BOOL try_prepare_chain(AneModel* m, NSString** out_err) {
    AneBuffer *ib = NULL, *ob = NULL;
    if (ane_buffer_create_for_input (m, 0, &ib) != ANE_OK) return NO;
    if (ane_buffer_create_for_output(m, 0, &ob) != ANE_OK) {
        ane_buffer_release(ib); return NO;
    }
    AneRequest* rq = NULL;
    ane_request_create(m, &rq);
    ane_request_bind_input (rq, 0, ib);
    ane_request_bind_output(rq, 0, ob);
    AneChainStep step = { .request = rq };
    AneChain* chain = NULL;
    AneStatus st = ane_chain_create(&step, 1, &chain);
    if (st != ANE_OK) {
        if (out_err) *out_err = [NSString stringWithFormat:@"chain_create: %s",
                                          ane_last_error()];
        ane_request_release(rq);
        ane_buffer_release(ib); ane_buffer_release(ob);
        return NO;
    }
    st = ane_chain_prepare(chain, ANE_QOS_DEFAULT);
    if (st != ANE_OK && out_err) {
        *out_err = [NSString stringWithFormat:@"chain_prepare: %s",
                              ane_last_error()];
    }
    ane_chain_release(chain);
    ane_request_release(rq);
    ane_buffer_release(ib); ane_buffer_release(ob);
    return st == ANE_OK;
}

/* ---------------------------------------------------------------- *
 *  Variant runners — each takes (mil, wts) and returns one of R_*. *
 *  Side-effect: prints up to one descriptive line about NSError.   *
 * ---------------------------------------------------------------- */

/* Variant 1: try loadRealTimeModel:options:qos:error: against an
 * `_ANEModel` built via +modelWithCacheURLIdentifier:. */
static int v1_realtime_load(const char* mil, const char* wts) {
    @autoreleasepool {
        if (!load_framework()) return RX_NO_FRAMEWORK;
        AneModelOpenOptions o = { .mil_path=mil, .weights_path=wts,
                                  .compile_qos=ANE_QOS_DEFAULT };
        AneModel* m = NULL;
        if (ane_model_open(&o, &m) != ANE_OK) return RX_OPEN_FAIL;
        NSString* hex = hex_id_for(m);
        if (!hex) { ane_model_close(m); return RX_NO_HEX_ID; }
        SEL byId = sel_getUid("modelWithCacheURLIdentifier:");
        if (![g_modelCls respondsToSelector:byId]) {
            ane_model_close(m); return RX_NOT_SUPPORTED;
        }
        id xm = ((id(*)(Class, SEL, id))objc_msgSend)(g_modelCls, byId, hex);
        if (!xm) { ane_model_close(m); return RX_NO_MODEL_CLS; }

        id client = make_client(YES);
        if (!client) { ane_model_close(m); return RX_NO_CLIENT; }
        SEL rt = sel_getUid("loadRealTimeModel:options:qos:error:");
        if (![client respondsToSelector:rt]) {
            [client release]; ane_model_close(m); return RX_NOT_SUPPORTED;
        }
        BOOL ok = NO; NSError* e = nil;
        @try {
            ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError**))objc_msgSend)(
                client, rt, xm, @{}, (unsigned int)ANE_QOS_DEFAULT, &e);
        } @catch (NSException* ex) {
            fprintf(stderr, "    [v1] EXC: %s\n", [[ex description] UTF8String]);
            [client release]; ane_model_close(m); return RX_LOAD_EXC;
        }
        if (!ok) {
            fprintf(stderr, "    [v1] loadRealTimeModel NO: %s\n",
                    e ? [[e description] UTF8String] : "<no NSError>");
            [client release]; ane_model_close(m); return RX_LOAD_NO;
        }
        NSString* perr = nil;
        BOOL prep = try_prepare_chain(m, &perr);
        if (!prep) fprintf(stderr, "    [v1] prepare NO: %s\n",
                           perr ? [perr UTF8String] : "?");
        [client release]; ane_model_close(m);
        return prep ? RX_OK : RX_LOAD_YES;
    }
}

/* Variant 2: _ANEClient with initWithRestrictedAccessAllowed:NO. */
static int v2_restricted_no(const char* mil, const char* wts) {
    @autoreleasepool {
        if (!load_framework()) return RX_NO_FRAMEWORK;
        AneModelOpenOptions o = { .mil_path=mil, .weights_path=wts,
                                  .compile_qos=ANE_QOS_DEFAULT };
        AneModel* m = NULL;
        if (ane_model_open(&o, &m) != ANE_OK) return RX_OPEN_FAIL;
        NSString* hex = hex_id_for(m);
        if (!hex) { ane_model_close(m); return RX_NO_HEX_ID; }
        SEL byId = sel_getUid("modelWithCacheURLIdentifier:");
        id xm = ((id(*)(Class, SEL, id))objc_msgSend)(g_modelCls, byId, hex);
        if (!xm) { ane_model_close(m); return RX_NO_MODEL_CLS; }

        id client = make_client(NO);  /* the variation */
        if (!client) { ane_model_close(m); return RX_NO_CLIENT; }
        SEL ld = sel_getUid("loadModel:options:qos:error:");
        BOOL ok = NO; NSError* e = nil;
        @try {
            ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError**))objc_msgSend)(
                client, ld, xm, @{}, (unsigned int)ANE_QOS_DEFAULT, &e);
        } @catch (NSException* ex) {
            fprintf(stderr, "    [v2] EXC: %s\n", [[ex description] UTF8String]);
            [client release]; ane_model_close(m); return RX_LOAD_EXC;
        }
        if (!ok) {
            fprintf(stderr, "    [v2] loadModel NO: %s\n",
                    e ? [[e description] UTF8String] : "<no NSError>");
            [client release]; ane_model_close(m); return RX_LOAD_NO;
        }
        NSString* perr = nil;
        BOOL prep = try_prepare_chain(m, &perr);
        if (!prep) fprintf(stderr, "    [v2] prepare NO: %s\n",
                           perr ? [perr UTF8String] : "?");
        [client release]; ane_model_close(m);
        return prep ? RX_OK : RX_LOAD_YES;
    }
}

/* Variant 3: sessionHintWithModel:hint:options:report:error: on the
 * shared _ANEVirtualClient. This is a much less invasive XPC round-trip
 * than loadModel; if even this comes back NO with sandbox errors, we
 * know aned won't talk to us *at all* for cache-identified models. */
static int v3_session_hint(const char* mil, const char* wts) {
    @autoreleasepool {
        if (!load_framework()) return RX_NO_FRAMEWORK;
        if (!g_virtualClientCls) return RX_NOT_SUPPORTED;
        AneModelOpenOptions o = { .mil_path=mil, .weights_path=wts,
                                  .compile_qos=ANE_QOS_DEFAULT };
        AneModel* m = NULL;
        if (ane_model_open(&o, &m) != ANE_OK) return RX_OPEN_FAIL;
        NSString* hex = hex_id_for(m);
        if (!hex) { ane_model_close(m); return RX_NO_HEX_ID; }
        SEL byId = sel_getUid("modelWithCacheURLIdentifier:");
        id xm = ((id(*)(Class, SEL, id))objc_msgSend)(g_modelCls, byId, hex);
        if (!xm) { ane_model_close(m); return RX_NO_MODEL_CLS; }

        id vc = ((id(*)(Class, SEL))objc_msgSend)(g_virtualClientCls,
                                                  sel_getUid("sharedConnection"));
        if (!vc) {
            /* Fallback: obtain virtualClient via a regular _ANEClient. */
            id base = make_client(YES);
            if (base) {
                SEL vcSel = sel_getUid("virtualClient");
                if ([base respondsToSelector:vcSel]) {
                    vc = ((id(*)(id, SEL))objc_msgSend)(base, vcSel);
                }
                [base release];
            }
        }
        if (!vc) {
            /* Last resort: -initWithSingletonAccess directly. */
            id alloced = [g_virtualClientCls alloc];
            SEL initS = sel_getUid("initWithSingletonAccess");
            if ([alloced respondsToSelector:initS]) {
                vc = ((id(*)(id, SEL))objc_msgSend)(alloced, initS);
            } else {
                vc = ((id(*)(id, SEL))objc_msgSend)(alloced, sel_getUid("init"));
            }
            fprintf(stderr, "    [v3] fallback VC via init = %s\n",
                    vc ? [[vc description] UTF8String] : "(nil)");
        }
        if (!vc) { ane_model_close(m); return RX_NO_CLIENT; }
        SEL sh = sel_getUid("sessionHintWithModel:hint:options:report:error:");
        if (![vc respondsToSelector:sh]) {
            ane_model_close(m); return RX_NOT_SUPPORTED;
        }
        NSMutableDictionary* report = [NSMutableDictionary dictionary];
        NSError* e = nil;
        BOOL ok = NO;
        @try {
            ok = ((BOOL(*)(id, SEL, id, NSInteger, id, id, NSError**))objc_msgSend)(
                vc, sh, xm, (NSInteger)1 /* prefetch */, @{}, report, &e);
        } @catch (NSException* ex) {
            fprintf(stderr, "    [v3] EXC: %s\n", [[ex description] UTF8String]);
            ane_model_close(m); return RX_LOAD_EXC;
        }
        if (!ok) {
            fprintf(stderr, "    [v3] sessionHint NO: %s | report=%s\n",
                    e ? [[e description] UTF8String] : "<no NSError>",
                    [[report description] UTF8String]);
            ane_model_close(m); return RX_LOAD_NO;
        }
        fprintf(stderr, "    [v3] sessionHint YES: report=%s\n",
                [[report description] UTF8String]);
        /* Even if hint succeeds we still want to see if chain prepare works. */
        NSString* perr = nil;
        BOOL prep = try_prepare_chain(m, &perr);
        if (!prep) fprintf(stderr, "    [v3] prepare NO: %s\n",
                           perr ? [perr UTF8String] : "?");
        ane_model_close(m);
        return prep ? RX_OK : RX_LOAD_YES;
    }
}

/* Variant 4: +modelWithCacheURLIdentifier: then load via VirtualClient
 * rather than via _ANEClient. Some private interfaces route through a
 * "virtual" path that doesn't need the same sandbox extension. */
static int v4_virtual_client_load(const char* mil, const char* wts) {
    @autoreleasepool {
        if (!load_framework()) return RX_NO_FRAMEWORK;
        if (!g_virtualClientCls) return RX_NOT_SUPPORTED;
        AneModelOpenOptions o = { .mil_path=mil, .weights_path=wts,
                                  .compile_qos=ANE_QOS_DEFAULT };
        AneModel* m = NULL;
        if (ane_model_open(&o, &m) != ANE_OK) return RX_OPEN_FAIL;
        NSString* hex = hex_id_for(m);
        if (!hex) { ane_model_close(m); return RX_NO_HEX_ID; }
        SEL byId = sel_getUid("modelWithCacheURLIdentifier:");
        id xm = ((id(*)(Class, SEL, id))objc_msgSend)(g_modelCls, byId, hex);
        if (!xm) { ane_model_close(m); return RX_NO_MODEL_CLS; }

        id vc = ((id(*)(Class, SEL))objc_msgSend)(g_virtualClientCls,
                                                  sel_getUid("sharedConnection"));
        if (!vc) {
            id base = make_client(YES);
            if (base) {
                SEL vcSel = sel_getUid("virtualClient");
                if ([base respondsToSelector:vcSel]) {
                    vc = ((id(*)(id, SEL))objc_msgSend)(base, vcSel);
                }
                [base release];
            }
        }
        if (!vc) {
            id alloced = [g_virtualClientCls alloc];
            SEL initS = sel_getUid("initWithSingletonAccess");
            if ([alloced respondsToSelector:initS]) {
                vc = ((id(*)(id, SEL))objc_msgSend)(alloced, initS);
            } else {
                vc = ((id(*)(id, SEL))objc_msgSend)(alloced, sel_getUid("init"));
            }
            fprintf(stderr, "    [v4] fallback VC via init = %s\n",
                    vc ? [[vc description] UTF8String] : "(nil)");
        }
        if (!vc) { ane_model_close(m); return RX_NO_CLIENT; }

        /* Try a series of selector signatures that a virtual client might
         * carry. The first one found that takes (model, opts, qos, error*)
         * wins. */
        const char* candidates[] = {
            "loadModel:options:qos:error:",
            "loadRealTimeModel:options:qos:error:",
            NULL,
        };
        BOOL ok = NO; NSError* e = nil; const char* used = NULL;
        for (int i = 0; candidates[i]; i++) {
            SEL s = sel_getUid(candidates[i]);
            if (![vc respondsToSelector:s]) continue;
            used = candidates[i];
            @try {
                ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError**))objc_msgSend)(
                    vc, s, xm, @{}, (unsigned int)ANE_QOS_DEFAULT, &e);
            } @catch (NSException* ex) {
                fprintf(stderr, "    [v4 %s] EXC: %s\n", used,
                        [[ex description] UTF8String]);
                ane_model_close(m); return RX_LOAD_EXC;
            }
            break;
        }
        if (!used) {
            fprintf(stderr, "    [v4] VirtualClient has no loadModel: variant\n");
            ane_model_close(m); return RX_NOT_SUPPORTED;
        }
        if (!ok) {
            fprintf(stderr, "    [v4 %s] NO: %s\n", used,
                    e ? [[e description] UTF8String] : "<no NSError>");
            ane_model_close(m); return RX_LOAD_NO;
        }
        NSString* perr = nil;
        BOOL prep = try_prepare_chain(m, &perr);
        if (!prep) fprintf(stderr, "    [v4] prepare NO: %s\n",
                           perr ? [perr UTF8String] : "?");
        ane_model_close(m);
        return prep ? RX_OK : RX_LOAD_YES;
    }
}

/* Variant 5: build _ANEModel via +modelAtURL:key: pointing at our own
 * NSTemporaryDirectory() materialization of model.mil + weights/weight.bin,
 * then loadModel on _ANEClient.
 *
 * The hypothesis: aned needs a sandbox extension for the bundle URL it
 * sees. A path under /var/folders/.../T/<hex>/ is writable to *us*; if
 * aned can read it (it's our temp dir, in our user context), the load
 * may go through without the privileged extension that compiled-bundle
 * URLs require. */
static int v5_model_at_temp_url(const char* mil, const char* wts) {
    @autoreleasepool {
        if (!load_framework()) return RX_NO_FRAMEWORK;
        AneModelOpenOptions o = { .mil_path=mil, .weights_path=wts,
                                  .compile_qos=ANE_QOS_DEFAULT };
        AneModel* m = NULL;
        if (ane_model_open(&o, &m) != ANE_OK) return RX_OPEN_FAIL;
        NSString* hex = hex_id_for(m);
        if (!hex) { ane_model_close(m); return RX_NO_HEX_ID; }

        /* Always (re-)materialize. On cache hit the bridge does NOT write
         * a temp dir, so we must do it ourselves. */
        NSString* dir = materialize_dir(hex, mil, wts);
        if (!dir) { ane_model_close(m); return RX_OPEN_FAIL; }
        NSURL* url = [NSURL fileURLWithPath:dir isDirectory:YES];

        SEL byUrl = sel_getUid("modelAtURL:key:");
        if (![g_modelCls respondsToSelector:byUrl]) {
            ane_model_close(m); return RX_NOT_SUPPORTED;
        }
        id xm = ((id(*)(Class, SEL, id, id))objc_msgSend)(
            g_modelCls, byUrl, url, hex);
        if (!xm) {
            fprintf(stderr, "    [v5] modelAtURL:key: returned nil for %s\n",
                    [[url path] UTF8String]);
            ane_model_close(m); return RX_NO_MODEL_CLS;
        }

        id client = make_client(YES);
        if (!client) { ane_model_close(m); return RX_NO_CLIENT; }
        SEL ld = sel_getUid("loadModel:options:qos:error:");
        BOOL ok = NO; NSError* e = nil;
        @try {
            ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError**))objc_msgSend)(
                client, ld, xm, @{}, (unsigned int)ANE_QOS_DEFAULT, &e);
        } @catch (NSException* ex) {
            fprintf(stderr, "    [v5] EXC: %s\n", [[ex description] UTF8String]);
            [client release]; ane_model_close(m); return RX_LOAD_EXC;
        }
        if (!ok) {
            fprintf(stderr, "    [v5] loadModel NO at %s: %s\n",
                    [[url path] UTF8String],
                    e ? [[e description] UTF8String] : "<no NSError>");
            [client release]; ane_model_close(m); return RX_LOAD_NO;
        }
        NSString* perr = nil;
        BOOL prep = try_prepare_chain(m, &perr);
        if (!prep) fprintf(stderr, "    [v5] prepare NO: %s\n",
                           perr ? [perr UTF8String] : "?");
        [client release]; ane_model_close(m);
        return prep ? RX_OK : RX_LOAD_YES;
    }
}

/* Variant 6: cache existence probes. Not a load — just info. */
static int v6_cache_probes(const char* mil, const char* wts) {
    @autoreleasepool {
        if (!load_framework()) return RX_NO_FRAMEWORK;
        AneModelOpenOptions o = { .mil_path=mil, .weights_path=wts,
                                  .compile_qos=ANE_QOS_DEFAULT };
        AneModel* m = NULL;
        if (ane_model_open(&o, &m) != ANE_OK) return RX_OPEN_FAIL;
        NSString* hex = hex_id_for(m);
        fprintf(stderr, "    [v6] hex_id = %s\n",
                hex ? [hex UTF8String] : "(nil)");
        fprintf(stderr, "    [v6] ane_model_was_cached = %s\n",
                ane_model_was_cached(m) ? "YES" : "NO");
        fprintf(stderr, "    [v6] ane_cache_exists_for_hash(hex) = %s\n",
                hex && ane_cache_exists_for_hash([hex UTF8String]) ? "YES" : "NO");

        id client = make_client(YES);
        if (client) {
            SEL s1 = sel_getUid("compiledModelExistsMatchingHash:");
            if (hex && [client respondsToSelector:s1]) {
                BOOL r = ((BOOL(*)(id, SEL, id))objc_msgSend)(client, s1, hex);
                fprintf(stderr, "    [v6] _ANEClient.compiledModelExistsMatchingHash: -> %s\n",
                        r ? "YES" : "NO");
            } else {
                fprintf(stderr, "    [v6] no compiledModelExistsMatchingHash:\n");
            }
            SEL s2 = sel_getUid("compiledModelExistsFor:");
            if (hex && [client respondsToSelector:s2]) {
                @try {
                    BOOL r = ((BOOL(*)(id, SEL, id))objc_msgSend)(client, s2, hex);
                    fprintf(stderr, "    [v6] _ANEClient.compiledModelExistsFor: -> %s\n",
                            r ? "YES" : "NO");
                } @catch (NSException* ex) {
                    fprintf(stderr, "    [v6] compiledModelExistsFor: EXC %s\n",
                            [[ex description] UTF8String]);
                }
            }
            [client release];
        }
        ane_model_close(m);
        return RX_OK;
    }
}

/* ---------- fork+wait harness ---------- */
typedef int (*variant_fn)(const char*, const char*);

static int run_forked(const char* label, variant_fn fn,
                      const char* mil, const char* wts) {
    fflush(stdout); fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return RX_CRASH; }
    if (pid == 0) {
        int rc = fn(mil, wts);
        fflush(stdout); fflush(stderr);
        _exit(rc & 0x7F);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    int rc = WIFEXITED(status) ? WEXITSTATUS(status) : RX_CRASH;
    if (WIFSIGNALED(status)) {
        fprintf(stderr, "    [%s] signal=%d\n", label, WTERMSIG(status));
        rc = RX_CRASH;
    }
    return rc;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.mil> <weights.bin>\n", argv[0]);
        return 2;
    }
    const char* mil = argv[1];
    const char* wts = argv[2];

    struct { const char* label; variant_fn fn; } variants[] = {
        { "v1_realtime_load",         v1_realtime_load },
        { "v2_restricted_NO",         v2_restricted_no },
        { "v3_session_hint",          v3_session_hint  },
        { "v4_virtual_client_load",   v4_virtual_client_load },
        { "v5_modelAtURL_temp",       v5_model_at_temp_url },
        { "v6_cache_probes",          v6_cache_probes  },
    };

    printf("%-26s %-4s %s\n", "variant", "rc", "status");
    printf("------------------------------------------------------------\n");
    int any_ok = 0;
    for (size_t i = 0; i < sizeof(variants)/sizeof(variants[0]); i++) {
        int rc = run_forked(variants[i].label, variants[i].fn, mil, wts);
        printf("%-26s %-4d %s\n", variants[i].label, rc, describe(rc));
        if (rc == RX_OK || rc == RX_LOAD_YES) any_ok++;
        fflush(stdout);
    }
    printf("\nsummary: %d/%zu variants got past loadModel\n",
           any_ok, sizeof(variants)/sizeof(variants[0]));
    return any_ok ? 0 : 1;
}
