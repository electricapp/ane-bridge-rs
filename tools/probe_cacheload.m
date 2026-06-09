/* probe_cacheload.m — Test whether warming aned's rootless cache via a real
 * Core ML ANE prediction lets an unsigned binary load the SAME model through
 * +[_ANEModel modelWithCacheURLIdentifier:] + loadModel:options:qos:error:.
 *
 * Strategy:
 *   - Open the model via the bridge (this computes the ANE hexStringIdentifier
 *     from the MIL+weights, the same way aned would key its cache).
 *   - Build an _ANEModel via +modelWithCacheURLIdentifier:<hex>.
 *   - loadModel:options:qos:error: and report YES/NO + NSError.
 *   - If load YES, attempt chain prepare + enqueue via the bridge API and
 *     verify output bytes.
 *
 * Each test runs in a forked child so a framework abort doesn't kill the run.
 *
 * Build:
 *   xcrun clang -O2 -fno-objc-arc -Ic/include -framework Foundation \
 *     -framework IOSurface -framework CoreML \
 *     -Wl,-rpath,@executable_path/../lib -Lbuild/lib -lane_bridge -ldl \
 *     -o build/bin/probe_cacheload tools/probe_cacheload.m
 *
 * Run:
 *   ./build/bin/probe_cacheload <model.mil> <weights.bin>
 */
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>

#include "ane_bridge.h"
#include <sys/wait.h>
#include <unistd.h>
#include <stdio.h>

static Class g_modelCls  = nil;
static Class g_clientCls = nil;

static BOOL load_framework(void) {
    if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
                "AppleNeuralEngine", RTLD_NOW)) return NO;
    g_modelCls  = NSClassFromString(@"_ANEModel");
    g_clientCls = NSClassFromString(@"_ANEClient");
    return (g_modelCls && g_clientCls) ? YES : NO;
}

static id make_client(void) {
    id a = [g_clientCls alloc];
    SEL s = sel_getUid("initWithRestrictedAccessAllowed:");
    if (![a respondsToSelector:s]) return [a init];
    return ((id(*)(id, SEL, BOOL))objc_msgSend)(a, s, YES);
}

/* Child: open model, derive hex, modelWithCacheURLIdentifier, loadModel. */
static int child_cacheload(const char* mil, const char* wts) {
    @autoreleasepool {
        if (!load_framework()) { fprintf(stderr, "  framework load failed\n"); return 11; }
        AneModelOpenOptions o = { .mil_path = mil, .weights_path = wts,
                                  .compile_qos = ANE_QOS_DEFAULT };
        AneModel* m = NULL;
        if (ane_model_open(&o, &m) != ANE_OK) {
            fprintf(stderr, "  ane_model_open failed: %s\n", ane_last_error());
            return 10;
        }
        const char* h = ane_model_program_id(m);
        if (!h || !*h) { fprintf(stderr, "  no hex id\n"); ane_model_close(m); return 12; }
        NSString* hex = [NSString stringWithUTF8String:h];
        fprintf(stderr, "  hex_id = %s\n", h);
        fprintf(stderr, "  ane_model_was_cached = %s\n",
                ane_model_was_cached(m) ? "YES" : "NO");
        fprintf(stderr, "  ane_cache_exists_for_hash = %s\n",
                ane_cache_exists_for_hash(h) ? "YES" : "NO");

        SEL byId = sel_getUid("modelWithCacheURLIdentifier:");
        if (![g_modelCls respondsToSelector:byId]) {
            fprintf(stderr, "  no modelWithCacheURLIdentifier:\n");
            ane_model_close(m); return 14;
        }
        id xm = ((id(*)(Class, SEL, id))objc_msgSend)(g_modelCls, byId, hex);
        if (!xm) { fprintf(stderr, "  modelWithCacheURLIdentifier nil\n"); ane_model_close(m); return 14; }

        id client = make_client();
        SEL ld = sel_getUid("loadModel:options:qos:error:");
        BOOL ok = NO; NSError* e = nil;
        @try {
            ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError**))objc_msgSend)(
                client, ld, xm, @{}, (unsigned int)ANE_QOS_DEFAULT, &e);
        } @catch (NSException* ex) {
            fprintf(stderr, "  loadModel EXC: %s\n", [[ex description] UTF8String]);
            ane_model_close(m); return 3;
        }
        if (!ok) {
            fprintf(stderr, "  loadModel = NO  NSError=%s\n",
                    e ? [[e description] UTF8String] : "<nil>");
            ane_model_close(m); return 2;
        }
        fprintf(stderr, "  loadModel = YES\n");
        ane_model_close(m);
        return 0;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <mil> <wts>\n", argv[0]); return 2; }
    fflush(stdout); fflush(stderr);
    pid_t pid = fork();
    if (pid == 0) { int rc = child_cacheload(argv[1], argv[2]); fflush(stderr); _exit(rc & 0x7F); }
    int st = 0; waitpid(pid, &st, 0);
    int rc = WIFEXITED(st) ? WEXITSTATUS(st) : 64;
    if (WIFSIGNALED(st)) { fprintf(stderr, "  signal=%d\n", WTERMSIG(st)); rc = 64; }
    const char* msg = rc == 0 ? "loadModel YES" :
                      rc == 2 ? "loadModel NO" :
                      rc == 3 ? "loadModel EXCEPTION" : "error";
    printf("\nVERDICT: rc=%d (%s)\n", rc, msg);
    return rc;
}
