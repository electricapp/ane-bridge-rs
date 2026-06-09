/* probe_cachemodel_url.m — Inspect the _ANEModel returned by
 * +modelWithCacheURLIdentifier: to see what URL/path it carries and why
 * loadModel: tries to issue a sandbox extension. Also probe
 * compiledModelExistsMatchingHash: on a freshly warmed model. */
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#include "ane_bridge.h"
#include <sys/wait.h>
#include <unistd.h>
#include <stdio.h>

static Class g_modelCls, g_clientCls;

static BOOL load_fw(void) {
    if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
                "AppleNeuralEngine", RTLD_NOW)) return NO;
    g_modelCls = NSClassFromString(@"_ANEModel");
    g_clientCls = NSClassFromString(@"_ANEClient");
    return g_modelCls && g_clientCls;
}

static void dump_props(id obj) {
    if (!obj) { fprintf(stderr, "  (nil model)\n"); return; }
    fprintf(stderr, "  class = %s\n", class_getName(object_getClass(obj)));
    const char* sels[] = {
        "modelURL", "cacheURLIdentifier", "url", "key", "hexStringIdentifier",
        "modelAttributes", "identifierSource", "compiledURL", "path", NULL
    };
    for (int i = 0; sels[i]; i++) {
        SEL s = sel_getUid(sels[i]);
        if (![obj respondsToSelector:s]) continue;
        @try {
            id v = ((id(*)(id, SEL))objc_msgSend)(obj, s);
            fprintf(stderr, "  -%s = %s\n", sels[i],
                    v ? [[v description] UTF8String] : "(nil)");
        } @catch (NSException* ex) {
            fprintf(stderr, "  -%s threw %s\n", sels[i], [[ex name] UTF8String]);
        }
    }
}

static int child(const char* mil, const char* wts) {
    @autoreleasepool {
        if (!load_fw()) return 11;
        AneModelOpenOptions o = { .mil_path=mil, .weights_path=wts, .compile_qos=ANE_QOS_DEFAULT };
        AneModel* m = NULL;
        if (ane_model_open(&o, &m) != ANE_OK) { fprintf(stderr, "open fail: %s\n", ane_last_error()); return 10; }
        const char* h = ane_model_program_id(m);
        NSString* hex = [NSString stringWithUTF8String:h];
        fprintf(stderr, "hex=%s\n", h);
        id xm = ((id(*)(Class,SEL,id))objc_msgSend)(g_modelCls,
                  sel_getUid("modelWithCacheURLIdentifier:"), hex);
        fprintf(stderr, "== modelWithCacheURLIdentifier result ==\n");
        dump_props(xm);

        id client = [[g_clientCls alloc] init];
        SEL exMatch = sel_getUid("compiledModelExistsMatchingHash:");
        if ([client respondsToSelector:exMatch]) {
            BOOL r = ((BOOL(*)(id,SEL,id))objc_msgSend)(client, exMatch, hex);
            fprintf(stderr, "compiledModelExistsMatchingHash: = %s\n", r?"YES":"NO");
        }
        ane_model_close(m);
        return 0;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s mil wts\n", argv[0]); return 2; }
    pid_t pid = fork();
    if (pid == 0) { int rc = child(argv[1], argv[2]); fflush(stderr); _exit(rc & 0x7F); }
    int st = 0; waitpid(pid, &st, 0);
    if (WIFSIGNALED(st)) fprintf(stderr, "signal=%d\n", WTERMSIG(st));
    return 0;
}
