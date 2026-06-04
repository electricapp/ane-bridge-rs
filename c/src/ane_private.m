/* ane_private.m — Private framework loading.
 *
 * Thread safety: every globally-visible side effect here is gated by a
 * `dispatch_once`. The class globals (`g_*Cls`) are written *only* inside
 * the once block, so concurrent readers either see all-zero (before the
 * once fires) or all-final (after) — never a torn intermediate state.
 *
 * Without that guard, a fresh process where two threads simultaneously
 * called `ane_model_open` would race on `NSClassFromString` writes to
 * the class globals, and a third thread reading them via `objc_msgSend`
 * could observe a torn pointer on ARM64 and segfault dereferencing it.
 */
#import "ane_private.h"
#import <dispatch/dispatch.h>
#import <dlfcn.h>
#import <objc/runtime.h>

Class g_AneDescriptorCls        = nil;
Class g_AneInMemoryCls          = nil;
Class g_AneRequestCls           = nil;
Class g_AneIOSurfaceCls         = nil;
Class g_AneClientCls            = nil;
Class g_AneModelCls             = nil;
Class g_AneVirtualClientCls     = nil;
Class g_AnePerfStatsCls         = nil;
Class g_AneSharedEventsCls      = nil;
Class g_AneSharedSignalEventCls = nil;
Class g_AneSharedWaitEventCls   = nil;
Class g_AneChainingRequestCls   = nil;
Class g_AneDeviceInfoCls        = nil;

static id stub_getUUID(id self, SEL _cmd) {
    (void)self; (void)_cmd;
    static dispatch_once_t once;
    static id u = nil;
    dispatch_once(&once, ^{
        u = [[NSUUID alloc] init];
    });
    return u;
}

/* `_ANEChainingRequest validate` calls -symbolIndex on every member
 * of its `inputs:` and `outputSets:` arrays. None of the documented
 * buffer wrapper classes carry that selector natively; instead, the
 * framework expects callers to set a symbol index on each wrapper
 * before passing it. We stash the index per-instance via an
 * associated object and install a -symbolIndex getter at framework
 * load time. */
char g_ane_symbol_index_key;
unsigned int ane_stub_symbol_index(id self, SEL _cmd) {
    (void)_cmd;
    id v = objc_getAssociatedObject(self, &g_ane_symbol_index_key);
    return v ? (unsigned int)[v unsignedIntValue] : 0;
}

static void install_symbol_index_on(Class cls) {
    if (!cls) return;
    SEL s = sel_registerName("symbolIndex");
    if ([cls instancesRespondToSelector:s]) return;
    class_addMethod(cls, s, (IMP)ane_stub_symbol_index, "I@:");
}

BOOL ane_private_load(void) {
    static dispatch_once_t once;
    static BOOL ok = NO;
    dispatch_once(&once, ^{
        if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
                    RTLD_NOW)) {
            return;
        }
        Class d   = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class im  = NSClassFromString(@"_ANEInMemoryModel");
        Class rq  = NSClassFromString(@"_ANERequest");
        Class iio = NSClassFromString(@"_ANEIOSurfaceObject");
        Class cl  = NSClassFromString(@"_ANEClient");
        if (!d || !im || !rq || !iio || !cl) {
            return;
        }
        /* Install the getUUID stub on the class while we still hold
         * exclusive write access (no other thread can be reading the
         * globals because none of them are published yet). */
        SEL sel = sel_registerName("getUUID");
        if (![im instancesRespondToSelector:sel]) {
            class_addMethod(im, sel, (IMP)stub_getUUID, "@@:");
        }
        /* Publish. Writes inside `dispatch_once` happen-before reads
         * outside it (dispatch_once includes the necessary fences), so
         * by the time another thread observes `ok = YES` it must also
         * observe these class globals as their final values. */
        g_AneDescriptorCls        = d;
        g_AneInMemoryCls          = im;
        g_AneRequestCls           = rq;
        g_AneIOSurfaceCls         = iio;
        g_AneClientCls            = cl;
        g_AneModelCls             = NSClassFromString(@"_ANEModel");
        g_AneVirtualClientCls     = NSClassFromString(@"_ANEVirtualClient");
        g_AnePerfStatsCls         = NSClassFromString(@"_ANEPerformanceStats");
        g_AneSharedEventsCls      = NSClassFromString(@"_ANESharedEvents");
        g_AneSharedSignalEventCls = NSClassFromString(@"_ANESharedSignalEvent");
        g_AneSharedWaitEventCls   = NSClassFromString(@"_ANESharedWaitEvent");
        g_AneChainingRequestCls   = NSClassFromString(@"_ANEChainingRequest");
        g_AneDeviceInfoCls        = NSClassFromString(@"_ANEDeviceInfo");

        /* Make every class that ane_chain_create may put into
         * `inputs:` / `outputSets:` respond to `-symbolIndex`. NSArray
         * is included because the chain uses array-of-array structure
         * for inputs (each inner array is one "input set"). */
        install_symbol_index_on(g_AneIOSurfaceCls);
        install_symbol_index_on(NSClassFromString(@"_ANEIOSurfaceOutputSets"));
        install_symbol_index_on(NSClassFromString(@"NSArray"));

        ok = YES;
    });
    return ok;
}
