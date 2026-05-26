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

Class g_AneDescriptorCls = nil;
Class g_AneInMemoryCls   = nil;
Class g_AneRequestCls    = nil;
Class g_AneIOSurfaceCls  = nil;
Class g_AneClientCls     = nil;

static id stub_getUUID(id self, SEL _cmd) {
    (void)self; (void)_cmd;
    static dispatch_once_t once;
    static id u = nil;
    dispatch_once(&once, ^{
        u = [[NSUUID alloc] init];
    });
    return u;
}

/* Public: kept for back-compat with the header, but the actual install
 * happens inside `ane_private_load`'s once block. Calling this directly
 * after `ane_private_load` is a no-op. */
void ane_private_install_uuid_stub(void) {
    if (!g_AneInMemoryCls) return;
    SEL sel = sel_registerName("getUUID");
    if ([g_AneInMemoryCls instancesRespondToSelector:sel]) return;
    /* `class_addMethod` itself is documented thread-safe in the
     * Objective-C runtime, but we still gate it via the once block in
     * `ane_private_load` so the call ordering is well-defined. */
    class_addMethod(g_AneInMemoryCls, sel, (IMP)stub_getUUID, "@@:");
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
        g_AneDescriptorCls = d;
        g_AneInMemoryCls   = im;
        g_AneRequestCls    = rq;
        g_AneIOSurfaceCls  = iio;
        g_AneClientCls     = cl;
        ok = YES;
    });
    return ok;
}
