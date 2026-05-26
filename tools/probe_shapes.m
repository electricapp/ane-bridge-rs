/* probe_shapes.m — Open a real model and call every promising
 * shape-introspection method on `_ANEInMemoryModel` after compile +
 * load. Whatever each returns gets pretty-printed; the goal is to
 * find a method that exposes input/output shapes so we can replace
 * the user-provided `TensorSpec` with framework-derived truth.
 *
 * Build (also via `make probe-shapes`):
 *   xcrun clang -O0 -fno-objc-arc -o build/bin/probe_shapes \
 *     tools/probe_shapes.m -framework Foundation -framework IOSurface -ldl
 *
 * Run:
 *   ./build/bin/probe_shapes <mil> <weights>
 */
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <stdio.h>

static id g_AneDescriptorCls;
static id g_AneInMemoryCls;

static id stub_getUUID(id self, SEL _cmd) {
    (void)self; (void)_cmd;
    static id u = nil;
    if (!u) u = [[NSUUID alloc] init];
    return u;
}

static void describe(const char* label, id obj) {
    printf("\n--- %s ---\n", label);
    if (!obj) { printf("  (nil)\n"); return; }
    printf("  class: %s\n", class_getName([obj class]));
    NSString* desc = [obj description];
    printf("  desc:  %s\n", [desc UTF8String]);
}

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: %s <mil> <weights>\n", argv[0]);
            return 1;
        }
        if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
                    RTLD_NOW)) {
            fprintf(stderr, "dlopen failed\n"); return 1;
        }
        g_AneDescriptorCls = (id)NSClassFromString(@"_ANEInMemoryModelDescriptor");
        g_AneInMemoryCls   = (id)NSClassFromString(@"_ANEInMemoryModel");
        Class im = NSClassFromString(@"_ANEInMemoryModel");
        SEL uuidSel = sel_registerName("getUUID");
        if (![im instancesRespondToSelector:uuidSel])
            class_addMethod(im, uuidSel, (IMP)stub_getUUID, "@@:");

        NSData* mil = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]];
        NSData* wts = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[2]]];
        NSDictionary* wdict = @{ @"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wts} };

        id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            (Class)g_AneDescriptorCls, sel_getUid("modelWithMILText:weights:optionsPlist:"),
            mil, wdict, nil);
        if (!desc) { fprintf(stderr, "descriptor nil\n"); return 1; }
        [desc retain];

        id mdl = ((id(*)(Class, SEL, id))objc_msgSend)(
            (Class)g_AneInMemoryCls, sel_getUid("inMemoryModelWithDescriptor:"), desc);
        if (!mdl) { fprintf(stderr, "model nil\n"); return 1; }
        [mdl retain];

        /* Mirror our hex+dir setup so compile/load succeed. */
        id hxid = ((id(*)(id, SEL))objc_msgSend)(mdl, sel_getUid("hexStringIdentifier"));
        NSString* td = [NSTemporaryDirectory() stringByAppendingPathComponent:hxid];
        NSFileManager* fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wts writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        NSError* e = nil;
        BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
            mdl, sel_getUid("compileWithQoS:options:error:"), 21, @{}, &e);
        if (!ok) { fprintf(stderr, "compile fail: %s\n", [[e localizedDescription] UTF8String]); return 1; }
        ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
            mdl, sel_getUid("loadWithQoS:options:error:"), 21, @{}, &e);
        if (!ok) { fprintf(stderr, "load fail: %s\n", [[e localizedDescription] UTF8String]); return 1; }

        /* ===== Probe every shape-ish accessor we can find. ===== */
        describe("modelAttributes",
            ((id(*)(id, SEL))objc_msgSend)(mdl, sel_getUid("modelAttributes")));
        describe("program",
            ((id(*)(id, SEL))objc_msgSend)(mdl, sel_getUid("program")));
        describe("model (inner _ANEModel)",
            ((id(*)(id, SEL))objc_msgSend)(mdl, sel_getUid("model")));
        describe("descriptor",
            ((id(*)(id, SEL))objc_msgSend)(mdl, sel_getUid("descriptor")));

        /* procedureInfoForProcedureIndex: returns id — call for index 0. */
        id pinfo = ((id(*)(id, SEL, unsigned int))objc_msgSend)(
            mdl, sel_getUid("procedureInfoForProcedureIndex:"), 0);
        describe("procedureInfoForProcedureIndex:0", pinfo);

        id inIdx = ((id(*)(id, SEL, unsigned int))objc_msgSend)(
            mdl, sel_getUid("inputSymbolIndicesForProcedureIndex:"), 0);
        describe("inputSymbolIndicesForProcedureIndex:0", inIdx);

        id outIdx = ((id(*)(id, SEL, unsigned int))objc_msgSend)(
            mdl, sel_getUid("outputSymbolIndicesForProcedureIndex:"), 0);
        describe("outputSymbolIndicesForProcedureIndex:0", outIdx);

        /* If pinfo is a dictionary, dump each key separately so we
         * can see nested shape arrays. */
        if (pinfo && [pinfo isKindOfClass:[NSDictionary class]]) {
            printf("\n--- procedureInfo keys ---\n");
            NSDictionary* d = (NSDictionary*)pinfo;
            for (id k in d) {
                printf("  [%s] -> %s :: %s\n",
                    [[k description] UTF8String],
                    class_getName([d[k] class]),
                    [[d[k] description] UTF8String]);
            }
        }

        /* `_ANEModel` (inner) also has procedureInfo + symbol indices —
         * sometimes the inner has richer info than the wrapper. */
        id inner = ((id(*)(id, SEL))objc_msgSend)(mdl, sel_getUid("model"));
        if (inner) {
            describe("inner.procedureInfoForProcedureIndex:0",
                ((id(*)(id, SEL, unsigned int))objc_msgSend)(
                    inner, sel_getUid("procedureInfoForProcedureIndex:"), 0));
        }
    }
    return 0;
}
