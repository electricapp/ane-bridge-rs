/* inspect_classes.m — Dump every public selector + property on the
 * private AppleNeuralEngine classes we care about. One-off discovery
 * tool: helps us find shape-introspection APIs we don't currently
 * call.
 *
 * Build (also wired into top-level Makefile via `make inspect`):
 *   xcrun clang -O0 -fno-objc-arc -o build/bin/inspect \
 *     tools/inspect_classes.m -framework Foundation -ldl
 */
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <dlfcn.h>
#import <stdio.h>

static void dump_methods(const char* prefix, Class c) {
    unsigned int count = 0;
    Method* methods = class_copyMethodList(c, &count);
    /* Sort by selector for stable diffs. */
    for (unsigned int i = 0; i < count; i++) {
        for (unsigned int j = i + 1; j < count; j++) {
            if (strcmp(sel_getName(method_getName(methods[i])),
                       sel_getName(method_getName(methods[j]))) > 0) {
                Method t = methods[i]; methods[i] = methods[j]; methods[j] = t;
            }
        }
    }
    for (unsigned int i = 0; i < count; i++) {
        SEL sel = method_getName(methods[i]);
        const char* types = method_getTypeEncoding(methods[i]);
        printf("  %s%s  ::  %s\n", prefix, sel_getName(sel), types ? types : "?");
    }
    free(methods);
}

static void dump_properties(Class c) {
    unsigned int count = 0;
    objc_property_t* props = class_copyPropertyList(c, &count);
    for (unsigned int i = 0; i < count; i++) {
        const char* name  = property_getName(props[i]);
        const char* attrs = property_getAttributes(props[i]);
        printf("  @property %s  ::  %s\n", name, attrs ? attrs : "?");
    }
    free(props);
}

static void dump_ivars(Class c) {
    unsigned int count = 0;
    Ivar* ivars = class_copyIvarList(c, &count);
    for (unsigned int i = 0; i < count; i++) {
        const char* name = ivar_getName(ivars[i]);
        const char* type = ivar_getTypeEncoding(ivars[i]);
        printf("  ivar %s  ::  %s\n", name, type ? type : "?");
    }
    free(ivars);
}

static void dump_class(const char* name) {
    Class c = NSClassFromString([NSString stringWithUTF8String:name]);
    printf("\n========== %s ==========\n", name);
    if (!c) { printf("  NOT FOUND\n"); return; }
    printf("  superclass: %s\n", class_getName(class_getSuperclass(c)));

    printf("\n--- Instance methods ---\n");
    dump_methods("-", c);

    printf("\n--- Class methods ---\n");
    dump_methods("+", object_getClass((id)c));

    printf("\n--- Properties ---\n");
    dump_properties(c);

    printf("\n--- Ivars ---\n");
    dump_ivars(c);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
                RTLD_NOW)) {
        fprintf(stderr, "dlopen failed\n");
        return 1;
    }

    const char* classes[] = {
        "_ANEInMemoryModelDescriptor",
        "_ANEInMemoryModel",
        "_ANEModelDescriptor",
        "_ANEModel",
        "_ANERequest",
        "_ANEIOSurfaceObject",
        "_ANEClient",
        "_ANEProgramForEvaluation",
        "_ANEProcedure",
        "_ANEProcedureSignature",
        NULL,
    };
    for (int i = 0; classes[i]; i++) dump_class(classes[i]);
    return 0;
}
