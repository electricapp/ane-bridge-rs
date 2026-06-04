/* explore_unknowns.m — Probe the 9 unexplored corners of
 * AppleNeuralEngine.framework documented in the README discussion.
 *
 * Targets:
 *   1. _ANEChainingRequest        — multi-model chained dispatch
 *   2. _ANESharedEvents / _ANESharedSignalEvent / _ANESharedWaitEvent — sync primitives
 *   3. _ANEPerformanceStats + the `perfStats:` request param
 *   4. _ANEVirtualClient          — per-process virtualized access
 *   5. queue depth (advertised 127)
 *   6. weightsBuffer: per-request param  — dynamic weights without reload
 *   7. procedureIndex: per-request param — multi-entry-point models
 *   8. weights as NSDictionary keys      — multi-blob naming
 *   9. file-based _ANEModel modelAtURL:key:
 *
 * Build:
 *   xcrun clang -O0 -fobjc-arc -fmodules -o /tmp/explore_unknowns \
 *     tools/explore_unknowns.m -framework Foundation -framework IOSurface -ldl
 *
 * Run:
 *   /tmp/explore_unknowns
 *
 * Output is verbose — pipe through `tee /tmp/explore.log` for review.
 */
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <stdio.h>

static void hr(const char* title) {
    printf("\n================================================================\n");
    printf("== %s\n", title);
    printf("================================================================\n");
}

static void list_all_ane_classes(void) {
    hr("All Obj-C classes whose name contains \"ANE\"");
    unsigned int n = 0;
    Class* all = objc_copyClassList(&n);
    /* Collect matching names. */
    NSMutableArray* names = [NSMutableArray array];
    for (unsigned int i = 0; i < n; i++) {
        const char* nm = class_getName(all[i]);
        if (!nm) continue;
        if (strstr(nm, "ANE")) {
            [names addObject:[NSString stringWithUTF8String:nm]];
        }
    }
    free(all);
    [names sortUsingSelector:@selector(compare:)];
    for (NSString* s in names) printf("  %s\n", s.UTF8String);
    printf("  (%lu total)\n", (unsigned long)names.count);
}

static void dump_methods(const char* prefix, Class c) {
    unsigned int count = 0;
    Method* methods = class_copyMethodList(c, &count);
    NSMutableArray* lines = [NSMutableArray array];
    for (unsigned int i = 0; i < count; i++) {
        SEL sel = method_getName(methods[i]);
        const char* types = method_getTypeEncoding(methods[i]);
        [lines addObject:[NSString stringWithFormat:@"  %s%s  ::  %s",
                                                    prefix, sel_getName(sel),
                                                    types ? types : "?"]];
    }
    free(methods);
    [lines sortUsingSelector:@selector(compare:)];
    for (NSString* s in lines) printf("%s\n", s.UTF8String);
}

static void dump_class(const char* name) {
    Class c = NSClassFromString([NSString stringWithUTF8String:name]);
    printf("\n---------- %s ----------\n", name);
    if (!c) { printf("  NOT FOUND\n"); return; }
    Class sup = class_getSuperclass(c);
    printf("  superclass: %s\n", sup ? class_getName(sup) : "(nil)");
    unsigned int pc = 0;
    __unsafe_unretained Protocol** prot = class_copyProtocolList(c, &pc);
    if (pc) {
        printf("  protocols:");
        for (unsigned int i = 0; i < pc; i++) printf(" %s", protocol_getName(prot[i]));
        printf("\n");
    }
    free(prot);
    printf("  -- instance methods --\n");
    dump_methods("-", c);
    printf("  -- class methods --\n");
    dump_methods("+", object_getClass((id)c));
}

/* Find selectors on a class whose name contains any of the given needles.
 * Searches both instance- and meta-class. Used to discover newly added
 * variants like requestWithInputs:...perfStats: vs alternatives. */
static void find_selectors(const char* className, NSArray<NSString*>* needles) {
    Class c = NSClassFromString([NSString stringWithUTF8String:className]);
    if (!c) { printf("  %s: NOT FOUND\n", className); return; }
    Class metas[] = { c, object_getClass((id)c) };
    const char* tags[]  = { "-", "+" };
    for (int m = 0; m < 2; m++) {
        unsigned int count = 0;
        Method* methods = class_copyMethodList(metas[m], &count);
        for (unsigned int i = 0; i < count; i++) {
            const char* sel = sel_getName(method_getName(methods[i]));
            NSString* selS = [NSString stringWithUTF8String:sel];
            for (NSString* needle in needles) {
                if ([selS rangeOfString:needle
                                options:NSCaseInsensitiveSearch].location != NSNotFound) {
                    printf("    %s[%s %s]\n", tags[m], className, sel);
                    break;
                }
            }
        }
        free(methods);
    }
}

/* Try every selector on a (live or class) object that takes zero args and
 * returns an object. Prints what comes back. Useful for poking at e.g.
 * `queueDepth`, `programHandle`, perfStats accessors. */
static void probe_zero_arg_getters(id target, NSArray<NSString*>* selNames) {
    for (NSString* nm in selNames) {
        SEL s = NSSelectorFromString(nm);
        if (![target respondsToSelector:s]) {
            printf("    %s: (does not respond)\n", nm.UTF8String);
            continue;
        }
        NSMethodSignature* sig = [target methodSignatureForSelector:s];
        if (!sig) { printf("    %s: (no signature)\n", nm.UTF8String); continue; }
        const char* ret = [sig methodReturnType];
        @try {
            if (ret[0] == '@') {
                id (*fn)(id, SEL) = (id (*)(id, SEL))objc_msgSend;
                id r = fn(target, s);
                printf("    %s -> (id) %s\n", nm.UTF8String,
                       r ? [[r description] UTF8String] : "nil");
            } else if (ret[0] == 'q' || ret[0] == 'l' || ret[0] == 'i') {
                NSInteger (*fn)(id, SEL) = (NSInteger (*)(id, SEL))objc_msgSend;
                NSInteger r = fn(target, s);
                printf("    %s -> (int) %ld\n", nm.UTF8String, (long)r);
            } else if (ret[0] == 'Q' || ret[0] == 'L' || ret[0] == 'I') {
                NSUInteger (*fn)(id, SEL) = (NSUInteger (*)(id, SEL))objc_msgSend;
                NSUInteger r = fn(target, s);
                printf("    %s -> (uint) %lu\n", nm.UTF8String, (unsigned long)r);
            } else if (ret[0] == 'B' || ret[0] == 'c') {
                BOOL (*fn)(id, SEL) = (BOOL (*)(id, SEL))objc_msgSend;
                BOOL r = fn(target, s);
                printf("    %s -> (bool) %s\n", nm.UTF8String, r ? "YES" : "NO");
            } else {
                printf("    %s -> (return-type '%s', skipped)\n", nm.UTF8String, ret);
            }
        } @catch (NSException* e) {
            printf("    %s -> EXCEPTION: %s\n", nm.UTF8String,
                   [[e description] UTF8String]);
        }
    }
}

int main(void) {
    @autoreleasepool {
        if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
                    RTLD_NOW)) {
            fprintf(stderr, "dlopen failed\n"); return 1;
        }

        list_all_ane_classes();

        /* Item 1, 2, 3, 4: dump the 4 new classes (if they exist) */
        hr("New class shapes");
        const char* targets[] = {
            "_ANEChainingRequest",
            "_ANESharedEvents",
            "_ANESharedSignalEvent",
            "_ANESharedWaitEvent",
            "_ANEPerformanceStats",
            "_ANEVirtualClient",
            "_ANEModel",
            "_ANEProgramForEvaluation",
            "_ANEProcedure",
            "_ANEProcedureSignature",
            NULL,
        };
        for (int i = 0; targets[i]; i++) dump_class(targets[i]);

        /* Item 6, 7, 8: hunt for selectors on _ANERequest and friends */
        hr("Selector search across the request/model classes");
        const char* searchClasses[] = {
            "_ANERequest", "_ANEChainingRequest",
            "_ANEClient",  "_ANEVirtualClient",
            "_ANEModel",   "_ANEInMemoryModel",
            "_ANEInMemoryModelDescriptor", "_ANEModelDescriptor",
            NULL,
        };
        NSArray<NSString*>* keywords = @[
            @"weights", @"procedure", @"perfStat", @"perfStats",
            @"chain", @"event", @"signal", @"wait",
            @"queueDepth", @"queue", @"programHandle", @"handle",
            @"modelAtURL", @"modelAt", @"atURL", @"key:",
            @"virtual", @"unload", @"compile", @"load",
        ];
        for (int i = 0; searchClasses[i]; i++) {
            printf("\n  ----- %s -----\n", searchClasses[i]);
            find_selectors(searchClasses[i], keywords);
        }

        /* Item 5: queue depth — probe the model + client for getters. */
        hr("Zero-arg getters on _ANEClient (sharedConnection, current)");
        Class clientCls = NSClassFromString(@"_ANEClient");
        if (clientCls) {
            id client = nil;
            SEL shared = NSSelectorFromString(@"sharedConnection");
            if ([clientCls respondsToSelector:shared]) {
                id (*fn)(id, SEL) = (id (*)(id, SEL))objc_msgSend;
                client = fn(clientCls, shared);
            }
            if (!client) {
                /* Fall back to initWithRestrictedAccessAllowed:NO */
                id alloced = [clientCls alloc];
                SEL initS = NSSelectorFromString(@"initWithRestrictedAccessAllowed:");
                if ([alloced respondsToSelector:initS]) {
                    id (*fn)(id, SEL, BOOL) = (id (*)(id, SEL, BOOL))objc_msgSend;
                    client = fn(alloced, initS, NO);
                }
            }
            printf("  client = %s\n", client ? [[client description] UTF8String] : "nil");
            if (client) {
                probe_zero_arg_getters(client, @[
                    @"queueDepth", @"maxQueueDepth", @"clientID",
                    @"isRestricted", @"deviceID", @"programHandle",
                    @"systemQueueDepth",
                ]);
            }
        } else {
            printf("  _ANEClient not found\n");
        }

        /* If _ANEModel exists, what zero-arg getters does it expose? */
        hr("Zero-arg getters on _ANEModel meta + a sample instance");
        Class modelCls = NSClassFromString(@"_ANEModel");
        if (modelCls) {
            printf("  -- on the Class itself --\n");
            probe_zero_arg_getters((id)modelCls, @[
                @"version", @"defaultKey",
            ]);
            /* We cannot easily construct one without a compiled bundle on disk;
             * skip the instance probe in this exploration pass. */
        } else printf("  _ANEModel: NOT FOUND\n");

        /* Item 9: scan _ANEModel's class methods for any modelAt*: variant. */
        hr("_ANEModel class methods (full)");
        if (modelCls) dump_methods("+", object_getClass((id)modelCls));
        else printf("  _ANEModel: NOT FOUND\n");

        printf("\nDONE.\n");
    }
    return 0;
}
