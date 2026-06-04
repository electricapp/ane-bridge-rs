/* gpu_to_ane.m — Zero-copy GPU → ANE handoff via shared IOSurface.
 *
 * Verifies end-to-end:
 *   1. An IOSurface can be wrapped as an MTLBuffer for GPU writes AND
 *      adopted by ane-bridge for ANE reads — same kernel object, no copy.
 *   2. Bytes Metal writes through the MTLBuffer are visible to ANE when
 *      the framework runs an identity model on the adopted surface.
 *
 * Synchronization between the GPU and ANE runs in this example is done
 * via `[MTLCommandBuffer waitUntilCompleted]` on the host side. The
 * private `_ANESharedWaitEvent` / `_ANESharedSignalEvent` interop with
 * MTLSharedEvent is exposed through the bridge API but the framework
 * currently aborts inside the eval thread when an MTLSharedEvent is
 * attached to a request — that integration path is left as future
 * work; the API surface is in place once the framework-side recipe is
 * understood.
 *
 * Build (via top-level Makefile):
 *   make examples
 *
 * Run:
 *   uv run python tools/make_identity_model.py build/identity
 *   ./build/bin/gpu_to_ane build/identity/model.mil build/identity/weights.bin
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>
#import <IOSurface/IOSurfaceRef.h>
#import "ane_bridge.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static IOSurfaceRef make_iosurface(size_t nbytes) {
    size_t W, H;
    if (nbytes == 0) {
        W = 1;
        H = 1;
    } else if (nbytes <= 16384) {
        W = nbytes;
        H = 1;
    } else {
        W = 4096;
        while (nbytes % W) {
            W /= 2;
        }
        H = nbytes / W;
    }
    NSDictionary* props = @{
        (id)kIOSurfaceWidth: @(W),
        (id)kIOSurfaceHeight: @(H),
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(W),
        (id)kIOSurfaceAllocSize: @(nbytes ? nbytes : 1),
        (id)kIOSurfacePixelFormat: @0,
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

__attribute__((noreturn)) static void die(const char* what, AneStatus st) {
    fprintf(stderr, "%s: status=%d, last_error=%s\n", what, (int)st, ane_last_error());
    exit(1);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.mil> <weights.bin>\n", argv[0]);
        return 2;
    }

    @autoreleasepool {
        AneModelOpenOptions opts = {
            .mil_path = argv[1],
            .weights_path = argv[2],
            .compile_qos = ANE_QOS_DEFAULT,
        };
        AneModel* model = NULL;
        AneStatus st = ane_model_open(&opts, &model);
        if (st != ANE_OK) {
            die("ane_model_open", st);
        }

        size_t nbytes_in = ane_model_input_nbytes(model, 0);
        size_t nbytes_out = ane_model_output_nbytes(model, 0);
        size_t n_floats = nbytes_in / sizeof(float);
        if (nbytes_in != nbytes_out) {
            fprintf(stderr, "identity model expected; got in=%zu out=%zu\n", nbytes_in, nbytes_out);
            return 1;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "no Metal device\n");
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];

        IOSurfaceRef src_surface = make_iosurface(nbytes_in);
        IOSurfaceLock(src_surface, (IOSurfaceLockOptions)0, NULL);
        float* src = (float*)IOSurfaceGetBaseAddress(src_surface);
        for (size_t i = 0; i < n_floats; i++) {
            src[i] = (float)((i % 100) + 1); /* fp16-exact */
        }
        IOSurfaceUnlock(src_surface, (IOSurfaceLockOptions)0, NULL);
        id<MTLBuffer> src_mtl = [device newBufferWithBytesNoCopy:src
                                                          length:nbytes_in
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        IOSurfaceRef ane_in_surface = make_iosurface(nbytes_in);
        AneBuffer* ane_in_buf = NULL;
        st = ane_buffer_adopt_iosurface((void*)ane_in_surface, nbytes_in, &ane_in_buf);
        if (st != ANE_OK) {
            die("ane_buffer_adopt_iosurface", st);
        }
        CFRelease(ane_in_surface);

        IOSurfaceRef ane_in_back = (IOSurfaceRef)ane_buffer_iosurface_ref(ane_in_buf);
        if (ane_in_back != ane_in_surface) {
            fprintf(stderr, "iosurface_ref round-trip mismatch\n");
            return 1;
        }
        void* ane_in_base = IOSurfaceGetBaseAddress(ane_in_back);
        id<MTLBuffer> ane_in_mtl = [device newBufferWithBytesNoCopy:ane_in_base
                                                             length:nbytes_in
                                                            options:MTLResourceStorageModeShared
                                                        deallocator:nil];

        AneBuffer* ane_out_buf = NULL;
        st = ane_buffer_create_for_output(model, 0, &ane_out_buf);
        if (st != ANE_OK) {
            die("ane_buffer_create_for_output", st);
        }

        AneRequest* req = NULL;
        st = ane_request_create(model, &req);
        if (st != ANE_OK) {
            die("ane_request_create", st);
        }
        if ((st = ane_request_bind_input(req, 0, ane_in_buf)) != ANE_OK) {
            die("bind_input", st);
        }
        if ((st = ane_request_bind_output(req, 0, ane_out_buf)) != ANE_OK) {
            die("bind_output", st);
        }

        /* GPU writes through MTLBuffer that aliases ANE's input IOSurface. */
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:src_mtl
                 sourceOffset:0
                     toBuffer:ane_in_mtl
            destinationOffset:0
                         size:nbytes_in];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        st = ane_request_run(req, ANE_QOS_DEFAULT);
        if (st != ANE_OK) {
            die("ane_request_run", st);
        }

        void* out_ptr = NULL;
        ane_buffer_lock(ane_out_buf, ANE_LOCK_READ, &out_ptr);
        const float* out = (const float*)out_ptr;
        size_t matches = 0, mismatches = 0;
        float max_abs_err = 0.0f;
        for (size_t i = 0; i < n_floats; i++) {
            float expected = (float)((i % 100) + 1);
            float err = fabsf(out[i] - expected);
            if (err > max_abs_err) {
                max_abs_err = err;
            }
            if (err < 1e-3f) {
                matches++;
            } else {
                mismatches++;
            }
        }
        ane_buffer_unlock(ane_out_buf);

        printf("=== GPU → ANE handoff via shared IOSurface ===\n");
        printf("model           : %s\n", argv[1]);
        printf("input bytes     : %zu\n", nbytes_in);
        printf("fp32 elements   : %zu\n", n_floats);
        printf("matches         : %zu / %zu\n", matches, n_floats);
        printf("mismatches      : %zu\n", mismatches);
        printf("max abs error   : %.6f\n", max_abs_err);
        int rc = (mismatches == 0) ? 0 : 1;
        printf("%s: GPU writes to a shared IOSurface are visible to ANE.\n",
               rc == 0 ? "PASS" : "FAIL");

        /* Bonus: attaching shared events to a direct request returns
         * `Unsupported` at the API instead of crashing. */
        AneSharedEvents* ev = NULL;
        ane_shared_events_create(&ev);
        id<MTLSharedEvent> mtl_evt = [device newSharedEvent];
        ane_shared_events_add_wait(ev, 1, (__bridge void*)mtl_evt, ANE_EVT_DEFAULT);
        AneRequest* req2 = NULL;
        ane_request_create(model, &req2);
        ane_request_bind_input(req2, 0, ane_in_buf);
        ane_request_set_shared_events(req2, ev);
        AneStatus reject = ane_request_run(req2, ANE_QOS_DEFAULT);
        if (reject == ANE_ERR_UNSUPPORTED) {
            printf("PASS: shared events on a direct request reject with UNSUPPORTED\n"
                   "      (use ane_chain_* for shared-event sync).\n");
        } else {
            printf("FAIL: expected UNSUPPORTED rejection, got status=%d\n", (int)reject);
            rc = 1;
        }
        ane_request_release(req2);
        ane_shared_events_release(ev);

        ane_request_release(req);
        ane_buffer_release(ane_in_buf);
        ane_buffer_release(ane_out_buf);
        ane_model_close(model);
        CFRelease(src_surface);
        return rc;
    }
}
