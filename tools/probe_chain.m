/* probe_chain.m — Validate chained dispatch end-to-end.
 *
 *   Phase A: bare chain (no events) — does `_ANEChainingRequest` +
 *            `doPrepareChaining…` + `doEnqueueSets…` actually run an
 *            identity model and return correct bytes?
 *   Phase B: chain + MTLSharedEvent signal — does ANE actually emit
 *            the signal when the step finishes (Metal command buffer
 *            waiting on the same MTLSharedEvent value should unblock)?
 *
 * Each phase runs in a forked child so a framework abort doesn't kill
 * the parent's matrix walk.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>
#import <IOSurface/IOSurfaceRef.h>
#import "ane_bridge.h"

#include <sys/wait.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

enum {
    OUT_OK             = 0,
    OUT_OPEN_FAIL      = 10,
    OUT_BUF_FAIL       = 11,
    OUT_CHAIN_CREATE   = 12,
    OUT_PREPARE_FAIL   = 13,
    OUT_ENQUEUE_FAIL   = 14,
    OUT_WAIT_FAIL      = 15,
    OUT_MISMATCH       = 16,
    OUT_METAL_TIMEOUT  = 17,
    OUT_CRASH_MARKER   = 64,
};

static IOSurfaceRef make_iosurface(size_t nbytes) {
    size_t W, H;
    if (nbytes <= 16384) { W = nbytes ? nbytes : 1; H = 1; }
    else { W = 4096; while (nbytes % W) W /= 2; H = nbytes / W; }
    NSDictionary* p = @{
        (id)kIOSurfaceWidth:@(W), (id)kIOSurfaceHeight:@(H),
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(W),
        (id)kIOSurfaceAllocSize:@(nbytes ? nbytes : 1), (id)kIOSurfacePixelFormat:@0,
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)p);
}

static int phase_a_bare(const char* mil, const char* wts) {
    @autoreleasepool {
        AneModelOpenOptions o = { .mil_path=mil, .weights_path=wts, .compile_qos=ANE_QOS_DEFAULT };
        AneModel* model = NULL;
        if (ane_model_open(&o, &model) != ANE_OK) return OUT_OPEN_FAIL;

        size_t nbytes = ane_model_input_nbytes(model, 0);
        size_t nf = nbytes / sizeof(float);

        IOSurfaceRef ins = make_iosurface(nbytes);
        IOSurfaceLock(ins, 0, NULL);
        float* p = (float*)IOSurfaceGetBaseAddress(ins);
        for (size_t i = 0; i < nf; i++) p[i] = (float)((i % 100) + 1);
        IOSurfaceUnlock(ins, 0, NULL);

        AneBuffer* ib = NULL;
        if (ane_buffer_adopt_iosurface((void*)ins, nbytes, &ib) != ANE_OK) return OUT_BUF_FAIL;
        CFRelease(ins);

        AneBuffer* ob = NULL;
        if (ane_buffer_create_for_output(model, 0, &ob) != ANE_OK) return OUT_BUF_FAIL;

        AneRequest* rq = NULL;
        ane_request_create(model, &rq);
        ane_request_bind_input(rq, 0, ib);
        ane_request_bind_output(rq, 0, ob);

        AneChainStep steps[1] = { {
            .request = rq,
            .lb_input_symbol_id = 0,
            .lb_output_symbol_id = 0,
            .fw_enqueue_delay = 0,
            .memory_pool_id = 0,
        } };
        AneChain* chain = NULL;
        if (ane_chain_create(steps, 1, &chain) != ANE_OK) return OUT_CHAIN_CREATE;

        if (ane_chain_prepare(chain, ANE_QOS_DEFAULT) != ANE_OK) return OUT_PREPARE_FAIL;
        if (ane_chain_enqueue(chain, ANE_QOS_DEFAULT) != ANE_OK) return OUT_ENQUEUE_FAIL;
        if (ane_chain_wait(chain, 5000) != ANE_OK) return OUT_WAIT_FAIL;

        void* op = NULL;
        ane_buffer_lock(ob, ANE_LOCK_READ, &op);
        const float* of = (const float*)op;
        int bad = 0;
        for (size_t i = 0; i < nf; i++) {
            if (fabsf(of[i] - (float)((i % 100) + 1)) > 1e-3f) { bad = 1; break; }
        }
        ane_buffer_unlock(ob);
        ane_chain_release(chain);
        ane_request_release(rq);
        ane_buffer_release(ib);
        ane_buffer_release(ob);
        ane_model_close(model);
        return bad ? OUT_MISMATCH : OUT_OK;
    }
}

static int phase_b_with_signal(const char* mil, const char* wts,
                               uint64_t event_type, uint64_t agent_mask) {
    @autoreleasepool {
        AneModelOpenOptions o = { .mil_path=mil, .weights_path=wts, .compile_qos=ANE_QOS_DEFAULT };
        AneModel* model = NULL;
        if (ane_model_open(&o, &model) != ANE_OK) return OUT_OPEN_FAIL;

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLSharedEvent> mtl_evt = [dev newSharedEvent];

        size_t nbytes = ane_model_input_nbytes(model, 0);
        size_t nf = nbytes / sizeof(float);
        IOSurfaceRef ins = make_iosurface(nbytes);
        IOSurfaceLock(ins, 0, NULL);
        float* p = (float*)IOSurfaceGetBaseAddress(ins);
        for (size_t i = 0; i < nf; i++) p[i] = (float)((i % 100) + 1);
        IOSurfaceUnlock(ins, 0, NULL);

        AneBuffer* ib = NULL;
        ane_buffer_adopt_iosurface((void*)ins, nbytes, &ib);
        CFRelease(ins);
        AneBuffer* ob = NULL;
        ane_buffer_create_for_output(model, 0, &ob);

        AneSharedEvents* ev = NULL;
        ane_shared_events_create(&ev);
        ane_shared_events_add_signal(ev, /*value=*/42, /*symbol_index=*/0,
                                     (AneEventType)event_type,
                                     (__bridge void*)mtl_evt, agent_mask);

        AneRequest* rq = NULL;
        ane_request_create(model, &rq);
        ane_request_bind_input(rq, 0, ib);
        ane_request_bind_output(rq, 0, ob);
        ane_request_set_shared_events(rq, ev);

        AneChainStep steps[1] = { {
            .request = rq,
            .lb_input_symbol_id = 0,
            .lb_output_symbol_id = 0,
            .fw_enqueue_delay = 0,
            .memory_pool_id = 0,
        } };
        AneChain* chain = NULL;
        if (ane_chain_create(steps, 1, &chain) != ANE_OK) return OUT_CHAIN_CREATE;
        if (ane_chain_prepare(chain, ANE_QOS_DEFAULT) != ANE_OK) return OUT_PREPARE_FAIL;

        /* Metal command buffer waits for value=42 then signals 43 so we
         * can detect completion via Metal. */
        id<MTLCommandBuffer> cb = [q commandBuffer];
        [cb encodeWaitForEvent:mtl_evt value:42];
        [cb encodeSignalEvent:mtl_evt value:43];
        [cb commit];

        if (ane_chain_enqueue(chain, ANE_QOS_DEFAULT) != ANE_OK) return OUT_ENQUEUE_FAIL;
        if (ane_chain_wait(chain, 5000) != ANE_OK) return OUT_WAIT_FAIL;

        /* Wait up to 2s for Metal to observe value=43. */
        struct timespec start, now;
        clock_gettime(CLOCK_MONOTONIC, &start);
        int saw_43 = 0;
        for (;;) {
            uint64_t cur = mtl_evt.signaledValue;
            if (cur >= 43) { saw_43 = 1; break; }
            clock_gettime(CLOCK_MONOTONIC, &now);
            double dt = (now.tv_sec - start.tv_sec) + (now.tv_nsec - start.tv_nsec) / 1e9;
            if (dt > 2.0) break;
            usleep(1000);
        }

        ane_chain_release(chain);
        ane_request_release(rq);
        ane_shared_events_release(ev);
        ane_buffer_release(ib);
        ane_buffer_release(ob);
        ane_model_close(model);
        return saw_43 ? OUT_OK : OUT_METAL_TIMEOUT;
    }
}

static const char* describe(int code) {
    switch (code) {
        case OUT_OK:             return "OK";
        case OUT_OPEN_FAIL:      return "open failed";
        case OUT_BUF_FAIL:       return "buffer failed";
        case OUT_CHAIN_CREATE:   return "chain_create failed";
        case OUT_PREPARE_FAIL:   return "prepare failed (framework rejected)";
        case OUT_ENQUEUE_FAIL:   return "enqueue failed";
        case OUT_WAIT_FAIL:      return "wait failed";
        case OUT_MISMATCH:       return "output mismatch (chain ran but wrong bytes)";
        case OUT_METAL_TIMEOUT:  return "Metal never observed the signal";
        case OUT_CRASH_MARKER:   return "framework abort (crash)";
        default:                 return "unknown";
    }
}

static int run_in_child(int (*fn)(const char*, const char*),
                        const char* mil, const char* wts) {
    pid_t pid = fork();
    if (pid == 0) _exit(fn(mil, wts));
    int status = 0; waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : OUT_CRASH_MARKER;
}

static int run_phase_b(const char* mil, const char* wts, uint64_t et, uint64_t am) {
    pid_t pid = fork();
    if (pid == 0) _exit(phase_b_with_signal(mil, wts, et, am));
    int status = 0; waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : OUT_CRASH_MARKER;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.mil> <weights.bin>\n", argv[0]);
        return 2;
    }

    printf("Phase A: bare chain (no events)\n");
    int a = run_in_child(phase_a_bare, argv[1], argv[2]);
    printf("  rc=%d  %s\n\n", a, describe(a));

    printf("Phase B: chain + signal event matrix\n");
    printf("%-12s %-22s %-3s   %s\n", "eventType", "agentMask", "rc", "result");
    printf("------------------------------------------------------\n");
    uint64_t event_types[] = {0, 1, 2, 3, 4, 8, 16};
    uint64_t agent_masks[] = {0, 1, 0xffull, ~0ull};
    int worked = 0;
    for (size_t e = 0; e < sizeof(event_types)/sizeof(event_types[0]); e++) {
        for (size_t a2 = 0; a2 < sizeof(agent_masks)/sizeof(agent_masks[0]); a2++) {
            int rc = run_phase_b(argv[1], argv[2], event_types[e], agent_masks[a2]);
            printf("%-12llu 0x%-20llx %3d   %s\n",
                   event_types[e], agent_masks[a2], rc, describe(rc));
            if (rc == OUT_OK) worked++;
        }
    }
    printf("\nphase B working configurations: %d\n", worked);
    return (a == OUT_OK && worked > 0) ? 0 : 1;
}
