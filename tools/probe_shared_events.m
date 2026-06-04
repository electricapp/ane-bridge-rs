/* probe_shared_events.m — Find a working configuration for shared
 * MTLSharedEvent ↔ ANE wait events.
 *
 * The framework crashes inside the worker thread on bad configurations
 * (bad eventType, bad agentMask, wrong code path), so each candidate
 * runs in a forked child. Parent collects child exit status and prints
 * a matrix.
 *
 * Build: see Makefile (`make probe-shared-events`)
 * Run:   ./build/bin/probe_shared_events build/identity/model.mil build/identity/weights.bin
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

/* Outcome codes carried through child exit status. */
enum { OUT_OK=0, OUT_OPEN_FAIL=10, OUT_EVENTS_CREATE_FAIL=11,
       OUT_ADD_WAIT_FAIL=12, OUT_SET_EVENTS_FAIL=13, OUT_SUBMIT_FAIL=14,
       OUT_WAIT_TIMEOUT=15, OUT_EVAL_FAIL=16, OUT_MISMATCH=17,
       OUT_CRASH_MARKER=64 };

/* In the child: try a single (eventType, agentMask, use_signal_too) configuration. */
static int try_config(const char* mil, const char* wts,
                      uint64_t event_type, uint64_t agent_mask,
                      int use_signal_too) {
    @autoreleasepool {
        AneModelOpenOptions o = { .mil_path=mil, .weights_path=wts, .compile_qos=ANE_QOS_DEFAULT };
        AneModel* model = NULL;
        if (ane_model_open(&o, &model) != ANE_OK) return OUT_OPEN_FAIL;

        size_t nbytes = ane_model_input_nbytes(model, 0);
        size_t nfloats = nbytes / sizeof(float);

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLSharedEvent> mtl_evt = [dev newSharedEvent];

        IOSurfaceRef in_s = make_iosurface(nbytes);
        IOSurfaceLock(in_s, 0, NULL);
        float* p = (float*)IOSurfaceGetBaseAddress(in_s);
        for (size_t i = 0; i < nfloats; i++) p[i] = (float)((i % 100) + 1);
        IOSurfaceUnlock(in_s, 0, NULL);

        AneBuffer* ib = NULL;
        ane_buffer_adopt_iosurface((void*)in_s, nbytes, &ib);
        CFRelease(in_s);
        AneBuffer* ob = NULL;
        ane_buffer_create_for_output(model, 0, &ob);

        AneSharedEvents* events = NULL;
        if (ane_shared_events_create(&events) != ANE_OK) return OUT_EVENTS_CREATE_FAIL;
        if (ane_shared_events_add_wait(events, 1, (__bridge void*)mtl_evt,
                                       (AneEventType)event_type) != ANE_OK)
            return OUT_ADD_WAIT_FAIL;
        if (use_signal_too) {
            ane_shared_events_add_signal(events, 2, 0,
                                         (AneEventType)event_type,
                                         (__bridge void*)mtl_evt, agent_mask);
        }

        AneRequest* req = NULL;
        ane_request_create(model, &req);
        ane_request_bind_input (req, 0, ib);
        ane_request_bind_output(req, 0, ob);
        if (ane_request_set_shared_events(req, events) != ANE_OK) return OUT_SET_EVENTS_FAIL;

        if (ane_request_submit(req, ANE_QOS_DEFAULT) != ANE_OK) return OUT_SUBMIT_FAIL;

        /* Signal the MTLSharedEvent from a Metal command buffer so ANE unblocks. */
        id<MTLCommandBuffer> cb = [q commandBuffer];
        [cb encodeSignalEvent:mtl_evt value:1];
        [cb commit];

        AneStatus rs = ane_request_wait(req, 3000);
        if (rs == ANE_ERR_TIMEOUT) return OUT_WAIT_TIMEOUT;
        if (rs != ANE_OK) return OUT_EVAL_FAIL;

        void* op = NULL;
        ane_buffer_lock(ob, ANE_LOCK_READ, &op);
        const float* of = (const float*)op;
        int bad = 0;
        for (size_t i = 0; i < nfloats; i++) {
            float exp = (float)((i % 100) + 1);
            if (fabsf(of[i] - exp) > 1e-3f) { bad = 1; break; }
        }
        ane_buffer_unlock(ob);
        ane_request_release(req);
        ane_shared_events_release(events);
        ane_buffer_release(ib);
        ane_buffer_release(ob);
        ane_model_close(model);
        return bad ? OUT_MISMATCH : OUT_OK;
    }
}

static const char* describe(int code) {
    switch (code) {
        case OUT_OK:                  return "OK";
        case OUT_OPEN_FAIL:           return "open failed";
        case OUT_EVENTS_CREATE_FAIL:  return "events_create failed";
        case OUT_ADD_WAIT_FAIL:       return "add_wait failed";
        case OUT_SET_EVENTS_FAIL:     return "set_shared_events failed";
        case OUT_SUBMIT_FAIL:         return "submit failed";
        case OUT_WAIT_TIMEOUT:        return "wait timed out (ANE blocked permanently)";
        case OUT_EVAL_FAIL:           return "eval failed (framework rejected)";
        case OUT_MISMATCH:            return "output mismatch";
        default:                      return "framework abort (crash)";
    }
}

static int run_child(const char* mil, const char* wts,
                     uint64_t event_type, uint64_t agent_mask, int signal_too) {
    pid_t pid = fork();
    if (pid == 0) {
        int rc = try_config(mil, wts, event_type, agent_mask, signal_too);
        _exit(rc);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status))   return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return OUT_CRASH_MARKER;
    return 99;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.mil> <weights.bin>\n", argv[0]);
        return 2;
    }

    uint64_t event_types[]  = {0, 1, 2, 3, 4, 8, 16, 32};
    uint64_t agent_masks[]  = {0, 1, 0xffull, 0xffffffffull, ~0ull};
    int      signal_combos[] = {0, 1};

    printf("Probing eventType × agentMask × {wait, wait+signal} ...\n");
    printf("(any non-OK row tells us what NOT to use; an OK row is a recipe.)\n\n");
    printf("%-12s %-22s %-9s %-3s   %s\n",
           "eventType", "agentMask", "signal?", "rc", "result");
    printf("--------------------------------------------------------------------\n");

    int ok_count = 0;
    for (size_t e = 0; e < sizeof(event_types)/sizeof(event_types[0]); e++) {
        for (size_t a = 0; a < sizeof(agent_masks)/sizeof(agent_masks[0]); a++) {
            for (size_t s = 0; s < sizeof(signal_combos)/sizeof(signal_combos[0]); s++) {
                int rc = run_child(argv[1], argv[2],
                                   event_types[e], agent_masks[a], signal_combos[s]);
                printf("%-12llu 0x%-20llx %-9s %3d   %s\n",
                       event_types[e], agent_masks[a],
                       signal_combos[s] ? "yes" : "no", rc, describe(rc));
                if (rc == OUT_OK) ok_count++;
            }
        }
    }
    printf("\nworking configurations: %d\n", ok_count);
    return ok_count > 0 ? 0 : 1;
}
