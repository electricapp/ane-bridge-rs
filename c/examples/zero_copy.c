/* zero_copy.c — Demonstrate the IOSurface bind path.
 *
 * Unlike `identity.c` which memcpys data via `ane_request_set/get_*_bytes`,
 * this example allocates IOSurface-backed input/output buffers, writes
 * directly into the mapped memory, binds them once, and then runs the
 * model repeatedly — no per-call host-side copies on the boundary
 * between user code and the ANE.
 */
#include "ane_bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <mach/mach_time.h>

static double ms_since(uint64_t t0, mach_timebase_info_data_t tb) {
    return (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <mil_path> <weights_path>\n", argv[0]);
        return 1;
    }
    mach_timebase_info_data_t tb; mach_timebase_info(&tb);

    AneModelOpenOptions opts = {
        .mil_path = argv[1], .weights_path = argv[2],
        .compile_qos = ANE_QOS_DEFAULT,
    };

    AneModel* model = NULL;
    if (ane_model_open(&opts, &model) != ANE_OK) {
        fprintf(stderr, "open failed: %s\n", ane_last_error()); return 1;
    }

    AneBuffer* in_buf  = NULL;
    AneBuffer* out_buf = NULL;
    if (ane_buffer_create_for_input(model, 0, &in_buf)   != ANE_OK ||
        ane_buffer_create_for_output(model, 0, &out_buf) != ANE_OK) {
        fprintf(stderr, "buffer create failed: %s\n", ane_last_error()); return 1;
    }
    printf("[buf] in IOSurfaceID=%u  out IOSurfaceID=%u  nbytes=%zu/%zu\n",
        ane_buffer_iosurface_id(in_buf), ane_buffer_iosurface_id(out_buf),
        ane_buffer_nbytes(in_buf), ane_buffer_nbytes(out_buf));

    AneRequest* req = NULL;
    if (ane_request_create(model, &req) != ANE_OK) {
        fprintf(stderr, "request create failed: %s\n", ane_last_error()); return 1;
    }
    if (ane_request_bind_input (req, 0, in_buf)  != ANE_OK ||
        ane_request_bind_output(req, 0, out_buf) != ANE_OK) {
        fprintf(stderr, "bind failed: %s\n", ane_last_error()); return 1;
    }

    /* Write input ONCE into the mapped IOSurface. */
    size_t n_elem = ane_buffer_nbytes(in_buf) / sizeof(float);
    float* p = NULL;
    if (ane_buffer_lock(in_buf, ANE_LOCK_WRITE, (void**)&p) != ANE_OK) {
        fprintf(stderr, "in lock failed: %s\n", ane_last_error()); return 1;
    }
    for (size_t i = 0; i < n_elem; i++) p[i] = (float)i * 0.01f;
    ane_buffer_unlock(in_buf);

    /* Warm up + bench. */
    for (int i = 0; i < 3; i++) ane_request_run(req, ANE_QOS_DEFAULT);
    uint64_t t0 = mach_absolute_time();
    int N = 100;
    for (int i = 0; i < N; i++) {
        if (ane_request_run(req, ANE_QOS_DEFAULT) != ANE_OK) {
            fprintf(stderr, "run %d failed: %s\n", i, ane_last_error()); return 1;
        }
    }
    double total = ms_since(t0, tb);
    printf("[bench] zero-copy: %d iters in %.2f ms (avg %.3f ms/iter)\n",
           N, total, total / N);

    /* Read output directly out of the mapped IOSurface. */
    if (ane_buffer_lock(out_buf, ANE_LOCK_READ, (void**)&p) != ANE_OK) {
        fprintf(stderr, "out lock failed: %s\n", ane_last_error()); return 1;
    }
    int ok = 1;
    for (size_t i = 0; i < 8; i++) {
        float exp = (float)i * 0.01f;
        float diff = p[i] - exp; if (diff < 0) diff = -diff;
        if (diff > 1e-2f) ok = 0;
    }
    printf("[check] first 8 values: %s   (out[0..3]: %.4f %.4f %.4f %.4f)\n",
           ok ? "OK" : "MISMATCH", p[0], p[1], p[2], p[3]);
    ane_buffer_unlock(out_buf);

    ane_request_release(req);
    ane_buffer_release(in_buf);
    ane_buffer_release(out_buf);
    ane_model_close(model);
    return ok ? 0 : 1;
}
