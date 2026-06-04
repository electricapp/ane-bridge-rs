/* identity.c — Minimal end-to-end test of ane-bridge.
 *
 * Loads <mil_path> + <weights_path>, declares a single fp32 input/output of
 * shape [1, 64, 1, 16], fills the input with a ramp, runs once synchronously
 * (run = submit+wait), and prints a few output values + the eval latency.
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
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);

    AneModelOpenOptions opts = {
        .mil_path = argv[1],
        .weights_path = argv[2],
        .compile_qos = ANE_QOS_DEFAULT,
    };

    AneModel* model = NULL;
    AneStatus s = ane_model_open(&opts, &model);
    if (s != ANE_OK) {
        fprintf(stderr, "open failed: %s\n", ane_last_error());
        return 1;
    }
    printf("[open] OK  inputs=%d outputs=%d  in_bytes=%zu out_bytes=%zu\n",
           ane_model_num_inputs(model), ane_model_num_outputs(model),
           ane_model_input_nbytes(model, 0), ane_model_output_nbytes(model, 0));

    AneRequest* req = NULL;
    s = ane_request_create(model, &req);
    if (s != ANE_OK) {
        fprintf(stderr, "request create failed: %s\n", ane_last_error());
        return 1;
    }

    size_t n_in = ane_model_input_nbytes(model, 0);
    size_t n_out = ane_model_output_nbytes(model, 0);
    float* in_buf = (float*)malloc(n_in);
    float* out_buf = (float*)malloc(n_out);
    for (size_t i = 0; i < n_in / sizeof(float); i++) {
        in_buf[i] = (float)i * 0.01f;
    }

    /* Use the byte-ptr fast path. */
    s = ane_request_set_input_bytes(req, 0, in_buf, n_in);
    if (s != ANE_OK) {
        fprintf(stderr, "set_input failed: %s\n", ane_last_error());
        return 1;
    }

    /* Warm up, then time 10 runs. */
    for (int i = 0; i < 3; i++) {
        s = ane_request_run(req, ANE_QOS_DEFAULT);
        if (s != ANE_OK) {
            fprintf(stderr, "warmup run failed: %s\n", ane_last_error());
            return 1;
        }
    }
    uint64_t t0 = mach_absolute_time();
    int N = 10;
    for (int i = 0; i < N; i++) {
        s = ane_request_run(req, ANE_QOS_DEFAULT);
        if (s != ANE_OK) {
            fprintf(stderr, "run failed: %s\n", ane_last_error());
            return 1;
        }
    }
    double total_ms = ms_since(t0, tb);

    s = ane_request_get_output_bytes(req, 0, out_buf, n_out);
    if (s != ANE_OK) {
        fprintf(stderr, "get_output failed: %s\n", ane_last_error());
        return 1;
    }

    int ok = 1;
    for (size_t i = 0; i < 8; i++) {
        float exp = (float)i * 0.01f;
        float got = out_buf[i];
        float diff = got - exp;
        if (diff < 0) {
            diff = -diff;
        }
        if (diff > 1e-2f) {
            ok = 0;
        }
    }
    printf("[run] %d iters in %.2f ms (avg %.3f ms/iter)\n", N, total_ms, total_ms / N);
    printf("[check] first 8 values: %s\n", ok ? "OK" : "MISMATCH");
    printf("  in[0..3]:  %.4f %.4f %.4f %.4f\n", in_buf[0], in_buf[1], in_buf[2], in_buf[3]);
    printf("  out[0..3]: %.4f %.4f %.4f %.4f\n", out_buf[0], out_buf[1], out_buf[2], out_buf[3]);

    ane_request_release(req);
    ane_model_close(model);
    free(in_buf);
    free(out_buf);
    return ok ? 0 : 1;
}
