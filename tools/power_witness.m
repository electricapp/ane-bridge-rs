/* power_witness.m — Drive BNNS-INT8 then real-ANE inference in
 * sequence with timestamped phase markers on stderr. Pair with
 * `sudo powermetrics --samplers ane,cpu_power -i 500` to attribute
 * the resulting energy traces to each phase.
 *
 * Hypothesis under test: BNNS uses the CPU; the ANE driver path
 * uses the ANE coprocessor. Falsifiable: ANE energy during the
 * BNNS phase must stay at ~0 mW (BNNS is CPU); CPU energy during
 * the ANE phase must drop while ANE energy rises. */
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import "ane_bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

#define MARK(label) do { \
    fprintf(stderr, "PHASE %s %.6f\n", label, now_sec()); \
    fflush(stderr); \
} while (0)

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.mil> <weights.bin>\n", argv[0]);
        return 2;
    }

    /* ---- Open the real ANE model up-front ---- */
    AneModelOpenOptions opts = {
        .mil_path = argv[1], .weights_path = argv[2], .compile_qos = ANE_QOS_DEFAULT,
    };
    AneModel* m = NULL;
    if (ane_model_open(&opts, &m) != ANE_OK) {
        fprintf(stderr, "ane_model_open failed: %s\n", ane_last_error());
        return 1;
    }
    AneRequest* req = NULL;
    ane_request_create(m, &req);
    AneBuffer* ib = NULL;
    AneBuffer* ob = NULL;
    ane_buffer_create_for_input(m, 0, &ib);
    ane_buffer_create_for_output(m, 0, &ob);
    ane_request_bind_input(req, 0, ib);
    ane_request_bind_output(req, 0, ob);

    /* ---- Set up a CPU-side matrix-multiply rig via `cblas_sgemm` ----
     *
     * The original BNNS-INT8 path Josh's harness uses is deprecated
     * starting macOS 15 (Apple now points at `BNNSGraph*`, a
     * fundamentally different API). `cblas_sgemm` is the
     * non-deprecated Accelerate equivalent and dispatches to the
     * same underlying SME/AMX backend on M-series — exactly the
     * "is this BNNS-on-CPU or is this the ANE?" question, no API
     * suppression required. */
    const int N = 1024;
    size_t bytes = (size_t)N * N * sizeof(float);
    float* A = aligned_alloc(64, bytes);
    float* B = aligned_alloc(64, bytes);
    float* C = aligned_alloc(64, bytes);
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)((i % 7) + 1);
        B[i] = (float)((i % 5) + 1);
    }

    /* Warm-up so first-call cold paths don't pollute samples. */
    for (int i = 0; i < 5; i++) ane_request_run(req, ANE_QOS_DEFAULT);
    for (int i = 0; i < 5; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
    }

    /* ---- Phase 0: idle baseline ---- */
    MARK("IDLE_START");
    sleep(3);
    MARK("IDLE_END");

    /* ---- Phase 1: pure BNNS INT8, no ANE call ---- */
    MARK("CPU_MATMUL_START");
    double t0 = now_sec();
    int cpu_n = 0;
    while (now_sec() - t0 < 5.0) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
        cpu_n++;
    }
    MARK("CPU_MATMUL_END");
    fprintf(stderr, "PHASE CPU_MATMUL_COUNT %d\n", cpu_n);

    /* ---- Phase 2: idle again (separates the phases in samples) ---- */
    MARK("GAP_START");
    sleep(3);
    MARK("GAP_END");

    /* ---- Phase 3: pure ANE driver path, no BNNS call ---- */
    MARK("ANE_START");
    t0 = now_sec();
    int ane_n = 0;
    while (now_sec() - t0 < 5.0) {
        ane_request_run(req, ANE_QOS_DEFAULT);
        ane_n++;
    }
    MARK("ANE_END");
    fprintf(stderr, "PHASE ANE_COUNT %d\n", ane_n);

    free(A); free(B); free(C);
    ane_request_release(req);
    ane_buffer_release(ib);
    ane_buffer_release(ob);
    ane_model_close(m);
    return 0;
}
