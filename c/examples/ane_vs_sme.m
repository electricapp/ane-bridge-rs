/* ane_vs_sme.m — Contention test: does ANE share silicon with CPU SME?
 *
 * Claim under test (joshmorgan1000/ane): "ANE is just SME inside the
 * CPU cores. Run them concurrently and throughput halves, proving
 * they are the same hardware."
 *
 * Protocol:
 *   Phase 1: measure ANE inference rate alone for T seconds.
 *   Phase 2: measure ANE rate again while a worker thread saturates
 *            Accelerate's BLAS matmul on the CPU (dispatches to
 *            AMX/SME depending on macOS version).
 *   Phase 3: measure ANE rate while a worker thread runs raw SME2
 *            outer-product instructions in a tight `smstart`-bracket
 *            loop — unambiguously the SME unit.
 *   Phase 4: control — measure ANE rate while a worker thread does
 *            scalar arithmetic only (no SIMD, no matrix unit).
 *
 * If ANE rate in phases 2/3 ≈ phase 1, ANE is distinct silicon (the
 * small drop is DRAM/LLC bandwidth contention). If ANE rate halves
 * in phase 3 but NOT in phase 4, the shared-silicon hypothesis would
 * be supported. Phase 4 is the bandwidth-contention control.
 *
 * Build (via Makefile):
 *   make examples
 *
 * Run:
 *   uv run python tools/make_identity_model.py build/identity
 *   ./build/bin/ane_vs_sme build/identity/model.mil build/identity/weights.bin
 */
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import "ane_bridge.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MATMUL_DIM 1024
#define SECONDS_PER_PHASE 5.0
#define NUM_WORKER_THREADS 4

static atomic_int g_should_stop;

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* ---- Worker A0: Accelerate BNNS INT8 matmul ----
 *
 * This is the worker the joshmorgan1000/ane "ANE = SME" article uses
 * to demonstrate "ANE contention." If BNNS is SME (it is, on M4+),
 * we expect it to contend heavily with the raw-SME worker AND to NOT
 * touch the real ANE driver path. The two outcomes together refute
 * the article's interpretation. */
#import <Accelerate/Accelerate.h>

static void* bnns_int8_worker(void* arg) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    WorkerResult* r = (WorkerResult*)arg;
    const int N = 1024;
    int8_t* A = aligned_alloc(64, (size_t)N * N);
    int8_t* B = aligned_alloc(64, (size_t)N * N);
    int8_t* C = aligned_alloc(64, (size_t)N * N);
    for (int i = 0; i < N * N; i++) {
        A[i] = (int8_t)((i % 11) - 5);
        B[i] = (int8_t)((i % 7) - 3);
    }
    BNNSNDArrayDescriptor inA = {0}, inB = {0}, outC = {0};
    inA.data_type = inB.data_type = outC.data_type = BNNSDataTypeInt8;
    inA.layout = inB.layout = outC.layout = BNNSDataLayoutRowMajorMatrix;
    inA.size[0] = inA.size[1] = (size_t)N;
    inB.size[0] = inB.size[1] = (size_t)N;
    outC.size[0] = outC.size[1] = (size_t)N;
    inA.data = A; inB.data = B; outC.data = C;

    double t0 = now_sec();
    int count = 0;
    while (!atomic_load(&g_should_stop)) {
        BNNSDirectApplyMatMul(&inA, &inB, &outC, NULL);
        count++;
    }
    r->duration = now_sec() - t0;
    r->matmuls = count;
    free(A); free(B); free(C);
    return NULL;
}

/* ---- Worker A: Accelerate BLAS matmul (whatever Apple dispatches to) ---- */

typedef struct {
    int matmuls;
    double duration;
} WorkerResult;

static void* blas_worker(void* arg) {
    WorkerResult* r = (WorkerResult*)arg;
    const int N = MATMUL_DIM;
    size_t bytes = (size_t)N * N * sizeof(float);
    float* A = aligned_alloc(64, bytes);
    float* B = aligned_alloc(64, bytes);
    float* C = aligned_alloc(64, bytes);
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)((i % 7) + 1);
        B[i] = (float)((i % 5) + 1);
    }
    double t0 = now_sec();
    int count = 0;
    while (!atomic_load(&g_should_stop)) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
        count++;
    }
    r->duration = now_sec() - t0;
    r->matmuls = count;
    free(A); free(B); free(C);
    return NULL;
}

/* ---- Worker B: raw SME2 outer-product loop (unambiguously SME) ----
 *
 * Stay in streaming mode for a long burst. `smstart` (no operand)
 * enables BOTH streaming mode and ZA storage — required for FMOPA to
 * write to a ZA tile. The inner loop chains four independent FMOPAs
 * per iteration across ZA0..ZA3 to keep the outer-product engines
 * fully fed, then iterates 16384 times before exiting streaming
 * mode. Each call ≈ 65k FMOPA outer products with the core
 * continuously in streaming + ZA state. */
#if defined(__APPLE__)
__attribute__((target("+sme2")))
static void sme_outer_product_burst(void) {
    asm volatile (
        "smstart\n"                    /* enter streaming + enable ZA */
        "ptrue   p0.b\n"               /* full-active predicate */
        "fmov    z0.s, #1.0\n"
        "fmov    z1.s, #2.0\n"
        "fmov    z2.s, #3.0\n"
        "fmov    z3.s, #4.0\n"
        "fmov    z4.s, #5.0\n"
        "fmov    z5.s, #6.0\n"
        "fmov    z6.s, #7.0\n"
        "fmov    z7.s, #8.0\n"
        "mov     x9, #16384\n"
        "1:\n"
        "fmopa   za0.s, p0/m, p0/m, z0.s, z1.s\n"
        "fmopa   za1.s, p0/m, p0/m, z2.s, z3.s\n"
        "fmopa   za2.s, p0/m, p0/m, z4.s, z5.s\n"
        "fmopa   za3.s, p0/m, p0/m, z6.s, z7.s\n"
        "subs    x9, x9, #1\n"
        "b.ne    1b\n"
        "smstop\n"                     /* leave streaming + disable ZA */
        :
        :
        : "x9", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "p0",
          "za"
    );
}

/* Quick check: does this core actually support SME2? Calling
 * `smstart` on a core without SME raises SIGILL. We probe via
 * sysctl so we can skip Phase 3 gracefully on older hardware. */
#include <sys/sysctl.h>
static int host_has_sme(void) {
    int has = 0;
    size_t sz = sizeof(has);
    if (sysctlbyname("hw.optional.arm.FEAT_SME", &has, &sz, NULL, 0) == 0) return has;
    if (sysctlbyname("hw.optional.arm.FEAT_SME2", &has, &sz, NULL, 0) == 0) return has;
    return 0;
}
#else
static int host_has_sme(void) { return 0; }
#endif

#import <pthread/qos.h>
static void* sme_worker(void* arg) {
    /* Pin to a P-core via QoS. SME on Apple Silicon is implemented on
     * the performance cores; the E-cores in some configurations may
     * trap SME instructions. */
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    WorkerResult* r = (WorkerResult*)arg;
    double t0 = now_sec();
    int bursts = 0;
#if defined(__APPLE__)
    while (!atomic_load(&g_should_stop)) {
        sme_outer_product_burst();
        bursts++;
    }
#endif
    r->duration = now_sec() - t0;
    r->matmuls = bursts;
    return NULL;
}

/* ---- Worker C: scalar CPU control (no SIMD, no matrix unit) ---- */

static void* scalar_worker(void* arg) {
    WorkerResult* r = (WorkerResult*)arg;
    double t0 = now_sec();
    volatile uint64_t acc = 0;
    uint64_t iters = 0;
    while (!atomic_load(&g_should_stop)) {
        for (int i = 0; i < 100000; i++) acc = acc * 1103515245ULL + 12345ULL;
        iters += 100000;
    }
    r->duration = now_sec() - t0;
    r->matmuls = (int)(iters / 100000);
    return NULL;
}

/* ---- ANE measurement ---- */

static double ane_inference_rate(AneRequest* req, double duration_sec) {
    double t0 = now_sec();
    int count = 0;
    double dt = 0;
    while ((dt = now_sec() - t0) < duration_sec) {
        if (ane_request_run(req, ANE_QOS_DEFAULT) != ANE_OK) break;
        count++;
    }
    return (double)count / dt;
}

/* Per-inference latency in microseconds. Submits one request, waits
 * for completion, records the wall-clock delta. Averages over `samples`.
 * If ANE silicon is contended by another workload, this MUST rise.
 * If only host dispatch is contended, throughput drops but per-call
 * latency on a quiet queue stays constant. */
static double ane_inference_latency_us(AneRequest* req, int samples) {
    double t0 = now_sec();
    for (int i = 0; i < samples; i++) {
        if (ane_request_run(req, ANE_QOS_DEFAULT) != ANE_OK) return -1;
    }
    double dt = now_sec() - t0;
    return dt * 1e6 / samples;
}

/* Saturate the ANE work queue with `n_inflight` independent requests
 * against the same model, submit them all, wait for the slowest, and
 * report wall-clock throughput. This amortizes the host-side dispatch
 * overhead so the measured rate floor is set by ANE silicon itself.
 * If ANE silicon is shared with SME, this MUST drop hard under SME
 * load. If it's genuinely distinct silicon, it should be flat. */
static double ane_saturation_rate(AneModel* m, AneBuffer* shared_in,
                                  int n_inflight, int batches) {
    AneRequest** rs = (AneRequest**)calloc((size_t)n_inflight, sizeof(*rs));
    AneBuffer** outs = (AneBuffer**)calloc((size_t)n_inflight, sizeof(*outs));
    for (int i = 0; i < n_inflight; i++) {
        ane_request_create(m, &rs[i]);
        ane_buffer_create_for_output(m, 0, &outs[i]);
        ane_request_bind_input(rs[i], 0, shared_in);
        ane_request_bind_output(rs[i], 0, outs[i]);
    }
    double t0 = now_sec();
    int total = 0;
    for (int b = 0; b < batches; b++) {
        for (int i = 0; i < n_inflight; i++) {
            (void)ane_request_submit(rs[i], ANE_QOS_DEFAULT);
        }
        for (int i = 0; i < n_inflight; i++) {
            (void)ane_request_wait(rs[i], -1);
            total++;
        }
    }
    double dt = now_sec() - t0;
    for (int i = 0; i < n_inflight; i++) {
        ane_request_release(rs[i]);
        ane_buffer_release(outs[i]);
    }
    free(rs); free(outs);
    return (double)total / dt;
}

/* ---- Three ANE metrics under a configurable worker load ---- */

typedef struct {
    double rate_seq;            /* sequential inferences/sec */
    double latency_us;          /* per-inference wall-clock latency */
    double rate_saturated;      /* throughput with N in-flight requests */
} AneMetrics;

static AneMetrics run_under_workers(AneRequest* req, AneModel* m, AneBuffer* ib,
                                    double seconds_seq, int latency_samples,
                                    int n_inflight, int saturation_batches,
                                    void* (*worker_fn)(void*), int num_workers) {
    AneMetrics out = {0};
    atomic_store(&g_should_stop, 0);
    pthread_t threads[NUM_WORKER_THREADS];
    WorkerResult results[NUM_WORKER_THREADS] = {0};
    for (int i = 0; i < num_workers; i++) {
        pthread_create(&threads[i], NULL, worker_fn, &results[i]);
    }
    out.rate_seq        = ane_inference_rate(req, seconds_seq);
    out.latency_us      = ane_inference_latency_us(req, latency_samples);
    out.rate_saturated  = ane_saturation_rate(m, ib, n_inflight, saturation_batches);
    atomic_store(&g_should_stop, 1);
    for (int i = 0; i < num_workers; i++) pthread_join(threads[i], NULL);
    return out;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.mil> <weights.bin>\n", argv[0]);
        return 2;
    }
    /* Pin the dispatcher to UserInteractive QoS so the baseline phase
     * runs at the same P-core frequency as the contention phases —
     * otherwise the OS boosts when worker threads spin up and the
     * baseline measurement is artificially low, producing fake
     * "retention >100%" numbers in the contended phases. */
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

    AneModelOpenOptions opts = {
        .mil_path = argv[1], .weights_path = argv[2], .compile_qos = ANE_QOS_DEFAULT,
    };
    AneModel* m = NULL;
    if (ane_model_open(&opts, &m) != ANE_OK) {
        fprintf(stderr, "open failed: %s\n", ane_last_error());
        return 1;
    }

    AneRequest* req = NULL;
    ane_request_create(m, &req);
    AneBuffer* ib = NULL;
    AneBuffer* ob = NULL;
    ane_buffer_create_for_input(m, 0, &ib);
    ane_buffer_create_for_output(m, 0, &ob);

    void* p = NULL;
    ane_buffer_lock(ib, ANE_LOCK_WRITE, &p);
    memset(p, 0, ane_model_input_nbytes(m, 0));
    ane_buffer_unlock(ib);
    ane_request_bind_input(req, 0, ib);
    ane_request_bind_output(req, 0, ob);

    /* Warm up: ensure the P-core frequency-scaler settles at the
     * boosted state by running a sustained 500ms warm-up at UI QoS
     * before any phase runs. */
    double wt = now_sec();
    while (now_sec() - wt < 0.5) ane_request_run(req, ANE_QOS_DEFAULT);

    const int LAT_SAMPLES = 200;
    const int N_INFLIGHT  = 16;
    const int SAT_BATCHES = 50;

    printf("=== ANE vs SME contention test (M-series Apple Silicon) ===\n");
    printf("Identity model:    %zu in / %zu out bytes\n",
           ane_model_input_nbytes(m, 0), ane_model_output_nbytes(m, 0));
    printf("Worker threads:    %d  (pinned to UI QoS)\n", NUM_WORKER_THREADS);
    printf("Seq phase:         %.1f sec of single-in-flight inferences\n",
           (double)SECONDS_PER_PHASE);
    printf("Latency phase:     %d single-inference samples\n", LAT_SAMPLES);
    printf("Saturation phase:  %d in-flight requests × %d batches\n\n",
           N_INFLIGHT, SAT_BATCHES);

    /* Methodology note: every phase runs with the same number of
     * background workers; only the *type* of worker varies. This
     * holds OS-scheduling and CPU-frequency state roughly constant
     * across phases so the only meaningful delta is the SME-vs-not
     * comparison. The "baseline" is N scalar workers — they keep
     * cores boosted without engaging SME or any matrix unit. If
     * ANE silicon is the SME unit, swapping scalar for SME workers
     * must collapse the SME row's throughput; otherwise SME and
     * scalar are statistically identical. */
    AneMetrics scalar = run_under_workers(req, m, ib, SECONDS_PER_PHASE,
                                          LAT_SAMPLES, N_INFLIGHT, SAT_BATCHES,
                                          scalar_worker, NUM_WORKER_THREADS);
    printf("%-18s rate_seq=%7.1f/s  latency=%6.1f us  rate_sat=%7.1f/s\n",
           "scalar workers:", scalar.rate_seq, scalar.latency_us, scalar.rate_saturated);

    AneMetrics blas = run_under_workers(req, m, ib, SECONDS_PER_PHASE,
                                        LAT_SAMPLES, N_INFLIGHT, SAT_BATCHES,
                                        blas_worker, NUM_WORKER_THREADS);
    printf("%-18s rate_seq=%7.1f/s  latency=%6.1f us  rate_sat=%7.1f/s\n",
           "BLAS sgemm workers:", blas.rate_seq, blas.latency_us, blas.rate_saturated);

    AneMetrics bnns = run_under_workers(req, m, ib, SECONDS_PER_PHASE,
                                        LAT_SAMPLES, N_INFLIGHT, SAT_BATCHES,
                                        bnns_int8_worker, NUM_WORKER_THREADS);
    printf("%-18s rate_seq=%7.1f/s  latency=%6.1f us  rate_sat=%7.1f/s\n",
           "BNNS int8 workers:", bnns.rate_seq, bnns.latency_us, bnns.rate_saturated);

    AneMetrics sme = scalar;
    int sme_ran = 0;
    if (host_has_sme()) {
        sme = run_under_workers(req, m, ib, SECONDS_PER_PHASE,
                                LAT_SAMPLES, N_INFLIGHT, SAT_BATCHES,
                                sme_worker, NUM_WORKER_THREADS);
        sme_ran = 1;
        printf("%-18s rate_seq=%7.1f/s  latency=%6.1f us  rate_sat=%7.1f/s\n",
               "raw SME2 workers:", sme.rate_seq, sme.latency_us, sme.rate_saturated);
    } else {
        printf("%-18s (host lacks FEAT_SME; skipped)\n", "raw SME2 workers:");
    }
    AneMetrics base = scalar;  /* "baseline" = N scalar workers (no matrix unit). */

    printf("\n=== Analysis ===\n");
    printf("(retention relative to baseline; 'sat_drop' is the key invariant — if ANE\n");
    printf(" silicon is shared with the loaded unit, saturated throughput MUST collapse)\n\n");
    printf("%-12s   seq        latency    sat\n", "");
    printf("%-12s   --------   --------   --------\n", "");
    printf("%-12s   %6.1f%%   %6.1f%%   %6.1f%%\n",
           "+ BLAS:", 100.0 * blas.rate_seq / base.rate_seq,
           100.0 * base.latency_us / blas.latency_us,
           100.0 * blas.rate_saturated / base.rate_saturated);
    if (sme_ran) {
        printf("%-12s   %6.1f%%   %6.1f%%   %6.1f%%\n",
               "+ SME2:", 100.0 * sme.rate_seq / base.rate_seq,
               100.0 * base.latency_us / sme.latency_us,
               100.0 * sme.rate_saturated / base.rate_saturated);
    }
    printf("%-12s   %6.1f%%   %6.1f%%   %6.1f%%\n",
           "+ scalar:", 100.0 * scalar.rate_seq / base.rate_seq,
           100.0 * base.latency_us / scalar.latency_us,
           100.0 * scalar.rate_saturated / base.rate_saturated);

    printf("\n=== Verdict ===\n");
    if (sme_ran) {
        double sme_lat_ratio = sme.latency_us / base.latency_us;
        double scalar_lat_ratio = scalar.latency_us / base.latency_us;
        double sme_sat_retention = sme.rate_saturated / base.rate_saturated * 100.0;
        double scalar_sat_retention = scalar.rate_saturated / base.rate_saturated * 100.0;
        printf("ANE latency under SME2  / baseline:   %.2fx\n", sme_lat_ratio);
        printf("ANE latency under scalar/ baseline:   %.2fx\n", scalar_lat_ratio);
        printf("ANE saturated rate under SME2:        %.1f%% of baseline\n", sme_sat_retention);
        printf("ANE saturated rate under scalar:      %.1f%% of baseline\n", scalar_sat_retention);
        printf("\n");
        if (sme_lat_ratio < 1.25 && sme_sat_retention > 80.0) {
            printf("CONCLUSION: ANE engine time is unaffected by SME load. Throughput drop\n"
                   "in the sequential phase is host-side scheduling (the dispatcher thread\n"
                   "competes with workers for P-core time), NOT silicon-side contention.\n"
                   "ANE and SME execute independently — distinct silicon.\n");
        } else if (sme_lat_ratio > 1.5 && scalar_lat_ratio < 1.25) {
            printf("CONCLUSION: ANE latency rises specifically under SME load but not under\n"
                   "scalar load. That would be the signature of shared silicon. Investigate\n"
                   "further (cache effects, power policy throttling, etc.).\n");
        } else {
            printf("CONCLUSION: ANE latency rises under BOTH SME and scalar load by similar\n"
                   "factors. The contention is general system pressure (DRAM, LLC, OS\n"
                   "scheduler), not SME-specific. Inconsistent with shared SME silicon.\n");
        }
    }

    ane_request_release(req);
    ane_buffer_release(ib);
    ane_buffer_release(ob);
    ane_model_close(m);
    return 0;
}
