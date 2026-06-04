/* probe_perfstats.m — Verify that _ANEPerformanceStats is actually
 * populated end-to-end by the AppleNeuralEngine framework when attached
 * to an _ANERequest via the bridge.
 *
 * Strategy:
 *   1. Open the identity model via the bridge's C API.
 *   2. For each candidate perfStatsMask (0, 1, 0xff, 0xffffffff), create
 *      a fresh AnePerfStats, attach it to a fresh AneRequest, bind
 *      buffers, run the request, then read:
 *        - hw_execution_ns
 *        - counters_nbytes (+ dump first bytes)
 *      Report which mask values change the observed values.
 *
 * Build:
 *   xcrun clang -O2 -fobjc-arc -Ic/include -framework Foundation \
 *     -framework IOSurface -Wl,-rpath,@executable_path/../lib \
 *     -Lbuild/lib -lane_bridge -o build/bin/probe_perfstats \
 *     tools/probe_perfstats.m
 *
 * Run (from repo root):
 *   ./build/bin/probe_perfstats build/identity/model.mil build/identity/weights.bin
 */
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <string.h>
#include "ane_bridge.h"

static void log_status(const char* what, AneStatus s) {
    if (s != ANE_OK) {
        fprintf(stderr, "  [ERR] %s -> status=%d: %s\n",
                what, (int)s, ane_last_error());
    }
}

static void hex_dump(const unsigned char* p, size_t n, size_t cap) {
    size_t k = n < cap ? n : cap;
    fprintf(stderr, "    first %zu bytes:", k);
    for (size_t i = 0; i < k; i++) fprintf(stderr, " %02x", p[i]);
    fprintf(stderr, "\n");
}

/* Run a single trial with a given mask. Returns hw_ns and counters_nbytes via out params. */
static void run_trial(AneModel* model, uint32_t mask,
                      uint64_t* out_hw_ns, size_t* out_counter_bytes) {
    fprintf(stderr, "\n=== Trial: perfStatsMask = 0x%08x ===\n", mask);

    AneStatus s = ane_model_set_perf_stats_mask(model, mask);
    log_status("ane_model_set_perf_stats_mask", s);
    uint32_t got = ane_model_get_perf_stats_mask(model);
    fprintf(stderr, "  get_perf_stats_mask -> 0x%08x (set=0x%08x, match=%s)\n",
            got, mask, (got == mask ? "YES" : "NO"));

    AnePerfStats* ps = NULL;
    s = ane_perf_stats_create(&ps);
    log_status("ane_perf_stats_create", s);
    if (!ps) { *out_hw_ns = 0; *out_counter_bytes = 0; return; }

    /* Sanity: before eval. */
    uint64_t hw_before = ane_perf_stats_hw_execution_ns(ps);
    size_t   cb_before = ane_perf_stats_counters_nbytes(ps);
    fprintf(stderr, "  pre-eval: hw_ns=%llu  counters_nbytes=%zu\n",
            (unsigned long long)hw_before, cb_before);

    AneRequest* req = NULL;
    s = ane_request_create(model, &req);
    log_status("ane_request_create", s);
    if (!req) { ane_perf_stats_release(ps); return; }

    /* Bind buffers for the identity model: 1 input, 1 output. */
    int32_t ni = ane_model_num_inputs(model);
    int32_t no = ane_model_num_outputs(model);
    fprintf(stderr, "  model: %d inputs / %d outputs\n", ni, no);

    AneBuffer* in_buf  = NULL;
    AneBuffer* out_buf = NULL;
    s = ane_buffer_create_for_input(model, 0, &in_buf);
    log_status("ane_buffer_create_for_input", s);
    s = ane_buffer_create_for_output(model, 0, &out_buf);
    log_status("ane_buffer_create_for_output", s);

    /* Fill the input with something non-zero so the framework doesn't
     * potentially short-circuit. */
    void* p = NULL;
    s = ane_buffer_lock(in_buf, ANE_LOCK_WRITE, &p);
    log_status("ane_buffer_lock(in)", s);
    if (p) {
        size_t n = ane_buffer_nbytes(in_buf);
        memset(p, 0x3c, n);  /* FP16 ~ 1.0 */
    }
    ane_buffer_unlock(in_buf);

    s = ane_request_bind_input(req, 0, in_buf);
    log_status("ane_request_bind_input", s);
    s = ane_request_bind_output(req, 0, out_buf);
    log_status("ane_request_bind_output", s);

    /* Attach perf stats BEFORE the eval. */
    s = ane_request_set_perf_stats(req, ps);
    log_status("ane_request_set_perf_stats", s);

    /* Run synchronously. */
    s = ane_request_run(req, ANE_QOS_DEFAULT);
    log_status("ane_request_run", s);
    if (s != ANE_OK) {
        fprintf(stderr, "  request_last_error: %s\n", ane_request_last_error(req));
    }

    /* Post-eval readout. */
    uint64_t hw_ns = ane_perf_stats_hw_execution_ns(ps);
    size_t   cb    = ane_perf_stats_counters_nbytes(ps);
    fprintf(stderr, "  POST-eval: hw_ns=%llu  counters_nbytes=%zu\n",
            (unsigned long long)hw_ns, cb);

    if (cb > 0) {
        unsigned char buf[64];
        size_t copied = ane_perf_stats_counters_copy(ps, buf, sizeof(buf));
        fprintf(stderr, "    copied=%zu\n", copied);
        hex_dump(buf, copied, sizeof(buf));
    }

    *out_hw_ns = hw_ns;
    *out_counter_bytes = cb;

    ane_buffer_release(in_buf);
    ane_buffer_release(out_buf);
    ane_request_release(req);
    ane_perf_stats_release(ps);
}

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: %s <model.mil> <weights.bin>\n", argv[0]);
            return 1;
        }
        fprintf(stderr, "ane_bridge_version: %s\n", ane_bridge_version());

        AneModelOpenOptions opts = {
            .mil_path     = argv[1],
            .weights_path = argv[2],
            .compile_qos  = ANE_QOS_DEFAULT,
        };
        AneModel* model = NULL;
        AneStatus s = ane_model_open(&opts, &model);
        if (s != ANE_OK || !model) {
            fprintf(stderr, "ane_model_open FAILED: status=%d err=%s\n",
                    (int)s, ane_last_error());
            return 2;
        }
        fprintf(stderr, "model opened: cached=%s\n",
                ane_model_was_cached(model) ? "yes" : "no");

        /* Confirm initial mask. */
        uint32_t initial = ane_model_get_perf_stats_mask(model);
        fprintf(stderr, "initial perfStatsMask = 0x%08x\n", initial);

        uint32_t masks[] = { 0u, 1u, 0xffu, 0xffffffffu };
        const char* names[] = { "0 (off)", "1", "0xff", "0xffffffff" };
        uint64_t hw[4] = {0};
        size_t   cb[4] = {0};
        for (int i = 0; i < 4; i++) {
            run_trial(model, masks[i], &hw[i], &cb[i]);
        }

        fprintf(stderr, "\n=== Summary ===\n");
        fprintf(stderr, "  mask              hw_ns                counters_nbytes\n");
        for (int i = 0; i < 4; i++) {
            fprintf(stderr, "  %-16s  %-20llu %zu\n",
                    names[i], (unsigned long long)hw[i], cb[i]);
        }

        bool any_hw     = (hw[0] | hw[1] | hw[2] | hw[3]) != 0;
        bool any_counter= (cb[0] | cb[1] | cb[2] | cb[3]) != 0;
        bool mask_diff_hw = !(hw[0] == hw[1] && hw[1] == hw[2] && hw[2] == hw[3]);
        bool mask_diff_cb = !(cb[0] == cb[1] && cb[1] == cb[2] && cb[2] == cb[3]);

        fprintf(stderr, "\nVerdict:\n");
        fprintf(stderr, "  hw_execution_ns ever non-zero?   %s\n", any_hw ? "YES" : "NO");
        fprintf(stderr, "  counters_nbytes ever non-zero?   %s\n", any_counter ? "YES" : "NO");
        fprintf(stderr, "  mask changes hw_ns observation?  %s\n", mask_diff_hw ? "YES" : "NO");
        fprintf(stderr, "  mask changes counter bytes?      %s\n", mask_diff_cb ? "YES" : "NO");

        ane_model_close(model);
    }
    return 0;
}
