/* chain_identity.c — Run the identity model via a single-step chain.
 *
 * Validates `ane_chain_create` + `ane_chain_prepare` + `ane_chain_enqueue`
 * end-to-end. Should print the same byte-for-byte round-trip as the
 * direct-eval identity example. */
#include "ane_bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.mil> <weights.bin>\n", argv[0]);
        return 2;
    }

    AneModelOpenOptions o = {
        .mil_path = argv[1], .weights_path = argv[2], .compile_qos = ANE_QOS_DEFAULT,
    };
    AneModel* model = NULL;
    AneStatus st = ane_model_open(&o, &model);
    if (st != ANE_OK) {
        fprintf(stderr, "open failed (%d): %s\n", st, ane_last_error()); return 1;
    }
    size_t nbytes_in  = ane_model_input_nbytes(model, 0);
    size_t nbytes_out = ane_model_output_nbytes(model, 0);
    size_t n_floats   = nbytes_in / sizeof(float);

    AneBuffer* ib = NULL;
    ane_buffer_create_for_input(model, 0, &ib);
    AneBuffer* ob = NULL;
    ane_buffer_create_for_output(model, 0, &ob);

    void* ip = NULL;
    ane_buffer_lock(ib, ANE_LOCK_WRITE, &ip);
    float* fp = (float*)ip;
    for (size_t i = 0; i < n_floats; i++) fp[i] = (float)((i % 100) + 1);
    ane_buffer_unlock(ib);

    AneRequest* req = NULL;
    ane_request_create(model, &req);
    ane_request_bind_input(req, 0, ib);
    ane_request_bind_output(req, 0, ob);

    AneChainStep step = {
        .request = req,
        .lb_input_symbol_id  = 0,
        .lb_output_symbol_id = 0,
        .fw_enqueue_delay = 0,
        .memory_pool_id = 0,
    };
    AneChain* chain = NULL;
    st = ane_chain_create(&step, 1, &chain);
    if (st != ANE_OK) { fprintf(stderr, "chain_create (%d): %s\n", st, ane_last_error()); return 1; }
    fprintf(stderr,"chain_create OK\n");

    st = ane_chain_prepare(chain, ANE_QOS_DEFAULT);
    if (st != ANE_OK) { fprintf(stderr, "chain_prepare (%d): %s\n", st, ane_last_error()); return 1; }
    fprintf(stderr,"chain_prepare OK\n");

    st = ane_chain_enqueue(chain, ANE_QOS_DEFAULT);
    if (st != ANE_OK) { fprintf(stderr, "chain_enqueue (%d): %s\n", st, ane_last_error()); return 1; }
    fprintf(stderr,"chain_enqueue OK\n");

    void* op = NULL;
    ane_buffer_lock(ob, ANE_LOCK_READ, &op);
    float* of = (float*)op;
    size_t matches = 0, bad = 0;
    float max_err = 0;
    for (size_t i = 0; i < n_floats; i++) {
        float expected = (float)((i % 100) + 1);
        float e = of[i] - expected;
        if (e < 0) e = -e;
        if (e > max_err) max_err = e;
        if (e < 1e-3) matches++; else bad++;
    }
    ane_buffer_unlock(ob);

    fprintf(stderr,"input bytes:  %zu\n", nbytes_in);
    fprintf(stderr,"output bytes: %zu\n", nbytes_out);
    fprintf(stderr,"matches:      %zu / %zu\n", matches, n_floats);
    fprintf(stderr,"mismatches:   %zu\n", bad);
    fprintf(stderr,"max err:      %.6f\n", max_err);
    fprintf(stderr,"%s: chained dispatch reproduces direct-eval identity.\n",
           bad == 0 ? "PASS" : "FAIL");

    ane_chain_release(chain);
    ane_request_release(req);
    ane_buffer_release(ib);
    ane_buffer_release(ob);
    ane_model_close(model);
    return bad == 0 ? 0 : 1;
}
