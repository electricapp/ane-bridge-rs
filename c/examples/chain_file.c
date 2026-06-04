/* chain_file.c — Verify chained dispatch with a file-opened model.
 *
 * Pairs ane_model_open_file (which returns an `_ANEModel` —
 * NSSecureCoding-conformant) with ane_chain_*. The bench fixture's
 * `.mlmodelc` bundle is the input.
 *
 * Build: `make examples`
 * Run:   ./build/bin/chain_file build/bench/bench.mlmodelc
 */
#include "ane_bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.mlmodelc>\n", argv[0]);
        return 2;
    }
    AneModelFileOpenOptions o = {
        .model_url = argv[1],
        .cache_key = "ane-bridge-chain-test",
        .cache_url_identifier = NULL,
        .identifier_source = ANE_IDENT_DEFAULT,
        .compile_qos = ANE_QOS_DEFAULT,
    };
    AneModel* model = NULL;
    AneStatus st = ane_model_open_file(&o, &model);
    if (st != ANE_OK) {
        fprintf(stderr, "open_file failed (%d): %s\n", (int)st, ane_last_error());
        return 1;
    }
    fprintf(stderr, "open_file OK; inputs=%d outputs=%d\n", ane_model_num_inputs(model),
            ane_model_num_outputs(model));

    AneBuffer* ib = NULL;
    ane_buffer_create_for_input(model, 0, &ib);
    AneBuffer* ob = NULL;
    ane_buffer_create_for_output(model, 0, &ob);

    AneRequest* req = NULL;
    ane_request_create(model, &req);
    ane_request_bind_input(req, 0, ib);
    ane_request_bind_output(req, 0, ob);

    AneChainStep step = {.request = req};
    AneChain* chain = NULL;
    st = ane_chain_create(&step, 1, &chain);
    if (st != ANE_OK) {
        fprintf(stderr, "chain_create (%d): %s\n", (int)st, ane_last_error());
        return 1;
    }
    fprintf(stderr, "chain_create OK\n");

    st = ane_chain_prepare(chain, ANE_QOS_DEFAULT);
    fprintf(stderr, "chain_prepare: %d  err=%s\n", (int)st, ane_last_error());

    st = ane_chain_enqueue(chain, ANE_QOS_DEFAULT);
    fprintf(stderr, "chain_enqueue: %d  err=%s\n", (int)st, ane_last_error());

    ane_chain_release(chain);
    ane_request_release(req);
    ane_buffer_release(ib);
    ane_buffer_release(ob);
    ane_model_close(model);
    return st == ANE_OK ? 0 : 1;
}
