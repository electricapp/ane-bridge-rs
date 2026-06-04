/* probe_lifecycle.m — End-to-end validation of bridge APIs that were
 * never exercised against the live AppleNeuralEngine.framework.
 *
 * Each check runs in a forked child so a framework abort/SIGABRT/SEGV
 * doesn't kill the harness; the parent reports the exit code and any
 * captured stdout/stderr from the child.
 *
 * Probed APIs (one per check):
 *   1) ane_cache_exists_for_hash
 *   2) ane_cache_purge_for_hash
 *   3) ane_model_new_instance (run a request on each)
 *   4) ane_decompress_weights (zlib-compressed input + garbage input)
 *   5) ane_realtime_task_begin / _end (begin/end/begin/end)
 *   6) ane_session_hint_create + ane_model_apply_session_hint (all kinds)
 *   7) ane_model_open_file against build/bench/bench.mlmodelc
 *   8) Identity accessors (uuid, source_url, model_url, key,
 *      cache_url_identifier, identifier_source, is_virtual_client)
 */
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <IOSurface/IOSurfaceRef.h>
#import "ane_bridge.h"

#include <sys/wait.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

/* ----- shared utilities ------------------------------------------------ */

/* Open the identity model. Returns NULL on failure (prints reason). */
static AneModel* open_identity(const char* mil, const char* wts) {
    AneModelOpenOptions o = {
        .mil_path = mil,
        .weights_path = wts,
        .compile_qos = ANE_QOS_DEFAULT,
    };
    AneModel* m = NULL;
    AneStatus s = ane_model_open(&o, &m);
    if (s != ANE_OK) {
        const char* msg = ane_last_error();
        fprintf(stderr, "    open failed: status=%d msg=%s\n",
                (int)s, msg ? msg : "");
        return NULL;
    }
    return m;
}

/* Run a single fast-path inference on `model`. Returns 0 on byte-exact
 * round-trip, nonzero otherwise. */
static int run_one(AneModel* model, const char* tag) {
    size_t nb_in = ane_model_input_nbytes(model, 0);
    size_t nb_out = ane_model_output_nbytes(model, 0);
    if (nb_in == 0 || nb_out == 0) {
        fprintf(stderr, "    %s: zero nbytes (in=%zu out=%zu)\n",
                tag, nb_in, nb_out);
        return 1;
    }

    AneRequest* rq = NULL;
    if (ane_request_create(model, &rq) != ANE_OK) {
        fprintf(stderr, "    %s: request_create failed: %s\n",
                tag, ane_last_error());
        return 2;
    }

    size_t nf = nb_in / sizeof(float);
    float* in = (float*)malloc(nb_in);
    float* out = (float*)malloc(nb_out);
    for (size_t i = 0; i < nf; i++) in[i] = (float)((i % 100) + 1);

    AneStatus s = ane_request_set_input_bytes(rq, 0, in, nb_in);
    if (s != ANE_OK) {
        fprintf(stderr, "    %s: set_input_bytes: %s\n", tag, ane_last_error());
        free(in); free(out); ane_request_release(rq); return 3;
    }
    s = ane_request_run(rq, ANE_QOS_DEFAULT);
    if (s != ANE_OK) {
        fprintf(stderr, "    %s: run: status=%d msg=%s\n",
                tag, (int)s, ane_last_error());
        free(in); free(out); ane_request_release(rq); return 4;
    }
    s = ane_request_get_output_bytes(rq, 0, out, nb_out);
    if (s != ANE_OK) {
        fprintf(stderr, "    %s: get_output_bytes: %s\n",
                tag, ane_last_error());
        free(in); free(out); ane_request_release(rq); return 5;
    }
    int bad = 0;
    for (size_t i = 0; i < nf; i++) {
        if (fabsf(out[i] - in[i]) > 1e-3f) { bad = 1; break; }
    }
    free(in); free(out);
    ane_request_release(rq);
    return bad;
}

/* fork+waitpid wrapper. Returns child exit code (or 128|signal on crash). */
static int run_in_child(int (*fn)(void*), void* arg, const char* tag) {
    fflush(stdout); fflush(stderr);
    pid_t pid = fork();
    if (pid == 0) {
        int rc = fn(arg);
        fflush(stdout); fflush(stderr);
        _exit(rc);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) {
        int rc = WEXITSTATUS(status);
        printf("  [%s] exit=%d\n", tag, rc);
        return rc;
    }
    if (WIFSIGNALED(status)) {
        int sig = WTERMSIG(status);
        printf("  [%s] CRASHED with signal %d\n", tag, sig);
        return 128 + sig;
    }
    printf("  [%s] unknown exit status\n", tag);
    return -1;
}

/* ----- paths captured at startup --------------------------------------- */

static char g_mil[1024];
static char g_wts[1024];
static char g_bench[1024];

/* ----- 1) ane_cache_exists_for_hash ------------------------------------ */

static int probe_cache_exists(void* _unused) {
    (void)_unused;
    @autoreleasepool {
        AneModel* m = open_identity(g_mil, g_wts);
        if (!m) return 10;
        const char* hash = ane_model_program_id(m);
        if (!hash || hash[0] == 0) {
            printf("    program_id returned empty/NULL\n");
            ane_model_close(m);
            return 11;
        }
        printf("    program_id (hash) = %s\n", hash);
        char hash_copy[256];
        snprintf(hash_copy, sizeof(hash_copy), "%s", hash);

        bool exists_real = ane_cache_exists_for_hash(hash_copy);
        printf("    cache_exists_for_hash(real) = %s\n",
               exists_real ? "true" : "false");

        const char* zeros =
            "0000000000000000000000000000000000000000000000000000000000000000";
        bool exists_garbage = ane_cache_exists_for_hash(zeros);
        printf("    cache_exists_for_hash(garbage) = %s\n",
               exists_garbage ? "true" : "false");

        ane_model_close(m);
        if (!exists_real) return 20;     /* expected true */
        if (exists_garbage) return 21;   /* expected false */
        return 0;
    }
}

/* ----- 2) ane_cache_purge_for_hash ------------------------------------- */

static int probe_cache_purge(void* _unused) {
    (void)_unused;
    @autoreleasepool {
        AneModel* m = open_identity(g_mil, g_wts);
        if (!m) return 10;
        const char* hash = ane_model_program_id(m);
        if (!hash || hash[0] == 0) {
            printf("    program_id empty/NULL\n");
            ane_model_close(m);
            return 11;
        }
        char hash_copy[256];
        snprintf(hash_copy, sizeof(hash_copy), "%s", hash);
        printf("    program_id (hash) = %s\n", hash_copy);

        /* Close the model first; the framework may keep the cache pinned. */
        ane_model_close(m);

        bool before = ane_cache_exists_for_hash(hash_copy);
        printf("    exists before purge = %s\n", before ? "true" : "false");

        AneStatus ps = ane_cache_purge_for_hash(hash_copy);
        printf("    purge status        = %d (%s)\n",
               (int)ps, ane_last_error() ? ane_last_error() : "");

        bool after = ane_cache_exists_for_hash(hash_copy);
        printf("    exists after purge  = %s\n", after ? "true" : "false");

        if (ps != ANE_OK) return 30;
        if (before && after) return 31;  /* purge had no effect */
        return 0;
    }
}

/* ----- 3) ane_model_new_instance --------------------------------------- */

static int probe_new_instance(void* _unused) {
    (void)_unused;
    @autoreleasepool {
        AneModel* src = open_identity(g_mil, g_wts);
        if (!src) return 10;
        printf("    src opened\n");

        AneModel* inst = NULL;
        AneStatus s = ane_model_new_instance(src, NULL, ANE_QOS_DEFAULT, &inst);
        printf("    new_instance status = %d (%s)\n",
               (int)s, ane_last_error() ? ane_last_error() : "");
        if (s != ANE_OK || inst == NULL) {
            ane_model_close(src);
            return 40;
        }
        printf("    new_instance returned non-null handle\n");

        int r_src = run_one(src, "src");
        printf("    src request rc = %d\n", r_src);

        int r_inst = run_one(inst, "inst");
        printf("    instance request rc = %d\n", r_inst);

        printf("    closing instance...\n"); fflush(stdout);
        ane_model_close(inst);
        printf("    closed instance OK\n");
        printf("    closing src...\n"); fflush(stdout);
        ane_model_close(src);
        printf("    closed src OK\n");
        if (r_src != 0 || r_inst != 0) return 41;
        return 0;
    }
}

/* ----- 4) ane_decompress_weights --------------------------------------- */

static int probe_decompress(void* _unused) {
    (void)_unused;
    @autoreleasepool {
        const char* secret = "the quick brown fox jumps over the lazy dog "
                             "the quick brown fox jumps over the lazy dog "
                             "the quick brown fox jumps over the lazy dog";
        size_t orig_len = strlen(secret);

        /* zlib (gzip header) compress */
        uLongf cap = compressBound((uLong)orig_len) + 32;
        unsigned char* buf = (unsigned char*)malloc(cap);
        z_stream zs = {0};
        deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED,
                     15 + 16 /* gzip wrapper */, 8, Z_DEFAULT_STRATEGY);
        zs.next_in = (Bytef*)secret;
        zs.avail_in = (uInt)orig_len;
        zs.next_out = buf;
        zs.avail_out = (uInt)cap;
        int zr = deflate(&zs, Z_FINISH);
        deflateEnd(&zs);
        if (zr != Z_STREAM_END) {
            printf("    deflate failed: %d\n", zr);
            free(buf);
            return 50;
        }
        size_t comp_len = zs.total_out;
        printf("    gzip-compressed %zu -> %zu bytes\n", orig_len, comp_len);

        /* Good path */
        void* out_bytes = NULL;
        size_t out_n = 0;
        AneStatus s = ane_decompress_weights(buf, comp_len, &out_bytes, &out_n);
        printf("    decompress(gzip) status=%d nbytes=%zu msg=%s\n",
               (int)s, out_n, ane_last_error() ? ane_last_error() : "");
        int good = 0;
        if (s == ANE_OK && out_bytes != NULL && out_n == orig_len &&
            memcmp(out_bytes, secret, orig_len) == 0) {
            good = 1;
            printf("    round-trip: PASS\n");
        } else {
            printf("    round-trip: FAIL\n");
        }
        if (out_bytes) free(out_bytes);

        /* Try raw deflate (no gzip header) for comparison. */
        unsigned char* buf2 = (unsigned char*)malloc(cap);
        z_stream zs2 = {0};
        deflateInit2(&zs2, Z_DEFAULT_COMPRESSION, Z_DEFLATED,
                     -15 /* raw */, 8, Z_DEFAULT_STRATEGY);
        zs2.next_in = (Bytef*)secret;
        zs2.avail_in = (uInt)orig_len;
        zs2.next_out = buf2;
        zs2.avail_out = (uInt)cap;
        deflate(&zs2, Z_FINISH);
        size_t raw_len = zs2.total_out;
        deflateEnd(&zs2);
        void* o2 = NULL; size_t n2 = 0;
        AneStatus s2 = ane_decompress_weights(buf2, raw_len, &o2, &n2);
        printf("    decompress(raw-deflate) status=%d nbytes=%zu msg=%s\n",
               (int)s2, n2, ane_last_error() ? ane_last_error() : "");
        if (o2) free(o2);
        free(buf2);

        /* Garbage path */
        unsigned char garbage[64];
        for (size_t i = 0; i < sizeof(garbage); i++) garbage[i] = (unsigned char)(i ^ 0x5a);
        void* og = NULL; size_t ng = 0;
        AneStatus sg = ane_decompress_weights(garbage, sizeof(garbage), &og, &ng);
        printf("    decompress(garbage) status=%d nbytes=%zu msg=%s\n",
               (int)sg, ng, ane_last_error() ? ane_last_error() : "");
        int garbage_ok = (sg != ANE_OK); /* we expect failure, not a crash */
        if (og) free(og);

        free(buf);
        if (!good) return 51;
        if (!garbage_ok) return 52;
        return 0;
    }
}

/* ----- 5) ane_realtime_task_begin / _end ------------------------------- */

static int probe_realtime(void* _unused) {
    (void)_unused;
    @autoreleasepool {
        AneStatus s;
        for (int i = 0; i < 2; i++) {
            s = ane_realtime_task_begin();
            printf("    begin#%d = %d (%s)\n", i, (int)s,
                   ane_last_error() ? ane_last_error() : "");
            if (s != ANE_OK) return 60 + i;
            s = ane_realtime_task_end();
            printf("    end#%d   = %d (%s)\n", i, (int)s,
                   ane_last_error() ? ane_last_error() : "");
            if (s != ANE_OK) return 70 + i;
        }
        return 0;
    }
}

/* ----- 6) ane_session_hint --------------------------------------------- */

static int probe_session_hints(void* _unused) {
    (void)_unused;
    @autoreleasepool {
        AneModel* m = open_identity(g_mil, g_wts);
        if (!m) return 10;

        AneSessionHintKind kinds[] = {
            ANE_HINT_PREFETCH,
            ANE_HINT_LOW_LATENCY,
            ANE_HINT_HIGH_THROUGHPUT,
        };
        const char* names[] = {"PREFETCH", "LOW_LATENCY", "HIGH_THROUGHPUT"};
        int any_failed_create = 0, any_failed_apply = 0;
        for (size_t i = 0; i < 3; i++) {
            AneSessionHint* h = NULL;
            AneStatus s = ane_session_hint_create(kinds[i], &h);
            printf("    hint_create(%s) status=%d (%s)\n",
                   names[i], (int)s, ane_last_error() ? ane_last_error() : "");
            if (s != ANE_OK || h == NULL) {
                any_failed_create = 1;
                continue;
            }
            char* report = NULL;
            AneStatus sa = ane_model_apply_session_hint(m, h, &report);
            printf("    apply_session_hint(%s) status=%d (%s)\n",
                   names[i], (int)sa, ane_last_error() ? ane_last_error() : "");
            if (report) {
                printf("    report = %s\n", report);
                free(report);
            } else {
                printf("    report = (NULL)\n");
            }
            if (sa != ANE_OK) any_failed_apply = 1;
            ane_session_hint_release(h);
        }
        ane_model_close(m);
        if (any_failed_create) return 80;
        if (any_failed_apply) return 81;
        return 0;
    }
}

/* ----- 7) ane_model_open_file ------------------------------------------ */

static void print_accessors(AneModel* m, const char* tag) {
    const char* uuid = ane_model_uuid(m);
    const char* surl = ane_model_source_url(m);
    const char* murl = ane_model_model_url(m);
    const char* key  = ane_model_key(m);
    const char* cui  = ane_model_cache_url_identifier(m);
    int64_t isrc     = ane_model_identifier_source(m);
    bool vc          = ane_model_is_virtual_client(m);
    printf("    [%s] uuid                 = '%s'\n", tag, uuid ? uuid : "(null)");
    printf("    [%s] source_url           = '%s'\n", tag, surl ? surl : "(null)");
    printf("    [%s] model_url            = '%s'\n", tag, murl ? murl : "(null)");
    printf("    [%s] key                  = '%s'\n", tag, key ? key : "(null)");
    printf("    [%s] cache_url_identifier = '%s'\n", tag, cui ? cui : "(null)");
    printf("    [%s] identifier_source    = %lld\n", tag, (long long)isrc);
    printf("    [%s] is_virtual_client    = %s\n", tag, vc ? "true" : "false");
}

static int probe_open_file(void* _unused) {
    (void)_unused;
    @autoreleasepool {
        AneModelFileOpenOptions o = {
            .model_url = g_bench,
            .cache_key = NULL,
            .cache_url_identifier = NULL,
            .identifier_source = ANE_IDENT_DEFAULT,
            .compile_qos = ANE_QOS_DEFAULT,
        };
        AneModel* m = NULL;
        AneStatus s = ane_model_open_file(&o, &m);
        printf("    open_file('%s') status=%d (%s)\n",
               g_bench, (int)s, ane_last_error() ? ane_last_error() : "");
        if (s != ANE_OK || m == NULL) {
            return 90;
        }
        print_accessors(m, "file");
        ane_model_close(m);
        return 0;
    }
}

/* ----- 8) accessors on in-memory open ---------------------------------- */

static int probe_accessors_mem(void* _unused) {
    (void)_unused;
    @autoreleasepool {
        AneModel* m = open_identity(g_mil, g_wts);
        if (!m) return 10;
        print_accessors(m, "mem");
        ane_model_close(m);
        return 0;
    }
}

/* ----- driver --------------------------------------------------------- */

int main(int argc, char** argv) {
    const char* root = ".";
    if (argc >= 2) root = argv[1];
    snprintf(g_mil, sizeof(g_mil), "%s/build/identity/model.mil", root);
    snprintf(g_wts, sizeof(g_wts), "%s/build/identity/weights.bin", root);
    snprintf(g_bench, sizeof(g_bench), "%s/build/bench/bench.mlmodelc", root);

    printf("ane-bridge version: %s\n", ane_bridge_version());
    printf("mil:    %s\n", g_mil);
    printf("wts:    %s\n", g_wts);
    printf("bench:  %s\n\n", g_bench);

    printf("=== 1) ane_cache_exists_for_hash ===\n");
    run_in_child(probe_cache_exists, NULL, "cache_exists");
    printf("\n=== 2) ane_cache_purge_for_hash ===\n");
    run_in_child(probe_cache_purge, NULL, "cache_purge");
    printf("\n=== 3) ane_model_new_instance ===\n");
    run_in_child(probe_new_instance, NULL, "new_instance");
    printf("\n=== 4) ane_decompress_weights ===\n");
    run_in_child(probe_decompress, NULL, "decompress");
    printf("\n=== 5) ane_realtime_task_begin / _end ===\n");
    run_in_child(probe_realtime, NULL, "realtime");
    printf("\n=== 6) ane_session_hint_create + apply ===\n");
    run_in_child(probe_session_hints, NULL, "session_hint");
    printf("\n=== 7) ane_model_open_file (bench.mlmodelc) ===\n");
    run_in_child(probe_open_file, NULL, "open_file");
    printf("\n=== 8) accessors on in-memory open ===\n");
    run_in_child(probe_accessors_mem, NULL, "accessors_mem");

    return 0;
}
