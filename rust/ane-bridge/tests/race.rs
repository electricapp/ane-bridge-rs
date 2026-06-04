//! Regression tests for concurrency invariants at the FFI boundary.
//!
//! Each `#[test]` here runs in a *separate* process (cargo's
//! integration-test convention) — so the `dispatch_once` inside
//! `ane_private_load` is in its initial pre-fired state. That lets us
//! reliably hit the racy first-open window from many threads
//! simultaneously, which is exactly the scenario that produced a
//! transient SIGSEGV before the fix.
//!
//! These tests are intentionally minimal: their purpose is to exercise
//! the race window and assert no crash, not to validate inference.

#![allow(
    clippy::allow_attributes,
    clippy::allow_attributes_without_reason,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::panic,
    clippy::print_stdout,
    clippy::dbg_macro,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::missing_assert_message,
    clippy::missing_docs_in_private_items,
    clippy::tests_outside_test_module,
    clippy::std_instead_of_core,
    clippy::std_instead_of_alloc,
    clippy::separated_literal_suffix,
    clippy::unseparated_literal_suffix,
    clippy::unreadable_literal,
    clippy::shadow_unrelated,
    clippy::shadow_reuse,
    clippy::shadow_same,
    clippy::min_ident_chars,
    clippy::float_arithmetic,
    clippy::float_cmp,
    clippy::arithmetic_side_effects,
    clippy::integer_division,
    clippy::default_numeric_fallback,
    clippy::pattern_type_mismatch,
    clippy::if_then_some_else_none,
    clippy::single_call_fn,
    clippy::needless_pass_by_value,
    clippy::let_underscore_must_use,
    clippy::let_underscore_untyped,
    clippy::redundant_pub_crate,
    clippy::semicolon_outside_block,
    clippy::semicolon_inside_block,
    clippy::semicolon_if_nothing_returned,
    clippy::cast_precision_loss,
    reason = "integration tests use idiomatic `.unwrap()` / indexing / `as` / \
              `println!` — assertion failure IS the test failure mode"
)]

mod common;

use std::sync::{Arc, Barrier};
use std::thread;

use ane_bridge::{Model, OpenOptions, QoS};
use common::Fixture;

/// N threads stampede `Model::open` at the same instant. Before the
/// fix this could segfault on ARM64 due to torn pointer reads on the
/// `g_AneInMemoryCls` etc. globals that `ane_private_load` was
/// writing without synchronization.
#[test]
fn parallel_first_open_no_segfault() {
    const THREADS: usize = 16;
    let barrier = Arc::new(Barrier::new(THREADS));

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let b = barrier.clone();
            thread::spawn(move || {
                // Pre-build the fixture so the only operation racing is
                // `Model::open` itself (and the first one inside it
                // calls `ane_private_load`).
                let fx = Fixture::identity();
                let opts = OpenOptions::new(fx.mil_path(), fx.weights_path());
                b.wait(); // release all threads at once
                Model::open(&opts).expect("open should not crash on race")
            })
        })
        .collect();

    let mut count = 0;
    for h in handles {
        let _model = h.join().expect("thread");
        count += 1;
    }
    assert_eq!(count, THREADS);
}

/// Hammer `Model::open` + `Buffer::create` + `Request::create` from
/// many threads without coordinating on the framework state. Catches
/// any global write-without-synchronization regression.
#[test]
fn parallel_init_paths_no_crash() {
    const THREADS: usize = 16;
    let barrier = Arc::new(Barrier::new(THREADS));

    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let b = barrier.clone();
            thread::spawn(move || {
                let fx = Fixture::identity();
                let opts = OpenOptions::new(fx.mil_path(), fx.weights_path());
                b.wait();
                let model = Model::open(&opts).expect("open");
                // Now exercise every code path that calls
                // `ane_private_load` (buffer create, request create).
                let _buf = model.input_buffer(0).expect("buf");
                let mut req = model.request().expect("req");
                if t % 2 == 0 {
                    req.run(QoS::Default).ok(); // best-effort; not validating output
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("thread");
    }
}
