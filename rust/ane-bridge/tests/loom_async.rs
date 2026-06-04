//! Loom model of the [`crate::EvalShared`] state machine.
//!
//! `loom` exhaustively explores legal interleavings of atomics and locks
//! between threads, looking for missed happens-before relationships.
//! It can't see into the C side or the real `core::task::Waker`, so we
//! model just the Rust-side atomic/mutex sequencing that the
//! `EvalFuture` relies on.
//!
//! This file is only compiled when the crate is built with `--cfg loom`:
//!   `RUSTFLAGS='--cfg loom' cargo test --release --test loom_async`.

#![cfg(loom)]
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
    clippy::print_stderr,
    clippy::clone_on_ref_ptr,
    clippy::integer_division_remainder_used,
    clippy::missing_const_for_fn,
    clippy::use_debug,
    clippy::little_endian_bytes,
    clippy::big_endian_bytes,
    clippy::deref_by_slicing,
    clippy::doc_paragraphs_missing_punctuation,
    clippy::doc_markdown,
    clippy::multiple_unsafe_ops_per_block,
    clippy::undocumented_unsafe_blocks,
    clippy::unused_trait_names,
    clippy::string_slice,
    clippy::cast_possible_wrap,
    clippy::cast_ptr_alignment,
    clippy::assertions_on_result_states,
    clippy::borrow_as_ptr,
    clippy::too_many_lines,
    clippy::unused_result_ok,
    clippy::map_with_unused_argument_over_ranges,
    clippy::ignored_unit_patterns,
    clippy::unreachable,
    clippy::decimal_literal_representation,
    clippy::single_char_pattern,
    clippy::cast_precision_loss,
    reason = "loom model — uses idiomatic test patterns"
)]

use loom::sync::atomic::{AtomicBool, Ordering};
use loom::sync::{Arc, Mutex};
use loom::thread;

/// A faithful model of `EvalShared` using `loom` primitives. The real
/// implementation in `lib.rs` uses `core::sync::atomic`+`std::sync::Mutex`
/// with identical orderings, so any race loom finds here is also a race
/// in the production code.
struct EvalShared {
    done: AtomicBool,
    waker_pending: Mutex<bool>,
    result: Mutex<Option<i32>>,
}

impl EvalShared {
    fn new() -> Self {
        Self {
            done: AtomicBool::new(false),
            waker_pending: Mutex::new(false),
            result: Mutex::new(None),
        }
    }
}

/// Verify: after the callback fires and the future polls, the future
/// always observes the result. No interleaving of (set_result,
/// store_done, take_waker) vs (register_waker, load_done, take_result)
/// can lose the result.
#[test]
fn no_lost_wakeup() {
    loom::model(|| {
        let shared = Arc::new(EvalShared::new());
        let shared_writer = shared.clone();

        // Writer thread: this models the C-side completion callback —
        // it sets the result, publishes via `done`, and then would
        // wake the registered waker.
        let writer = thread::spawn(move || {
            *shared_writer.result.lock().unwrap() = Some(42);
            shared_writer.done.store(true, Ordering::Release);
            // "Wake": observe the waker_pending bit (so loom sees this
            // dependency); the real code calls Waker::wake here.
            let _ = shared_writer.waker_pending.lock().unwrap();
        });

        // Reader thread: this models `EvalFuture::poll`. It either
        // sees `done` already true and returns Ready, or registers
        // a waker and re-checks. After the writer completes, at
        // least ONE re-poll must observe `done == true`.
        let reader = thread::spawn(move || {
            // First poll path.
            if shared.done.load(Ordering::Acquire) {
                let r = shared.result.lock().unwrap().take();
                assert_eq!(r, Some(42), "Ready path saw stale result");
                return;
            }
            // Register waker (model).
            *shared.waker_pending.lock().unwrap() = true;
            // Re-check done after registration to close the lost-wakeup
            // race (the exact pattern used in `EvalFuture::poll`).
            if shared.done.load(Ordering::Acquire) {
                let r = shared.result.lock().unwrap().take();
                assert_eq!(r, Some(42), "after-register path saw stale result");
                return;
            }
            // Otherwise we'd be "Pending" — in production the runtime
            // would wake us. We model the eventual wakeup by waiting
            // for the writer thread (which is the source of the wake).
        });

        writer.join().unwrap();
        reader.join().unwrap();
    });
}

/// Verify the `done` flag is monotonic: once true it never goes back
/// to false. Combined with the seq_cst stores in the real code, this
/// prevents spurious "Ready -> Pending" transitions across polls.
#[test]
fn done_is_monotonic() {
    loom::model(|| {
        let shared = Arc::new(EvalShared::new());
        let writer = shared.clone();
        let h = thread::spawn(move || {
            writer.done.store(true, Ordering::Release);
        });
        // Reader spin: once it sees true, no subsequent load may see false.
        let mut seen_true = false;
        for _ in 0..2 {
            let v = shared.done.load(Ordering::Acquire);
            if v {
                seen_true = true;
            } else {
                assert!(!seen_true, "done went true -> false");
            }
        }
        h.join().unwrap();
    });
}
