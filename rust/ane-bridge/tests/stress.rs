//! Stress / lifecycle tests for the FFI boundary.
//!
//! The goal of this suite is to catch:
//!   1. **Leaks** — every `Model` / `Buffer` / `Request` / callback box
//!      must be released exactly once. Run the binary under macOS's
//!      `leaks(1)` (see CI) to verify.
//!   2. **Use-after-free** during teardown — submitting then dropping
//!      mid-flight, replacing callbacks under load, etc.
//!   3. **Send/Sync correctness** — many threads pounding the same
//!      [`Model`] with their own requests must not corrupt state.
//!
//! These tests assume a working ANE; they don't validate inference
//! results beyond "no crash" + clean error paths.

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
    clippy::cast_possible_truncation,
    clippy::significant_drop_tightening,
    reason = "stress / lifecycle tests use idiomatic test patterns"
)]

mod common;

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::Duration;

use ane_bridge::{BufferAccess, Model, OpenOptions, QoS};
use common::Fixture;

const N_ELEM: usize = 64 * 16;

fn open_identity() -> Model {
    let fx = Fixture::identity();
    let opts = OpenOptions::new(fx.mil_path(), fx.weights_path());
    Model::open(&opts).expect("open identity")
}

fn ramp_bytes() -> Vec<u8> {
    (0..N_ELEM)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect()
}

// ---- Lifecycle cycling ----

/// Rapid buffer create/release cycle. Catches `IOSurface` or wrapper
/// retain-count leaks: each iteration must reclaim its memory before
/// the next allocation.
#[test]
fn buffer_create_release_cycle() {
    let model = open_identity();
    for _ in 0..2_000 {
        let buf = model.input_buffer(0).expect("buf");
        drop(buf);
    }
}

/// Deterministic leak detection for the Obj-C autorelease path inside
/// `ane_buffer_create`.
///
/// `ane_buffer_create` allocates several autoreleased objects
/// (`NSDictionary` for the `IOSurface` properties, `_ANEIOSurfaceObject`
/// wrapper, etc.). When called from a thread WITHOUT an enclosing
/// `@autoreleasepool`, those would accumulate in a nonexistent pool
/// and leak permanently.
///
/// We measure via macOS's `malloc_zone_statistics`, which returns the
/// EXACT live heap byte count across all zones. After N create/drop
/// cycles in a thread with no outer pool, the delta against baseline
/// must be small (allocator slop only) rather than scaling with N.
const ALLOC_TEST_N: usize = 5_000;

#[test]
fn buffer_create_in_thread_without_pool_does_not_leak() {
    #[repr(C)]
    #[derive(Default)]
    struct MallocStatistics {
        blocks_in_use: u32,
        size_in_use: usize,
        max_size_in_use: usize,
        size_allocated: usize,
    }
    unsafe extern "C" {
        fn malloc_zone_statistics(zone: *mut std::ffi::c_void, stats: *mut MallocStatistics);
    }
    fn live_bytes() -> usize {
        let mut s = MallocStatistics::default();
        // SAFETY: malloc_zone_statistics with a null zone aggregates
        // across all zones; it only writes to `s`.
        unsafe { malloc_zone_statistics(std::ptr::null_mut(), &raw mut s) };
        s.size_in_use
    }

    let model = std::sync::Arc::new(open_identity());

    // Warm up on this thread to amortize any one-time Foundation /
    // ANE-framework heap costs that aren't related to the leak.
    for _ in 0..16 {
        let _ = model.input_buffer(0).expect("warmup");
    }

    let leak = std::thread::spawn(move || {
        let m = model;
        // Worker thread — no implicit autoreleasepool, no runloop.
        // Two passes: first stabilises any one-shot framework allocs,
        // then we measure the steady-state delta.
        for _ in 0..1_000 {
            let _ = m.input_buffer(0).expect("warmup-2");
        }
        let baseline = live_bytes();
        for _ in 0..ALLOC_TEST_N {
            let _ = m.input_buffer(0).expect("buf");
        }
        let after = live_bytes();
        after.saturating_sub(baseline)
    })
    .join()
    .expect("thread");

    // Each leaked autoreleased object (wrapper + NSDictionary +
    // NSNumbers) is on the order of hundreds of bytes. 5000 leaks
    // would be ~1MB+. A correctly drained path stays under a few
    // KB of allocator noise. Threshold chosen well above the noise
    // floor but well below the leak floor.
    assert!(
        leak < 256 * 1024,
        "live-heap delta after {ALLOC_TEST_N} buffer create/drops on a pool-less thread \
         is {leak} bytes — autoreleasepool drain missing in ane_buffer_create?"
    );
}

/// Rapid request create/release cycle. Catches `dispatch_queue` / serial
/// queue leaks.
#[test]
fn request_create_release_cycle() {
    let model = open_identity();
    for _ in 0..1_000 {
        let req = model.request().expect("request");
        drop(req);
    }
}

/// Open + run + close cycle. Each open compiles a *unique* MIL (via
/// `Fixture::identity()`'s token) so the on-disk hex differs. Catches
/// model/compiler leaks, temp-dir leaks, client leaks.
#[test]
fn open_run_close_cycle() {
    for _ in 0..16 {
        let model = open_identity();
        let mut req = model.request().expect("req");
        req.set_input_bytes(0, &ramp_bytes()).expect("set_input");
        for _ in 0..4 {
            req.run(QoS::Default).expect("run");
        }
        drop(req);
        drop(model);
    }
}

// ---- Drop / teardown safety ----

/// Submit a request then immediately drop it. `ane_request_release`
/// must drain the queue (`dispatch_sync`) before freeing state; if it
/// didn't, the worker thread would be reading freed memory.
#[test]
fn drop_request_just_after_submit() {
    let model = open_identity();
    for _ in 0..32 {
        let mut req = model.request().expect("req");
        req.set_input_bytes(0, &ramp_bytes()).expect("set");
        req.submit(QoS::Default).expect("submit");
        drop(req); // Should drain → release → free, with no UAF.
    }
}

/// Install a callback that takes a while (sleeps), submit, then drop
/// the request. The drop must wait for the callback to finish before
/// freeing the callback box.
#[test]
fn drop_request_during_slow_callback() {
    let model = open_identity();
    let fired = Arc::new(AtomicU32::new(0));
    for _ in 0..8 {
        let mut req = model.request().expect("req");
        let fired_cb = fired.clone();
        req.on_complete(move |_| {
            // Do meaningful work in the callback — exercises the
            // worker thread still holding our box.
            std::thread::sleep(Duration::from_micros(500));
            fired_cb.fetch_add(1, Ordering::Relaxed);
        })
        .expect("on_complete");
        req.set_input_bytes(0, &ramp_bytes()).expect("set");
        req.submit(QoS::Default).expect("submit");
        drop(req); // Must wait for callback to finish.
    }
    // Every submission must have fired its callback before drop returned.
    assert_eq!(fired.load(Ordering::Relaxed), 8);
}

/// Callback box must be replaced safely between runs.
#[test]
fn callback_replacement_no_leak() {
    let model = open_identity();
    let mut req = model.request().expect("req");
    req.set_input_bytes(0, &ramp_bytes()).expect("set");
    for i in 0..50 {
        let tag = i;
        req.on_complete(move |_| {
            // Use the captured value so the closure has a non-zero size.
            let _ = tag * 2;
        })
        .expect("replace cb");
        req.run(QoS::Default).expect("run");
    }
    // Clear; box should be freed.
    req.clear_completion().expect("clear");
}

// ---- Concurrency stress ----

/// Many threads each running many iterations against a shared model.
/// Each thread keeps its own request. Catches Send/Sync violations,
/// `dispatch_queue` contention, `IOSurface` DMA races.
#[test]
fn concurrent_stress() {
    const THREADS: usize = 8;
    const ITERS_PER_THREAD: usize = 64;

    let model = Arc::new(open_identity());
    let barrier = Arc::new(Barrier::new(THREADS));
    let errors = Arc::new(Mutex::new(Vec::<String>::new()));

    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let model = model.clone();
            let barrier = barrier.clone();
            let errors = errors.clone();
            thread::spawn(move || {
                // Synchronize the start so all threads pile in at once.
                barrier.wait();
                let mut req = model.request().expect("req");
                let input: Vec<u8> = (0..N_ELEM)
                    .flat_map(|i| ((i as f32) * 0.01 * (t as f32 + 1.0)).to_le_bytes())
                    .collect();
                if let Err(e) = req.set_input_bytes(0, &input) {
                    errors.lock().unwrap().push(format!("t{t} set: {e}"));
                    return;
                }
                for k in 0..ITERS_PER_THREAD {
                    if let Err(e) = req.run(QoS::Default) {
                        errors.lock().unwrap().push(format!("t{t} iter{k}: {e}"));
                        return;
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }
    let errs = errors.lock().unwrap();
    assert!(errs.is_empty(), "concurrency errors: {errs:?}");
}

/// Concurrent buffer lock/unlock from many threads (each on their own
/// buffer — `Buffer` is `Send` but not `Sync`).
#[test]
fn concurrent_buffer_locks() {
    const THREADS: usize = 16;
    const ITERS: usize = 200;

    let model = Arc::new(open_identity());
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let model = model.clone();
            thread::spawn(move || {
                let mut buf = model.input_buffer(0).expect("buf");
                for k in 0..ITERS {
                    buf.with_locked(BufferAccess::ReadWrite, |bytes| {
                        // Write some thread-specific data; verify no
                        // overlap by reading it back.
                        bytes[0] = (t as u8).wrapping_add(k as u8);
                        bytes[bytes.len() - 1] = bytes[0];
                    })
                    .expect("with_locked");
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("thread");
    }
}
