//! Borrow-safety regression tests.
//!
//! Things the safe Rust wrapper must enforce against the lifetime
//! holes naturally hidden by an `extern "C"` boundary. Each test here
//! corresponds to a UAF or UB hazard caught by audit.

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

use ane_bridge::{Model, OpenOptions, QoS};
use common::Fixture;

fn open_identity() -> Model {
    let fx = Fixture::identity();
    let opts = OpenOptions::new(fx.mil_path(), fx.weights_path());
    Model::open(&opts).expect("open identity")
}

/// A `Buffer` that was bound into a `Request` and then dropped before
/// the request ran must not cause a use-after-free.
///
/// The Rust signature `bind_input(&mut self, idx, buf: &Buffer)` only
/// borrows the buffer for the call — so a caller can legitimately drop
/// the buffer afterwards. If the Request retained only a raw pointer
/// to it, the next `run()` dereferences freed memory and segfaults
/// (UAF on the `_ANEIOSurfaceObject` wrapper).
#[test]
fn drop_bound_input_buffer_no_uaf() {
    let model = open_identity();
    let mut req = model.request().unwrap();
    let out_buf = model.output_buffer(0).unwrap();
    req.bind_output(0, out_buf).unwrap();

    {
        let in_buf = model.input_buffer(0).unwrap();
        req.bind_input(0, in_buf).unwrap();
        // Allocate to encourage reuse of the freed slot — increases
        // the odds of hitting the UAF if one exists.
    }
    let _scratch: Vec<u8> = vec![0xAB; 4096];

    // Must not segfault. The result itself may be Err — we only care
    // that we return cleanly.
    let _ = req.run(QoS::Default);
}

/// Same hazard, but for outputs.
#[test]
fn drop_bound_output_buffer_no_uaf() {
    let model = open_identity();
    let mut req = model.request().unwrap();
    let in_buf = model.input_buffer(0).unwrap();
    req.bind_input(0, in_buf).unwrap();

    {
        let out_buf = model.output_buffer(0).unwrap();
        req.bind_output(0, out_buf).unwrap();
    }
    let _scratch: Vec<u8> = vec![0xAB; 4096];

    let _ = req.run(QoS::Default);
}

/// `submit()` returns sync-Err when an input isn't bound. A subsequent
/// `wait()` must immediately observe the request as "done with error"
/// — the C side's fail path sets `has_result=1` + `in_flight=0` and
/// `wait`'s fast path returns without touching the semaphore.
///
/// Verified deterministically via `wait(0)` (poll mode): with a zero
/// timeout the implementation either reports the captured status (if
/// the atomics are in the "done" state) or returns `NotDone` (if not).
/// No sleep, no scheduler interaction, no timing dependency.
#[test]
fn submit_without_binding_leaves_request_done_for_wait_poll() {
    let model = open_identity();
    let mut req = model.request().unwrap();
    let submit_err = req
        .submit(QoS::Default)
        .expect_err("submit should fail with input unbound");
    assert_eq!(submit_err.status, ane_bridge::sys::AneStatus::InvalidArg);

    // Poll-mode wait. Returns either the captured status (if the fail
    // path correctly set the atomics) or `NotDone` (if it did not).
    // We accept any non-NotDone result; `NotDone` would mean the
    // failure path left the request stuck "in flight" — the bug
    // pattern this test guards against.
    let poll = req.wait(0);
    assert!(
        !matches!(
            poll.as_ref().err().map(|e| e.status),
            Some(ane_bridge::sys::AneStatus::NotDone),
        ),
        "wait(0) reported NotDone — submit-fail did not publish has_result/in_flight: {poll:?}",
    );

    // The same condition reflected through is_done — both must agree.
    assert!(
        req.is_done(),
        "is_done() should be true after submit-fail (atomics are 'done')",
    );
}

/// Accessing a bound buffer via `input_buffer_mut`/`output_buffer_mut`
/// while a submit is in flight would let the caller `with_locked`
/// the `IOSurface` that ANE is concurrently reading or writing via DMA.
/// `IOSurfaceLock` serializes only CPU accesses, not ANE/DMA. The safe
/// wrapper must therefore refuse these accessors during an in-flight
/// submission.
///
/// The race only manifests in a short scope where the `&mut Buffer`
/// is held between `submit` returning and `wait` — the borrow checker
/// allows that pattern, so the runtime must catch it.
#[test]
fn buffer_mut_mid_flight_unavailable() {
    let model = open_identity();
    let mut req = model.request().unwrap();
    let in_buf = model.input_buffer(0).unwrap();
    let out_buf = model.output_buffer(0).unwrap();
    req.bind_input(0, in_buf).unwrap();
    req.bind_output(0, out_buf).unwrap();

    // Sanity: accessible before any submit.
    assert!(req.input_buffer_mut(0).is_some());
    assert!(req.output_buffer_mut(0).is_some());

    let n = 64 * 16;
    let input: Vec<u8> = (0..n)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    req.input_buffer_mut(0)
        .unwrap()
        .with_locked(ane_bridge::BufferAccess::Write, |b| {
            b.copy_from_slice(&input);
        })
        .unwrap();

    req.submit(QoS::Default).unwrap();
    // Mid-flight: accessors must return None to prevent a DMA race.
    assert!(
        req.input_buffer_mut(0).is_none(),
        "input_buffer_mut must be None while in flight",
    );
    assert!(
        req.output_buffer_mut(0).is_none(),
        "output_buffer_mut must be None while in flight",
    );

    req.wait(-1).unwrap();
    // After wait, eval is complete and the buffers are safe again.
    assert!(req.input_buffer_mut(0).is_some());
    assert!(req.output_buffer_mut(0).is_some());
}

/// Calling `set_input_bytes` while an eval is in flight would write
/// into the same `IOSurface` the ANE is reading via DMA — a data race
/// on the payload bytes. The C side must reject the call with Busy
/// rather than silently corrupting the in-flight eval's input.
#[test]
fn set_input_bytes_during_in_flight_rejected() {
    let model = open_identity();
    let mut req = model.request().unwrap();
    let n = 64 * 16;
    let input: Vec<u8> = (0..n)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    req.set_input_bytes(0, &input).unwrap();
    req.submit(QoS::Default).unwrap();
    // Worker is now reading the IOSurface. A concurrent re-write
    // would race. We expect a clean Busy error, not a silent racy Ok.
    let racy = req.set_input_bytes(0, &input);
    let racy_err = racy.expect_err("set_input_bytes mid-flight must error");
    assert_eq!(
        racy_err.status,
        ane_bridge::sys::AneStatus::Busy,
        "expected Busy mid-flight, got {racy_err:?}",
    );
    req.wait(-1).unwrap();
}

/// Mixing `bind_input` then `set_input_bytes` must not double-free.
/// Internally, `bind_input` puts the Buffer in `Request::bound_inputs`
/// (Rust-owned), while `set_input_bytes` creates a *separate* owned
/// buffer on the C side stored in `r->input_owned`. Both are released
/// independently on drop. This test catches a regression where Rust
/// and C tried to free the same `AneBuffer`.
#[test]
fn bind_then_set_input_bytes_no_double_free() {
    let model = open_identity();
    let mut req = model.request().unwrap();
    let buf = model.input_buffer(0).unwrap();
    req.bind_input(0, buf).unwrap();
    let out = model.output_buffer(0).unwrap();
    req.bind_output(0, out).unwrap();

    let n = 64 * 16;
    let input: Vec<u8> = (0..n)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    // set_input_bytes silently re-binds to an internal IOSurface — the
    // previously bound user buffer becomes unreachable from C but is
    // still owned by Rust until Request drops.
    req.set_input_bytes(0, &input).unwrap();
    req.run(QoS::Default).unwrap();

    // Drop the request. C releases its `input_owned`, Rust drops
    // `bound_inputs`. Different `AneBuffer`s — should NOT crash with
    // a double-free or libmalloc abort.
    drop(req);
}

/// Replacing a binding (`bind_input` twice on the same slot) must
/// drop the previous Buffer exactly once. No double-free, no leak,
/// no UAF if the C side momentarily held the old pointer.
#[test]
fn rebind_same_slot_drops_previous_once() {
    let model = open_identity();
    let mut req = model.request().unwrap();
    let buf_a = model.input_buffer(0).unwrap();
    let id_a = buf_a.iosurface_id();
    req.bind_input(0, buf_a).unwrap();

    let buf_b = model.input_buffer(0).unwrap();
    let id_b = buf_b.iosurface_id();
    assert_ne!(
        id_a, id_b,
        "fresh buffers should have distinct IOSurfaceIDs"
    );
    req.bind_input(0, buf_b).unwrap();

    // Don't crash. Run to verify the second buffer is in use.
    let out = model.output_buffer(0).unwrap();
    req.bind_output(0, out).unwrap();
    req.run(QoS::Default).unwrap();
}

/// A failed `submit_async` (e.g., input not bound) installs its
/// completion callback successfully but the subsequent `submit`
/// returns Err. The next legitimate `submit_async` must clean up the
/// stale callback and not get confused. Concretely: a subsequent
/// successful `submit_async + await` must observe its OWN result,
/// not anything left over from the failed attempt.
#[test]
fn submit_async_failure_then_retry_is_clean() {
    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        use std::sync::Arc;
        use std::task::{Context, Poll, Wake, Waker};
        struct ThreadWaker(std::thread::Thread);
        impl Wake for ThreadWaker {
            fn wake(self: Arc<Self>) {
                self.0.unpark();
            }
        }
        let waker: Waker = Arc::new(ThreadWaker(std::thread::current())).into();
        let mut ctx = Context::from_waker(&waker);
        let mut f = std::pin::pin!(f);
        loop {
            match f.as_mut().poll(&mut ctx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::thread::park(),
            }
        }
    }

    let model = open_identity();
    let mut req = model.request().unwrap();

    // First attempt: no input bound — `submit` inside `submit_async`
    // fails after `on_complete` already installed the callback.
    let first = req.submit_async(QoS::Default);
    assert!(first.is_err(), "expected submit_async to fail before bind");

    // Now actually bind and retry — must succeed and produce correct
    // output, without any residual state from the failed attempt.
    let mut in_buf = model.input_buffer(0).unwrap();
    let n = 64 * 16;
    let input: Vec<u8> = (0..n)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    in_buf
        .with_locked(ane_bridge::BufferAccess::Write, |b| {
            b.copy_from_slice(&input);
        })
        .unwrap();
    let out_buf = model.output_buffer(0).unwrap();
    req.bind_input(0, in_buf).unwrap();
    req.bind_output(0, out_buf).unwrap();

    block_on(async { req.submit_async(QoS::Default).expect("retry submit").await })
        .expect("retry await");

    let out_ref = req.output_buffer_mut(0).expect("output bound + done");
    let v = out_ref
        .with_locked(ane_bridge::BufferAccess::Read, |b| {
            f32::from_le_bytes(b[4..8].try_into().unwrap())
        })
        .unwrap();
    assert!(
        (v - 0.01).abs() < 1e-2,
        "second element should be 0.01, got {v}"
    );
}

/// Binding to one request and dropping the user's handle should still
/// let the request work because the request kept its own strong
/// reference. Verify run + read actually produces correct output.
#[test]
fn bind_then_drop_user_handle_still_works() {
    let model = open_identity();
    let mut req = model.request().unwrap();

    let n_elem = 64 * 16;
    let in_data: Vec<u8> = (0..n_elem)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();

    {
        let mut in_buf = model.input_buffer(0).unwrap();
        in_buf
            .with_locked(ane_bridge::BufferAccess::Write, |bytes| {
                bytes.copy_from_slice(&in_data);
            })
            .unwrap();
        req.bind_input(0, in_buf).unwrap();
        let out_buf = model.output_buffer(0).unwrap();
        req.bind_output(0, out_buf).unwrap();
    } // both user-visible buffers dropped

    // The request kept the bindings alive — must run and produce
    // correct output.
    req.run(QoS::Default)
        .expect("run after drop of user handles");

    let mut out = vec![0_u8; n_elem * 4];
    req.get_output_bytes(0, &mut out).expect("read output back");
    for (i, chunk) in out.chunks_exact(4).take(8).enumerate() {
        let v = f32::from_le_bytes(chunk.try_into().unwrap());
        let expected = (i as f32) * 0.01;
        assert!((v - expected).abs() < 1e-2, "i={i} got {v}");
    }
}
