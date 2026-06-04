//! End-to-end tests covering the full FFI surface.
//!
//! Each test compiles a small MIL program through `_ANECompiler`, which
//! takes ~200-500 ms; tests are therefore slower than typical unit tests.
//! Run with `cargo test --release` for the lowest overhead.

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

use std::sync::{Arc, Mutex};
use std::thread;

use ane_bridge::{BufferAccess, Dtype, Model, OpenOptions, QoS, TensorSpec, sys};
use common::Fixture;

const N_ELEM: usize = 64 * 16;
const N_BYTES: usize = N_ELEM * 4;

fn ramp(scale: f32) -> Vec<f32> {
    (0..N_ELEM).map(|i| (i as f32) * scale).collect()
}
fn as_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and every bit pattern is a valid u8.
    unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}
fn as_bytes_mut(v: &mut [f32]) -> &mut [u8] {
    // SAFETY: same as `as_bytes`; exclusive borrow promoted to mut.
    unsafe {
        core::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(v))
    }
}

fn open_identity() -> Model {
    let fx = Fixture::identity();
    let opts = OpenOptions::new(fx.mil_path(), fx.weights_path());
    // The library copies MIL+weights into its own NSTemporaryDirectory
    // at open time, so the fixture's TempDir can be dropped after open.
    Model::open(&opts).expect("open identity")
}

// ---- Schema / lifecycle ----

#[test]
fn schema_matches_declared() {
    let m = open_identity();
    assert_eq!(m.num_inputs(), 1);
    assert_eq!(m.num_outputs(), 1);
    assert_eq!(m.input_nbytes(0), N_BYTES);
    assert_eq!(m.output_nbytes(0), N_BYTES);
    assert_eq!(m.input_nbytes(99), 0, "out-of-range idx should return 0");
}

#[test]
fn open_with_invalid_mil_path_returns_io_error() {
    let opts = OpenOptions::new("/nonexistent/model.mil", "/nonexistent/weights.bin");
    let err = Model::open(&opts).expect_err("should fail");
    assert_eq!(err.status, sys::AneStatus::Io);
    assert!(!err.message.is_empty(), "error message should not be empty");
}

#[test]
fn open_with_invalid_mil_returns_compile_error() {
    let dir = tempfile::tempdir().unwrap();
    let mil = dir.path().join("model.mil");
    std::fs::write(&mil, "this is not MIL text").unwrap();
    let wts = dir.path().join("weights.bin");
    std::fs::write(&wts, b"\x00").unwrap();
    let opts = OpenOptions::new(mil.to_str().unwrap(), wts.to_str().unwrap());
    let err = Model::open(&opts).expect_err("should fail compile");
    assert!(
        matches!(err.status, sys::AneStatus::Compile | sys::AneStatus::Load),
        "expected Compile/Load, got {:?}",
        err.status
    );
}

#[test]
fn tensor_spec_nbytes_is_product() {
    let s = TensorSpec::new("x", Dtype::Fp16, &[2, 3, 4, 5]);
    assert_eq!(s.nbytes(), 2 * 3 * 4 * 5 * 2);
}

/// A shape large enough to overflow `usize` multiplication must
/// return 0 (sentinel for "invalid"), not a wrapped small value. A
/// wrapped value would cause downstream `Buffer` allocation to be
/// smaller than the actual tensor → ANE writes past the buffer end.
#[test]
fn tensor_spec_nbytes_overflow_returns_zero() {
    let s = TensorSpec::new("x", Dtype::Fp32, &[i64::MAX, 2, 2, 2]);
    assert_eq!(
        s.nbytes(),
        0,
        "expected 0 on overflow, got wrapped {}",
        s.nbytes(),
    );
}

/// Negative shape entries must also yield 0 (invalid), not be cast
/// to a huge positive `size_t`.
#[test]
fn tensor_spec_nbytes_negative_dim_returns_zero() {
    let s = TensorSpec::new("x", Dtype::Fp32, &[2, -1, 4]);
    assert_eq!(s.nbytes(), 0);
}

/// Rank-0 (scalar) specs should report a single-element byte count.
#[test]
fn tensor_spec_nbytes_rank_zero_is_element_size() {
    assert_eq!(TensorSpec::new("x", Dtype::Fp32, &[]).nbytes(), 4);
    assert_eq!(TensorSpec::new("x", Dtype::Fp16, &[]).nbytes(), 2);
    assert_eq!(TensorSpec::new("x", Dtype::UInt8, &[]).nbytes(), 1);
}

/// A spec whose product overflows `usize` on the C side (via the
/// C `spec_nbytes`) must be rejected at `Model::open`, not silently
/// accepted with a wrapped byte count.
///
/// `Model::input` / `Model::output` return the framework-derived
/// schema. For the identity MIL of shape [1, 64, 1, 16] we expect to
/// see fp32 NCHW with N=1, C=64, H=1, W=16 reflected back.
#[test]
fn model_input_output_reflect_framework_schema() {
    let m = open_identity();
    let inp = m.input(0).expect("input 0");
    let out = m.output(0).expect("output 0");

    assert_eq!(inp.dtype(), Dtype::Fp32);
    assert_eq!(inp.shape(), &[1, 64, 1, 16]);
    assert_eq!(inp.name(), "x");
    assert_eq!(inp.nbytes(), N_BYTES);

    assert_eq!(out.dtype(), Dtype::Fp32);
    assert_eq!(out.shape(), &[1, 64, 1, 16]);
    // The framework reports outputs as "<name>@output" — the C
    // parser strips the suffix so users see the MIL-declared name.
    assert_eq!(out.name(), "y");
    assert_eq!(out.nbytes(), N_BYTES);
}

#[test]
#[allow(clippy::multiple_unsafe_ops_per_block)]
fn dtype_size_matches() {
    // SAFETY:
    // `ane_dtype_size` is a pure function over a `#[repr(C)]` enum
    // value; it dereferences no pointers, takes no resources, and has
    // no preconditions. Calling it repeatedly is therefore sound.
    let sizes = unsafe {
        [
            sys::ane_dtype_size(sys::AneDtype::Fp32),
            sys::ane_dtype_size(sys::AneDtype::Fp16),
            sys::ane_dtype_size(sys::AneDtype::Int32),
            sys::ane_dtype_size(sys::AneDtype::Int64),
            sys::ane_dtype_size(sys::AneDtype::UInt8),
            sys::ane_dtype_size(sys::AneDtype::Int8),
        ]
    };
    assert_eq!(sizes, [4, 2, 4, 8, 1, 1]);
}

// ---- Byte-ptr fast path ----

#[test]
fn round_trip_byte_path() {
    let m = open_identity();
    let mut req = m.request().unwrap();
    let input = ramp(0.01);
    let mut output = vec![0_f32; N_ELEM];
    req.set_input_bytes(0, as_bytes(&input)).unwrap();
    req.run(QoS::Default).unwrap();
    req.get_output_bytes(0, as_bytes_mut(&mut output)).unwrap();
    for i in 0..16 {
        assert!(
            (output[i] - input[i]).abs() < 1e-2,
            "i={} in={} out={}",
            i,
            input[i],
            output[i]
        );
    }
}

#[test]
fn set_input_with_wrong_size_returns_invalid_arg() {
    let m = open_identity();
    let mut req = m.request().unwrap();
    let bad = vec![0_u8; N_BYTES - 1];
    let err = req
        .set_input_bytes(0, &bad)
        .expect_err("size mismatch should fail");
    assert_eq!(err.status, sys::AneStatus::InvalidArg);
}

#[test]
fn get_output_before_submit_returns_error() {
    let m = open_identity();
    let mut req = m.request().unwrap();
    let mut out = vec![0_u8; N_BYTES];
    let err = req
        .get_output_bytes(0, &mut out)
        .expect_err("output not ready");
    // Either NotDone (no result captured yet) or InvalidArg (no buffer
    // bound) is acceptable — both reflect "you can't read an output
    // before submit + bind". The C side picks one priority order.
    assert!(
        matches!(
            err.status,
            sys::AneStatus::NotDone | sys::AneStatus::InvalidArg
        ),
        "expected NotDone/InvalidArg, got {:?}",
        err.status,
    );
}

// ---- Zero-copy bind path ----

#[test]
fn round_trip_zero_copy() {
    let m = open_identity();
    let mut in_buf = m.input_buffer(0).unwrap();
    let out_buf = m.output_buffer(0).unwrap();
    in_buf
        .with_locked(BufferAccess::Write, |bytes| {
            for (i, chunk) in bytes.chunks_exact_mut(4).enumerate() {
                let v = (i as f32) * 0.02;
                chunk.copy_from_slice(&v.to_le_bytes());
            }
        })
        .unwrap();
    let mut req = m.request().unwrap();
    req.bind_input(0, in_buf).unwrap();
    req.bind_output(0, out_buf).unwrap();

    req.run(QoS::Default).unwrap();

    let out_ref = req.output_buffer_mut(0).expect("output bound");
    let first = out_ref
        .with_locked(BufferAccess::Read, |bytes| {
            f32::from_le_bytes(bytes[0..4].try_into().unwrap())
        })
        .unwrap();
    assert!(first.abs() < 1e-2);
    let eighth = out_ref
        .with_locked(BufferAccess::Read, |bytes| {
            f32::from_le_bytes(bytes[28..32].try_into().unwrap())
        })
        .unwrap();
    assert!((eighth - 0.14).abs() < 1e-2, "got {eighth}");
}

#[test]
fn buffer_iosurface_id_is_nonzero() {
    let m = open_identity();
    let b = m.input_buffer(0).unwrap();
    assert!(b.iosurface_id() != 0);
    assert_eq!(b.nbytes(), N_BYTES);
}

// ---- Async submit / wait ----

#[test]
fn submit_then_wait_completes() {
    let m = open_identity();
    let mut req = m.request().unwrap();
    req.set_input_bytes(0, as_bytes(&ramp(0.01))).unwrap();
    req.submit(QoS::Default).unwrap();
    req.wait(-1).unwrap();
    assert!(req.is_done());
}

#[test]
fn second_submit_while_in_flight_returns_busy() {
    let m = open_identity();
    let mut req = m.request().unwrap();
    req.set_input_bytes(0, as_bytes(&ramp(0.01))).unwrap();
    req.submit(QoS::Default).unwrap();
    // Race: the eval is ~100 µs so the BUSY path is hit if we submit fast.
    let err = req.submit(QoS::Default);
    // Either it raced and we get Busy, or it completed already. Both fine.
    if let Err(e) = err {
        assert_eq!(e.status, sys::AneStatus::Busy);
    }
    req.wait(-1).unwrap();
}

#[test]
fn wait_with_zero_timeout_polls() {
    let m = open_identity();
    let mut req = m.request().unwrap();
    req.set_input_bytes(0, as_bytes(&ramp(0.01))).unwrap();
    req.run(QoS::Default).unwrap();
    // After run completes, polling should return Ok immediately.
    req.wait(0).unwrap();
}

#[test]
fn resubmit_without_intervening_wait_stays_balanced() {
    // Regression: submitting again after a completion but *before* calling
    // wait() used to leave a stale `done_sem` count. The next timed wait()
    // then consumed that stale signal and returned *before* the new eval
    // finished — so `is_done()` could still be false right after `wait`
    // returned Ok. `ane_request_submit`/`run` now drain the semaphore on
    // entry, keeping signal/consume balanced 1:1 per submission. Repeated
    // to shake out the timing window.
    let m = open_identity();
    let mut req = m.request().unwrap();
    let input = ramp(0.01);
    let mut out = vec![0_f32; N_ELEM];
    for k in 0..128 {
        req.set_input_bytes(0, as_bytes(&input)).unwrap();
        req.submit(QoS::Default).unwrap();
        // Spin until the worker has completed AND signaled the semaphore,
        // so the next submit exercises the stale-signal window.
        while !req.is_done() {
            std::hint::spin_loop();
        }
        // Re-arm and submit again with no intervening wait().
        req.set_input_bytes(0, as_bytes(&input)).unwrap();
        req.submit(QoS::Default).unwrap();
        req.wait(-1).unwrap();
        assert!(
            req.is_done(),
            "wait() returned before the resubmitted eval completed (iter {k})"
        );
        req.get_output_bytes(0, as_bytes_mut(&mut out)).unwrap();
        assert!(
            (out[0] - input[0]).abs() < 1e-2 && (out[15] - input[15]).abs() < 1e-2,
            "stale/premature output at iter {k}: {:?}",
            &out[..4]
        );
    }
}

// ---- Completion callback ----

#[test]
fn completion_callback_fires_on_success() {
    let m = open_identity();
    let mut req = m.request().unwrap();
    let counter = Arc::new(Mutex::new(0_u32));
    let counter_cb = counter.clone();
    req.on_complete(move |result| {
        assert!(result.is_ok());
        *counter_cb.lock().unwrap() += 1;
    })
    .unwrap();
    req.set_input_bytes(0, as_bytes(&ramp(0.01))).unwrap();
    for _ in 0..5 {
        req.run(QoS::Default).unwrap();
    }
    assert_eq!(*counter.lock().unwrap(), 5);
}

#[test]
fn completion_callback_can_be_replaced_when_idle() {
    let m = open_identity();
    let mut req = m.request().unwrap();
    req.on_complete(|_| {}).unwrap();
    req.set_input_bytes(0, as_bytes(&ramp(0.01))).unwrap();
    req.run(QoS::Default).unwrap();
    // Replace (request is idle after run).
    let hit = Arc::new(Mutex::new(false));
    let hit_cb = hit.clone();
    req.on_complete(move |_| *hit_cb.lock().unwrap() = true)
        .unwrap();
    req.run(QoS::Default).unwrap();
    assert!(*hit.lock().unwrap());
}

#[test]
fn last_error_is_empty_on_success() {
    let m = open_identity();
    let mut req = m.request().unwrap();
    req.set_input_bytes(0, as_bytes(&ramp(0.01))).unwrap();
    req.run(QoS::Default).unwrap();
    assert!(req.last_error().is_empty());
}

// ---- Async / Future ----

/// Minimal `block_on` so we don't need a runtime dependency in tests.
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

#[test]
fn submit_async_resolves_to_ok() {
    let m = open_identity();
    let mut req = m.request().unwrap();
    req.set_input_bytes(0, as_bytes(&ramp(0.01))).unwrap();
    block_on(async { req.submit_async(QoS::Default).expect("submit_async").await })
        .expect("future resolved Err");
    assert!(req.is_done());
}

#[test]
fn submit_async_many_in_a_row() {
    let m = open_identity();
    let mut req = m.request().unwrap();
    req.set_input_bytes(0, as_bytes(&ramp(0.01))).unwrap();
    block_on(async {
        for _ in 0..10 {
            req.submit_async(QoS::Default)
                .expect("submit_async")
                .await
                .expect("await");
        }
    });
    let mut out = vec![0_f32; N_ELEM];
    req.get_output_bytes(0, as_bytes_mut(&mut out)).unwrap();
    let input = ramp(0.01);
    for i in 0..8 {
        assert!((out[i] - input[i]).abs() < 1e-2);
    }
}

// ---- Multi-IO ----

#[test]
fn two_inputs_two_outputs() {
    let fx = Fixture::dual_io();
    let opts = OpenOptions::new(fx.mil_path(), fx.weights_path());
    let m = Model::open(&opts).unwrap();
    assert_eq!(m.num_inputs(), 2);
    assert_eq!(m.num_outputs(), 2);

    let mut req = m.request().unwrap();
    let a = ramp(0.01);
    let b = ramp(-0.02);
    let mut ya = vec![0_f32; N_ELEM];
    let mut yb = vec![0_f32; N_ELEM];
    req.set_input_bytes(0, as_bytes(&a)).unwrap();
    req.set_input_bytes(1, as_bytes(&b)).unwrap();
    req.run(QoS::Default).unwrap();
    req.get_output_bytes(0, as_bytes_mut(&mut ya)).unwrap();
    req.get_output_bytes(1, as_bytes_mut(&mut yb)).unwrap();
    for i in 0..8 {
        assert!((ya[i] - a[i]).abs() < 1e-2);
        assert!((yb[i] - b[i]).abs() < 1e-2);
    }
}

// ---- Concurrent requests against the same model ----

#[test]
fn concurrent_requests() {
    let m = Arc::new(open_identity());
    let n_threads = 4;
    let n_iters = 10;
    let handles: Vec<_> = (0..n_threads)
        .map(|t| {
            let m = m.clone();
            thread::spawn(move || {
                let mut req = m.request().unwrap();
                let input = ramp(0.01 * (t + 1) as f32);
                req.set_input_bytes(0, as_bytes(&input)).unwrap();
                let mut out = vec![0_f32; N_ELEM];
                for _ in 0..n_iters {
                    req.run(QoS::Default).unwrap();
                }
                req.get_output_bytes(0, as_bytes_mut(&mut out)).unwrap();
                // Each thread had its own input — verify they don't bleed.
                for i in 0..8 {
                    assert!(
                        (out[i] - input[i]).abs() < 1e-2,
                        "thread {} idx {}: in={} out={}",
                        t,
                        i,
                        input[i],
                        out[i]
                    );
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("thread panic");
    }
}
