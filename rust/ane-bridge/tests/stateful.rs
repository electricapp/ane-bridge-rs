//! End-to-end coverage for the CoreML-backed stateful inference API
//! ([`ane_bridge::StateModel`] / [`ane_bridge::State`]).
//!
//! The thesis: a model's state (KV cache) lives ANE-resident across calls, so
//! only the small input/output crosses the host boundary each step — and it
//! needs no ANE entitlement (pure CoreML / MLE5Engine path). The fixture is an
//! in-place accumulator (`cache += x`, output = mean) built hermetically by
//! `tools/make_state_fixture.py`. Calling it K times with x=1.0 must make the
//! mean climb 1..K, proving the state persisted and updated in place.
//!
//! Skips cleanly if the fixture toolchain (uv / coremltools) or the ANE is
//! unavailable.

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
    clippy::question_mark_used,
    clippy::implicit_return,
    reason = "integration tests use idiomatic `.unwrap()` / indexing / `as` / \
              `println!` — assertion failure IS the test failure mode"
)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use ane_bridge::{
    Buffer, StateModel, e5rt_error_string, fourcc_for_surface_format, surface_format_for_fourcc,
};
use tempfile::TempDir;

/// Repo root: two levels up from this crate's manifest dir.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("repo root two levels above the crate manifest")
        .to_path_buf()
}

/// Build the stateful `.mlmodelc` fixture via the PEP 723 tool, or return
/// `None` (skip) if uv / coremltools / ANE is unavailable.
fn build_fixture(dir: &Path) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ANE_STATE_FIXTURE") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            return Some(p);
        }
    }
    let out = dir.join("state_fixture.mlmodelc");
    let tool = repo_root().join("tools/make_state_fixture.py");
    let status = Command::new("uv")
        .arg("run")
        .arg(&tool)
        .arg(&out)
        .env("OS_ACTIVITY_MODE", "disable")
        .status();
    match status {
        Ok(s) if s.success() && out.is_dir() => Some(out),
        Ok(s) => {
            println!("skipping: fixture build exited {s:?} (no coremltools/ANE?)");
            None
        }
        Err(e) => {
            println!("skipping: could not launch `uv run` ({e})");
            None
        }
    }
}

/// Schema reflects the state model: one input, one output, one state buffer.
#[test]
fn schema_reports_one_state_buffer() {
    let dir = TempDir::new().expect("tempdir");
    let Some(path) = build_fixture(dir.path()) else {
        return;
    };
    let m = StateModel::open(&path).expect("open state model");
    assert_eq!(m.input_names().len(), 1, "expected exactly one input");
    assert_eq!(m.output_names().len(), 1, "expected exactly one output");
    assert_eq!(
        m.state_names().len(),
        1,
        "model must expose its resident state buffer, got {:?}",
        m.state_names()
    );
}

/// The core thesis: state persists ANE-resident and updates in place across
/// calls. Feed x=1.0 K times against one `State`; the mean must climb 1..K.
/// We pass only the scalar input + a state handle — never the cache contents.
#[test]
fn state_accumulates_in_place_across_calls() {
    let dir = TempDir::new().expect("tempdir");
    let Some(path) = build_fixture(dir.path()) else {
        return;
    };
    let m = StateModel::open(&path).expect("open state model");
    let in_name = m.input_names()[0].clone();
    let out_name = m.output_names()[0].clone();

    let mut state = m.new_state().expect("new state");
    for k in 1..=5_i32 {
        let mut out = [0.0_f32];
        m.predict(
            &mut state,
            &[(in_name.as_str(), &[1.0_f32])],
            &mut [(out_name.as_str(), &mut out[..])],
        )
        .expect("predict step");
        assert!(
            (out[0] - k as f32).abs() < 1e-3,
            "after {k} steps the resident state mean should be {k}, got {}",
            out[0]
        );
    }
}

/// E5RT escape hatch: after a predict has built the engine's stream, the live
/// `e5rt_execution_stream` is reachable and reports a valid stream id — the
/// beachhead for driving the raw `e5rt_*` runtime on the same stream.
#[test]
fn e5rt_stream_is_reachable_after_predict() {
    let dir = TempDir::new().expect("tempdir");
    let Some(path) = build_fixture(dir.path()) else {
        return;
    };
    let m = StateModel::open(&path).expect("open state model");
    let in_name = m.input_names()[0].clone();
    let out_name = m.output_names()[0].clone();

    // The engine builds its execution stream lazily — none before first predict.
    let mut state = m.new_state().expect("new state");
    let mut out = [0.0_f32];
    m.predict(
        &mut state,
        &[(in_name.as_str(), &[1.0])],
        &mut [(out_name.as_str(), &mut out[..])],
    )
    .expect("predict");

    assert!(
        !m.e5rt_stream_handle().is_null(),
        "a stateful model routes through MLE5Engine — its e5rt_execution_stream \
         should be reachable after the first predict"
    );
    // The raw `e5rt_*` C-ABI is callable on the borrowed handle. The id value
    // itself is engine-internal (the pool may hold several streams), so we only
    // assert it reads back without UB, not a specific value.
    let id = m
        .e5rt_stream_id()
        .expect("stream id readable on the borrowed stream");
    println!("borrowed e5rt_execution_stream id = {id}");
}

/// A fresh `State` starts from zero — proving each state is independent and the
/// resident buffer is per-`State`, not global.
#[test]
fn fresh_state_resets_accumulation() {
    let dir = TempDir::new().expect("tempdir");
    let Some(path) = build_fixture(dir.path()) else {
        return;
    };
    let m = StateModel::open(&path).expect("open state model");
    let in_name = m.input_names()[0].clone();
    let out_name = m.output_names()[0].clone();

    // Advance one state a few steps.
    let mut a = m.new_state().expect("state a");
    for _ in 0..3 {
        let mut out = [0.0_f32];
        m.predict(
            &mut a,
            &[(in_name.as_str(), &[1.0])],
            &mut [(out_name.as_str(), &mut out[..])],
        )
        .expect("predict a");
    }
    // A brand-new state must start fresh.
    let mut b = m.new_state().expect("state b");
    let mut out = [0.0_f32];
    m.predict(
        &mut b,
        &[(in_name.as_str(), &[1.0])],
        &mut [(out_name.as_str(), &mut out[..])],
    )
    .expect("predict b");
    assert!(
        (out[0] - 1.0).abs() < 1e-3,
        "a fresh state must accumulate from zero, got {}",
        out[0]
    );
}

/// Warm the model's E5RT runtime with one throwaway predict, so the
/// `e5rt_buffer_object_*` factory has a live runtime to build against.
fn warm(m: &StateModel) {
    let in_name = m.input_names()[0].clone();
    let out_name = m.output_names()[0].clone();
    let mut state = m.new_state().expect("new state");
    let mut out = [0.0_f32];
    m.predict(
        &mut state,
        &[(in_name.as_str(), &[1.0])],
        &mut [(out_name.as_str(), &mut out[..])],
    )
    .expect("warmup predict");
}

/// The zero-copy headline: an `IOSurface`-backed [`Buffer`] wrapped as an
/// E5RT buffer object exposes the *same* surface — no copy. After warming the
/// runtime, `wrap_buffer` must round-trip the surface pointer and report a
/// size at least the logical request.
#[test]
fn e5rt_buffer_wraps_iosurface_zero_copy() {
    let dir = TempDir::new().expect("tempdir");
    let Some(path) = build_fixture(dir.path()) else {
        return;
    };
    let m = StateModel::open(&path).expect("open state model");
    warm(&m);

    let buf = Buffer::new(4096).expect("allocate IOSurface buffer");
    let e5 = m
        .wrap_buffer(&buf)
        .expect("wrap buffer as e5rt buffer object");

    assert_eq!(
        e5.iosurface_ref(),
        buf.iosurface_ref(),
        "the e5rt buffer must wrap the very same IOSurface (zero copy)"
    );
    assert!(
        !e5.iosurface_ref().is_null(),
        "wrapped IOSurface must be reachable"
    );
    assert!(
        e5.size() >= buf.nbytes(),
        "e5rt buffer size {} should cover the {}-byte surface",
        e5.size(),
        buf.nbytes()
    );
    println!(
        "wrapped IOSurface zero-copy: e5rt size={}, surface round-trips",
        e5.size()
    );
}

/// A fresh `alloc_buffer` gives an ANE-resident buffer with a real,
/// host-visible data pointer and the requested size.
#[test]
fn e5rt_buffer_alloc_is_resident() {
    let dir = TempDir::new().expect("tempdir");
    let Some(path) = build_fixture(dir.path()) else {
        return;
    };
    let m = StateModel::open(&path).expect("open state model");
    warm(&m);

    let e5 = m.alloc_buffer(2048).expect("alloc e5rt buffer");
    assert!(
        e5.size() >= 2048,
        "alloc'd buffer should be at least 2048 bytes, got {}",
        e5.size()
    );
    assert!(
        !e5.data_ptr().is_null(),
        "a freshly allocated buffer should expose a host data pointer"
    );
}

/// The safety gate: creating an E5RT buffer before any predict (cold runtime)
/// must return an error, not crash — the C factory would segfault on an
/// uninitialized provider, so the wrapper refuses up front.
#[test]
fn e5rt_buffer_creation_requires_warm_runtime() {
    let dir = TempDir::new().expect("tempdir");
    let Some(path) = build_fixture(dir.path()) else {
        return;
    };
    let m = StateModel::open(&path).expect("open state model");
    // Deliberately NO predict: the runtime is cold.
    let err = m
        .alloc_buffer(2048)
        .expect_err("cold-runtime buffer creation must be refused, not crash");
    assert!(
        matches!(err.status, ane_bridge::sys::AneStatus::Unsupported),
        "expected Unsupported for a cold runtime, got {err:?}"
    );
}

/// E5RT timeline event: signal raises the value, the query reads it back, and a
/// wait on an already-reached value returns at once. The active future value
/// round-trips a set/get, and the event remembers its name.
#[test]
fn e5rt_event_signal_wait_round_trips() {
    let dir = TempDir::new().expect("tempdir");
    let Some(path) = build_fixture(dir.path()) else {
        return;
    };
    let m = StateModel::open(&path).expect("open state model");
    warm(&m);

    let evt = m.create_event("kv-step").expect("create event");
    assert_eq!(
        evt.name().as_deref(),
        Some("kv-step"),
        "event keeps its name"
    );

    evt.signal(7).expect("signal");
    assert_eq!(evt.last_signaled_value(), 7, "signal must raise the value");
    // Already at 7, so waiting for 7 returns immediately rather than blocking.
    evt.wait(7, Duration::from_secs(1))
        .expect("wait for reached value");

    evt.set_active_future_value(9).expect("set future value");
    assert_eq!(
        evt.active_future_value(),
        9,
        "active future value must round-trip"
    );
    println!("e5rt event round-trip ok (last_signaled=7, future=9)");
}

/// The same cold-runtime safety gate applies to events: creation before any
/// predict is refused, not crashed.
#[test]
fn e5rt_event_creation_requires_warm_runtime() {
    let dir = TempDir::new().expect("tempdir");
    let Some(path) = build_fixture(dir.path()) else {
        return;
    };
    let m = StateModel::open(&path).expect("open state model");
    let err = m
        .create_event("cold")
        .expect_err("cold-runtime event creation must be refused, not crash");
    assert!(
        matches!(err.status, ane_bridge::sys::AneStatus::Unsupported),
        "expected Unsupported for a cold runtime, got {err:?}"
    );
}

/// `e5rt_error_string` maps codes to the framework's own legible text. Pure
/// function — needs no model or warm runtime, so it always runs.
#[test]
fn e5rt_error_string_is_legible() {
    assert_eq!(e5rt_error_string(0), "OK", "code 0 is success");
    for code in 1..=6 {
        assert!(
            !e5rt_error_string(code).is_empty(),
            "code {code} should map to a non-empty message"
        );
    }
}

/// The borrowed E5RT stream accepts scheduling tuning: QoS (a Darwin
/// `qos_class_t`) and ANE execution priority both apply without error.
#[test]
fn e5rt_stream_qos_and_priority_tunable() {
    let dir = TempDir::new().expect("tempdir");
    let Some(path) = build_fixture(dir.path()) else {
        return;
    };
    let m = StateModel::open(&path).expect("open state model");
    warm(&m);

    // 0x19 == QOS_CLASS_USER_INITIATED.
    m.set_stream_qos(0x19).expect("set QoS on borrowed stream");
    m.set_stream_ane_priority(0)
        .expect("set ANE priority on borrowed stream");
    println!("tuned borrowed stream: qos=user-initiated, ane_priority=0");
}

/// Stream tuning is gated on a warm runtime too: before any predict there is no
/// stream, so tuning is refused (not a null-deref crash).
#[test]
fn e5rt_stream_tuning_requires_warm_runtime() {
    let dir = TempDir::new().expect("tempdir");
    let Some(path) = build_fixture(dir.path()) else {
        return;
    };
    let m = StateModel::open(&path).expect("open state model");
    let err = m
        .set_stream_qos(0x19)
        .expect_err("cold-runtime stream tuning must be refused, not crash");
    assert!(
        matches!(err.status, ane_bridge::sys::AneStatus::Unsupported),
        "expected Unsupported for a cold runtime, got {err:?}"
    );
}

// The system Metal device factory — reachable because `ane-bridge-sys` links
// the Metal framework.
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *mut core::ffi::c_void;
}

/// An `MTLDevice` wrapped as an E5RT compute device round-trips back to the same
/// device. Needs no model or warm runtime (GPU-device handles are standalone).
#[test]
fn e5rt_gpu_device_wraps_mtl_device() {
    // SAFETY: returns the process's shared `MTLDevice` (or null on a host
    // without one); we only retain it (via the wrapper) and compare the pointer.
    let mtl = unsafe { MTLCreateSystemDefaultDevice() };
    if mtl.is_null() {
        println!("skipping: no system MTLDevice");
        return;
    }
    // SAFETY: `mtl` is a live `id<MTLDevice>` from `MTLCreateSystemDefaultDevice`.
    let dev = unsafe { ane_bridge::E5rtGpuDevice::from_mtl_device(mtl) }
        .expect("wrap MTLDevice as E5RT compute device");
    assert_eq!(
        dev.mtl_device(),
        mtl,
        "the compute device must round-trip the same MTLDevice"
    );
    assert!(
        !dev.as_raw().is_null(),
        "wrapped device handle must be live"
    );
    println!("wrapped MTLDevice round-trips through E5rtGpuDevice");
}

/// The CVPixelBuffer 4CC <-> E5RT surface-format converters round-trip. Pure
/// functions — no model or warm runtime, so this always runs.
#[test]
fn e5rt_surface_format_fourcc_round_trips() {
    let bgra: u32 = 0x4247_5241; // 'BGRA'
    let fmt = surface_format_for_fourcc(bgra).expect("BGRA must map to a surface format");
    assert_eq!(
        fourcc_for_surface_format(fmt),
        Some(bgra),
        "surface format {fmt} must map back to 'BGRA'"
    );
}

/// Read-only introspection of the live E5RT program library the engine loaded:
/// its function list and on-disk e5 bundle path.
#[test]
fn e5rt_program_library_introspectable() {
    let dir = TempDir::new().expect("tempdir");
    let Some(path) = build_fixture(dir.path()) else {
        return;
    };
    let m = StateModel::open(&path).expect("open state model");
    warm(&m);

    let lib = m
        .e5rt_program_library()
        .expect("a warmed MLE5 model exposes its program library");
    assert!(
        lib.function_names().iter().any(|n| n == "main"),
        "expected a 'main' function, got {:?}",
        lib.function_names()
    );
    assert!(
        lib.num_functions() >= 1,
        "library must report >= 1 function"
    );
    let bundle = lib.bundle_path().expect("library has an e5 bundle path");
    assert!(
        bundle.contains("e5bundlecache") || bundle.ends_with(".bundle"),
        "bundle path should point into the e5 bundle cache, got {bundle}"
    );
    println!("program library: functions={:?}", lib.function_names());
}

/// Read-only introspection of the live E5RT operation(s) the engine built: the
/// op name and its I/O port names, which must match the fixture's schema.
#[test]
fn e5rt_operations_introspectable() {
    let dir = TempDir::new().expect("tempdir");
    let Some(path) = build_fixture(dir.path()) else {
        return;
    };
    let m = StateModel::open(&path).expect("open state model");
    warm(&m);

    let ops = m.e5rt_operations();
    assert!(!ops.is_empty(), "a warmed model must expose >= 1 operation");
    let op = &ops[0];
    assert!(op.name().is_some(), "operation should report a name");
    assert_eq!(op.input_names(), vec!["x"], "operation input names");
    assert_eq!(
        op.output_names(),
        vec!["reduce_mean_0"],
        "operation output names"
    );
    println!(
        "operation {:?}: in={:?} out={:?} inout={:?}",
        op.name(),
        op.input_names(),
        op.output_names(),
        op.inout_names()
    );
}

/// Operations are built lazily: before any predict there is no stream, so the
/// introspection list is empty (not a crash).
#[test]
fn e5rt_operations_empty_before_predict() {
    let dir = TempDir::new().expect("tempdir");
    let Some(path) = build_fixture(dir.path()) else {
        return;
    };
    let m = StateModel::open(&path).expect("open state model");
    // No predict: the engine has not built its stream / operations yet.
    assert!(
        m.e5rt_operations().is_empty(),
        "no operations should be reachable before the first predict"
    );
}

/// The capstone: drive ANE inference ourselves via [`ane_bridge::E5rtRunner`],
/// reusing the engine's already-compiled operation with our own buffers —
/// entitlement-free. The resident KV-cache accumulates across drives, so feeding
/// input 1.0 each time must raise the output mean by exactly 1.0 per drive,
/// proving the state stays on-device and our I/O buffers carry input/output.
#[test]
fn e5rt_runner_drives_inference() {
    let dir = TempDir::new().expect("tempdir");
    let Some(path) = build_fixture(dir.path()) else {
        return;
    };
    let m = StateModel::open(&path).expect("open state model");

    // Warm, keeping a state alive so the engine's resident state buffer persists.
    let in_feat = m.input_names()[0].clone();
    let out_feat = m.output_names()[0].clone();
    let mut state = m.new_state().expect("new state");
    let mut o = [0.0_f32];
    m.predict(
        &mut state,
        &[(in_feat.as_str(), &[1.0])],
        &mut [(out_feat.as_str(), &mut o[..])],
    )
    .expect("warm predict");

    // The runner binds by the operation's port names (what bind expects).
    let ops = m.e5rt_operations();
    let in_port = ops[0].input_names()[0].clone();
    let out_port = ops[0].output_names()[0].clone();

    let mut runner = m.e5rt_runner().expect("build runner");
    let inbuf = m.alloc_buffer(256).expect("alloc input buffer");
    let outbuf = m.alloc_buffer(256).expect("alloc output buffer");
    inbuf
        .write_f32(&[1.0])
        .expect("write input (host-visible buffer)");

    let mut prev: Option<f32> = None;
    for step in 0..4 {
        runner
            .execute(
                &[(in_port.as_str(), &inbuf)],
                &[(out_port.as_str(), &outbuf)],
            )
            .expect("drive inference");
        let mut out_val = [0.0_f32];
        outbuf.read_f32(&mut out_val).expect("read output");
        let got = out_val[0];
        if let Some(p) = prev {
            assert!(
                (got - p - 1.0).abs() < 1e-3,
                "each drive adds input 1.0 to the resident-state mean: step {step}, prev {p}, got {got}"
            );
        }
        prev = Some(got);
    }
    drop(state);
    println!("drove ANE inference via E5rtRunner; final output = {prev:?}");
}

/// The runner is gated on a warm runtime: before any predict there is no loaded
/// operation, so building one is refused (not a crash).
#[test]
fn e5rt_runner_requires_warm_runtime() {
    let dir = TempDir::new().expect("tempdir");
    let Some(path) = build_fixture(dir.path()) else {
        return;
    };
    let m = StateModel::open(&path).expect("open state model");
    let err = m
        .e5rt_runner()
        .expect_err("a cold runtime has no operation to drive");
    assert!(
        matches!(err.status, ane_bridge::sys::AneStatus::Unsupported),
        "expected Unsupported for a cold runtime, got {err:?}"
    );
}

/// Our drive and ordinary `predict` coexist on the same model and SHARE the
/// resident state: a `predict`, then a drive, then a `predict` — each step (all
/// feeding 1.0) raises the mean by exactly 1.0, proving `predict` still works
/// after we reset/drive the borrowed stream and that both see the same on-device
/// KV-cache.
#[test]
fn e5rt_drive_and_predict_share_state() {
    let dir = TempDir::new().expect("tempdir");
    let Some(path) = build_fixture(dir.path()) else {
        return;
    };
    let m = StateModel::open(&path).expect("open state model");
    let in_feat = m.input_names()[0].clone();
    let out_feat = m.output_names()[0].clone();
    let mut state = m.new_state().expect("new state");

    // predict #1 (warms; cache 0 -> 1).
    let mut o1 = [0.0_f32];
    m.predict(
        &mut state,
        &[(in_feat.as_str(), &[1.0])],
        &mut [(out_feat.as_str(), &mut o1[..])],
    )
    .expect("predict #1");

    // our drive (input 1.0): reads the resident state, cache -> 2.
    let ops = m.e5rt_operations();
    let in_port = ops[0].input_names()[0].clone();
    let out_port = ops[0].output_names()[0].clone();
    let mut runner = m.e5rt_runner().expect("runner");
    let inbuf = m.alloc_buffer(256).expect("input buffer");
    let outbuf = m.alloc_buffer(256).expect("output buffer");
    inbuf.write_f32(&[1.0]).expect("write input");
    runner
        .execute(
            &[(in_port.as_str(), &inbuf)],
            &[(out_port.as_str(), &outbuf)],
        )
        .expect("drive");
    let mut drive_arr = [0.0_f32];
    outbuf.read_f32(&mut drive_arr).expect("read drive output");
    let drive_out = drive_arr[0];

    // predict #2 with the SAME state: must still work AND see the drive's update.
    let mut o2 = [0.0_f32];
    m.predict(
        &mut state,
        &[(in_feat.as_str(), &[1.0])],
        &mut [(out_feat.as_str(), &mut o2[..])],
    )
    .expect("predict #2 after our drive");

    assert!(
        (drive_out - o1[0] - 1.0).abs() < 1e-3,
        "drive must read the state predict left: predict {} -> drive {drive_out}",
        o1[0]
    );
    assert!(
        (o2[0] - drive_out - 1.0).abs() < 1e-3,
        "predict after drive must work and see the drive's state update: drive {drive_out} -> predict {}",
        o2[0]
    );
    drop(state);
    println!(
        "predict<->drive share state: {} -> {drive_out} -> {}",
        o1[0], o2[0]
    );
}
