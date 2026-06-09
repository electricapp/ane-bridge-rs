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

use ane_bridge::StateModel;
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
