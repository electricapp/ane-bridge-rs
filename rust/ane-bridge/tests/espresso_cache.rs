//! End-to-end coverage for the Espresso-framework ANE program cache:
//! [`ane_bridge::espresso_cache_has_network`] /
//! [`ane_bridge::espresso_cache_purge_network`].
//!
//! These wrap `espresso_ane_cache_has_network` / `_purge_network`, which take a
//! model path (`model_path_to_model_url`) and query / evict an ANE network
//! cache via `_ANEClient`'s shared connection.
//!
//! What is verified here: the binding and signature (recovered from
//! disassembly), the return convention (`rc == 0` ok; presence via the
//! out-param), NUL handling, and that both calls run without UB — an unknown
//! path reports not-cached and purging it is a no-op.
//!
//! What is NOT yet verified: a positive `has == true`. Empirically this cache
//! reports not-cached for every path tried, including Apple-shipped models the
//! OS has ANE-run, so it is *not* aned's CoreML/`_ANE` program cache (the one
//! `cache_warm_start.rs` exercises via the hash API). It is most likely the
//! E5RT/Espresso compiler bundle cache (cf. the `e5rt_*_cache_bundle_location`
//! / `force_fetch_from_cache` symbols), populated by the Espresso plan path,
//! not a CoreML load. The round-trip test is `#[ignore]`d until that
//! population path is wired; `tools/make_cache_fixture.py` builds the fixture
//! scaffold it will use.

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
    reason = "integration tests use idiomatic `.unwrap()` / indexing / `as` / \
              `println!` — assertion failure IS the test failure mode"
)]

use std::path::{Path, PathBuf};
use std::process::Command;

use ane_bridge::{espresso_cache_has_network, espresso_cache_purge_network};
use tempfile::TempDir;

/// Repo root: two levels up from this crate's manifest dir (`rust/ane-bridge`).
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("repo root two levels above the crate manifest")
        .to_path_buf()
}

/// Build a compiled identity `.mlmodelc` into `dir` via the PEP 723 fixture
/// tool (it also runs one ANE inference as a best-effort cache warm).
/// Returns the `.mlmodelc` path, or `None` if the toolchain (uv / coremltools)
/// or ANE is unavailable — in which case the caller skips.
fn build_fixture(dir: &Path) -> Option<PathBuf> {
    // Escape hatch: let CI (or a diagnostic run) supply a prebuilt compiled
    // `.mlmodelc` instead of shelling out to coremltools.
    if let Ok(p) = std::env::var("ANE_CACHE_FIXTURE") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            return Some(p);
        }
        println!(
            "ANE_CACHE_FIXTURE={} is not a directory; ignoring",
            p.display()
        );
    }
    let out = dir.join("cache_fixture.mlmodelc");
    let tool = repo_root().join("tools/make_cache_fixture.py");
    let status = Command::new("uv")
        .arg("run")
        .arg(&tool)
        .arg(&out)
        // Espresso/CoreML are chatty on os_log; keep the test output clean.
        .env("OS_ACTIVITY_MODE", "disable")
        .status();
    match status {
        Ok(s) if s.success() && out.is_dir() => Some(out),
        Ok(s) => {
            println!("skipping: fixture build exited {s:?} (no coremltools/ANE?)");
            None
        }
        Err(e) => {
            println!("skipping: could not launch `uv run` ({e}) — is uv installed?");
            None
        }
    }
}

/// Contract: a path aned has never compiled reports "not cached" with no
/// error. This always runs — it needs only the framework + daemon, no fixture.
#[test]
fn has_network_is_false_for_unknown_path() {
    let dir = TempDir::new().expect("tempdir");
    // A plausible-looking but never-compiled model path.
    let bogus = dir.path().join("never-compiled.mlmodelc");
    let present = espresso_cache_has_network(&bogus).expect("has-network query must not error");
    assert!(
        !present,
        "a path that was never compiled must report not-cached, got cached"
    );
}

/// Purging a path with nothing cached is a no-op, not an error.
#[test]
fn purge_unknown_path_is_ok() {
    let dir = TempDir::new().expect("tempdir");
    let bogus = dir.path().join("never-compiled.mlmodelc");
    espresso_cache_purge_network(&bogus).expect("purging an absent entry must succeed");
}

/// Full round-trip through the Espresso cache C ABI: `has` reports cached →
/// `purge` evicts it → `has` reports not-cached.
///
/// IGNORED pending a reliable way to *populate* this cache. Empirically
/// `espresso_ane_cache_has_network` reports not-cached for every model path
/// tried — including Apple-shipped models the OS has ANE-run and a model
/// warmed via `CompiledMLModel` inference — so it does not read aned's
/// CoreML/`_ANE` program cache. Together with the `e5rt_*_cache_bundle_location`
/// / `force_fetch_from_cache` symbols in the framework, this points at the
/// E5RT/Espresso *compiler bundle cache*, which is populated by driving the
/// Espresso plan path (`espresso_plan_add_network` + build), not a CoreML
/// load. Wiring that path (and/or fixing the by-URL `Model::open_file` load,
/// which currently fails `loadModel:` even on system models) is the unblock.
/// The fixture scaffold below is kept so this runs once population works.
#[test]
#[ignore = "cache population path unresolved — see doc comment (E5RT plan path / open_file)"]
fn cached_network_is_visible_then_purgeable() {
    let dir = TempDir::new().expect("tempdir");
    let Some(model_path) = build_fixture(dir.path()) else {
        return; // skipped: see build_fixture's notice
    };

    let cached = espresso_cache_has_network(&model_path).expect("has-network query");
    assert!(
        cached,
        "the warmed fixture must report cached for {}",
        model_path.display()
    );

    espresso_cache_purge_network(&model_path).expect("purge cached network");

    let after = espresso_cache_has_network(&model_path).expect("has-network query after purge");
    assert!(
        !after,
        "purge should have evicted the cache entry for {}",
        model_path.display()
    );
}
