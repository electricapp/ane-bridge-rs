//! Real Apple-shipped MIL corpus.
//!
//! Probes a list of `model.mil` paths that macOS ships under
//! `/System/Library/Frameworks/...` and `/System/Library/PrivateFrameworks/...`.
//! These are produced by Apple's own coremltools pipeline and are the
//! ground-truth corpus for the `modelAttributes` parser. Hand-rolled
//! MIL strings (see `tests/corpus.rs`) only exercise the shapes *we*
//! think Apple emits — these files exercise the shapes Apple actually
//! emits, including pixel-buffer inputs, multi-function programs, and
//! whatever else the framework decides to put in the live IO lists.
//!
//! Each entry pins an expected verdict in `CANDIDATES`; the test
//! asserts the actual outcome matches. A mismatch in either direction
//! is a real signal:
//!   * a previously-ANE-loadable model now fails  → bridge regression,
//!   * a previously-not-ANE-targetable model now loads  → Apple
//!     loosened compiler policy; flip the entry to `Expect::Ane`,
//!   * the path resolves to a different model version on this Mac
//!     → expected outcome may legitimately have drifted.
//!
//! Missing-path entries are tolerated (macOS versions and SKUs ship
//! different models). Loaded entries are also schema-validated:
//! non-empty name, known dtype, positive dims, rank ≤ 5. We do not
//! pin specific shapes/names — those are Apple's to change.

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

use std::path::Path;

use ane_bridge::{Dtype, Model, OpenOptions};

/// Expected verdict for each candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Expect {
    /// Should load and produce a well-formed schema.
    Ane,
    /// Apple's `_ANECompiler` should reject with `CompilationFailure`
    /// — valid MIL the ANE compiler can't lower to bytecode (e.g.
    /// `fp32` ops, palettized weights it doesn't support yet).
    /// `CoreML` routes these to CPU/GPU via the public `MLModel` API.
    CompilationFailure,
    /// Apple's `_ANECompiler` should reject with `InvalidMILProgram`
    /// — MIL the ANE compiler considers structurally invalid (e.g.
    /// dynamic-shape inputs, unsupported MIL constructs). Same
    /// "valid `CoreML` / not ANE-targetable" story.
    InvalidMilProgram,
    /// `weights/weight.bin` is not present on this Mac (the model
    /// is shipped but its weights live elsewhere).
    UnreadableWeights,
}

/// Candidate Apple-shipped MIL files. Each entry pins the verdict we
/// expect from Apple's ANE compiler today — a flip in either direction
/// (regression in our bridge OR Apple making a model ANE-targetable
/// in a future OS) trips the test.
///
/// Ordering matters only for the report; we exercise every present
/// path. New macOS releases may add or remove entries — keep the
/// list focused on frameworks that have been stable for several
/// releases (`Vision`, `SoundAnalysis`, `Spotlight`).
const CANDIDATES: &[(&str, Expect, &str)] = &[
    (
        "Vision/vmtracker",
        Expect::Ane,
        "/System/Library/Frameworks/Vision.framework/Versions/A/Resources/vmtracker_model_v1_6.mlmodelc/model.mil",
    ),
    (
        "SoundAnalysis/SNLanguageAlignedAudioEncoder",
        Expect::CompilationFailure,
        "/System/Library/Frameworks/SoundAnalysis.framework/Versions/A/Resources/SNLanguageAlignedAudioEncoder.mlmodelc/model.mil",
    ),
    (
        "SoundAnalysis/SNLanguageAlignedAVFuser",
        Expect::Ane,
        "/System/Library/Frameworks/SoundAnalysis.framework/Versions/A/Resources/SNLanguageAlignedAVFuserModel.mlmodelc/model.mil",
    ),
    (
        "DuetExpert/ATXIntentPrediction",
        Expect::InvalidMilProgram,
        "/System/Library/DuetExpertCenter/Assets/Assets.bundle/AssetData/ATXIntentPredictionMLModel.mlmodelc/model.mil",
    ),
    (
        "DuetExpert/ATXAppPrediction",
        Expect::InvalidMilProgram,
        "/System/Library/DuetExpertCenter/Assets/Assets.bundle/AssetData/ATXAppPredictionMLModel.mlmodelc/model.mil",
    ),
    (
        "HDRProcessing/sceneLuxB2DItp",
        Expect::CompilationFailure,
        "/System/Library/PrivateFrameworks/HDRProcessing.framework/Versions/A/Resources/sceneLuxB2DItpMLModel.mlmodelc/model.mil",
    ),
    (
        "TextRecognition/cr_td_model_v3_e5",
        Expect::Ane,
        "/System/Library/PrivateFrameworks/TextRecognition.framework/Versions/A/Resources/cr_td_model_v3_e5.mlmodelc.bundle/model.mil",
    ),
    (
        "IntelligencePlatform/MentionGeneration",
        Expect::InvalidMilProgram,
        "/System/Library/PrivateFrameworks/IntelligencePlatform.framework/Versions/A/Resources/MentionGenerationModel.mlmodelc/model.mil",
    ),
    (
        "IntelligencePlatform/EntityReranker",
        Expect::UnreadableWeights,
        "/System/Library/PrivateFrameworks/IntelligencePlatform.framework/Versions/A/Resources/EntityRerankerModel.mlmodelc/model.mil",
    ),
    (
        "IntelligenceFlow/PlanResolution",
        Expect::InvalidMilProgram,
        "/System/Library/PrivateFrameworks/IntelligenceFlowPlannerRuntime.framework/Versions/A/Resources/PlanResolutionModel.mlmodelc/model.mil",
    ),
    (
        "VoiceActions/aa_encoder",
        Expect::Ane,
        "/System/Library/PrivateFrameworks/VoiceActions.framework/Versions/A/Resources/aa_encoder_125141826.mlmodelc/model.mil",
    ),
];

/// Maximum sane rank for an ANE tensor; nothing real ships above 5-D.
const MAX_REASONABLE_RANK: usize = 5;

/// Result of probing one candidate.
struct Probe {
    label: &'static str,
    expect: Expect,
    mil: &'static str,
    outcome: Outcome,
}

enum Outcome {
    Missing,
    Unreadable(String),
    OpenFailed(String),
    Loaded {
        num_inputs: i32,
        num_outputs: i32,
        io_dump: Vec<String>,
    },
}

/// Classify a compile-error message into the Apple verdict category
/// it represents. We match on the underlying-error tail surfaced by
/// `ane_bridge.m` (`| underlying: ... err=(\n    <Class>\n)`).
fn classify(msg: &str) -> Option<Expect> {
    if msg.contains("InvalidMILProgram") {
        Some(Expect::InvalidMilProgram)
    } else if msg.contains("CompilationFailure") {
        Some(Expect::CompilationFailure)
    } else {
        None
    }
}

fn dtype_is_known(d: Dtype) -> bool {
    matches!(
        d,
        Dtype::Fp16 | Dtype::Fp32 | Dtype::Int8 | Dtype::UInt8 | Dtype::Int32 | Dtype::Int64
    )
}

fn probe(label: &'static str, expect: Expect, mil: &'static str) -> Probe {
    let mil_p = Path::new(mil);
    if !mil_p.exists() {
        return Probe {
            label,
            expect,
            mil,
            outcome: Outcome::Missing,
        };
    }
    let weights = mil_p.parent().map(|d| d.join("weights").join("weight.bin"));
    let Some(weights) = weights else {
        return Probe {
            label,
            expect,
            mil,
            outcome: Outcome::Unreadable("no parent directory".into()),
        };
    };
    if !weights.exists() {
        return Probe {
            label,
            expect,
            mil,
            outcome: Outcome::Unreadable(format!("missing {}", weights.display())),
        };
    }
    let Some(mil_str) = mil_p.to_str() else {
        return Probe {
            label,
            expect,
            mil,
            outcome: Outcome::Unreadable("non-utf8 path".into()),
        };
    };
    let Some(wts_str) = weights.to_str() else {
        return Probe {
            label,
            expect,
            mil,
            outcome: Outcome::Unreadable("non-utf8 path".into()),
        };
    };
    let opts = OpenOptions::new(mil_str, wts_str);
    match Model::open(&opts) {
        Ok(model) => {
            let num_inputs = model.num_inputs();
            let num_outputs = model.num_outputs();
            let mut io_dump = Vec::new();
            for i in 0..num_inputs {
                let spec = model
                    .input(i)
                    .unwrap_or_else(|| panic!("[{label}] input {i} missing despite count > 0"));
                validate_spec(label, "input", i, &spec);
                io_dump.push(format!(
                    "    in[{i}]  {} {:?} {:?}",
                    spec.name(),
                    spec.dtype(),
                    spec.shape()
                ));
            }
            for i in 0..num_outputs {
                let spec = model
                    .output(i)
                    .unwrap_or_else(|| panic!("[{label}] output {i} missing despite count > 0"));
                validate_spec(label, "output", i, &spec);
                io_dump.push(format!(
                    "    out[{i}] {} {:?} {:?}",
                    spec.name(),
                    spec.dtype(),
                    spec.shape()
                ));
            }
            Probe {
                label,
                expect,
                mil,
                outcome: Outcome::Loaded {
                    num_inputs,
                    num_outputs,
                    io_dump,
                },
            }
        }
        Err(e) => Probe {
            label,
            expect,
            mil,
            outcome: Outcome::OpenFailed(format!("{e:?}")),
        },
    }
}

fn validate_spec(label: &str, kind: &str, idx: i32, spec: &ane_bridge::TensorSpec) {
    let name = spec.name();
    assert!(!name.is_empty(), "[{label}] {kind} {idx}: empty name");
    assert!(
        dtype_is_known(spec.dtype()),
        "[{label}] {kind} {idx} '{name}': unknown dtype {:?}",
        spec.dtype(),
    );
    let shape = spec.shape();
    assert!(
        !shape.is_empty(),
        "[{label}] {kind} {idx} '{name}': empty shape"
    );
    assert!(
        shape.len() <= MAX_REASONABLE_RANK,
        "[{label}] {kind} {idx} '{name}': implausible rank {}",
        shape.len()
    );
    for (d, dim) in shape.iter().enumerate() {
        assert!(
            *dim >= 1,
            "[{label}] {kind} {idx} '{name}': non-positive dim[{d}] = {dim}"
        );
    }
}

/// Map an outcome onto the expected category — `None` for outcomes
/// that can't be turned into a comparable `Expect` (open succeeded
/// when we expected an open-failure, or vice versa).
fn actual_for(outcome: &Outcome) -> Option<Expect> {
    match outcome {
        Outcome::Loaded { .. } => Some(Expect::Ane),
        Outcome::Unreadable(_) => Some(Expect::UnreadableWeights),
        Outcome::OpenFailed(why) => classify(why),
        Outcome::Missing => None,
    }
}

#[test]
fn parser_survives_apple_shipped_mil() {
    let probes: Vec<Probe> = CANDIDATES
        .iter()
        .map(|(label, expect, mil)| probe(label, *expect, mil))
        .collect();

    let mut present = 0;
    let mut mismatches: Vec<String> = Vec::new();

    println!();
    println!("== Apple system MIL corpus ==");
    for p in &probes {
        let actual = actual_for(&p.outcome);
        match &p.outcome {
            Outcome::Missing => {
                println!("  [skip   ] {} ({}: not present)", p.label, p.mil);
            }
            Outcome::Unreadable(why) => {
                println!(
                    "  [unread ] {} ({why})  [expected: {:?}]",
                    p.label, p.expect
                );
                present += 1;
            }
            Outcome::OpenFailed(why) => {
                let tag = match actual {
                    Some(Expect::CompilationFailure) => "[compfail]",
                    Some(Expect::InvalidMilProgram) => "[invalmil]",
                    _ => "[fail    ]",
                };
                println!(
                    "  {tag} {} [expected: {:?}]\n           reason: {why}",
                    p.label, p.expect
                );
                present += 1;
            }
            Outcome::Loaded {
                num_inputs,
                num_outputs,
                io_dump,
            } => {
                println!(
                    "  [ ok    ] {} (inputs={num_inputs}, outputs={num_outputs}) [expected: {:?}]",
                    p.label, p.expect
                );
                for line in io_dump {
                    println!("{line}");
                }
                present += 1;
            }
        }
        // Per-case verdict. `Missing` is allowed — different macOS
        // versions ship different models. Everything else must match
        // the expected verdict exactly.
        if matches!(p.outcome, Outcome::Missing) {
            continue;
        }
        match actual {
            Some(a) if a == p.expect => {}
            Some(a) => {
                mismatches.push(format!("{}: expected {:?}, got {:?}", p.label, p.expect, a));
            }
            None => {
                mismatches.push(format!(
                    "{}: expected {:?}, got an outcome we couldn't classify",
                    p.label, p.expect
                ));
            }
        }
    }
    println!(
        "== reachable={present}/{total} mismatches={} ==",
        mismatches.len(),
        total = CANDIDATES.len()
    );

    // If literally nothing was reachable, the test taught us nothing
    // — flag it but don't fail (some Macs are minimal by design).
    if present == 0 {
        eprintln!(
            "warning: no Apple-shipped MIL was reachable on this machine; \
             parser was not exercised against real Apple corpus"
        );
        return;
    }

    // A mismatch is either:
    //   * our bridge regressed (a previously-ANE-loadable model now
    //     fails to load, or a new error class appeared), or
    //   * Apple changed compiler policy (a model that previously got
    //     `CompilationFailure` now loads to ANE — good news, update
    //     `Expect::Ane`), or
    //   * the macOS version on this machine shipped a different
    //     version of the model.
    //
    // All three deserve a human look. Hard-fail with the full list.
    assert!(
        mismatches.is_empty(),
        "verdict mismatches detected — bridge regression, Apple policy change, \
         or macOS version drift. Review the report above and update Expect:\n  {}",
        mismatches.join("\n  ")
    );
}
