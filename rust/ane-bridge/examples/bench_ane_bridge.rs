//! Steady-state per-call latency benchmark for ane-bridge.
//!
//! Loads a real Apple-shipped `mlmodelc` bundle, allocates inputs/output
//! once, then times `req.run(QoS)` over N iterations. Prints percentile
//! stats in a single line of TSV so a shell driver can diff against
//! the matching `CoreML` run.
//!
//! Output format (single line, tab-separated):
//! `ane-bridge / warmup / measured / min_us / p50_us / p90_us / p99_us / max_us / mean_us / std_us`

// Benchmark scratch: percentile math intentionally lossy.
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
    clippy::cast_sign_loss,
    reason = "benchmark — percentile math is intentionally lossy and panic on bad input is fine"
)]

use std::path::Path;
use std::time::Instant;

use ane_bridge::{Model, OpenOptions, QoS};

const WARMUP: usize = 200;
const MEASURED: usize = 2000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: {} <mlmodelc_dir>\n(takes .mlmodelc directory; reads model.mil + weights/weight.bin)",
            args[0]
        );
        std::process::exit(1);
    }
    let bundle = Path::new(&args[1]);
    let mil = bundle.join("model.mil");
    let weights = bundle.join("weights/weight.bin");
    if !mil.exists() || !weights.exists() {
        eprintln!(
            "missing model.mil or weights/weight.bin under {}",
            bundle.display()
        );
        std::process::exit(1);
    }
    let opts = OpenOptions::new(
        mil.to_str().expect("utf-8 mil path"),
        weights.to_str().expect("utf-8 weights path"),
    )
    .qos(QoS::UserInteractive);
    let model = Model::open(&opts)?;

    let mut req = model.request()?;

    // Allocate input buffers and bind once, then measure only the
    // submit/wait round-trip. This matches typical MLModel usage
    // (the caller reuses an MLMultiArray across predictions).
    let nin = model.num_inputs();
    let nout = model.num_outputs();
    for i in 0..nin {
        let buf = model.input_buffer(i)?;
        req.bind_input(i, buf)?;
    }
    for i in 0..nout {
        let buf = model.output_buffer(i)?;
        req.bind_output(i, buf)?;
    }

    for _ in 0..WARMUP {
        req.run(QoS::UserInteractive)?;
    }

    let mut samples = Vec::with_capacity(MEASURED);
    for _ in 0..MEASURED {
        let t = Instant::now();
        req.run(QoS::UserInteractive)?;
        samples.push(t.elapsed().as_nanos() as u64);
    }

    report("ane-bridge", &mut samples);
    Ok(())
}

fn report(label: &str, samples: &mut [u64]) {
    samples.sort_unstable();
    let n = samples.len() as f64;
    let min = samples[0];
    let max = samples[samples.len() - 1];
    let p50 = samples[samples.len() / 2];
    let p90 = samples[samples.len() * 90 / 100];
    let p99 = samples[samples.len() * 99 / 100];
    let mean = samples.iter().sum::<u64>() as f64 / n;
    let var = samples
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let std = var.sqrt();
    // Print microseconds with 3 decimals so we keep sub-microsecond
    // resolution if it's there.
    println!(
        "{label}\t{warmup}\t{measured}\t{min:.3}\t{p50:.3}\t{p90:.3}\t{p99:.3}\t{max:.3}\t{mean:.3}\t{std:.3}",
        warmup = WARMUP,
        measured = MEASURED,
        min = min as f64 / 1e3,
        p50 = p50 as f64 / 1e3,
        p90 = p90 as f64 / 1e3,
        p99 = p99 as f64 / 1e3,
        max = max as f64 / 1e3,
        mean = mean / 1e3,
        std = std / 1e3,
    );
}
