//! Drive ANE inference via E5RT and compare its per-call latency to CoreML
//! `predict`.
//!
//! Loads a stateful `.mlmodelc`, warms it, then times two ways of running the
//! same compiled operation step after step. `StateModel::predict` is the CoreML
//! path (MLState + feature providers); `E5rtRunner::execute` is our path,
//! re-driving the engine's loaded op on the borrowed stream with our own buffer
//! objects (entitlement-free, the KV cache kept ANE-resident). Both feed 0.0 so
//! the resident state stays bounded; only the call overhead differs. Prints
//! p50/p90/p99 for each and the drive-vs-predict speedup.
//!
//! Usage: `cargo run --example e5rt_drive -- <state.mlmodelc>`
//! (or set `ANE_STATE_FIXTURE` to a prebuilt bundle).

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

use std::time::Instant;

use ane_bridge::StateModel;

const WARMUP: usize = 200;
const MEASURED: usize = 2000;

fn p(sorted: &[u128], pct: f64) -> f64 {
    let i = ((sorted.len() - 1) as f64 * pct) as usize;
    sorted[i] as f64 / 1000.0 // ns -> us
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("ANE_STATE_FIXTURE").ok())
        .expect("usage: e5rt_drive <state.mlmodelc>  (or set ANE_STATE_FIXTURE)");
    let m = StateModel::open(&path)?;
    let in_feat = m.input_names()[0].clone();
    let out_feat = m.output_names()[0].clone();
    let mut state = m.new_state()?;
    let mut out = [0.0_f32];

    // Warm (compiles + loads the operation, builds the stream).
    m.predict(
        &mut state,
        &[(in_feat.as_str(), &[0.0])],
        &mut [(out_feat.as_str(), &mut out[..])],
    )?;

    // Our drive: reuse the loaded op with our own buffers.
    let ops = m.e5rt_operations();
    let in_port = ops[0].input_names()[0].clone();
    let out_port = ops[0].output_names()[0].clone();
    let mut runner = m.e5rt_runner()?;
    let inbuf = m.alloc_buffer(256)?;
    let outbuf = m.alloc_buffer(256)?;
    inbuf.write_f32(&[0.0]).expect("write input buffer");

    // ---- time CoreML predict ----
    for _ in 0..WARMUP {
        m.predict(
            &mut state,
            &[(in_feat.as_str(), &[0.0])],
            &mut [(out_feat.as_str(), &mut out[..])],
        )?;
    }
    let mut pred = Vec::with_capacity(MEASURED);
    for _ in 0..MEASURED {
        let t = Instant::now();
        m.predict(
            &mut state,
            &[(in_feat.as_str(), &[0.0])],
            &mut [(out_feat.as_str(), &mut out[..])],
        )?;
        pred.push(t.elapsed().as_nanos());
    }

    // ---- time E5RT drive ----
    for _ in 0..WARMUP {
        runner.execute(
            &[(in_port.as_str(), &inbuf)],
            &[(out_port.as_str(), &outbuf)],
        )?;
    }
    let mut drv = Vec::with_capacity(MEASURED);
    for _ in 0..MEASURED {
        let t = Instant::now();
        runner.execute(
            &[(in_port.as_str(), &inbuf)],
            &[(out_port.as_str(), &outbuf)],
        )?;
        drv.push(t.elapsed().as_nanos());
    }

    pred.sort_unstable();
    drv.sort_unstable();
    println!("path        p50_us   p90_us   p99_us");
    println!(
        "predict     {:7.1}  {:7.1}  {:7.1}",
        p(&pred, 0.5),
        p(&pred, 0.9),
        p(&pred, 0.99)
    );
    println!(
        "e5rt_drive  {:7.1}  {:7.1}  {:7.1}",
        p(&drv, 0.5),
        p(&drv, 0.9),
        p(&drv, 0.99)
    );
    println!(
        "drive p50 speedup vs predict: {:.2}x",
        p(&pred, 0.5) / p(&drv, 0.5)
    );
    drop(state);
    Ok(())
}
