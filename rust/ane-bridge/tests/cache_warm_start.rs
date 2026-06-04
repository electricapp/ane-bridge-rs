//! Warm-start cache: opening the same model twice in-process should hit
//! aned's lowered-program cache on the second call, skipping the
//! `compileWithQoS:` step.

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

mod common;

use std::time::Instant;

use ane_bridge::{Model, OpenOptions, QoS};
use common::Fixture;

const N_ELEM: usize = 64 * 16;

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

/// Run one eval on a model and return the output bytes. The fixture's
/// identity MIL is `y = cast(fp32, cast(fp16, x))`, so output should
/// match input modulo fp16 rounding.
fn run_once(m: &Model, input: &[f32]) -> Vec<f32> {
    let mut out = vec![0_f32; N_ELEM];
    let mut req = m.request().expect("request");
    req.set_input_bytes(0, as_bytes(input)).expect("set input");
    req.run(QoS::Default).expect("run");
    req.get_output_bytes(0, as_bytes_mut(&mut out))
        .expect("get output");
    out
}

/// (1) The original timing check: warm open reports `cache_hit` and is
/// much faster than cold. We assert "at least 4× faster" OR "warm
/// under 200 ms" so the test survives both small graphs (where both
/// opens are quick) and large graphs (where cold is 10+ s).
#[test]
fn double_open_hits_cache() {
    let fx = Fixture::identity();
    let opts = OpenOptions::new(fx.mil_path(), fx.weights_path());

    let t = Instant::now();
    let m1 = Model::open(&opts).expect("cold open");
    let cold = t.elapsed();
    assert!(
        !m1.was_cached(),
        "first open of a unique-token MIL must report cache_miss"
    );

    let t = Instant::now();
    let m2 = Model::open(&opts).expect("warm open");
    let warm = t.elapsed();
    assert!(
        m2.was_cached(),
        "second open of the same MIL must report cache_hit \
         (cold={cold:?}, warm={warm:?}) — aned didn't keep the lowering"
    );

    let warm_ok = warm.as_millis() < 200 || warm * 4 < cold;
    assert!(
        warm_ok,
        "warm open should be much faster than cold (cold={cold:?}, warm={warm:?})"
    );

    drop(m1);
    drop(m2);
}

/// (2) Eval correctness on the warm path. If a future refactor
/// accidentally short-circuits something `loadWithQoS:` depends on,
/// the warm model would build clean but produce garbage. We run the
/// same input through cold + warm models and assert the outputs are
/// bit-identical (same lowering, same weights → same numerics).
#[test]
fn warm_open_produces_same_output_as_cold() {
    let fx = Fixture::identity();
    let opts = OpenOptions::new(fx.mil_path(), fx.weights_path());

    let m_cold = Model::open(&opts).expect("cold open");
    assert!(!m_cold.was_cached());
    let input = ramp(0.01);
    let out_cold = run_once(&m_cold, &input);

    let m_warm = Model::open(&opts).expect("warm open");
    assert!(m_warm.was_cached(), "second open must hit cache");
    let out_warm = run_once(&m_warm, &input);

    assert_eq!(
        out_cold.len(),
        out_warm.len(),
        "output length must match between cold and warm"
    );
    // Identity model uses fp16 cast — outputs are deterministic given
    // the same input and weights, so the byte-for-byte comparison is
    // the right discipline. A floating-point tolerance would mask the
    // exact regression we're trying to catch.
    assert_eq!(
        out_cold, out_warm,
        "warm-path eval produced different output than cold-path eval"
    );
}

/// (3) Cache flag isolation. Open model A (caches), then open a
/// *different* model B in the same process. B has a different
/// `hexStringIdentifier`, so its `was_cached()` must be `false` —
/// otherwise the bit was sticky / set on the wrong thing.
#[test]
fn cache_flag_is_per_model_not_global() {
    let fx_a = Fixture::identity();
    let opts_a = OpenOptions::new(fx_a.mil_path(), fx_a.weights_path());

    // Open A cold then warm so the cache is definitely populated.
    let _ma_cold = Model::open(&opts_a).expect("A cold");
    let ma_warm = Model::open(&opts_a).expect("A warm");
    assert!(
        ma_warm.was_cached(),
        "model A should be cached by second open"
    );

    // Now open a brand-new model B. The fixture builder embeds a
    // unique nanosecond + counter token in the MIL's `buildInfo`, so
    // B's `networkTextHash` differs from A's even though the program
    // shape is identical.
    let fx_b = Fixture::identity();
    assert_ne!(
        std::fs::read(fx_a.mil_path()).unwrap(),
        std::fs::read(fx_b.mil_path()).unwrap(),
        "fixtures must produce distinct MIL bytes"
    );
    let opts_b = OpenOptions::new(fx_b.mil_path(), fx_b.weights_path());
    let mb = Model::open(&opts_b).expect("B cold");
    assert!(
        !mb.was_cached(),
        "model B has a different content hash and must report cache_miss"
    );
}
