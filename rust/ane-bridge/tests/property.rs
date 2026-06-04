//! Property tests against the safe FFI surface.
//!
//! These tests do not try to verify *correctness* of inference — that
//! is the job of `integration.rs`. Their purpose is to exercise the
//! FFI boundary with random and malformed inputs, to confirm that the
//! safe wrapper always either succeeds or returns a well-formed
//! [`Error`], and *never* segfaults / panics / corrupts the heap.
//!
//! Each property runs many random cases; the overall suite therefore
//! does many model-opens, which is slow. Run with `--release`.

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
    clippy::cast_sign_loss,
    reason = "property tests use idiomatic test patterns"
)]

mod common;

use ane_bridge::{BufferAccess, Dtype, Error, Model, OpenOptions, QoS, TensorSpec, sys};
use common::Fixture;
use proptest::prelude::*;

const N_BYTES: usize = 64 * 16 * 4;

fn open_identity() -> Model {
    let fx = Fixture::identity();
    let opts = OpenOptions::new(fx.mil_path(), fx.weights_path());
    Model::open(&opts).expect("open identity")
}

fn assert_clean_error(e: &Error, context: &str) {
    // Status must be a known, non-OK variant.
    assert!(
        !matches!(e.status, sys::AneStatus::Ok),
        "{context}: Ok returned in error path",
    );
    // The Display impl must produce a non-empty string.
    let s = format!("{e}");
    assert!(!s.is_empty(), "{context}: Display produced empty string");
    // Status itself must round-trip through Debug without panicking.
    let _ = format!("{:?}", e.status);
}

proptest! {
    #![proptest_config(ProptestConfig {
        // Cases are expensive (~200ms each: compile + run). Keep small.
        cases: 32,
        max_shrink_iters: 16,
        ..ProptestConfig::default()
    })]

    /// Buffers of any reasonable size lock/unlock cleanly and the slice
    /// view has the requested length.
    #[test]
    fn buffer_lock_unlock_always_safe(nbytes in 0usize..(1 << 20)) {
        let model = open_identity();
        let mut buf = model.buffer(nbytes).expect("buffer create");
        prop_assert_eq!(buf.nbytes(), nbytes);
        // Lock + write first/last byte via with_locked. Returning the
        // observed length out of the closure lets us assert on it.
        let observed_len = buf
            .with_locked(BufferAccess::ReadWrite, |bytes| {
                if let Some(last) = bytes.last_mut() {
                    *last = 0xAA;
                    *bytes.first_mut().expect("non-empty") = 0xBB;
                }
                bytes.len()
            })
            .expect("with_locked");
        prop_assert_eq!(observed_len, nbytes);
        // RAII guard variant.
        {
            let mut g = buf.lock(BufferAccess::ReadWrite).expect("lock");
            prop_assert_eq!(g.len(), nbytes);
            if !g.is_empty() {
                g[0] = 0x55;
            }
        }
    }

    /// Out-of-range tensor indices on any operation return a clean error,
    /// not a crash.
    #[test]
    fn out_of_range_idx_errors_cleanly(idx in (i32::MIN..i32::MAX).prop_filter("not 0", |&i| i != 0)) {
        let model = open_identity();
        let mut req = model.request().expect("request");

        // bind_input/bind_output take ownership, so we create one
        // buffer per call. Out-of-range idx must fail cleanly without
        // dropping the buffer prematurely or panicking.
        if let Err(e) = req.bind_input(idx, model.buffer(N_BYTES).expect("buf")) {
            assert_clean_error(&e, "bind_input");
        }
        if let Err(e) = req.bind_output(idx, model.buffer(N_BYTES).expect("buf")) {
            assert_clean_error(&e, "bind_output");
        }
        if let Err(e) = req.set_input_bytes(idx, &vec![0; N_BYTES]) {
            assert_clean_error(&e, "set_input_bytes");
        }
        let mut out = vec![0_u8; N_BYTES];
        if let Err(e) = req.get_output_bytes(idx, &mut out) {
            assert_clean_error(&e, "get_output_bytes");
        }
    }

    /// `set_input_bytes` with a wrong byte count never crashes; it must
    /// either return InvalidArg (size mismatch) or succeed if it happens
    /// to land on the right size by coincidence (= N_BYTES).
    #[test]
    fn arbitrary_byte_count_safe(n in 0usize..(2 * N_BYTES + 17)) {
        let model = open_identity();
        let mut req = model.request().expect("request");
        let buf = vec![0xAB_u8; n];
        match req.set_input_bytes(0, &buf) {
            Ok(()) => prop_assert_eq!(n, N_BYTES),
            Err(e) => {
                prop_assert_ne!(n, N_BYTES, "expected Ok at right size");
                assert_clean_error(&e, "set_input_bytes wrong-size");
            }
        }
    }

    /// `TensorSpec::nbytes()` is the product of shape * dtype size for
    /// any reasonable rank/shape and never overflows for small inputs.
    #[test]
    fn tensor_spec_nbytes_matches_product(
        dims in proptest::collection::vec(1_i64..=64, 1..=6),
    ) {
        let dtype = Dtype::Fp32;
        let s = TensorSpec::new("x", dtype, &dims);
        let expected: usize = (dims.iter().map(|&d| d as usize).product::<usize>()) * 4;
        prop_assert_eq!(s.nbytes(), expected);
    }

    /// Repeated submit/run cycles with random ramps don't accumulate
    /// state — first 8 outputs always match the input ramp.
    #[test]
    fn repeated_runs_independent(scale in 0.001_f32..=0.1_f32, n_iter in 1usize..=8) {
        let model = open_identity();
        let mut req = model.request().expect("request");
        let mut input = vec![0_f32; 64 * 16];
        for (i, v) in input.iter_mut().enumerate() {
            *v = (i as f32) * scale;
        }
        let in_bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();
        req.set_input_bytes(0, &in_bytes).expect("set_input");
        for _ in 0..n_iter {
            req.run(QoS::Default).expect("run");
        }
        let mut out_bytes = vec![0_u8; N_BYTES];
        req.get_output_bytes(0, &mut out_bytes).expect("get_output");
        for i in 0..8 {
            let v = f32::from_le_bytes(
                out_bytes[i*4..i*4+4].try_into().expect("slice"),
            );
            prop_assert!(
                (v - input[i]).abs() < 1e-2,
                "i={i} in={} out={v}", input[i],
            );
        }
    }
}
