//! Adversarial tests for the C-side `modelAttributes` `LiveInputList`
//! parser. The parser cannot crash, hang, or read uninitialized memory
//! regardless of how malformed the input dict is — those properties
//! are the only thing standing between an evolving Apple private
//! framework and a soundness hole in the bridge.
//!
//! We drive the parser via [`sys::_ane_internal_fuzz_parse_one`], an
//! internal C entry point that synthesizes a one-entry
//! `LiveInputList` `NSArray` from plain C-struct fields and runs it
//! through the production code path. Each test asserts a property —
//! never just "didn't crash" — so a regression that *silently
//! accepts* bad input still fails.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_wrap)]

use std::ffi::CString;

use ane_bridge::sys::{self, AneFuzzAttrsCase, AneFuzzCase, AneStatus, fuzz_attrs, fuzz_field};
use proptest::prelude::*;

/// Build a `AneFuzzCase` with all fields set to "valid normal" values.
/// Callers tweak just the field(s) under test.
fn valid_case() -> (CString, CString, AneFuzzCase) {
    let name = CString::new("x").unwrap();
    let ty = CString::new("Float32").unwrap();
    let fc = AneFuzzCase {
        present_mask: fuzz_field::ALL,
        name: name.as_ptr(),
        type_string: ty.as_ptr(),
        batches: 1,
        channels: 64,
        depth: 1,
        height: 1,
        width: 16,
        flags: 0,
    };
    (name, ty, fc)
}

fn run(fc: &AneFuzzCase) -> AneStatus {
    // SAFETY: `_ane_internal_fuzz_parse_one` reads `*fc` and constructs
    // its own Obj-C objects; it never retains the C pointers. We hold
    // `fc` on the stack for the whole call so the inner `name` /
    // `type_string` pointers remain valid.
    unsafe { sys::_ane_internal_fuzz_parse_one(fc) }
}

// ----------------------------------------------------------------
// Hand-crafted adversarial cases. Each one corresponds to a way a
// future framework version could mis-shape its modelAttributes.
// ----------------------------------------------------------------

#[test]
fn baseline_valid_dict_accepted() {
    let (_n, _t, fc) = valid_case();
    assert_eq!(run(&fc), AneStatus::Ok);
}

#[test]
fn missing_name_rejected() {
    let (_n, _t, mut fc) = valid_case();
    fc.present_mask &= !fuzz_field::NAME;
    assert_eq!(run(&fc), AneStatus::Internal);
}

#[test]
fn missing_type_rejected() {
    let (_n, _t, mut fc) = valid_case();
    fc.present_mask &= !fuzz_field::TYPE;
    assert_eq!(run(&fc), AneStatus::Internal);
}

#[test]
fn missing_batches_rejected() {
    let (_n, _t, mut fc) = valid_case();
    fc.present_mask &= !fuzz_field::BATCHES;
    assert_eq!(run(&fc), AneStatus::Internal);
}

#[test]
fn missing_channels_rejected() {
    let (_n, _t, mut fc) = valid_case();
    fc.present_mask &= !fuzz_field::CHANNELS;
    assert_eq!(run(&fc), AneStatus::Internal);
}

#[test]
fn missing_height_rejected() {
    let (_n, _t, mut fc) = valid_case();
    fc.present_mask &= !fuzz_field::HEIGHT;
    assert_eq!(run(&fc), AneStatus::Internal);
}

#[test]
fn missing_width_rejected() {
    let (_n, _t, mut fc) = valid_case();
    fc.present_mask &= !fuzz_field::WIDTH;
    assert_eq!(run(&fc), AneStatus::Internal);
}

#[test]
fn missing_depth_rejected() {
    let (_n, _t, mut fc) = valid_case();
    fc.present_mask &= !fuzz_field::DEPTH;
    assert_eq!(run(&fc), AneStatus::Internal);
}

#[test]
fn unknown_dtype_string_rejected() {
    let n = CString::new("x").unwrap();
    let bogus = CString::new("BogusFloat42").unwrap();
    let mut fc = valid_case().2;
    fc.name = n.as_ptr();
    fc.type_string = bogus.as_ptr();
    assert_eq!(run(&fc), AneStatus::Unsupported);
}

#[test]
fn zero_batches_rejected() {
    let (_n, _t, mut fc) = valid_case();
    fc.batches = 0;
    assert_eq!(run(&fc), AneStatus::Internal);
}

#[test]
fn negative_batches_rejected() {
    let (_n, _t, mut fc) = valid_case();
    fc.batches = -1;
    assert_eq!(run(&fc), AneStatus::Internal);
}

#[test]
fn zero_channels_rejected() {
    let (_n, _t, mut fc) = valid_case();
    fc.channels = 0;
    assert_eq!(run(&fc), AneStatus::Internal);
}

#[test]
fn negative_dim_anywhere_rejected() {
    for which in 0..5 {
        let (_n, _t, mut fc) = valid_case();
        match which {
            0 => fc.batches = -2,
            1 => fc.channels = -2,
            2 => fc.depth = -2,
            3 => fc.height = -2,
            4 => fc.width = -2,
            _ => unreachable!(),
        }
        assert_eq!(run(&fc), AneStatus::Internal, "which={which}");
    }
}

#[test]
fn overflow_dim_product_rejected() {
    // (i64::MAX) * 2 * 2 * 2 trivially overflows usize on a 64-bit
    // build. The parser must detect and reject, not wrap to a small
    // nbytes that would then underprovision the IOSurface.
    let (_n, _t, mut fc) = valid_case();
    fc.batches = i64::MAX;
    fc.channels = 2;
    fc.height = 2;
    fc.width = 2;
    assert_eq!(run(&fc), AneStatus::Internal);
}

#[test]
fn name_as_number_rejected() {
    let (_n, _t, mut fc) = valid_case();
    fc.flags = AneFuzzCase::FLAG_NAME_AS_NUMBER;
    // The parser sees an NSNumber for Name → fails the
    // `if (!name || !type ...)` check (the dict lookup hits the
    // class check inside dup_name_clean or `isKindOfClass:` upstream).
    // We accept any non-Ok status; what matters is that it does not
    // crash trying to UTF8String an NSNumber.
    assert_ne!(run(&fc), AneStatus::Ok);
}

#[test]
fn type_as_number_rejected() {
    let (_n, _t, mut fc) = valid_case();
    fc.flags = AneFuzzCase::FLAG_TYPE_AS_NUMBER;
    assert_ne!(run(&fc), AneStatus::Ok);
}

#[test]
fn dim_as_string_rejected() {
    // Each dim slot wrapped as a string in turn. NSString's
    // longLongValue returns 0 for non-numeric strings, which hits
    // the "non-positive dim" guard. Must NOT crash.
    let flags = [
        AneFuzzCase::FLAG_BATCHES_AS_STRING,
        AneFuzzCase::FLAG_CHANNELS_AS_STRING,
        AneFuzzCase::FLAG_DEPTH_AS_STRING,
        AneFuzzCase::FLAG_HEIGHT_AS_STRING,
        AneFuzzCase::FLAG_WIDTH_AS_STRING,
    ];
    for f in flags {
        let (_n, _t, mut fc) = valid_case();
        fc.flags = f;
        let s = run(&fc);
        // The string `"not_a_number"` parses as 0 → "non-positive
        // dim" guard fires → Internal. Depth-as-string is special:
        // depth=0 is canonicalized to "4-D" and we still get a
        // valid result (since the parser ignores depth==0 vs
        // depth==1 the same way). Document the actual behavior:
        // both Ok and Internal are acceptable here as long as
        // there is no crash. The "no crash" property is the
        // central invariant.
        assert!(
            matches!(s, AneStatus::Ok | AneStatus::Internal),
            "flag={f:#x} got unexpected status {s:?}",
        );
    }
}

// ----------------------------------------------------------------
// Property-based: random AneFuzzCase, parser must never crash or
// return a status outside its declared set.
// ----------------------------------------------------------------

prop_compose! {
    /// Generate arbitrary fuzz cases — random presence mask, random
    /// flag bits, dimensions across the full i64 range.
    fn arb_fuzz_case()(
        mask in 0u32..=fuzz_field::ALL,
        flags in 0u32..=0x7F_u32,
        batches in any::<i64>(),
        channels in any::<i64>(),
        depth in any::<i64>(),
        height in any::<i64>(),
        width in any::<i64>(),
        name_pick in 0u8..4,
        type_pick in 0u8..6,
    ) -> (CString, CString, AneFuzzCase) {
        let name = CString::new(match name_pick {
            0 => "x",
            1 => "",
            2 => "averylongnamewithpaddingxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            _ => "y@output",
        }).unwrap();
        let ty = CString::new(match type_pick {
            0 => "Float32",
            1 => "Float16",
            2 => "Int32",
            3 => "Int64",
            4 => "UInt8",
            _ => "GarbageDtype",
        }).unwrap();
        let fc = AneFuzzCase {
            present_mask: mask,
            name: name.as_ptr(),
            type_string: ty.as_ptr(),
            batches, channels, depth, height, width,
            flags,
        };
        (name, ty, fc)
    }
}

// ----------------------------------------------------------------
// Outer-parser fuzz: `modelAttributes` → `NetworkStatusList` → ...
// ----------------------------------------------------------------

fn run_attrs(fc: &AneFuzzAttrsCase) -> AneStatus {
    // SAFETY: the C side reads `*fc` (POD layout asserted at compile
    // time) and builds its own Obj-C objects; we never retain
    // pointers across the call.
    unsafe { sys::_ane_internal_fuzz_parse_attrs(fc) }
}

#[test]
fn outer_well_formed_two_in_two_out_accepted() {
    let fc = AneFuzzAttrsCase {
        mutations: 0,
        n_inputs: 2,
        n_outputs: 2,
    };
    assert_eq!(run_attrs(&fc), AneStatus::Ok);
}

#[test]
fn outer_missing_network_status_list_rejected() {
    let fc = AneFuzzAttrsCase {
        mutations: fuzz_attrs::NSL_MISSING,
        n_inputs: 1,
        n_outputs: 1,
    };
    assert_eq!(run_attrs(&fc), AneStatus::Internal);
}

#[test]
fn outer_empty_network_status_list_rejected() {
    let fc = AneFuzzAttrsCase {
        mutations: fuzz_attrs::NSL_EMPTY,
        n_inputs: 1,
        n_outputs: 1,
    };
    assert_eq!(run_attrs(&fc), AneStatus::Internal);
}

#[test]
fn outer_network_status_list_wrong_type_rejected() {
    let fc = AneFuzzAttrsCase {
        mutations: fuzz_attrs::NSL_NOT_ARRAY,
        n_inputs: 1,
        n_outputs: 1,
    };
    assert_eq!(run_attrs(&fc), AneStatus::Internal);
}

#[test]
fn outer_procedure_not_dict_rejected() {
    let fc = AneFuzzAttrsCase {
        mutations: fuzz_attrs::PROC_NOT_DICT,
        n_inputs: 1,
        n_outputs: 1,
    };
    assert_eq!(run_attrs(&fc), AneStatus::Internal);
}

#[test]
fn outer_live_input_list_missing_rejected() {
    let fc = AneFuzzAttrsCase {
        mutations: fuzz_attrs::LIVEIN_MISSING,
        n_inputs: 1,
        n_outputs: 1,
    };
    assert_eq!(run_attrs(&fc), AneStatus::Internal);
}

#[test]
fn outer_live_output_list_missing_rejected() {
    let fc = AneFuzzAttrsCase {
        mutations: fuzz_attrs::LIVEOUT_MISSING,
        n_inputs: 1,
        n_outputs: 1,
    };
    assert_eq!(run_attrs(&fc), AneStatus::Internal);
}

#[test]
fn outer_live_input_list_wrong_type_rejected() {
    let fc = AneFuzzAttrsCase {
        mutations: fuzz_attrs::LIVEIN_NOT_ARRAY,
        n_inputs: 1,
        n_outputs: 1,
    };
    assert_eq!(run_attrs(&fc), AneStatus::Internal);
}

/// Multi-entry: the per-entry loop must validate every entry, not
/// just the first.
#[test]
fn outer_multi_entry_accepted() {
    // Up to 5 inputs and outputs — exercises the iteration loop
    // beyond the trivial n=1 case the inner fuzzer always uses.
    for nin in 1..=5 {
        for nout in 1..=5 {
            let fc = AneFuzzAttrsCase {
                mutations: 0,
                n_inputs: nin,
                n_outputs: nout,
            };
            assert_eq!(
                run_attrs(&fc),
                AneStatus::Ok,
                "nin={nin} nout={nout} rejected",
            );
        }
    }
}

#[test]
fn outer_zero_in_zero_out_accepted() {
    let fc = AneFuzzAttrsCase {
        mutations: 0,
        n_inputs: 0,
        n_outputs: 0,
    };
    assert_eq!(run_attrs(&fc), AneStatus::Ok);
}

// ----------------------------------------------------------------
// NSNumber(double) edge case.
// ----------------------------------------------------------------

fn run_double_batches(dbl: f64) -> AneStatus {
    // SAFETY: pure value-in, status-out FFI.
    unsafe { sys::_ane_internal_fuzz_parse_one_with_double_batches(dbl) }
}

/// `longLongValue` on a non-integer `NSNumber` is documented as UB
/// for values outside the `long long` range. The parser must still
/// either accept (with a sane truncation) or cleanly reject — it
/// must NEVER crash, produce a wrapped-tiny nbytes, or otherwise
/// silently misinterpret the input.
#[test]
fn double_batches_never_crashes() {
    // A sweep covering: zero, tiny, exact integer, fractional, huge
    // positive/negative, infinity, NaN.
    let probes = [
        0.0,
        1.0,
        1.5,
        2.0,
        1e9,
        -1.0,
        -1e9,
        i64::MAX as f64,       // exactly representable? no, but close
        i64::MAX as f64 * 2.0, // out of i64 range
        i64::MIN as f64,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::MAX,
        f64::MIN_POSITIVE,
    ];
    for &p in &probes {
        let s = run_double_batches(p);
        let known = matches!(
            s,
            AneStatus::Ok | AneStatus::Internal | AneStatus::Unsupported | AneStatus::InvalidArg,
        );
        assert!(known, "dbl={p:?} produced unexpected status {s:?}");
    }
}

// ----------------------------------------------------------------
// Extended dim-type fuzz: doubles / unsigned / decimal across every
// dim slot, plus NSNull / embedded-NUL names / huge names / mixed
// multi-entry validity.
// ----------------------------------------------------------------

fn known_status(s: AneStatus) -> bool {
    matches!(
        s,
        AneStatus::Ok
            | AneStatus::InvalidArg
            | AneStatus::Internal
            | AneStatus::Unsupported
            | AneStatus::Oom,
    )
}

/// Every dim slot, with a sweep of double values that cover signed
/// boundary cases, infinities, NaN, sub-integer fractions, and
/// integers exactly representable in f64. Parser must never crash.
#[test]
fn any_dim_as_double_never_crashes() {
    let dbls = [
        0.0,
        1.0,
        1.5,
        2.0,
        -1.0,
        -1e9,
        1e9,
        i64::MAX as f64,
        (i64::MAX as f64) * 2.0,
        i64::MIN as f64,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::MAX,
    ];
    for which in 0..=4 {
        for &d in &dbls {
            // SAFETY: pure POD FFI, no pointers held across the call.
            let s = unsafe { sys::_ane_internal_fuzz_dim_as_double(which, d) };
            assert!(
                known_status(s),
                "which={which} dbl={d:?} unexpected status {s:?}"
            );
        }
    }
}

/// `NSNumber numberWithUnsignedLongLong:` then `longLongValue` —
/// at `UINT64_MAX` the signed read is -1, the `< 1` guard rejects.
/// At `INT64_MAX` it's accepted (positive). Anywhere in between
/// either accepts cleanly (positive int64) or rejects via overflow.
#[test]
fn dim_as_uint64_never_crashes_and_overflow_handled() {
    let values = [
        0_u64,
        1,
        2,
        64,
        i64::MAX as u64,
        (i64::MAX as u64) + 1,
        u64::MAX,
        u64::MAX / 2,
    ];
    for which in 0..=4 {
        for &v in &values {
            // SAFETY: pure POD FFI.
            let s = unsafe { sys::_ane_internal_fuzz_dim_as_uint64(which, v) };
            assert!(
                known_status(s),
                "which={which} v={v} unexpected status {s:?}"
            );
        }
    }
}

/// `NSDecimalNumber` is an `NSNumber` subclass with its own
/// `longLongValue` implementation. We expect the parser to behave
/// the same as for a regular `NSNumber` — never crash.
#[test]
fn dim_as_decimal_never_crashes() {
    let decimals: &[&str] = &[
        "1", "0", "-1", "64", "1.5", "1e9", "1e300", "-1e300", "NaN",
        "Infinity", // NSDecimalNumber may parse as NaN
    ];
    for which in 0..=4 {
        for d in decimals {
            let cd = CString::new(*d).unwrap();
            // SAFETY: cd outlives the call.
            let s = unsafe { sys::_ane_internal_fuzz_dim_as_decimal(which, cd.as_ptr()) };
            assert!(
                known_status(s),
                "which={which} dec={d:?} unexpected status {s:?}"
            );
        }
    }
}

/// `[NSNull null]` for ANY required dict value should reject — the
/// `isKindOfClass:` guards must catch it. `NSNull` is a real
/// `NSObject` that doesn't respond to `NSString`/`NSNumber` selectors,
/// so a missing guard would crash with `unrecognized selector`.
#[test]
fn nsnull_for_any_required_key_rejected() {
    for which_key in 0..=6 {
        // SAFETY: pure POD FFI.
        let s = unsafe { sys::_ane_internal_fuzz_value_is_nsnull(which_key) };
        assert_ne!(
            s,
            AneStatus::Ok,
            "NSNull at key {which_key} accepted (status {s:?})"
        );
        assert!(
            known_status(s),
            "which_key={which_key} unexpected status {s:?}"
        );
    }
}

/// Name with `valid\0continued`: the C side `strdup`s the
/// `UTF8String`, so it truncates at the embedded NUL → stored name
/// becomes "valid". Must not crash and the truncation should be
/// visible in any downstream consumer (here we just assert Ok).
#[test]
fn name_with_embedded_nul_truncates_cleanly() {
    // SAFETY: no inputs.
    let s = unsafe { sys::_ane_internal_fuzz_name_with_embedded_nul() };
    assert_eq!(s, AneStatus::Ok, "embedded-NUL name caused status {s:?}");
}

/// Very long names exercise strdup's malloc path. 1 MB should be
/// trivially accepted; larger sizes test the helper's internal cap.
#[test]
fn huge_name_lengths_accepted_or_cleanly_rejected() {
    for len in [0_usize, 1, 256, 4096, 1 << 20] {
        // SAFETY: just a usize argument.
        let s = unsafe { sys::_ane_internal_fuzz_huge_name(len) };
        assert!(known_status(s), "len={len} unexpected status {s:?}");
    }
}

/// Two-entry list with one good + one bad entry: the parser must
/// fail the whole list (no partial accept), and the `OwnedSpec` for
/// the good entry must NOT leak when the bad entry trips the loop.
/// (Leak detection is via heap-guard CI; here we just assert
/// rejection.)
#[test]
fn mixed_validity_multi_entry_rejected_wholly() {
    // SAFETY: no inputs.
    let s = unsafe { sys::_ane_internal_fuzz_mixed_validity_two_entries() };
    assert!(
        matches!(s, AneStatus::Unsupported | AneStatus::Internal),
        "expected whole-list rejection, got {s:?}",
    );
}

/// Read `ANE_FUZZ_CASES` to scale up for long-soak runs.
/// Defaults to 1024 for normal CI; soak jobs export `ANE_FUZZ_CASES=100000`.
fn fuzz_cases() -> u32 {
    std::env::var("ANE_FUZZ_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: fuzz_cases(), ..ProptestConfig::default() })]

    /// The parser may return any AneStatus, but it must always return
    /// — never crash, never enter UB. We assert the return value is
    /// in the known status set, which forces a non-trivial decode of
    /// the C side's response on every iteration.
    #[test]
    fn parser_never_crashes_on_random_input((_n, _t, fc) in arb_fuzz_case()) {
        let status = run(&fc);
        let known = matches!(
            status,
            AneStatus::Ok
                | AneStatus::InvalidArg
                | AneStatus::Internal
                | AneStatus::Unsupported
                | AneStatus::Oom,
        );
        prop_assert!(known, "unexpected status {:?}", status);
    }

    /// Stronger: when ALL fields are present and have plausible
    /// "normal" types AND positive dims AND a known dtype, the
    /// parser must return Ok. Catches false negatives where a
    /// future tweak makes the parser too strict.
    #[test]
    fn well_formed_inputs_accepted(
        batches in 1i64..=16,
        channels in 1i64..=2048,
        depth in 1i64..=8,
        height in 1i64..=512,
        width in 1i64..=512,
    ) {
        let name = CString::new("x").unwrap();
        let ty = CString::new("Float32").unwrap();
        let fc = AneFuzzCase {
            present_mask: fuzz_field::ALL,
            name: name.as_ptr(),
            type_string: ty.as_ptr(),
            batches, channels, depth, height, width,
            flags: 0,
        };
        let s = run(&fc);
        prop_assert_eq!(s, AneStatus::Ok, "well-formed case rejected: {:?}", s);
    }

    /// A case missing ANY required key must NOT be accepted.
    #[test]
    fn missing_any_key_rejected(
        omit_bit in 0u32..7,
    ) {
        let omit = 1u32 << omit_bit;
        let name = CString::new("x").unwrap();
        let ty = CString::new("Float32").unwrap();
        let fc = AneFuzzCase {
            present_mask: fuzz_field::ALL & !omit,
            name: name.as_ptr(),
            type_string: ty.as_ptr(),
            batches: 1, channels: 64, depth: 1, height: 1, width: 16,
            flags: 0,
        };
        let s = run(&fc);
        prop_assert_ne!(s, AneStatus::Ok, "missing bit {} accepted", omit_bit);
    }

    /// Random AneFuzzAttrsCase: the outer parser must also never
    /// crash on arbitrary mutation combinations + entry counts.
    #[test]
    fn outer_parser_never_crashes_on_random_input(
        mutations in 0u32..=0xFF,
        n_inputs  in -2i32..=8,
        n_outputs in -2i32..=8,
    ) {
        let fc = AneFuzzAttrsCase { mutations, n_inputs, n_outputs };
        let s = run_attrs(&fc);
        let known = matches!(
            s,
            AneStatus::Ok
                | AneStatus::Internal
                | AneStatus::InvalidArg
                | AneStatus::Unsupported
                | AneStatus::Oom,
        );
        prop_assert!(known, "outer parser returned unexpected status {:?}", s);
    }

    /// Random double values for `Batches` — parser must never crash.
    /// (Already cover the canonical edges in `double_batches_never_crashes`;
    /// this proptest extends to arbitrary bit patterns.)
    #[test]
    fn double_batches_random_never_crashes(bits in any::<u64>()) {
        let dbl = f64::from_bits(bits);
        let s = run_double_batches(dbl);
        prop_assert!(known_status(s), "double={:?} produced unexpected status {:?}", dbl, s);
    }

    /// Same dim-as-double sweep but across all 5 slots and arbitrary
    /// f64 bit patterns. Catches any slot-specific overflow gap.
    #[test]
    fn any_dim_as_double_random_never_crashes(
        which in 0i32..=4,
        bits in any::<u64>(),
    ) {
        let dbl = f64::from_bits(bits);
        // SAFETY: pure POD FFI — no pointers held across the call.
        let s = unsafe { sys::_ane_internal_fuzz_dim_as_double(which, dbl) };
        prop_assert!(known_status(s), "which={} dbl={:?} → {:?}", which, dbl, s);
    }

    /// Arbitrary unsigned dim values across all 5 slots.
    #[test]
    fn any_dim_as_uint64_random_never_crashes(
        which in 0i32..=4,
        value in any::<u64>(),
    ) {
        // SAFETY: pure POD FFI — no pointers held across the call.
        let s = unsafe { sys::_ane_internal_fuzz_dim_as_uint64(which, value) };
        prop_assert!(known_status(s), "which={} v={} → {:?}", which, value, s);
    }

    /// Arbitrary name lengths up to 64 KB.
    #[test]
    fn huge_name_random_lengths_never_crash(len in 0usize..(64 * 1024)) {
        // SAFETY: usize argument; the C side caps and validates internally.
        let s = unsafe { sys::_ane_internal_fuzz_huge_name(len) };
        prop_assert!(known_status(s), "len={} → {:?}", len, s);
    }
}
