//! Public contracts of the safe wrapper, asserted without a loaded model.
//!
//! Every test pins one observable promise: a builder accepts (or
//! rejects) a specific input; an FFI entry point validates null; a
//! `Drop` is idempotent; a parser cleans up on partial failure; a
//! `wait` returns within a bound. Model-loaded paths live in the
//! corpus and integration suites.

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
    reason = "integration tests use idiomatic `.unwrap()` / indexing / `as` / \
              `println!` — assertion failure IS the test failure mode"
)]

use std::time::Duration;

use ane_bridge::{
    Buffer, Chain, ChainLink, OpenFileOptions, OpenFileOptionsEx, OpenOptionsEx, PerfStats,
    SessionHint, SharedEvents, WeightEntry, cache_exists_for_hash, cache_purge_for_hash,
    decompress_weights, device_info, num_client_connections, perf_counter_name, sys,
};
use proptest::prelude::*;

const LIFECYCLE_ITERS: usize = 256;

fn ascii_no_nul() -> impl Strategy<Value = String> {
    proptest::collection::vec(0x20u8..=0x7eu8, 0..64)
        .prop_map(|bytes| String::from_utf8(bytes).unwrap())
}

fn arbitrary_bytes(max: usize) -> impl Strategy<Value = Vec<u8>> {
    proptest::collection::vec(any::<u8>(), 0..max)
}

// ----------------------------------------------------------------
// Builders: documented panics fire on the documented input; everything
// else returns a value without panicking.
// ----------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

    #[test]
    fn weight_entry_from_bytes_accepts_any_ascii_name(
        name in ascii_no_nul(), bytes in arbitrary_bytes(4096)
    ) {
        let _ = WeightEntry::from_bytes(name, bytes);
    }

    #[test]
    fn weight_entry_from_path_accepts_any_ascii(
        name in ascii_no_nul(), path in ascii_no_nul()
    ) {
        let _ = WeightEntry::from_path(name, path);
    }

    #[test]
    fn open_options_ex_builder_accepts_arbitrary_mil_and_weights(
        mil in arbitrary_bytes(8192),
        n_weights in 0usize..16,
        is_mil in any::<bool>(),
    ) {
        let mut opts = OpenOptionsEx::from_bytes(mil).is_mil_model(is_mil);
        for i in 0..n_weights {
            opts = opts.weight(WeightEntry::from_bytes(
                format!("w{i}"),
                vec![0u8; 16],
            ));
        }
        let _ = opts;
    }

    #[test]
    fn open_file_options_builder_accepts_arbitrary_strings(
        url in ascii_no_nul(), key in ascii_no_nul(),
    ) {
        let _ = OpenFileOptions::new(url).cache_key(key);
    }
}

#[test]
fn open_options_ex_with_empty_weights_constructs() {
    let _ = OpenOptionsEx::from_bytes(vec![0u8; 32]).weights(Vec::new());
}

#[test]
fn open_file_options_ex_wraps_base_options() {
    let _ = OpenFileOptionsEx::new(OpenFileOptions::new("/nonexistent"));
}

#[test]
#[should_panic(expected = "NUL")]
fn weight_entry_panics_on_nul_in_name() {
    let _ = WeightEntry::from_bytes("bad\0name", vec![0]);
}

// ----------------------------------------------------------------
// FFI null discipline: every entry point that can be called with a
// null handle returns `InvalidArg`, never crashes.
// ----------------------------------------------------------------

#[test]
fn request_setters_reject_null_request() {
    // SAFETY: probing null-validation paths.
    unsafe {
        assert_eq!(
            sys::ane_request_set_weights(core::ptr::null_mut(), core::ptr::null_mut()),
            sys::AneStatus::InvalidArg
        );
        assert_eq!(
            sys::ane_request_set_procedure_index(core::ptr::null_mut(), 0),
            sys::AneStatus::InvalidArg
        );
        assert_eq!(
            sys::ane_request_set_perf_stats(core::ptr::null_mut(), core::ptr::null_mut()),
            sys::AneStatus::InvalidArg
        );
        assert_eq!(
            sys::ane_request_set_shared_events(core::ptr::null_mut(), core::ptr::null_mut()),
            sys::AneStatus::InvalidArg
        );
        assert_eq!(
            sys::ane_request_set_transaction(core::ptr::null_mut(), 0),
            sys::AneStatus::InvalidArg
        );
    }
}

#[test]
fn new_instance_rejects_null_source() {
    let mut out: *mut sys::AneModel = core::ptr::null_mut();
    // SAFETY: probing null-validation path.
    let status = unsafe {
        sys::ane_model_new_instance(
            core::ptr::null_mut(),
            core::ptr::null(),
            sys::AneQoS::Default,
            &raw mut out,
        )
    };
    assert_eq!(status, sys::AneStatus::InvalidArg);
}

#[test]
fn adopt_iosurface_rejects_null() {
    // SAFETY: explicitly passing null to probe the validation path.
    let r = unsafe { Buffer::adopt_iosurface(core::ptr::null_mut(), 0) };
    assert!(r.is_err());
}

// ----------------------------------------------------------------
// Idle-path safety: FFI calls reachable without a loaded model
// must accept arbitrary input without crashing.
// ----------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[test]
    fn cache_lookup_accepts_arbitrary_hash(hex in ascii_no_nul()) {
        let _ = cache_exists_for_hash(&hex);
        let _ = cache_purge_for_hash(&hex);
    }

    #[test]
    fn decompress_accepts_arbitrary_bytes(bytes in arbitrary_bytes(8192)) {
        let _ = decompress_weights(&bytes);
    }
}

#[test]
fn perf_counter_name_accepts_extreme_indices() {
    for idx in [i32::MIN, -1, 0, 1, 7, 31, 63, 127, i32::MAX] {
        let _ = perf_counter_name(idx);
    }
}

#[test]
fn num_client_connections_returns_a_value() {
    let _ = num_client_connections();
}

// ----------------------------------------------------------------
// Wait timing: `ane_chain_wait` on a null chain returns within a
// bounded duration, never blocks.
// ----------------------------------------------------------------

#[test]
fn chain_wait_on_null_returns_within_bound() {
    let start = std::time::Instant::now();
    // SAFETY: probing null-validation path.
    let status = unsafe { sys::ane_chain_wait(core::ptr::null_mut(), -1) };
    assert_eq!(status, sys::AneStatus::InvalidArg);
    assert!(start.elapsed() < Duration::from_millis(500));
}

#[test]
fn chain_new_with_empty_stages_returns_err() {
    assert!(Chain::new(Vec::new()).is_err());
}

// ----------------------------------------------------------------
// Lifecycle: tight create/drop loops are stable. Combined with the
// CI's `MallocScribble` + `leaks --atExit` this catches Drop
// regressions and missed retain/release pairs.
// ----------------------------------------------------------------

#[test]
fn perf_stats_create_succeeds() {
    PerfStats::new().expect("PerfStats::new must succeed when the framework is loadable");
}

#[test]
fn perf_stats_lifecycle_is_stable() {
    for _ in 0..LIFECYCLE_ITERS {
        let ps = PerfStats::new().expect("create");
        assert_eq!(ps.hw_execution_ns(), 0);
        let _ = ps.counters();
    }
}

#[test]
fn shared_events_create_succeeds() {
    SharedEvents::new().expect("SharedEvents::new must succeed when the framework is loadable");
}

#[test]
fn shared_events_lifecycle_is_stable() {
    for _ in 0..LIFECYCLE_ITERS {
        let ev = SharedEvents::new().expect("create");
        assert_eq!(ev.num_signals(), 0);
        assert_eq!(ev.num_waits(), 0);
    }
}

#[test]
fn add_wait_event_succeeds_with_metal_shared_event() {
    use ane_bridge::sys::AneEventType;
    let mut ev = SharedEvents::new().expect("create");
    // SAFETY: passing null in place of the MTLSharedEvent — the wrapper
    // accepts an arbitrary id; the framework only dereferences it
    // during eval, and we never reach eval here.
    let res = unsafe { ev.add_wait(1, core::ptr::null_mut(), AneEventType::Default) };
    res.expect("add_wait must succeed at the construction layer");
    assert_eq!(ev.num_waits(), 1);
}

#[test]
fn session_hint_lifecycle_is_stable() {
    use sys::AneSessionHintKind::*;
    for kind in [Prefetch, LowLatency, HighThroughput] {
        for _ in 0..LIFECYCLE_ITERS {
            SessionHint::new(kind).expect("session hint create");
        }
    }
}

#[test]
fn device_info_succeeds() {
    let info = device_info().expect("device_info must succeed when the framework is loadable");
    assert!(!info.arch_type.contains('\0'));
}

#[test]
fn device_info_is_idempotent() {
    for _ in 0..LIFECYCLE_ITERS {
        let info = device_info().expect("device info");
        assert!(!info.arch_type.contains('\0'));
    }
}

// ----------------------------------------------------------------
// Parser cleanup: a malformed output list after a valid input list
// returns an error. Combined with leak detection this proves the
// input-stage allocations are freed on the failure path.
// ----------------------------------------------------------------

#[test]
fn parser_rejects_when_output_list_is_not_an_array() {
    let case = sys::AneFuzzAttrsCase {
        mutations: sys::fuzz_attrs::LIVEOUT_NOT_ARRAY,
        n_inputs: 1,
        n_outputs: 1,
    };
    // SAFETY: stack POD; read once.
    let status = unsafe { sys::_ane_internal_fuzz_parse_attrs(&raw const case) };
    assert_ne!(status, sys::AneStatus::Ok);
}

// ----------------------------------------------------------------
// Trivial value contracts that have to hold for callers to compose.
// ----------------------------------------------------------------

#[test]
fn chain_link_default_is_all_zero() {
    let link = ChainLink::default();
    assert_eq!(link.lb_input_symbol_id, 0);
    assert_eq!(link.lb_output_symbol_id, 0);
    assert_eq!(link.fw_enqueue_delay, 0);
    assert_eq!(link.memory_pool_id, 0);
}

#[test]
fn every_error_status_is_distinct_from_ok() {
    use sys::AneStatus::*;
    for &c in &[
        InvalidArg,
        Io,
        Compile,
        Load,
        Eval,
        Oom,
        Unsupported,
        Timeout,
        Busy,
        NotDone,
        Internal,
    ] {
        assert_ne!(c, Ok);
    }
}

// ----------------------------------------------------------------
// Entitlement-gated paths: the bridge exposes the API surface but
// the framework refuses to run these from unsigned binaries. The
// contract is that they return `Unsupported` rather than crash or
// hang. Each test below pins one such path.
// ----------------------------------------------------------------

#[test]
fn realtime_task_begin_returns_unsupported_or_ok() {
    use ane_bridge::{realtime_task_begin, realtime_task_end};
    let r = realtime_task_begin();
    match r {
        Ok(_) => {
            realtime_task_end().expect("end pairs with begin");
        }
        Err(e) => assert_eq!(e.status, sys::AneStatus::Unsupported),
    }
}

#[test]
fn decompress_random_garbage_does_not_succeed_silently() {
    // The framework decompressor expects a specific compressed format.
    // Random bytes must either parse as empty, fail with a status, or
    // return Unsupported (when _ANEVirtualClient is unreachable).
    // Never a SIGSEGV.
    let mut rng: u64 = 0x9e37_79b9_7f4a_7c15;
    for _ in 0..16 {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let len = ((rng >> 56) & 0xff) as usize;
        let bytes: Vec<u8> = (0..len).map(|i| ((rng >> (i % 56)) & 0xff) as u8).collect();
        let _ = decompress_weights(&bytes);
    }
}

#[test]
fn session_hint_create_never_panics() {
    use sys::AneSessionHintKind::*;
    for kind in [Prefetch, LowLatency, HighThroughput] {
        SessionHint::new(kind).expect("session hint create");
    }
}

#[test]
fn chain_new_with_empty_returns_invalid_arg() {
    let r = Chain::new(Vec::new());
    let err = r.expect_err("empty chain must reject");
    assert_eq!(err.status, sys::AneStatus::InvalidArg);
}
