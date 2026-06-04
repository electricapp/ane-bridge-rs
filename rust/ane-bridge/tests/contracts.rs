//! Public contracts of the safe wrapper, asserted without a loaded model.
//!
//! Every test pins one observable promise: a builder accepts (or
//! rejects) a specific input; an FFI entry point validates null; a
//! `Drop` is idempotent; a parser cleans up on partial failure; a
//! `wait` returns within a bound. Model-loaded paths live in the
//! corpus and integration suites.

#![allow(clippy::cast_possible_truncation)]

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
fn perf_stats_lifecycle_is_stable() {
    if PerfStats::new().is_err() {
        return;
    }
    for _ in 0..LIFECYCLE_ITERS {
        let ps = PerfStats::new().expect("create");
        assert_eq!(ps.hw_execution_ns(), 0);
        let _ = ps.counters();
    }
}

#[test]
fn shared_events_lifecycle_is_stable() {
    if SharedEvents::new().is_err() {
        return;
    }
    for _ in 0..LIFECYCLE_ITERS {
        let ev = SharedEvents::new().expect("create");
        assert_eq!(ev.num_signals(), 0);
        assert_eq!(ev.num_waits(), 0);
    }
}

#[test]
fn session_hint_lifecycle_is_stable() {
    use sys::AneSessionHintKind::*;
    for kind in [Prefetch, LowLatency, HighThroughput] {
        for _ in 0..LIFECYCLE_ITERS {
            let _ = SessionHint::new(kind);
        }
    }
}

#[test]
fn device_info_is_idempotent() {
    if device_info().is_err() {
        return;
    }
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
        InvalidArg, Io, Compile, Load, Eval, Oom, Unsupported, Timeout, Busy, NotDone, Internal,
    ] {
        assert_ne!(c, Ok);
    }
}
