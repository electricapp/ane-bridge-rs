# Testing

## Test suites

| Suite                     | What it catches                                                                                                                                                                              |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `integration.rs` (24)     | Schema derived from framework, error paths, byte-ptr + zero-copy round-trips, multi-IO, concurrent threads                                                                                   |
| `contracts.rs` (27)       | Public-API contracts without a loaded model: builder rejects, FFI null validation, `Drop` idempotency, parser cleanup on partial failure, entitlement-gated paths return `Unsupported`       |
| `api_complete.rs` (3)     | C-header ↔ Rust FFI ↔ dylib symbol agreement — fails if any of the three drifts                                                                                                              |
| `cache_warm_start.rs` (3) | `Model::open` is a cache hit on the second call against `aned`; `Model::was_cached()` reflects it                                                                                            |
| `property.rs` (proptest)  | Random buffer sizes, random indices, random shapes, byte-count fuzzing through the safe wrapper                                                                                              |
| `parser_fuzz.rs` (36)     | Adversarial `modelAttributes` fuzz (proptest) — missing keys, wrong types, `NSNull`, overflow dims, `NSDecimal`, embedded-NUL names; case count tunable via `ANE_FUZZ_CASES`                 |
| `corpus.rs` (1)           | Hand-built MIL corpus (fp32/fp16, multi-IO, asymmetric shapes) through `Model::open` → `modelAttributes` derivation                                                                          |
| `system_corpus.rs` (1)    | Real Apple-shipped MIL under `/System/Library/Frameworks/...`; each entry pins an `Expect` verdict, test fails on drift                                                                      |
| `borrow_safety.rs` (9)    | UAF on drop-after-bind, mid-flight buffer access, double-free between Rust + C, `on_complete` drain                                                                                          |
| `race.rs` (2)             | Parallel `Model::open` from a fresh process — exercises the `dispatch_once` framework init                                                                                                   |
| `stress.rs` (9)           | Rapid create/destroy, drop-during-callback, deterministic autoreleasepool-leak check via `malloc_zone_statistics`                                                                            |
| `loom_async.rs` (2)       | Exhaustive interleaving of `EvalFuture` poll vs. completion callback                                                                                                                         |

## CI checks

| Check            | What it does                                                                        |
| ---------------- | ----------------------------------------------------------------------------------- |
| heap guards      | `MallocGuardEdges` + `MallocScribble` + `MallocCheckHeapEach` re-run of full matrix |
| TSAN             | C/Obj-C built with `-fsanitize=thread`; race + stress tests, zero warnings          |
| `leaks --atExit` | macOS `leaks` against the stress binary — zero unfreed bytes                        |

Two extra checks live behind `workflow_dispatch` triggers (not in the per-PR loop):

- **fuzz-soak**: `parser_fuzz` with `ANE_FUZZ_CASES=100000` per property (~6x default) under heap guards.
- **coverage**: rebuilds C side with LLVM source-based coverage, exercises the parser, uploads HTML report. Locally: `./scripts/coverage.sh`.

## Apple system MIL corpus

`tests/system_corpus.rs` points the bridge at 11 `model.mil` files
Apple ships under `/System/Library/Frameworks/...` and
`/System/Library/PrivateFrameworks/...`. Each entry pins an expected
verdict; the test fails on drift.

| Model                                 | OS target | dtype + notable          | Expected verdict   |
| ------------------------------------- | --------- | ------------------------ | ------------------ |
| `Vision/vmtracker`                    | ios18     | fp16 + `pixel_buffer`    | ANE-loadable       |
| `SoundAnalysis/AVFuser`               | ios18     | fp16                     | ANE-loadable       |
| `TextRecognition/cr_td`               | (current) | fp16 + uint8             | ANE-loadable       |
| `VoiceActions/aa_encoder`             | (current) | fp16                     | ANE-loadable       |
| `SoundAnalysis/AudioEncoder`          | ios18     | fp16 + LUT-palettized    | CompilationFailure |
| `HDRProcessing/sceneLux`              | ios15     | fp32 `[1, 4]`            | CompilationFailure |
| `DuetExpert/ATXIntentPrediction`      | ios15     | fp32 `[?, 17]` (dynamic) | InvalidMILProgram  |
| `DuetExpert/ATXAppPrediction`         | ios15     | fp32                     | InvalidMILProgram  |
| `IntelligencePlatform/MentionGen`     | ios15     | fp32                     | InvalidMILProgram  |
| `IntelligenceFlow/PlanResolution`     | ios15     | fp32                     | InvalidMILProgram  |
| `IntelligencePlatform/EntityReranker` | n/a       | n/a                      | UnreadableWeights  |

To add a candidate, append `(label, Expect::*, mil_path)` to
`CANDIDATES` in `tests/system_corpus.rs`.

## Running locally

```bash
cd rust
cargo test --release --workspace
```
