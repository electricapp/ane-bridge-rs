# ane-bridge

<div align="center">

[![CI](https://github.com/electricapp/ane-bridge-rs/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/electricapp/ane-bridge-rs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/ane-bridge.svg)](https://crates.io/crates/ane-bridge)
[![docs.rs](https://docs.rs/ane-bridge/badge.svg)](https://docs.rs/ane-bridge)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![macOS](https://img.shields.io/badge/macOS-14%2B-black?logo=apple)](https://www.apple.com/macos/)
[![Rust 1.95+](https://img.shields.io/badge/Rust-1.95%2B-orange?logo=rust)](https://www.rust-lang.org/)

</div>

Safe Rust bindings and C library for the Apple Neural Engine. A thin
bridge from MIL text + weights to ANE, built on
`AppleNeuralEngine.framework` (private), the same path CoreML uses
when it offloads to ANE.

- **Safe Rust bindings** + **C library** (`libane_bridge.dylib`) with a stable header (`ane_bridge.h`).
- **Rust workspace**: `ane-bridge-sys` (raw FFI) + `ane-bridge` (safe wrapper).
- **Schema derived from the framework** (via `modelAttributes`), not declared by the caller.
- **Async**: `submit` + `wait` / poll / callback / `Future`; up to 127 in-flight requests per model.
- **Zero-copy path**: `IOSurface` bind with ownership transfer for hot loops; byte-ptr memcpy path for convenience.
- **GPU↔ANE composition**: borrow the underlying `IOSurfaceRef`; queue Metal `MTLSharedEvent` signals/waits via `_ANESharedEvents`.
- **Telemetry**: hardware execution time + raw perf counters from `_ANEPerformanceStats`; live in-flight count from `_ANEProgramForEvaluation`.
- **Multi-procedure models**: per-procedure I/O schemas and per-request procedure index.
- **Multi-blob weights**: `NSDictionary` of named weight entries (file or in-memory).
- **Chained dispatch**: `_ANEChainingRequest` pipeline — prepare once, enqueue many times.
- **File-based open**: `_ANEModel modelAtURL:key:` family with explicit cache key, `mpsConstants`, and `identifierSource`.
- **Real-time class**: `loadRealTimeModel:` + `beginRealTimeTask` for latency-critical paths.
- **Warm-start cache**: `Model::open` skips recompilation when `aned` already has a lowering for the model hash. Observable via `Model::was_cached()`.

## Testing

`cargo test --release --workspace` runs integration, property, fuzz,
borrow-safety, race, stress, and loom tests. CI adds heap guards
(`MallocGuardEdges` + `MallocScribble`), TSAN, and `leaks --atExit`.

`tests/system_corpus.rs` validates against 11 Apple-shipped MIL files
(Vision, SoundAnalysis, TextRecognition, VoiceActions) and pins
expected verdicts so CI catches drift on OS updates.

## Benchmarks

8-block conv stack (Conv2d-ReLU-Conv2d-ReLU per block, 128ch, 32x32
spatial, fp16). M-series Apple Silicon, macOS 26.x. 200 warm-up +
2000 measured calls, inputs allocated once and reused.

```bash
uv run python tools/make_bench_model.py build/bench
bash scripts/bench_vs_coreml.sh build/bench/bench.mlmodelc
```

| Path          | min  | p50  | p90  | p99  | max  | mean | std |
| ------------- | ---- | ---- | ---- | ---- | ---- | ---- | --- |
| ane-bridge    | 238  | 265  | 303  | 329  | 417  | 271  | 21  |
| CoreML (.ane) | 262  | 293  | 311  | 333  | 512  | 293  | 16  |
| CoreML (.all) | 263  | 297  | 315  | 339  | 436  | 297  | 15  |
| CoreML (.cpu) | 1178 | 1191 | 1223 | 1265 | 1345 | 1198 | 19  |

All values in microseconds. The CPU row rules out either ANE path
silently falling back.

## Quick start

The input/output schema is derived from the framework after
`loadWithQoS:`. Query with `Model::input(idx)` / `Model::output(idx)`.

### C

```c
#include "ane_bridge.h"

AneModelOpenOptions opts = {
    .mil_path = "model.mil",
    .weights_path = "weights.bin",
    .compile_qos = ANE_QOS_DEFAULT,
};
AneModel* model = NULL;
ane_model_open(&opts, &model);

/* Sizes come from the loaded model */
size_t n_in  = ane_model_input_nbytes(model, 0);
size_t n_out = ane_model_output_nbytes(model, 0);

AneRequest* req = NULL;
ane_request_create(model, &req);
ane_request_set_input_bytes(req, 0, input_data, n_in);
ane_request_run(req, ANE_QOS_DEFAULT);          /* submit + wait */
ane_request_get_output_bytes(req, 0, output_data, n_out);

ane_request_release(req);
ane_model_close(model);
```

### Rust

```rust
use ane_bridge::{Model, OpenOptions, QoS};

let opts = OpenOptions::new("model.mil", "weights.bin");
let model = Model::open(&opts)?;

// Schema derived from the loaded model
let inp = model.input(0).unwrap();
let out = model.output(0).unwrap();
println!("input  = {} {:?} {} bytes", inp.name(), inp.shape(), inp.nbytes());
println!("output = {} {:?} {} bytes", out.name(), out.shape(), out.nbytes());

let mut req = model.request()?;
req.set_input_bytes(0, &input_bytes)?;
req.run(QoS::Default)?;
req.get_output_bytes(0, &mut output_bytes)?;
```

## Zero-copy path

`IOSurface`-backed buffers skip the memcpy. Bind them into a request
(ownership transfers); access via `input_buffer_mut` /
`output_buffer_mut` between runs.

```rust
let mut in_buf = model.input_buffer(0)?;
in_buf.with_locked(ane_bridge::BufferAccess::Write, |bytes| {
    bytes.copy_from_slice(my_input_bytes);
})?;

let out_buf = model.output_buffer(0)?;
req.bind_input(0, in_buf)?;     // moved
req.bind_output(0, out_buf)?;   // moved

req.run(QoS::Default)?;

// Read back via the request's view of the bound buffer
let out_ref = req.output_buffer_mut(0).expect("bound + done");
out_ref.with_locked(ane_bridge::BufferAccess::Read, |bytes| {
    // ... consume bytes ...
})?;
```

### Resident state (MLState)

A MIL program can declare state tensors that stay on the ANE across
evaluations. A state buffer is bound once and updated in place by the
program's `read_state` / `coreml.update_state` ops — it never crosses the
host boundary per call, so a streaming cache stays resident instead of
being shipped in and out every submit.

```rust
let mut req = model.request()?;
let cache = model.state_buffer(0)?;   // zeroed, sized for state slot 0
req.bind_state(0, cache)?;            // persists + updates in place across submits
req.bind_input(0, x)?;
req.run(QoS::Default)?;               // state survives into the next run
```

Bind every input, output, and state before the first submit. Initialize the
state buffer before the first run. The buffer must outlive the request.

> Note: the schema surface (`Model::num_states` / `state` / `state_nbytes` /
> `state_buffer`) is live, but `bind_state` on this `_ANE` path returns
> `Unsupported`: state ops do not compile through the private `_ANEModel` path
> on current macOS (`ANECCompile` rejects them). For working ANE-resident
> state, use the CoreML-backed `StateModel` below.

### Stateful inference via `StateModel` (CoreML-backed)

For models that declare state, `StateModel` drives CoreML's `MLModel` +
`MLState` — the MLE5Engine / E5RT path that keeps the state resident on the
Neural Engine across calls and needs no ANE entitlement. Only the (small)
named inputs/outputs cross the host boundary per step; the state never does.

```rust
let m = StateModel::open("model.mlmodelc")?;      // a model declaring state
let mut state = m.new_state()?;                   // KV cache, resident on the ANE
let mut y = vec![0.0_f32; m.output_count("y")];
m.predict(&mut state, &[("x", &x)], &mut [("y", &mut y)])?; // state updates in place
```

## Async

`submit` is non-blocking. Multiple `Request` objects can overlap
evaluations against the same model:

```rust
let mut req_a = model.request()?;
let mut req_b = model.request()?;
req_a.submit(QoS::Default)?;
req_b.submit(QoS::Default)?;
req_a.wait(-1)?;
req_b.wait(-1)?;
```

Completion callbacks available on the C side via `ane_request_set_completion`.

## Building

```bash
make                   # build libane_bridge.dylib
make examples          # build C example
make test              # build + run C identity example
make rust              # cargo build --workspace
make rust-test         # run Rust identity example
```

`cargo build` works standalone — `build.rs` compiles the C/Obj-C
sources via the `cc` crate.

## API surface

| C call                                   | Rust                                                            |
| ---------------------------------------- | --------------------------------------------------------------- |
| `ane_model_open`                         | `Model::open`                                                   |
| `ane_model_close`                        | `Drop` on `Model` (`Arc`)                                       |
| `ane_model_{input,output}_spec/_nbytes`  | `Model::input(idx)`, `Model::output(idx)` → `TensorSpec`        |
| `ane_buffer_create*`                     | `Model::buffer`, `Model::input_buffer`, `Model::output_buffer`  |
| `ane_buffer_lock`/`unlock`               | `Buffer::lock` (RAII guard) or `Buffer::with_locked` (closure)  |
| `ane_request_create`                     | `Model::request`                                                |
| `ane_request_bind_input/output`          | `Request::bind_input/output` (take `Buffer` by value)           |
| —                                        | `Request::input_buffer_mut` / `output_buffer_mut`               |
| `ane_model_num_states`/`state_spec/_nbytes` | `Model::num_states`, `Model::state(idx)`, `Model::state_nbytes` |
| `ane_buffer_create_for_state`            | `Model::state_buffer`                                          |
| `ane_request_bind_state`                 | `Request::bind_state` (take `Buffer` by value)                 |
| `ane_request_set/get_*_bytes`            | `Request::set_input_bytes`, `get_output_bytes`                  |
| `ane_request_submit` / `wait` / `run`    | `Request::submit` / `wait` / `run`                              |
| `ane_request_submit` + async glue        | `Request::submit_async` → `impl Future<Output=Result<()>>`      |
| `ane_request_set_completion`             | `Request::on_complete(FnMut+Send+'static)` / `clear_completion` |
| `ane_request_last_error`                 | `Request::last_error`                                           |
| `ane_model_open_ex`                      | `Model::open_ex(&OpenOptionsEx)`                                |
| `ane_model_open_file{,_ex}`              | `Model::open_file{,_ex}`                                        |
| `ane_model_open_realtime{,_ex}`          | `Model::open_realtime{,_ex}`                                    |
| `ane_realtime_task_{begin,end}`          | `realtime_task_{begin,end}()`                                   |
| `ane_model_num_procedures`               | `Model::num_procedures`                                         |
| `ane_model_*_for_procedure`              | `Model::*_for_procedure`                                        |
| `ane_request_set_procedure_index`        | `Request::set_procedure_index`                                  |
| `ane_request_set_weights`                | `Request::set_weights`                                          |
| `ane_perf_stats_*` + `*_perf_stats_mask` | `PerfStats`, `Model::set_perf_stats_mask`                       |
| `ane_request_set_perf_stats`             | `Request::set_perf_stats`                                       |
| `ane_shared_events_*`                    | `SharedEvents`                                                  |
| `ane_request_set_shared_events`          | `Request::set_shared_events`                                    |
| `ane_buffer_iosurface_ref`               | `Buffer::iosurface_ref`                                         |
| `ane_buffer_adopt_iosurface`             | `Buffer::adopt_iosurface` (unsafe)                              |
| `ane_chain_{create,prepare,enqueue}`     | `Chain::{new,prepare,enqueue}`                                  |
| `ane_model_queue_depth` / `in_flight`    | `Model::queue_depth` / `in_flight`                              |
| `ane_model_program_id` / `weights_hash`  | `Model::program_id` / `weights_hash`                            |
| `ane_cache_{exists,purge}_for_hash`      | `cache_{exists,purge}_for_hash`                                 |
| `ane_device_info`                        | `device_info()`                                                 |
| `ane_session_hint_*`                     | `SessionHint`, `Model::apply_session_hint`                      |
| `ane_model_new_instance`                 | `Model::new_instance`                                           |
| `ane_decompress_weights`                 | `decompress_weights`                                            |

## Threading model

- `Model` is `Send + Sync` (wraps `Arc<ModelInner>` internally).
- `Request` is `Send`, not `Sync`. One submit at a time per request.
- `Buffer` is `Send`, not `Sync`.

## Caveats

- Uses Apple's **private** `AppleNeuralEngine.framework`.
- MIL text is the input contract. No converter shipped — use `coremltools` or write MIL directly.
- Weight references resolve against the `NSDictionary` passed at open — one entry per named blob.

## Further reading

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — dispatch path diagrams, model/buffer/request lifecycle, error reporting, Rust safety invariants.
- [docs/TESTING.md](docs/TESTING.md) — test suites, CI checks, system corpus.
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — development setup, lint policy.
- [docs/DEBUNK.md](docs/DEBUNK.md) — why ANE ≠ SME, with measured evidence.

## License

MIT. See `LICENSE`.
