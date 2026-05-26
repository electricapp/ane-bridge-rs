# ane-bridge

[![CI](https://github.com/anthropics/ane-bridge/actions/workflows/ci.yml/badge.svg)](https://github.com/anthropics/ane-bridge/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/ane-bridge.svg)](https://crates.io/crates/ane-bridge)
[![docs.rs](https://docs.rs/ane-bridge/badge.svg)](https://docs.rs/ane-bridge)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![macOS](https://img.shields.io/badge/macOS-14%2B-black?logo=apple)](https://www.apple.com/macos/)
[![Rust 1.95+](https://img.shields.io/badge/Rust-1.95%2B-orange?logo=rust)](https://www.rust-lang.org/)

A thin bridge from MIL text + weights to the Apple Neural Engine.
Built on `AppleNeuralEngine.framework` (private), the same path CoreML
uses when it offloads to ANE.

- **C library** (`libane_bridge.dylib`) with a stable header (`ane_bridge.h`).
- **Rust workspace**: `ane-bridge-sys` (raw FFI) + `ane-bridge` (safe wrapper).
- **Schema derived from the framework** (via `modelAttributes`), not declared by the caller.
- **Async**: `submit` + `wait` / poll / callback / `Future`; multiple in-flight requests per model.
- **Zero-copy path**: `IOSurface` bind with ownership transfer for hot loops; byte-ptr memcpy path for convenience.
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

| Path              | min   | p50   | p90   | p99   | max   | mean  | std  |
| ----------------- | ----- | ----- | ----- | ----- | ----- | ----- | ---- |
| ane-bridge        | 238   | 265   | 303   | 329   | 417   | 271   | 21   |
| CoreML (.ane)     | 262   | 293   | 311   | 333   | 512   | 293   | 16   |
| CoreML (.all)     | 263   | 297   | 315   | 339   | 436   | 297   | 15   |
| CoreML (.cpu)     | 1178  | 1191  | 1223  | 1265  | 1345  | 1198  | 19   |

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

| C call                                  | Rust                                                            |
| --------------------------------------- | --------------------------------------------------------------- |
| `ane_model_open`                        | `Model::open`                                                   |
| `ane_model_close`                       | `Drop` on `Model` (`Arc`)                                       |
| `ane_model_{input,output}_spec/_nbytes` | `Model::input(idx)`, `Model::output(idx)` → `TensorSpec`        |
| `ane_buffer_create*`                    | `Model::buffer`, `Model::input_buffer`, `Model::output_buffer`  |
| `ane_buffer_lock`/`unlock`              | `Buffer::lock` (RAII guard) or `Buffer::with_locked` (closure)  |
| `ane_request_create`                    | `Model::request`                                                |
| `ane_request_bind_input/output`         | `Request::bind_input/output` (take `Buffer` by value)           |
| —                                       | `Request::input_buffer_mut` / `output_buffer_mut`               |
| `ane_request_set/get_*_bytes`           | `Request::set_input_bytes`, `get_output_bytes`                  |
| `ane_request_submit` / `wait` / `run`   | `Request::submit` / `wait` / `run`                              |
| `ane_request_submit` + async glue       | `Request::submit_async` → `impl Future<Output=Result<()>>`      |
| `ane_request_set_completion`            | `Request::on_complete(FnMut+Send+'static)` / `clear_completion` |
| `ane_request_last_error`                | `Request::last_error`                                           |

## Threading model

- `Model` is `Send + Sync` (wraps `Arc<ModelInner>` internally).
- `Request` is `Send`, not `Sync`. One submit at a time per request.
- `Buffer` is `Send`, not `Sync`.

## Caveats

- Uses Apple's **private** `AppleNeuralEngine.framework`. Not App Store safe.
- macOS only. Apple Silicon required.
- MIL text is the input contract. No converter shipped — use `coremltools` or write MIL directly.
- Weight references: `@model_path/weights/weight.bin` — one blob per model.

## Further reading

- [ARCHITECTURE.md](ARCHITECTURE.md) — dispatch path diagrams, model/buffer/request lifecycle, error reporting, Rust safety invariants.
- [TESTING.md](TESTING.md) — test suites, CI checks, system corpus.
- [CONTRIBUTING.md](CONTRIBUTING.md) — development setup, lint policy.

## License

MIT. See `LICENSE`.
