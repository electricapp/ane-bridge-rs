# Architecture

How ane-bridge gets from "MIL text + weights" to "tensor out of the ANE."

## The stack

```
  Rust / C caller
        │
        ▼
   ane_bridge.h                 ← stable C API
        │
        ▼
   ane_bridge.m  (Obj-C)        ← lifecycle, IOSurfaces, dispatch_queue
        │
        ▼
   _ANEClient, _ANERequest,     ← private Apple classes (objc_msgSend)
   _ANEInMemoryModel,
   _ANEInMemoryModelDescriptor,
   _ANEIOSurfaceObject
        │
        ▼
   AppleNeuralEngine.framework  ← dlopen'd at startup
        │
        ▼
   _ANECompiler                 ← MIL text → ANE bytecode
        │
        ▼
   ANE hardware                 ← matmul / conv / elementwise engines
```

## Dispatch path: ane-bridge

```
┌───────────────────────────────────────────────────────────────┐
│  Rust (or C) caller                                           │
│    let req = model.request()?; req.run(QoS::Default)?;        │
└───────────────────────┬───────────────────────────────────────┘
                        │ safe API
                        ▼
┌───────────────────────────────────────────────────────────────┐
│  ane-bridge   (safe Rust crate)                               │
│    Model::open / Request::run / Buffer::with_locked           │
└───────────────────────┬───────────────────────────────────────┘
                        │ extern "C" calls
                        ▼
┌───────────────────────────────────────────────────────────────┐
│  ane-bridge-sys   (raw FFI crate)                             │
│    pub fn ane_model_open(...) -> AneStatus;                   │
│    build.rs compiles ane_bridge.m via the cc crate            │
└───────────────────────┬───────────────────────────────────────┘
                        │ C ABI
                        ▼
┌───────────────────────────────────────────────────────────────┐
│  libane_bridge.a   (Obj-C, compiled into the Rust binary)     │
│    ane_bridge.m: lifecycle, dispatch_queue, IOSurfaces         │
│    ane_private.m: dlopen + class lookups + getUUID stub        │
└───────────────────────┬───────────────────────────────────────┘
                        │ objc_msgSend casts on private classes
                        ▼
┌───────────────────────────────────────────────────────────────┐
│  AppleNeuralEngine.framework (PrivateFrameworks)              │
│    _ANEInMemoryModelDescriptor                                │
│    _ANEInMemoryModel        (compile + load + getUUID)        │
│    _ANEClient               (doEvaluateDirectWithModel:…)     │
│    _ANERequest              (binds inputs/outputs by index)   │
│    _ANEIOSurfaceObject      (wraps IOSurface for ANE DMA)     │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────────┐
│  _ANECompiler  →  ANE bytecode  →  ANE silicon                │
└───────────────────────────────────────────────────────────────┘
```

| Layer                         | Role                                                     | Per-call cost                       |
| ----------------------------- | -------------------------------------------------------- | ----------------------------------- |
| Rust caller                   | Typed handle, RAII, async API                            | —                                   |
| `extern "C"`                  | Stable C ABI; pointer/int args                           | function call (LTO usually inlines) |
| Obj-C glue (`ane_bridge.m`)   | IOSurface lifetimes, dispatch_queue, `_ANERequest` build | µs (NSArray build only)             |
| `objc_msgSend` casts          | Selector invocation on private classes                   | virtual call                        |
| `AppleNeuralEngine.framework` | Private dispatcher; bridges to the ANE driver            | µs to enter                         |
| `_ANECompiler`                | MIL → ANE bytecode; cached by `hexStringIdentifier`      | 200-500 ms at open; 0 per call      |
| ANE bytecode → silicon        | Actual execution on matmul/conv/eltwise units            | runtime cost                        |

## Dispatch path: CoreML MLModel (for comparison)

```
┌───────────────────────────────────────────────────────────────┐
│  Python caller                                                │
│    out = model.predict({"x": numpy_array})                    │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────────┐
│  coremltools.models.MLModel  (Python)                         │
│    thin wrapper over Obj-C MLModel via pyobjc                 │
└───────────────────────┬───────────────────────────────────────┘
                        │ pyobjc bridge
                        ▼
┌───────────────────────────────────────────────────────────────┐
│  pyobjc marshalling                                           │
│    numpy.ndarray → MLMultiArray → IOSurface                   │
│    full memcpy each direction; cost scales with tensor size   │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────────┐
│  CoreML.framework  (public)                                   │
│    -[MLModel predictionFromFeatures:error:]                   │
└───────────────────────┬───────────────────────────────────────┘
                        │ per-op routing decision
                        ▼
┌───────────────────────────────────────────────────────────────┐
│  compute-unit dispatcher                                      │
│    splits the graph across CPU / GPU / ANE                    │
│    silent fallback if any op is ANE-ineligible                │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────────┐
│  AppleNeuralEngine.framework (PrivateFrameworks)              │
│    same hardware target — different entry path                │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────────┐
│  _ANECompiler  →  ANE bytecode  →  ANE silicon                │
└───────────────────────────────────────────────────────────────┘
```

### Key differences

|                            | ane-bridge                                 | CoreML MLModel (Python)                                    |
| -------------------------- | ------------------------------------------ | ---------------------------------------------------------- |
| Layers above ANE.framework | 4 (Rust → C → ObjC → msgSend)              | 6 (Python → coremltools → pyobjc → MLModel → asset → disp) |
| Cold start                 | ~200-500 ms (compile only)                 | 1-10 s first load, then cached                             |
| Per-call overhead (small)  | IOSurface bind (~µs)                       | numpy/MLMultiArray marshalling (~10s of µs)                |
| Per-call overhead (large)  | same — IOSurface is zero-copy              | proportional to bytes — full memcpy each direction         |
| Op routing                 | all-or-nothing ANE; hard fail on rejection | automatic per-op fallback to GPU/CPU                       |
| Model format               | MIL text + single weights blob             | `.mlpackage` (signed bundle, metadata, multifile)          |
| State management           | manual (caller owns buffers between calls) | `MLState` on iOS 18+ / macOS 15+                           |
| Op-fusion / layout control | whatever the MIL program specifies         | whatever `coremlc` chose                                   |
| App Store                  | no — links private symbols                 | yes                                                        |

## Model lifecycle

```
ane_model_open(opts):
    dlopen AppleNeuralEngine + look up the 5 private classes (once)
    read MIL  → NSData
    read wts  → NSData
    descriptor   = +[_ANEInMemoryModelDescriptor modelWithMILText:weights:optionsPlist:]
    in_memory    = +[_ANEInMemoryModel inMemoryModelWithDescriptor:]
    hex_id       = -[in_memory hexStringIdentifier]
    mkdir + write MIL + weights to NSTemporaryDirectory()/<hex_id>/...
    -[in_memory compileWithQoS:options:error:]   ← invokes _ANECompiler
    -[in_memory loadWithQoS:options:error:]      ← places weights on ANE
    client       = +[_ANEClient new]
    return AneModel*
```

The `hexStringIdentifier` write step is a quirk: even though the
"in-memory" descriptor is used, `_ANECompiler` reads MIL and weights
from disk under a path keyed by that hex ID. Missing file → compile
fails with an opaque error.

## Buffer lifecycle

```
ane_buffer_create(nbytes):
    surface = IOSurfaceCreate({width: ..., bytesPerElement: 1, ...})
    wrap    = +[_ANEIOSurfaceObject objectWithIOSurface:surface]
    return AneBuffer{surface, wrap, nbytes}

ane_buffer_lock(buf, access):
    IOSurfaceLock(buf->surface, access==READ ? kReadOnly : 0, NULL)
    return IOSurfaceGetBaseAddress(buf->surface)

ane_buffer_release(buf):
    [buf->wrap release]
    CFRelease(buf->surface)
```

IOSurfaces are kernel-shared, so once the wrapper is handed to ANE the
hardware can DMA directly to/from host memory. The `set_input_bytes`
convenience path is a memcpy into a library-owned IOSurface.

## Request + async dispatch

```
ane_request_create(model):
    queue    = dispatch_queue_create(... DISPATCH_QUEUE_SERIAL)
    done_sem = dispatch_semaphore_create(0)
    return AneRequest{model, queue, done_sem, ...}

ane_request_submit(req, qos):
    atomic CAS in_flight: 0 → 1, else return BUSY
    build _ANERequest from current bindings
    dispatch_async(req->queue, ^{
        ok = -[client doEvaluateDirectWithModel:options:request:qos:error:]
                (in_memory, @{}, req_obj, qos, &nserror)
        store status
        atomic_store(has_result, 1)
        atomic_store(in_flight, 0)
        dispatch_semaphore_signal(done_sem)
        if callback: callback(req, status, user)
    })
    return OK

ane_request_wait(req, timeout_ms):
    dispatch_semaphore_wait(done_sem, t)
    return last_status
```

The model is shared across requests; each request has its own serial
queue so request A's worker thread cannot block request B's submission.
ANE itself serializes hardware access internally.

## Error reporting

Thread-local last-error string set via `pthread_key_t`. The private
framework returns `NSError*` from compile/load/eval, copied into the
thread-local on failure. Read via `ane_last_error()`.

One quirk: `_ANEClient`'s `reportEvaluateFailure:` calls
`[in_memory getUUID]`, which the in-memory descriptor doesn't
implement. Without intervention an eval failure crashes via
`NSInvalidArgumentException` instead of returning an `NSError*`.
A stub `getUUID` is installed at startup (`ane_private_install_uuid_stub`)
returning a dummy `NSUUID`, which lets the framework surface the actual
problem.

## Rust safety invariants

- `Model` wraps `Arc<ModelInner>`; `Drop` calls `ane_model_close` (safe on any thread).
- `Request` is `Send` but not `Sync`. The C `in_flight` atomic prevents concurrent submits with `BUSY`, but does not protect concurrent `wait` against the worker thread. Single-owner `&mut self` is the correct discipline.
- `Buffer` is `Send` but not `Sync`. C-side lock count plus Rust `&mut self` borrow together ensure exclusive host access inside `with_locked`.
- Every `unsafe impl Send/Sync` and every `unsafe { ... }` block carries a `// SAFETY:` comment.
- The crate enables `clippy::pedantic + nursery + cargo`, `undocumented_unsafe_blocks = deny`, `multiple_unsafe_ops_per_block = deny`.

## Out of scope

- **Model conversion** — MIL is the input contract. No PyTorch / coremltools shim.
- **CoreML `MLState`** — ane-bridge is the direct-MIL path.
- **`.mlpackage` loading** — for the signed-bundle path, use CoreML.

## Pointers for hacking

- Private classes and selectors: `c/src/ane_private.h`.
- Dispatch model: `ane_bridge.m` under `ane_request_submit` / `ane_request_wait`.
- IOSurface layout calculation: `make_surface` — width <= 16384, bytes split into a 2D layout for larger buffers.
