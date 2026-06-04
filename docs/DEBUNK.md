# Is the Apple Neural Engine just SME on the CPU?

**No.** This document refutes the claim in
[joshmorgan1000/ane/docs/DISCOVERY.md](https://github.com/joshmorgan1000/ane/blob/main/docs/DISCOVERY.md)
that "ANE = SME inside the CPU cores." The claim rests on a category
error: the author labels Accelerate's BNNS (a CPU library) as "the
Neural Engine," then observes that BNNS contends with raw SME and
concludes the units are identical. They are not — BNNS _is_ SME on
the CPU; the Neural Engine is a separate coprocessor reached through
a different kernel driver.

The evidence below is direct measurement on Apple M5 Max under
macOS, reproducible from this repo.

## 1. Power telemetry: ANE channel stays at zero during BNNS

`tools/power_witness.m` drives two phases back-to-back on the same
process: 5 s of `BNNSFilterApply` (`BNNSDataTypeInt8` fully-connected,
the same path Josh's harness exercises), then 5 s of real ANE
inference via the ane-bridge driver path (`_ANEClient.doEvaluateDirectWithModel:`).
`powermetrics --samplers ane_power,cpu_power` runs concurrently.

| phase         |      CPU avg |  CPU max |      ANE avg | ANE max |
| ------------- | -----------: | -------: | -----------: | ------: |
| idle          |      4305 mW | 15319 mW |       0.0 mW |    0 mW |
| **BNNS INT8** | **15947 mW** | 18433 mW |   **0.2 mW** |    4 mW |
| gap           |      3440 mW | 11077 mW |       3.5 mW |   42 mW |
| **real ANE**  |  **1489 mW** |  4612 mW | **369.6 mW** |  625 mW |

During BNNS the ANE power channel is at the noise floor. During real
ANE inference, CPU power drops 10× and ANE power rises by orders of
magnitude. If BNNS were the Neural Engine, ANE power would spike
during the BNNS phase. It does not.

Reproduce:

```bash
make && make examples
make -C tools all
uv run python tools/make_identity_model.py build/identity
sudo -v
bash tools/run_power_witness.sh
```

## 2. Static framework boundaries

The Neural Engine path is gated by the private framework
`AppleNeuralEngine.framework`, the `aned` daemon, and the IOKit user
client `_ANEClient`. A process using the Neural Engine must allocate
an `_ANEClient`, set up `IOSurface`-backed buffers, and dispatch via
the daemon. None of these appear in BNNS.

```bash
otool -L /System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Versions/A/libBNNS.dylib
# → libBLAS, libvDSP, libLAPACK, libLinearAlgebra, libSystem
# → no AppleNeuralEngine.framework, no IOSurface usage
```

`BNNSFilterApply` is documented as part of Accelerate, which is the
CPU compute framework. The ANE driver entry point (`_ANEClient
doEvaluateDirectWithModel:`) lives in a separate, private framework
that BNNS does not link.

## 3. Throughput orders of magnitude

Josh's table reports `BNNS INT8 alone (4 threads) = 4.2 TOPS`. Apple
publishes the M4 Neural Engine at 38 TOPS INT8 and we measured 18.6
TOPS FP16 directly in this repo's bench. A real ANE call returns
~9000 inferences/sec on the identity model at ~140 µs latency.

If BNNS dispatched to the Neural Engine, BNNS would clear an order
of magnitude higher than raw CPU SME (which Josh measures at 8.5
TOPS). Instead BNNS is _roughly half_ of raw SME — exactly what one
expects from Accelerate's SME-backed path with its safety margins
relative to hand-tuned `smopa` asm.

The arithmetic from his own probe makes this concrete. M4's SME
unit has one matrix MAC pipeline, primarily wired for **FP16→FP32**
outer products. With SVL = 512 bits, an FP16→FP32 `fmopa` does

```
  32 × 32 × 2  =  2048 FLOPs per fmopa
```

(SVL = 64 bytes ÷ 2 bytes per fp16 = 32 lanes per vector, outer
product fills a 32×32 fp32 tile, `× 2` for the multiply + add of
each MAC.) `smopa` int8 runs through the same MAC pipeline rather
than a wider int8 datapath, so it does not gain the headline 4×
density an isolated int8 unit would imply. With M4 issuing the
matrix op once per ~4 cycles and ~2 SME-bearing P-core clusters at
~4.0 GHz, the achievable peak is

```
  2 clusters × (2048 ops / 4 cycles) × 4.0e9 cycles/s  ≈  4 TFLOPs
```

— and the int8 ceiling sits roughly twice that at most.

That puts Josh's 8.5 TOPS raw `smopa` right at the int8 ceiling for
the FP16-shaped MAC, and his BNNS at 4.2 TOPS at ~50 % of raw
`smopa`. The cascade fits a single hypothesis cleanly: **`smopa`
and BNNS share the SME unit; BNNS just pays more library
overhead**. The Neural Engine has nothing to do with either number —
its published INT8 peak is 38 TOPS, an order of magnitude above
both measurements.

## 4. His test never calls the Neural Engine

[`run_full_throughput_tests.sh`](https://github.com/joshmorgan1000/ane/blob/main/tests/run_full_throughput_tests.sh)
defines five workers: GPU (Metal), SME (`smopa` inline asm), BNNS
INT8, CBLAS SGEMM, NEON. The table reproduced as "evidence" in
DISCOVERY.md derives from this script. There is no CoreML invocation
in it. There is no `_ANEClient`, no `IOSurface`, no `aned` round-trip.
The `triple_threat.swift` file references a `coreml_worker` binary
whose source is **not present in the repository**.

The result he calls "ANE + SME contention" is **`BNNS + SME` contention** —
two SME paths on the same CPU silicon contending with each other,
exactly as expected, and unrelated to the Neural Engine.

## 5. The CPU-driven contention pattern is reproduced here, alongside the absence of ANE contention

`c/examples/ane_vs_sme.m` runs the identity model against four
background-worker types: scalar arithmetic, BLAS SGEMM, raw SME2,
and (optionally) BNNS INT8. The critical comparison is per-inference
latency under SME vs scalar contention — if the Neural Engine were
the SME unit, ANE latency would rise specifically under SME load.

Measured on M5 across multiple runs:

| worker type  |    rate_seq |     latency |    rate_sat |
| ------------ | ----------: | ----------: | ----------: |
| scalar       |     ~7000/s |     ~140 µs |     ~8240/s |
| BLAS         |     ~7470/s |     ~125 µs |    ~11215/s |
| **raw SME2** | **~6710/s** | **~144 µs** | **~6870/s** |

Per-inference ANE latency under SME load is statistically
indistinguishable from per-inference ANE latency under scalar load.
A shared compute unit _cannot_ produce that result.

## Conclusion

Five independent lines of evidence agree:

1. **Power telemetry** — ANE channel reads zero during BNNS, hundreds of mW during real ANE.
2. **Framework linkage** — BNNS does not link the ANE private framework or open the kernel driver.
3. **Throughput magnitude** — BNNS is half of raw SME; the Neural Engine is multiples of raw SME.
4. **Code path absence** — his benchmark never invokes any ANE-driver API.
5. **Latency under SME load** — real ANE per-inference latency is unaffected by saturated SME workers; a shared compute unit could not behave this way.

The Neural Engine and SME are not the same silicon.

BNNS uses SME.

That is all the cited benchmark shows.
