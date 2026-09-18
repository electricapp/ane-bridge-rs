# What the ANE will run, and what it costs

MIL text is this crate's input contract, and the framework accepts a good
deal less than MIL describes. Some of what it rejects it rejects loudly at
compile; some it accepts and then fails at dispatch; and one case runs
happily and returns wrong data. None of it is documented by Apple.

Everything below is direct measurement on Apple M5 under macOS 26.5.2,
through this crate's dispatch path. The timings come from a streaming
signal-processing workload — a polyphase filter bank feeding a DFT and a
demodulator, 7 to 14 ops over tensors of 0.8 to 6.6 MB — so they describe
the regime where a graph is a chain of medium tensors, not the regime
where it is one enormous matmul.

## The machine

| Bound                   | Value                              |
| ----------------------- | ---------------------------------- |
| ANE peak fp16           | ~15.0 TFLOP/s (large dense matmul) |
| ANE streaming bandwidth | ~100 GB/s DRAM, ~200 GB/s on-chip  |
| Ridge point             | ~150 FLOP/byte                     |
| Host memory bandwidth   | ~130 GB/s                          |
| Dispatch overhead       | ~60 µs                             |
| Advertised queue depth  | 127 (only 2 are useful)            |

## Hard constraints

`ane-bridge-mil` enforces the mechanical ones — width alignment, kernel width,
fp32 arithmetic, strided dense convolutions and dead ops — at the builder, so a
graph written through it cannot reach the compiler in a state the table below
describes.

| Constraint                               | Detail                                                                                                                                                                                                                                                                                                                                    |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Conv kernel width ≤ 15                   | `KW >= 16` fails with `CompilationFailure` at every padding style and grouping.                                                                                                                                                                                                                                                          |
| Output width must be a multiple of 32    | Silent when violated. Narrower than 32 compiles and then fails at dispatch. Not a multiple of 32 may _run and return wrong data_: rows are laid out at the next multiple of 32 while the schema still reports the requested width, so channel `k` is read from where `k+1` begins. `[1, 100, 1, 4082]` comes back with a row stride of 4096. |
| Input width must be a multiple of 32 too | An unaligned input width fails at dispatch.                                                                                                                                                                                                                                                                                              |
| Ports are reordered                      | The framework does not preserve MIL output order. A graph declaring `(audio, power)` hands back `power` at port 0. Resolve every port by name.                                                                                                                                                                                           |
| fp16 compute only                        | fp32 arithmetic is rejected. fp32 is fine at the boundary if immediately `cast`.                                                                                                                                                                                                                                                         |
| Static shapes only                       | Any `?` dimension yields `InvalidMILProgram`.                                                                                                                                                                                                                                                                                            |
| Rank 4 `[N, C, H, W]`                    | The I/O schema is canonicalized to NCHW whatever the MIL declares.                                                                                                                                                                                                                                                                       |
| No transcendentals                       | No `sin`, `cos`, `exp`, `log`, `sqrt`, `atan`. Compose from `mul`/`add`/`sub`/`relu`/`real_div`.                                                                                                                                                                                                                                          |
| `reduce_mean` is unusable                | A graph containing it passes `ANECCompile` and then fails at dispatch. Reduce with a cascade of strided averaging convolutions instead.                                                                                                                                                                                                  |
| No dead code                             | An unused subgraph that reduces to width 1 makes the whole program fail to compile.                                                                                                                                                                                                                                                      |
| No resident state                        | State ops are rejected on this path. A streaming filter's history must be passed in as an input. (`StateModel` is a separate, CoreML-backed path — see the README.)                                                                                                                                                                      |
| Graph must be ANE-eligible               | A purely elementwise program can be left on the CPU and refused at load. Anchor it with a `conv`, `linear` or `matmul`.                                                                                                                                                                                                                  |
| `buildInfo` on one line                  | The MIL parser rejects a line break inside it.                                                                                                                                                                                                                                                                                           |

## What a graph costs

```
time  ≈  60 µs  +  materialized_traffic / bandwidth(tensor size)
```

Materialized traffic is countable from the program text before anything is
dispatched: every op's result written once, every operand read once per
use, and weights included — they are re-read from memory on every
dispatch, not held on-chip. Attribute constants (strides, axes, pad
vectors) do not count.

A two-point fit over six formulations of one ~1.6 MB graph gave
`50 µs + traffic / 136 GB/s`. Treat the 60 µs / ~107 GB/s pair as the
conservative version of the same model; both predict within the run-to-run
noise.

The floor is not amortizable. It is ~20% of a 250 µs dispatch, and larger
blocks do not dilute it, because throughput peaks around 4096 frames of
width and falls off above that.

### Ops do not fuse

Hold a tensor fixed and vary only the length of a chain after it, over
3-tap depthwise convolutions — no reuse, nothing foldable. The slope is
the marginal cost of one op:

|  tensor | µs per op |  implied |
| ------: | --------: | -------: |
| 0.82 MB |      11.1 | 147 GB/s |
| 1.64 MB |      30.6 | 107 GB/s |
| 3.28 MB |      48.5 | 135 GB/s |
| 6.55 MB |      95.5 | 137 GB/s |

Linear in chain length at every size, so each op is a genuine separate
memory pass. The rate is flat at ~110–145 GB/s: there is no on-chip regime
for a streaming graph to fall into, and tensor size is not a lever.

Four things do fold, and a probe built out of them will report that
everything fuses:

- A `mul`/`add` against a **scalar** folds into the preceding
  convolution's weights and bias. Scalar operands are free.
- `relu` is idempotent; a chain of it collapses to one.
- `tanh` is an activation, and activations fuse into the op before them.
- A `mul`/`add` against a live input tensor cannot fold, but re-reads the
  same tensor every op, which stays hot. The apparent rate runs high at
  small sizes (192 GB/s at 0.82 MB) and converges to the real one
  (107 GB/s at 6.55 MB).

Holding more tensors live does not degrade throughput either — measured up
to 19.66 MB live — so restructuring a graph for shorter tensor lifetimes
gains nothing. A graph touching N full-size tensors moves N times its
compulsory traffic and reaches about `1/N` of its algorithmic bound. That
is a property of the formulation, not of the emission.

### What reduces traffic

**Emit only the rows that are read.** A dense `1x1` convolution's output
rows are independent, so which rows exist is a choice of which rows of the
weight matrix to write down. If K of C outputs are consumed downstream,
emit a K-row matrix and the full-width tensor is never materialized —
including for everything after it, which now runs K wide. Measured on an
11-op graph whose `1x1` emitted 200 rows: narrowed to 16, traffic fell
28.2 MB → 10.1 MB and time 270 µs → 90 µs, bit-identical on the rows kept.

**Slice before striding.** A strided *dense* convolution is pathologically
slow. The same graph, with the same output, took 215 µs at an effective
45 GB/s when a `groups = 1` convolution strided over its input, and ~90 µs
at ~104 GB/s when the input was sliced first and the convolution left
unstrided. Grouped and depthwise strided convolutions do not show this —
they are the normal way to decimate, and run at the usual rate. The cliff
is on dense ones.

**Interleave operands so a grouped convolution can absorb an elementwise
chain.** Arranging a tensor so that the values an elementwise expression
combines are adjacent in the channel axis turns that expression into one
grouped convolution instead of several full-tensor ops.

**Sample rather than reduce.** A statistic over a block rarely needs every
frame of it; a few hundred frames of a wide tensor cost a fraction of all
of them.

**Land widths on the 32 grid deliberately**, so no slice is needed later
to realign them.

What does not help:

- Shortening a convolution. 3 taps against 15 measures the same; the
  convolutions are not the cost.
- Widening a tensor to save an op. Packing four multiplies into one, at
  double the operand width, moved 35 MB instead of 28 MB.
- Fusing a producer into a dense consumer. Removing a 1.6 MB intermediate
  (~22 µs) by turning a `1x1` into a width-8 dense convolution cost 8× the
  FLOPs (~175 µs).
- Queueing more than two dispatches. Throughput is flat from 2 to 8.
- Blocks larger than ~4096 wide. Throughput peaks there.

### When the graph is compute-bound instead

A dense `1x1` convolution over C channels is a `C × C` matmul applied at
every position: `O(C²)` work for `O(C)` output. Past a few hundred
channels it leaves the bandwidth regime. At C = 4096 the measured
*traffic* rate fell to 21 GB/s while every other graph on the machine held
~117, and the weights were only 9% of the bytes moved — a falling byte
rate with small weights is the signature of a graph that has stopped
moving bytes and started doing arithmetic. The effective rate there is
4–6 TFLOP/s.

The fix is fewer FLOPs, and the awkward part is usually that the cheaper
formulation needs a strided subset of the channel axis, which no grouped
convolution can address, since groups are contiguous.

**The unused height axis solves that.** Reshaping `[1, C, 1, W]` to
`[1, C/s, s, W]` moves the stride onto H. In NCHW that is the same
offsets — a pure renaming, which the hardware agrees with and charges
nothing for. A `1x1` convolution is applied independently at every height,
so a strided subset becomes a *dense* convolution over `C/s` channels.
Reshaping back makes the other half of the factorization contiguous, so it
is a grouped `1x1`.

Applied to a DFT as one Cooley–Tukey step (`C = c₁·c₂`, cutting `8C` FLOPs
per sample to `8(c₁+c₂)`), throughput went from 239 to 1824 Msample/s at
C = 2048 and stayed flat out to C = 8192, where the dense form no longer
runs at all. Note that such a factorization permutes the output order;
permuting it back costs a full tensor copy, so it is usually better to
carry the permutation in whatever indexes the results.

One consequence of weights being re-read every dispatch: the arithmetic
intensity of a dense `1x1` is `2·C²·W / 2C² = W`, exactly the width,
independent of C. Shrinking width as channels grow walks the op below the
ridge point and measures weight bandwidth rather than compute.

## Measuring any of this

- **Time best-of-N, never a long average.** These timings swing ±15–30%
  run to run with thermal state. Best-of-N repeats to under 1%.
- **Compare formulations inside one process.** The absolute level drifts
  between runs; the relative ordering within a process is stable to a
  percent.
- **Measure slopes.** Dividing total work by total time folds in the fixed
  dispatch cost. Timing two problem sizes and taking Δwork/Δtime cancels
  it.
- **Sweep wide enough to see the effect being fitted.** A dispatch floor
  fitted over 2048–8192 wide — where it is 20–35% of runtime against ±15%
  noise — comes out negative. It is unmistakable once the sweep reaches
  256.
- **Make the probe measure the thing.** An elementwise-fusion probe built
  from scalar constants measures constant folding, and reports that
  everything fuses.
- **Check that a faster graph is still a correct one.** Every one of the
  reformulations above is a claim that some tensor did not need to exist,
  and a graph that skips work is indistinguishable from a graph that is
  wrong until something compares them against a reference.

## Concurrency

`Model` is `Arc`-backed and hands out any number of `Request`s, each with
its own `IOSurface` buffers, over one compiled program. Two separate
`Model`s over the same MIL also overlap, but pay for a second compile, a
second copy of the weights and a second program resident on the device.

Two in-flight requests hide the host behind the ANE: 397 µs serial became
265 µs overlapped. A third adds nothing, which is why the advertised depth
of 127 is not worth chasing.

`submit_async`, `on_complete` and `is_done` did not beat `submit`/`wait`
across a ring of two in this workload, because the host loop had exactly
one thing to do while waiting. They matter when it has other work to
interleave.
