# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "coremltools==9.0",
#     "numpy<2",
#     "typer>=0.12",
#     "rich>=13",
# ]
# ///
#
# NOTE: pinned to <3.13 — coremltools 9.0 ships its native BlobWriter only for
# CPython 3.11/3.12 (see make_cache_fixture.py).
"""Validate ANE-resident stateful inference (KV-cache-style) with no entitlement.

The thesis behind the Espresso/MLE5Engine path: a model's *state* (e.g. a KV
cache) can live on the Neural Engine across calls, so per call only a tiny
input/output crosses the host<->device boundary — the big cache never does.
This runs entirely through `ct.MLModel.predict(..., state=...)` (the sanctioned
CoreML path → MLE5Engine → e5rt execution stream), so it needs NO ANE
entitlement of our own.

Two models, both `new = cache + x` over a cache of `size` fp16 elements:

  * STATE mode: `cache` is a State buffer. Input is one scalar `x`; output is
    the scalar mean of the updated cache. The cache stays resident; only 2
    floats cross per call. Calling K times with x=1.0 must make the mean climb
    1,2,3,...,K — proving the state persisted and updated *in place* on device.

  * IO mode (control): `cache` is a normal input AND the updated cache is an
    output. Same compute, but the whole cache crosses the bus both ways every
    call — what you'd pay without resident state.

If the state path is real, STATE per-call latency stays ~flat as `size` grows,
while IO latency scales with `size`. That gap is the validation.
"""

from __future__ import annotations

import time
from typing import Annotated

import coremltools as ct
import numpy as np
import typer
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from rich.console import Console
from rich.table import Table

console = Console()


def _state_model(size: int) -> ct.models.MLModel:
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1,)),
            mb.StateTensorSpec((1, size), dtype=types.fp16),
        ],
        opset_version=ct.target.iOS18,
    )
    def prog(x, cache):
        c = mb.cast(x=mb.read_state(input=cache), dtype="fp32")
        new = mb.add(x=c, y=x)  # x (1,) broadcasts across the cache
        mb.coreml_update_state(state=cache, value=mb.cast(x=new, dtype="fp16"))
        return mb.reduce_mean(x=new, axes=[0, 1], keep_dims=False)

    return ct.convert(prog, convert_to="mlprogram",
                      compute_units=ct.ComputeUnit.CPU_AND_NE,
                      minimum_deployment_target=ct.target.iOS18)


def _io_model(size: int) -> ct.models.MLModel:
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1,)), mb.TensorSpec(shape=(1, size))],
        opset_version=ct.target.iOS18,
    )
    def prog(x, cache_in):
        new = mb.add(x=cache_in, y=x)  # whole cache crosses in...
        return new  # ...and back out

    return ct.convert(prog, convert_to="mlprogram",
                      compute_units=ct.ComputeUnit.CPU_AND_NE,
                      minimum_deployment_target=ct.target.iOS18)


def _time_ms(fn, iters: int) -> float:
    for _ in range(5):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / iters


def main(
    sizes: Annotated[str, typer.Option(help="comma-sep cache element counts")] = "65536,1048576,8388608",
    iters: Annotated[int, typer.Option(help="timed iterations per size")] = 50,
) -> None:
    """Build, prove state persistence, and compare STATE vs IO per-call latency."""
    size_list = [int(s) for s in sizes.split(",")]

    # 1) Correctness / persistence — small cache, exact arithmetic.
    console.print("[bold]1) state persistence & in-place update[/bold]")
    m = _state_model(4096)
    st = m.make_state()
    ok = True
    for k in range(1, 6):
        out = m.predict({"x": np.array([1.0], dtype=np.float32)}, state=st)
        mean = float(list(out.values())[0])
        good = abs(mean - k) < 1e-3
        ok = ok and good
        console.print(f"   call {k}: cache mean = {mean:.4f} (expect {k})  {'OK' if good else 'BAD'}")
    console.print(f"   [{'green' if ok else 'red'}]state {'PERSISTS on device across calls' if ok else 'DID NOT persist'}[/]")

    # 2) IO-boundedness — per-call latency vs cache size.
    console.print("\n[bold]2) per-call latency vs cache size (STATE resident vs IO crossing)[/bold]")
    tbl = Table()
    tbl.add_column("cache elems", justify="right")
    tbl.add_column("cache MiB (fp16)", justify="right")
    tbl.add_column("STATE ms/call", justify="right")
    tbl.add_column("IO ms/call", justify="right")
    tbl.add_column("speedup", justify="right")
    for size in size_list:
        sm = _state_model(size)
        sst = sm.make_state()
        x = np.array([1.0], dtype=np.float32)
        s_ms = _time_ms(lambda: sm.predict({"x": x}, state=sst), iters)

        im = _io_model(size)
        cache = np.zeros((1, size), dtype=np.float32)
        i_ms = _time_ms(lambda: im.predict({"x": x, "cache_in": cache}), iters)

        tbl.add_row(f"{size:,}", f"{size * 2 / 2**20:.1f}",
                    f"{s_ms:.3f}", f"{i_ms:.3f}", f"{i_ms / s_ms:.1f}x")
    console.print(tbl)
    console.print("\n[dim]No _ANEClient, no ANE entitlement — pure CoreML predict (MLE5Engine path).[/dim]")


if __name__ == "__main__":
    typer.run(main)
