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
"""Build a tiny stateful `.mlmodelc` for the stateful-inference API tests.

The model is an in-place accumulator: a resident fp16 state `cache` of `size`
elements, one scalar input `x`, and a scalar output = mean(cache + x). Each
step does `cache += x` and returns the new mean. Calling it K times with x=1.0
makes the mean climb 1,2,...,K — a cheap, exact check that the state persists
ANE-resident across calls (see rust/ane-bridge/tests/stateful.rs).

This is the same model `validate_stateful_ane.py` benchmarks, emitted as a
standalone compiled bundle the Rust test opens via `StateModel::open`.

Usage:
  uv run tools/make_state_fixture.py <dest-dir.mlmodelc> [--size N]
"""

from __future__ import annotations

import shutil
from pathlib import Path  # noqa: TC003 -- used at runtime, not just in annotations
from typing import Annotated

import coremltools as ct
import typer
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from rich.console import Console

console = Console()


def main(
    dest: Annotated[Path, typer.Argument(help="output .mlmodelc directory")],
    size: Annotated[int, typer.Option(help="resident state element count")] = 4096,
) -> None:
    """Build the accumulator state model and copy its compiled .mlmodelc to dest."""
    console.print(f"building stateful fixture: [bold]cache={size} fp16[/bold]")

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1,)),
            mb.StateTensorSpec((1, size), dtype=types.fp16),
        ],
        opset_version=ct.target.iOS18,
    )
    def prog(x, cache):
        c = mb.cast(x=mb.read_state(input=cache), dtype="fp32")
        new = mb.add(x=c, y=x)  # x (1,) broadcasts across the resident cache
        mb.coreml_update_state(state=cache, value=mb.cast(x=new, dtype="fp16"))
        return mb.reduce_mean(x=new, axes=[0, 1], keep_dims=False)

    model = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    compiled = Path(model.get_compiled_model_path())
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(compiled, dest)
    console.print(f"RESULT:copied_to={dest}")


if __name__ == "__main__":
    typer.run(main)
