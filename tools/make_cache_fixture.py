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
# NOTE: pinned to <3.13 — coremltools 9.0 ships its native `libmilstoragepython`
# (the MIL weight BlobWriter) only for CPython 3.11/3.12. On 3.13+ uv would pick
# an interpreter with no matching wheel and conversion dies with
# "RuntimeError: BlobWriter not loaded".
"""Build a tiny ANE-eligible `.mlmodelc` aned can load by URL.

Companion to the `espresso_ane_cache_*` C ABI (see `c/include/espresso.h`).
Those functions key aned's compiled-network cache by *model path/URL*, so a
hermetic end-to-end test of them needs a compiled model on disk that aned will
actually lower onto the Neural Engine — otherwise nothing lands in the ANE
cache to query or purge.

Two constraints shape the fixture:

  * It must be a NeuralNetwork-style espresso bundle (`model.espresso.net` /
    `.shape` / `.weights`). The private `_ANEModel modelAtURL:` path rejects an
    `mlprogram` bundle (just `model.mil`) with a bare `loadModel:` failure.
  * It must contain an ANE-eligible op. A trivial elementwise identity is left
    on CPU and aned refuses to load it (`loadModel:` fails with no NSError), so
    we use a 1x1 convolution with identity weights — a no-op numerically, but a
    real conv the ANE places.

Built with coremltools' `NeuralNetworkBuilder` (no torch). `get_compiled_model_path()`
lowers it to the espresso bundle, which we copy to the destination; aned
compiles + caches the network when the caller opens the copied path.

Usage:
  uv run tools/make_cache_fixture.py <dest-dir.mlmodelc>

Prints `RESULT:copied_to=<path>` on success. The destination is what you pass
to `Model::open_file` and to `espresso_ane_cache_has_network` / `_purge_network`.
"""

from __future__ import annotations

import shutil
from pathlib import Path  # noqa: TC003 -- used at runtime, not just in annotations
from typing import Annotated

import coremltools as ct
import numpy as np
import typer
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder
from rich.console import Console

console = Console()

# (C, H, W) echoing the in-process MIL fixtures' [1, 64, 1, 16].
CHANNELS, HEIGHT, WIDTH = 64, 1, 16


def main(
    dest: Annotated[Path, typer.Argument(help="output .mlmodelc directory")],
) -> None:
    """Build a 1x1 identity-conv NeuralNetwork and copy its compiled .mlmodelc."""
    console.print(f"building identity-conv fixture: [bold]C={CHANNELS} H={HEIGHT} W={WIDTH}[/bold]")
    builder = NeuralNetworkBuilder(
        [("x", datatypes.Array(CHANNELS, HEIGHT, WIDTH))],
        [("y", datatypes.Array(CHANNELS, HEIGHT, WIDTH))],
    )
    # 1x1 conv with an identity channel matrix → numerically a no-op, but an
    # ANE-eligible op so aned lowers + caches the network.
    weights = np.eye(CHANNELS, dtype=np.float32).reshape(1, 1, CHANNELS, CHANNELS)
    builder.add_convolution(
        name="identity_conv",
        kernel_channels=CHANNELS,
        output_channels=CHANNELS,
        height=1,
        width=1,
        stride_height=1,
        stride_width=1,
        border_mode="valid",
        groups=1,
        W=weights,
        b=np.zeros(CHANNELS, dtype=np.float32),
        has_bias=True,
        input_name="x",
        output_name="y",
    )

    model = ct.models.MLModel(builder.spec, compute_units=ct.ComputeUnit.CPU_AND_NE)
    compiled = Path(model.get_compiled_model_path())
    console.print("compiled to a NeuralNetwork espresso bundle (model.espresso.*)")

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(compiled, dest)
    console.print(f"RESULT:compiled_path={compiled}")

    # Best-effort cache warm: load the *copied* bundle and run one ANE
    # inference under dest's path. NOTE: this does not (yet) make
    # `espresso_ane_cache_has_network(dest)` report cached — that cache appears
    # to be the E5RT compiler bundle cache, not the CoreML/aned program cache a
    # predict populates (see tests/espresso_cache.rs). Kept so the fixture is a
    # genuinely ANE-run bundle for whatever population path lands.
    warm = ct.models.CompiledMLModel(str(dest), compute_units=ct.ComputeUnit.CPU_AND_NE)
    warm.predict({"x": np.zeros((CHANNELS, HEIGHT, WIDTH), dtype=np.float32)})
    console.print("ran one CPU_AND_NE inference on the copied bundle")
    console.print(f"RESULT:copied_to={dest}")


if __name__ == "__main__":
    typer.run(main)
