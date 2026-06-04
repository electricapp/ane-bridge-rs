# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.2",
#     "coremltools>=8.0",
#     "rich>=13",
#     "typer>=0.12",
# ]
# ///
"""Generate an ANE-friendly benchmark model and compile to .mlmodelc.

Builds an 8-block conv stack in fp16 — the kind of compute-bound graph
ANE accelerates well — converts it via `coremltools.mlprogram`, and
compiles to `.mlmodelc` so both the ane-bridge direct path and the
public CoreML `MLModel` API can load the SAME compiled artifact.

After running, `outdir` contains:
* `bench.mlpackage/`  — source mlprogram, for `MLModel`
* `bench.mlmodelc/`   — compiled, for both paths
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

# coremltools lacks `py.typed`; ignore the untyped-import diagnostic.
import coremltools as ct  # type: ignore[import-untyped]
import torch
import typer
from rich.console import Console
from torch import nn
from torch.nn import functional as F  # noqa: N812 -- official torch alias

console = Console()


class Block(nn.Module):
    """Two-conv ReLU block — the unit cell of the benchmark stack."""

    def __init__(self, c: int) -> None:
        """Construct with `c` input/output channels."""
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv → ReLU → conv → ReLU."""
        return F.relu(self.conv2(F.relu(self.conv1(x))))


class Net(nn.Module):
    """Sequential stack of `n_blocks` `Block` instances at width `c`."""

    def __init__(self, c: int = 128, n_blocks: int = 8) -> None:
        """Construct with `c` channels and `n_blocks` blocks."""
        super().__init__()
        self.blocks = nn.Sequential(*[Block(c) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stack forward."""
        # nn.Sequential is typed as returning Any in torch's stubs.
        out: torch.Tensor = self.blocks(x)
        return out


def main(
    outdir: Annotated[Path, typer.Argument(help="output directory for fixture")],
    channels: Annotated[int, typer.Option(help="conv channel count")] = 128,
    spatial: Annotated[int, typer.Option(help="spatial extent (H=W)")] = 32,
    blocks: Annotated[int, typer.Option(help="number of conv blocks")] = 8,
) -> None:
    """Build the benchmark conv stack, save mlpackage, copy compiled mlmodelc."""
    outdir.mkdir(parents=True, exist_ok=True)
    console.print(
        f"building net: [bold]C={channels} H=W={spatial} blocks={blocks}[/bold]",
    )

    net = Net(c=channels, n_blocks=blocks).eval()
    example = torch.zeros(1, channels, spatial, spatial)
    # `torch.jit.trace` is untyped in torch's stubs.
    traced = torch.jit.trace(net, example)  # type: ignore[no-untyped-call]
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="x",
                shape=example.shape,
                dtype=ct.converters.mil.mil.types.fp16,
            ),
        ],
        outputs=[
            ct.TensorType(name="y", dtype=ct.converters.mil.mil.types.fp16),
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )

    pkg = outdir / "bench.mlpackage"
    if pkg.exists():
        shutil.rmtree(pkg)
    mlmodel.save(pkg.as_posix())
    console.print(f"wrote [bold]{pkg}[/bold]")

    target_modelc = outdir / "bench.mlmodelc"
    if target_modelc.exists():
        shutil.rmtree(target_modelc)
    src_modelc = Path(mlmodel.get_compiled_model_path())
    shutil.copytree(src_modelc, target_modelc)
    console.print(f"compiled to [bold]{target_modelc}[/bold]")


if __name__ == "__main__":
    typer.run(main)
