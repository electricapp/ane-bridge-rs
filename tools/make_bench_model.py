# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.2",
#     "coremltools>=8.0",
# ]
# ///
"""Generate an ANE-friendly benchmark model and compile to .mlmodelc.

Builds an 8-block conv stack in fp16 — the kind of compute-bound graph
ANE accelerates well — converts it via coremltools.mlprogram, and
compiles to .mlmodelc so both the ane-bridge direct path and the
public CoreML MLModel API can load the SAME compiled artifact.

Usage:
    uv run python tools/make_bench_model.py <outdir>

After running, <outdir> contains:
    bench.mlpackage/          (source mlprogram, for MLModel)
    bench.mlmodelc/           (compiled, for both paths)
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct


class Block(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv2(F.relu(self.conv1(x))))


class Net(nn.Module):
    def __init__(self, c: int = 128, n_blocks: int = 8) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[Block(c) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=pathlib.Path)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--spatial", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=8)
    args = ap.parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"building net: C={args.channels} H=W={args.spatial} blocks={args.blocks}"
    )
    net = Net(c=args.channels, n_blocks=args.blocks).eval()
    example = torch.zeros(1, args.channels, args.spatial, args.spatial)

    # Trace + convert to mlprogram in fp16.
    traced = torch.jit.trace(net, example)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="x",
                shape=example.shape,
                dtype=ct.converters.mil.mil.types.fp16,
            )
        ],
        outputs=[
            ct.TensorType(
                name="y",
                dtype=ct.converters.mil.mil.types.fp16,
            )
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )

    pkg = outdir / "bench.mlpackage"
    if pkg.exists():
        import shutil
        shutil.rmtree(pkg)
    mlmodel.save(pkg.as_posix())
    print(f"wrote {pkg}")

    # Compile to .mlmodelc so ane-bridge can read model.mil + weights/weight.bin
    # directly from the same artifact CoreML uses. coremltools exposes
    # the compiled path via `get_compiled_model_path()`, which uses the
    # public `MLModel.compileModel(at:)` under the hood — no Xcode
    # `coremlcompiler` binary required.
    target_modelc = outdir / "bench.mlmodelc"
    if target_modelc.exists():
        import shutil
        shutil.rmtree(target_modelc)
    src_modelc = pathlib.Path(mlmodel.get_compiled_model_path())
    import shutil
    shutil.copytree(src_modelc, target_modelc)
    print(f"compiled to {target_modelc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
