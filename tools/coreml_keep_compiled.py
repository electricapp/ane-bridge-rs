# /// script
# requires-python = "==3.11.*"
# dependencies = [
#     "coremltools==8.1",
#     "numpy<2",
# ]
# ///
"""Predict on ANE, then COPY the compiled .mlmodelc out of the temp dir to a
stable location before the process exits, and dump every file in it so we can
find the ANE program hash aned keyed its cache on."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import coremltools as ct

SHAPE = (1, 64, 1, 16)


def main() -> None:
    pkg = sys.argv[1]
    dest = Path(sys.argv[2])
    m = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
    inp = {"x": np.random.rand(*SHAPE).astype(np.float32)}
    m.predict(inp)  # force ANE compile + exec
    cpath = Path(m.get_compiled_model_path())
    print(f"RESULT:compiled_path={cpath}")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(cpath, dest)
    print(f"RESULT:copied_to={dest}")


if __name__ == "__main__":
    main()
