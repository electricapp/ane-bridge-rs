# /// script
# requires-python = "==3.11.*"
# dependencies = [
#     "coremltools==8.1",
#     "numpy<2",
# ]
# ///
"""Load an already-saved .mlpackage pinned to CPU_AND_NE and predict, to
force a fresh ANE compile+exec while a log stream watches. Prints the
compiled .mlmodelc path that Core ML produced (get_compiled_model_path)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import coremltools as ct

SHAPE = (1, 64, 1, 16)


def main() -> None:
    pkg = sys.argv[1] if len(sys.argv) > 1 else "build/coreml_identity/identity.mlpackage"
    m = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
    try:
        cpath = m.get_compiled_model_path()
        print(f"RESULT:compiled_path={cpath}")
    except Exception as e:  # noqa: BLE001
        print(f"RESULT:compiled_path=ERR:{e}")
    inp = {"x": np.random.rand(*SHAPE).astype(np.float32)}
    out = m.predict(inp)
    key = next(iter(out))
    arr = np.asarray(out[key]).reshape(SHAPE)
    print(f"RESULT:predicted=ok max_abs_diff={float(np.max(np.abs(arr - inp['x']))):.6g}")
    # Predict a few more times to be sure the ANE path is exercised.
    for _ in range(3):
        m.predict(inp)
    print("RESULT:done")


if __name__ == "__main__":
    main()
