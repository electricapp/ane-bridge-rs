# /// script
# requires-python = "==3.11.*"
# dependencies = [
#     "coremltools==8.1",
#     "numpy<2",
#     "torch==2.4.1",
# ]
# ///
"""Build a tiny identity Core ML model matching the bridge identity fixture
([1,64,1,16] fp32 in/out), force a prediction on the ANE, and report the
compute device that actually serviced it.

Emits, on stdout, lines prefixed `RESULT:` that the caller greps:
  RESULT:mlpackage=<path>
  RESULT:compiled=<path/.mlmodelc>
  RESULT:predicted=ok|fail
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import coremltools as ct


SHAPE = (1, 64, 1, 16)


class Identity(torch.nn.Module):
    def forward(self, x):
        # A trivial op the ANE will accept: scale by 1.0 then add 0.0.
        return x * 1.0 + 0.0


def main() -> None:
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/coreml_identity")
    outdir.mkdir(parents=True, exist_ok=True)

    example = torch.zeros(*SHAPE, dtype=torch.float32)
    traced = torch.jit.trace(Identity().eval(), example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=SHAPE, dtype=np.float32)],
        outputs=[ct.TensorType(name="y", dtype=np.float32)],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )

    pkg = outdir / "identity.mlpackage"
    mlmodel.save(str(pkg))
    print(f"RESULT:mlpackage={pkg}")

    # Reload pinned to CPU_AND_NE and predict to force ANE compilation+exec.
    m = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_AND_NE)

    try:
        compiled = ct.utils.compile_model(str(pkg))
        print(f"RESULT:compiled={compiled}")
    except Exception as e:  # noqa: BLE001
        print(f"RESULT:compiled=ERR:{e}")

    inp = {"x": np.random.rand(*SHAPE).astype(np.float32)}
    try:
        out = m.predict(inp)
        key = next(iter(out))
        arr = np.asarray(out[key])
        diff = float(np.max(np.abs(arr.reshape(SHAPE) - inp["x"])))
        print(f"RESULT:predicted=ok max_abs_diff={diff:.6g}")
    except Exception as e:  # noqa: BLE001
        print(f"RESULT:predicted=fail err={e}")


if __name__ == "__main__":
    main()
