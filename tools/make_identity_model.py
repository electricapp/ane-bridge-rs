"""Generate a trivial identity MIL model + (unused) weights blob for testing.

Writes:
  <outdir>/model.mil
  <outdir>/weights.bin   (1-byte placeholder)

The MIL declares one input and one output of identical shape [1, 64, 1, 16]
fp32, and the function body is a single identity-via-cast pair (fp32 -> fp16
-> fp32). This exercises the open / compile / load / eval pipeline without
relying on any weights.
"""
from __future__ import annotations
import argparse, pathlib

SHAPE = (1, 64, 1, 16)

MIL_TEMPLATE = """program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp32, [__SHAPE__]> x) {
        string td16 = const()[name = string("td16"), val = string("fp16")];
        tensor<fp16, [__SHAPE__]> x16 = cast(dtype = td16, x = x)[name = string("x16")];
        string td32 = const()[name = string("td32"), val = string("fp32")];
        tensor<fp32, [__SHAPE__]> y = cast(dtype = td32, x = x16)[name = string("y")];
    } -> (y);
}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=pathlib.Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    shape_str = ", ".join(str(d) for d in SHAPE)
    mil = MIL_TEMPLATE.replace("__SHAPE__", shape_str)
    (args.outdir / "model.mil").write_text(mil)
    (args.outdir / "weights.bin").write_bytes(b"\x00")
    print(f"wrote {args.outdir}/model.mil  shape={SHAPE}")


if __name__ == "__main__":
    main()
