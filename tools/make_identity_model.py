# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rich>=13",
#     "typer>=0.12",
# ]
# ///
"""Generate a trivial identity MIL model + placeholder weights blob.

The MIL declares one input and one output of identical shape
`[1, 64, 1, 16]` fp32, with a single identity-via-cast pair
(fp32 → fp16 → fp32). Exercises the open / compile / load / eval
pipeline without relying on any weights.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 -- used at runtime, not just in annotations
from typing import Annotated

import typer
from rich.console import Console

SHAPE: tuple[int, int, int, int] = (1, 64, 1, 16)

# The framework's MIL parser rejects line breaks inside `buildInfo`,
# so the buildInfo dict must stay on one line. Suppress E501 for the
# whole template since it's an opaque payload, not Python source.
_BUILDINFO = (
    'dict<string, string>('
    '{{"coremlc-component-MIL", "3510.2.1"},'
    '{"coremlc-version", "3505.4.1"},'
    '{"coremltools-component-milinternal", ""},'
    '{"coremltools-version", "9.0"}})'
)
MIL_TEMPLATE = f"""program(1.3)
[buildInfo = {_BUILDINFO}]
{{
    func main<ios18>(tensor<fp32, [__SHAPE__]> x) {{
        string td16 = const()[name = string("td16"), val = string("fp16")];
        tensor<fp16, [__SHAPE__]> x16 = cast(dtype = td16, x = x)[name = string("x16")];
        string td32 = const()[name = string("td32"), val = string("fp32")];
        tensor<fp32, [__SHAPE__]> y = cast(dtype = td32, x = x16)[name = string("y")];
    }} -> (y);
}}
"""

console = Console()


def main(
    outdir: Annotated[Path, typer.Argument(help="output directory for fixture")],
) -> None:
    """Write `model.mil` and `weights.bin` placeholder into `outdir`."""
    outdir.mkdir(parents=True, exist_ok=True)
    shape_str = ", ".join(str(d) for d in SHAPE)
    mil = MIL_TEMPLATE.replace("__SHAPE__", shape_str)
    (outdir / "model.mil").write_text(mil)
    (outdir / "weights.bin").write_bytes(b"\x00")
    console.print(f"wrote [bold]{outdir}/model.mil[/bold]  shape={SHAPE}")


if __name__ == "__main__":
    typer.run(main)
