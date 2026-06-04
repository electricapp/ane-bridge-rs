# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "torch>=2.2",
#     "mlx>=0.18 ; sys_platform == 'darwin' and platform_machine == 'arm64'",
#     "coremltools>=7.2 ; sys_platform == 'darwin'",
#     "numpy>=1.26",
#     "rich>=13",
# ]
# ///
"""
Apple Silicon FLOPS bench — continuous matmul stress + live TUI observer.

Run any single rail or comma-separated combo:
    uv run mps_matmul_bench.py --backend mps
    uv run mps_matmul_bench.py --backend mlx,ane
    uv run mps_matmul_bench.py --backend max          # mps + cpu-stress + ane

Stop: Ctrl-C
"""

from __future__ import annotations

import argparse
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ─── Types & constants ─────────────────────────────────────────────────


@dataclass
class Sample:
    t: float
    tflops: float


class Rail(StrEnum):
    MPS = "mps"
    MLX = "mlx"
    CPU_NEON = "cpu-neon"  # 1 process/core, NEON SIMD FMA chain (no SME/AMX)
    CPU_SME = "cpu-sme"  # torch CPU matmul → Accelerate sgemm → SME/AMX coprocessor
    ANE = "ane"

    @classmethod
    def values(cls) -> list[str]:
        return [r.value for r in cls]


# Aliases resolved before parsing — `cpu` is a friendlier alias for cpu-neon.
RAIL_ALIASES: dict[str, list[Rail]] = {
    "max": [Rail.MPS, Rail.CPU_NEON, Rail.ANE],
    "all": [Rail.MPS, Rail.CPU_NEON, Rail.ANE],
    "cpu": [Rail.CPU_NEON],
}
GPU_RAILS = {Rail.MPS, Rail.MLX}
CPU_RAILS = {Rail.CPU_NEON, Rail.CPU_SME}

SPARK = " ▁▂▃▄▅▆▇█"
RAIL_COLORS: dict[str, str] = {"gpu": "cyan", "cpu": "magenta", "ane": "yellow"}
RAIL_LABELS: dict[str, str] = {"gpu": "GPU", "cpu": "CPU", "ane": "ANE"}

ANE_CHANNELS = 256
ANE_HW = 64
ANE_LAYERS = 16
ANE_FLOPS_PER_INF = ANE_LAYERS * 2 * ANE_CHANNELS * ANE_CHANNELS * ANE_HW * ANE_HW * 9
ANE_CACHE_PATH = "/tmp/ane_stress.mlpackage"

MAX_PANEL_WIDTH = 76


# ─── Helpers ────────────────────────────────────────────────────────────


def sparkline(values: deque[float], width: int = 60) -> str:
    if not values:
        return ""
    vs = list(values)[-width:]
    mn, mx = min(vs), max(vs)
    rng = mx - mn if mx > mn else 1.0
    return "".join(SPARK[min(8, int((v - mn) / rng * 8))] for v in vs)


def _bar(frac: float, width: int, color: str) -> Text:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    t = Text()
    t.append("█" * filled, style=color)
    t.append("░" * (width - filled), style="grey30")
    return t


def _dedup(seq: list[Rail]) -> list[Rail]:
    seen: set[Rail] = set()
    out: list[Rail] = []
    for r in seq:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


# ─── TUI panel ──────────────────────────────────────────────────────────

BAR_W = 20


def _rule() -> Text:
    return Text("─" * (MAX_PANEL_WIDTH - 6), style="grey30")


def make_panel(
    *,
    dtype_label: str,
    breakdown: dict[str, float] | None = None,
    power: PowerMonitor | None = None,
    iters: int,
    elapsed: float,
    recent: deque[float],
    spark_data: deque[float],
    all_samples: list[Sample],
    first_window_avg: float | None,
) -> Panel:
    cur = recent[-1] if recent else 0.0
    avg_recent = sum(recent) / len(recent) if recent else 0.0
    avg_all = (
        sum(s.tflops for s in all_samples) / len(all_samples) if all_samples else 0.0
    )
    mn_r = min(recent) if recent else 0.0
    mx_r = max(recent) if recent else 0.0

    has_power = power is not None and power.active and power.combined_mw > 0
    total_w = power.combined_mw / 1000.0 if (power and has_power) else 0.0
    eff = cur / total_w if total_w > 0 else 0.0
    multi_rail = breakdown is not None and len(breakdown) > 1
    spark_w = MAX_PANEL_WIDTH - 8

    parts: list[Any] = []

    # ─── Headline metric row ──────────────────────────────────
    hl = Table.grid(padding=(0, 2))
    hl.add_column(style="dim", justify="right", min_width=6)
    hl.add_column(justify="right", min_width=6)
    hl.add_column(style="dim", justify="right", min_width=6)
    hl.add_column(justify="right", min_width=7)
    hl.add_column(style="dim", justify="right", min_width=4)
    hl.add_column(justify="right", min_width=9)
    if has_power:
        hl.add_row(
            "TFLOPS",
            Text(f"{cur:.2f}", style="bold bright_green"),
            "Power",
            Text(f"{total_w:.1f} W", style="bold"),
            "Eff",
            Text(f"{eff:.3f} T/W", style="bold yellow"),
        )
    else:
        hl.add_row(
            "TFLOPS",
            Text(f"{cur:.2f}", style="bold bright_green"),
            "",
            "",
            "",
            "",
        )
    parts.append(hl)

    # ─── Per-rail breakdown ───────────────────────────────────
    if multi_rail and breakdown is not None:
        parts.append(_rule())
        rail_keys = [k for k in ("gpu", "cpu", "ane") if k in breakdown]
        bar_max = max((breakdown[k] for k in rail_keys), default=1.0) or 1.0
        rt = Table.grid(padding=(0, 1))
        rt.add_column(justify="right", min_width=4)
        rt.add_column(justify="right", min_width=6)
        rt.add_column(no_wrap=True, min_width=BAR_W)
        rt.add_column(justify="right", min_width=7)
        rt.add_column(justify="right", min_width=9)
        rt.add_column(min_width=2)
        for key in rail_keys:
            c = RAIL_COLORS[key]
            tv = breakdown[key]
            rw = 0.0
            if has_power and power is not None:
                rw = {"gpu": power.gpu_mw, "cpu": power.cpu_mw, "ane": power.ane_mw}[
                    key
                ] / 1000.0
            re_ = tv / rw if rw > 0.1 else 0.0
            badge = "⚡" if (key == "ane" and re_ > 1.0) else ""
            rt.add_row(
                Text(RAIL_LABELS[key], style=f"bold {c}"),
                Text(f"{tv:5.2f}", style=f"bold {c}"),
                _bar(tv / bar_max, BAR_W, c),
                f"{rw:5.1f} W" if has_power else "",
                f"{re_:.3f} T/W" if has_power else "",
                Text(badge, style="yellow"),
            )
        parts.append(rt)

    # ─── Stats ────────────────────────────────────────────────
    parts.append(_rule())
    mins, secs = divmod(int(elapsed), 60)

    sg = Table.grid(padding=(0, 1))
    sg.add_column(style="dim", justify="right", min_width=5)
    sg.add_column(justify="right", min_width=5)
    sg.add_column(style="dim", justify="right", min_width=5)
    sg.add_column(justify="right", min_width=5)
    sg.add_column(style="dim", justify="right", min_width=4)
    sg.add_column(justify="right", min_width=5)
    sg.add_column(style="dim", justify="right", min_width=4)
    sg.add_column(justify="right", min_width=5)
    sg.add_row(
        "avg₆₀",
        Text(f"{avg_recent:.2f}", style="bold"),
        "avg∞",
        Text(f"{avg_all:.2f}", style="bold"),
        "min",
        Text(f"{mn_r:.2f}", style="bold"),
        "max",
        Text(f"{mx_r:.2f}", style="bold"),
    )
    if first_window_avg is not None and avg_recent > 0:
        dp = (avg_recent - first_window_avg) / first_window_avg * 100
        dc = "green" if dp >= -2 else ("yellow" if dp >= -10 else "red")
        gl = "▲" if dp >= 0 else "▼"
        deg = Text(f"{gl}{dp:+.1f}%", style=f"bold {dc}")
    else:
        deg = Text("…", style="dim")
    sg.add_row(
        "degr",
        deg,
        "time",
        Text(f"{mins}:{secs:02d}", style="bold"),
        "wins",
        Text(str(iters), style="bold"),
        "",
        "",
    )
    parts.append(sg)

    # ─── Sparkline ────────────────────────────────────────────
    parts.append(_rule())
    st = Text(" ")
    st.append(sparkline(spark_data, spark_w), style="bright_blue")
    parts.append(st)

    return Panel(
        Group(*parts),
        title=f"[bold blue] {dtype_label} [/]",
        subtitle="[dim] ctrl-c [/]",
        border_style="blue",
        width=MAX_PANEL_WIDTH,
        padding=(0, 1),
    )


# ─── Power monitor ─────────────────────────────────────────────────────


class PowerMonitor:
    def __init__(self) -> None:
        self.cpu_mw: int = 0
        self.gpu_mw: int = 0
        self.ane_mw: int = 0
        self.combined_mw: int = 0
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._stop: bool = False
        self.active: bool = False

    def start(self, console: Console) -> None:
        import re
        import shutil

        if shutil.which("powermetrics") is None:
            return
        probe = subprocess.run(
            [
                "sudo",
                "-n",
                "powermetrics",
                "-n",
                "1",
                "-i",
                "100",
                "--samplers",
                "cpu_power",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if probe.returncode != 0:
            console.print(
                "[yellow]power:[/] sudo not cached. "
                "Run [bold]sudo -v[/] first to enable TFLOPS/W."
            )
            return

        self._proc = subprocess.Popen(
            [
                "sudo",
                "-n",
                "powermetrics",
                "-i",
                "1000",
                "--samplers",
                "cpu_power,gpu_power,ane_power",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        cpu_re = re.compile(r"^CPU Power:\s+(\d+)\s*mW")
        gpu_re = re.compile(r"^GPU Power:\s+(\d+)\s*mW")
        ane_re = re.compile(r"^ANE Power:\s+(\d+)\s*mW")
        cmb_re = re.compile(r"^Combined Power.*?:\s+(\d+)\s*mW")

        def reader() -> None:
            assert self._proc is not None and self._proc.stdout is not None
            for line in self._proc.stdout:
                if self._stop:
                    return
                if m := cpu_re.match(line):
                    self.cpu_mw = int(m.group(1))
                elif m := gpu_re.match(line):
                    self.gpu_mw = int(m.group(1))
                elif m := ane_re.match(line):
                    self.ane_mw = int(m.group(1))
                elif m := cmb_re.match(line):
                    self.combined_mw = int(m.group(1))

        self._thread = threading.Thread(target=reader, daemon=True)
        self._thread.start()
        self.active = True
        console.print("[dim]power: powermetrics started (CPU + GPU + ANE)[/]")

    def stop(self) -> None:
        self._stop = True
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()


# ─── Workers (module-level for multiprocessing "spawn") ─────────────────


def _build_ane_model(verbose: bool = False) -> Any:
    import os

    import coremltools as ct

    if os.path.exists(ANE_CACHE_PATH):
        return ct.models.MLModel(
            ANE_CACHE_PATH, compute_units=ct.ComputeUnit.CPU_AND_NE
        )

    import torch
    import torch.nn as nn

    class ConvStack(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                *[
                    nn.Conv2d(ANE_CHANNELS, ANE_CHANNELS, 3, padding=1, bias=False)
                    for _ in range(ANE_LAYERS)
                ]
            )

        def forward(self, x: Any) -> Any:
            return self.layers(x)

    model = ConvStack().eval()
    example = torch.randn(1, ANE_CHANNELS, ANE_HW, ANE_HW)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, ANE_CHANNELS, ANE_HW, ANE_HW))],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    mlmodel.save(ANE_CACHE_PATH)
    return mlmodel


def _ignore_sigint() -> None:
    """Workers ignore SIGINT — Ctrl-C is delivered to every process in the
    foreground process group, but only main should act on it. Workers exit
    via the shared stop_flag."""
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _ane_worker(counter: Any, stop_flag: Any) -> None:
    _ignore_sigint()
    import numpy as np

    m = _build_ane_model()
    in_arr = np.random.randn(1, ANE_CHANNELS, ANE_HW, ANE_HW).astype(np.float32)
    for _ in range(3):
        m.predict({"input": in_arr})
    batch = 8
    while stop_flag.value == 0:
        for _ in range(batch):
            if stop_flag.value:
                return
            m.predict({"input": in_arr})
        with counter.get_lock():
            counter.value += batch * ANE_FLOPS_PER_INF


def _cpu_neon_worker(counter: Any, stop_flag: Any, dtype_str: str, size: int) -> None:
    _ignore_sigint()
    import torch

    torch.set_num_threads(1)
    dtype_t = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        dtype_str
    ]
    x = torch.rand(size, dtype=dtype_t)
    y = torch.rand(size, dtype=dtype_t)
    # Large inner loop keeps cores hot — smaller values cause lock contention on
    # the shared counter (14 workers fighting for the same Value lock = workers
    # sleeping in kernel wait = CPU% drops). Ctrl-C response is already fast
    # because workers ignore SIGINT and exit cleanly via stop_flag within ~10 ms.
    inner = 2000
    flops_per_inner = 3 * size
    while stop_flag.value == 0:
        for _ in range(inner):
            x.mul_(0.9).add_(y, alpha=0.1)
        with counter.get_lock():
            counter.value += inner * flops_per_inner


# ─── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="Apple Silicon FLOPS bench")
    ap.add_argument(
        "--backend",
        default=Rail.MPS.value,
        help=(
            "One rail or comma-separated combo. Rails: "
            + ", ".join(Rail.values())
            + ". Aliases: max/all. Example: --backend mlx,ane"
        ),
    )
    ap.add_argument("--n", type=int, default=4096, help="matrix size N (NxN)")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument(
        "--batch", type=int, default=8, help="matmuls per measurement window"
    )
    ap.add_argument(
        "--workers", type=int, default=0, help="cpu-stress worker count (default=ncpu)"
    )
    ap.add_argument("--log", type=str, default=None, help="CSV log path")
    args = ap.parse_args()

    n_size = args.n
    batch: int = args.batch
    console = Console()
    power = PowerMonitor()
    power.start(console)

    flops_per_matmul = 2.0 * (n_size**3)

    def cleanup() -> None:
        pass

    # ─── Parse rails ───────────────────────────────────────────
    raw = [s.strip() for s in args.backend.split(",") if s.strip()]
    expanded: list[Rail] = []
    for tok in raw:
        if tok in RAIL_ALIASES:
            expanded.extend(RAIL_ALIASES[tok])
        else:
            try:
                expanded.append(Rail(tok))
            except ValueError:
                raise SystemExit(
                    f"Unknown backend: {tok!r}. Valid: {Rail.values() + list(RAIL_ALIASES)}"
                )
    selected = _dedup(expanded)

    gpu_sel = [r for r in selected if r in GPU_RAILS]
    cpu_sel = [r for r in selected if r in CPU_RAILS]
    if len(gpu_sel) > 1:
        raise SystemExit("Pick at most one of {mps, mlx}.")
    if len(cpu_sel) > 1:
        raise SystemExit("Pick at most one of {cpu-neon, cpu-sme}.")
    is_combined = len(selected) > 1
    if is_combined and Rail.CPU_SME in cpu_sel:
        raise SystemExit("`cpu-sme` (AMX/SME) is single-rail only — use `cpu-neon`.")

    # ─── Single-rail backends ──────────────────────────────────
    sample: Any = None
    dtype_label = ""

    if not is_combined and selected[0] is Rail.MPS:
        import torch

        if not torch.backends.mps.is_available():
            raise SystemExit("MPS not available.")
        dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
            args.dtype
        ]
        dev = torch.device("mps")
        console.print(f"[dim]MPS: {n_size}² {args.dtype}[/]")
        a = torch.randn(n_size, n_size, device=dev, dtype=dt)
        b = torch.randn(n_size, n_size, device=dev, dtype=dt)
        c = torch.empty(n_size, n_size, device=dev, dtype=dt)
        for _ in range(5):
            torch.matmul(a, b, out=c)
        torch.mps.synchronize()

        def sample() -> tuple[float, float]:
            t0 = time.perf_counter()
            for _ in range(batch):
                torch.matmul(a, b, out=c)
            torch.mps.synchronize()
            return batch * flops_per_matmul, time.perf_counter() - t0

        dtype_label = f"MPS torch.{args.dtype} {n_size}²"

    elif not is_combined and selected[0] is Rail.MLX:
        import mlx.core as mx

        dm = {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}[args.dtype]
        console.print(f"[dim]MLX: {n_size}² {args.dtype}[/]")
        a = mx.random.normal((n_size, n_size)).astype(dm)
        b = mx.random.normal((n_size, n_size)).astype(dm)
        mx.eval(a, b)
        for _ in range(5):
            mx.eval(a @ b)

        def sample() -> tuple[float, float]:
            t0 = time.perf_counter()
            x = a
            for _ in range(batch):
                x = x @ b
            mx.eval(x)
            return batch * flops_per_matmul, time.perf_counter() - t0

        dtype_label = f"MLX {args.dtype} {n_size}²"

    elif not is_combined and selected[0] is Rail.CPU_SME:
        import os

        import torch

        nt = os.cpu_count() or 8
        torch.set_num_threads(nt)
        dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
            args.dtype
        ]
        console.print(f"[dim]CPU SME/AMX: {n_size}² {args.dtype} ×{nt} threads[/]")
        a = torch.randn(n_size, n_size, dtype=dt)
        b = torch.randn(n_size, n_size, dtype=dt)
        c = torch.empty(n_size, n_size, dtype=dt)
        for _ in range(2):
            torch.matmul(a, b, out=c)

        def sample() -> tuple[float, float]:
            t0 = time.perf_counter()
            for _ in range(batch):
                torch.matmul(a, b, out=c)
            return batch * flops_per_matmul, time.perf_counter() - t0

        dtype_label = f"CPU SME {args.dtype} ×{nt}"

    elif not is_combined and selected[0] is Rail.CPU_NEON:
        import multiprocessing as mp
        import os

        nw = args.workers if args.workers > 0 else (os.cpu_count() or 8)
        console.print(f"[dim]CPU-NEON: {nw} workers[/]")
        ctx = mp.get_context("spawn")
        counter: Any = ctx.Value("q", 0)
        stop_flag: Any = ctx.Value("b", 0)
        neon_procs = [
            ctx.Process(
                target=_cpu_neon_worker, args=(counter, stop_flag, args.dtype, 4096)
            )
            for _ in range(nw)
        ]
        for p in neon_procs:
            p.start()
        last_f = [0]

        def sample() -> tuple[float, float]:
            t0 = time.perf_counter()
            time.sleep(0.5)
            cur = counter.value
            delta = cur - last_f[0]
            last_f[0] = cur
            return float(delta), time.perf_counter() - t0

        def cleanup() -> None:
            stop_flag.value = 1
            for p in neon_procs:
                p.join(timeout=3)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=1)

        dtype_label = f"CPU NEON {args.dtype} ({nw} workers)"

    elif not is_combined and selected[0] is Rail.ANE:
        import numpy as np

        console.print("[dim]ANE: building/loading CoreML model…[/]")
        m = _build_ane_model()
        in_arr = np.random.randn(1, ANE_CHANNELS, ANE_HW, ANE_HW).astype(np.float32)
        for _ in range(3):
            m.predict({"input": in_arr})

        def sample() -> tuple[float, float]:
            t0 = time.perf_counter()
            for _ in range(batch):
                m.predict({"input": in_arr})
            return float(batch * ANE_FLOPS_PER_INF), time.perf_counter() - t0

        dtype_label = f"ANE {ANE_LAYERS}×Conv2d {ANE_CHANNELS}ch {ANE_HW}² fp16"

    else:
        # ─── Combined rail mode ────────────────────────────────
        import multiprocessing as mp
        import os

        gpu_rail = gpu_sel[0] if gpu_sel else None
        has_cpu = Rail.CPU_NEON in cpu_sel
        has_ane = Rail.ANE in selected

        gpu_do_batch: Any = None
        if gpu_rail is Rail.MPS:
            import torch

            if not torch.backends.mps.is_available():
                raise SystemExit("MPS not available.")
            dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
                args.dtype
            ]
            dev = torch.device("mps")
            console.print(f"[dim]GPU (MPS): {n_size}² {args.dtype}[/]")
            a = torch.randn(n_size, n_size, device=dev, dtype=dt)
            b = torch.randn(n_size, n_size, device=dev, dtype=dt)
            c = torch.empty(n_size, n_size, device=dev, dtype=dt)
            for _ in range(5):
                torch.matmul(a, b, out=c)
            torch.mps.synchronize()

            def gpu_do_batch() -> float:
                for _ in range(batch):
                    torch.matmul(a, b, out=c)
                torch.mps.synchronize()
                return float(batch * flops_per_matmul)

        elif gpu_rail is Rail.MLX:
            import mlx.core as mx

            dm = {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}[
                args.dtype
            ]
            console.print(f"[dim]GPU (MLX): {n_size}² {args.dtype}[/]")
            a = mx.random.normal((n_size, n_size)).astype(dm)
            b = mx.random.normal((n_size, n_size)).astype(dm)
            mx.eval(a, b)
            for _ in range(5):
                mx.eval(a @ b)

            def gpu_do_batch() -> float:
                x = a
                for _ in range(batch):
                    x = x @ b
                mx.eval(x)
                return float(batch * flops_per_matmul)

        ctx = mp.get_context("spawn")
        stop_flag = ctx.Value("b", 0) if (has_cpu or has_ane) else None
        procs: list[Any] = []
        nw = 0
        cpu_counter: Any = None
        ane_counter: Any = None
        last_cpu = [0]
        last_ane = [0]

        if has_cpu:
            nw = args.workers if args.workers > 0 else (os.cpu_count() or 8)
            console.print(f"[dim]CPU: {nw} NEON workers[/]")
            cpu_counter = ctx.Value("q", 0)
            for _ in range(nw):
                procs.append(
                    ctx.Process(
                        target=_cpu_neon_worker,
                        args=(cpu_counter, stop_flag, args.dtype, 4096),
                    )
                )

        if has_ane:
            console.print("[dim]ANE: pre-building CoreML model…[/]")
            _build_ane_model()
            console.print("[dim]ANE: spawning worker…[/]")
            ane_counter = ctx.Value("q", 0)
            procs.append(ctx.Process(target=_ane_worker, args=(ane_counter, stop_flag)))

        for p in procs:
            p.start()

        breakdown: dict[str, float] = {}
        if gpu_rail:
            breakdown["gpu"] = 0.0
        if has_cpu:
            breakdown["cpu"] = 0.0
        if has_ane:
            breakdown["ane"] = 0.0

        def sample() -> tuple[float, float]:
            t0 = time.perf_counter()
            gpu_flops = gpu_do_batch() if gpu_do_batch is not None else 0.0
            if gpu_do_batch is None:
                time.sleep(0.5)
            t1 = time.perf_counter()
            dt = t1 - t0
            cpu_flops = 0.0
            if cpu_counter is not None:
                cur = cpu_counter.value
                cpu_flops = float(cur - last_cpu[0])
                last_cpu[0] = cur
            ane_flops = 0.0
            if ane_counter is not None:
                cur = ane_counter.value
                ane_flops = float(cur - last_ane[0])
                last_ane[0] = cur
            if "gpu" in breakdown:
                breakdown["gpu"] = gpu_flops / dt / 1e12
            if "cpu" in breakdown:
                breakdown["cpu"] = cpu_flops / dt / 1e12
            if "ane" in breakdown:
                breakdown["ane"] = ane_flops / dt / 1e12
            return gpu_flops + cpu_flops + ane_flops, dt

        def cleanup() -> None:
            if stop_flag is not None:
                stop_flag.value = 1
            for p in procs:
                p.join(timeout=3)
                if p.is_alive():
                    p.terminate()

        sample.breakdown = breakdown  # type: ignore[attr-defined,unused-ignore]
        parts: list[str] = []
        if gpu_rail is not None:
            parts.append(f"{gpu_rail.value.upper()} {n_size}² {args.dtype}")
        if has_cpu:
            parts.append(f"{nw}× NEON")
        if has_ane:
            parts.append("ANE")
        dtype_label = " + ".join(parts)

    # ─── Main loop ─────────────────────────────────────────────
    assert sample is not None
    start = time.perf_counter()
    iters = 0
    recent: deque[float] = deque(maxlen=60)
    spark_data: deque[float] = deque(maxlen=120)
    all_samples: list[Sample] = []
    first_window_avg: float | None = None

    log_f = None
    if args.log:
        log_f = open(args.log, "w", buffering=1)
        log_f.write(
            "elapsed_s,total_tflops,gpu_tflops,cpu_tflops,ane_tflops,"
            "cpu_w,gpu_w,ane_w,total_w,tflops_per_w\n"
        )
        console.print(f"[dim]log → {args.log}[/]")

    bd: dict[str, float] | None = getattr(sample, "breakdown", None)

    try:
        with Live(
            make_panel(
                dtype_label=dtype_label,
                iters=0,
                elapsed=0.0,
                recent=recent,
                spark_data=spark_data,
                all_samples=all_samples,
                first_window_avg=None,
                breakdown=bd,
                power=power,
            ),
            console=console,
            refresh_per_second=6,
        ) as live:
            while True:
                flops, dt = sample()
                tflops = flops / dt / 1e12
                elapsed = time.perf_counter() - start

                recent.append(tflops)
                spark_data.append(tflops)
                all_samples.append(Sample(t=elapsed, tflops=tflops))
                iters += 1

                if log_f:
                    g = bd.get("gpu", 0.0) if bd else 0.0
                    cp = bd.get("cpu", 0.0) if bd else 0.0
                    an = bd.get("ane", 0.0) if bd else 0.0
                    cw = power.cpu_mw / 1000.0
                    gw = power.gpu_mw / 1000.0
                    aw = power.ane_mw / 1000.0
                    tw = power.combined_mw / 1000.0
                    ev = tflops / tw if tw > 0 else 0.0
                    log_f.write(
                        f"{elapsed:.3f},{tflops:.4f},{g:.4f},{cp:.4f},{an:.4f},"
                        f"{cw:.3f},{gw:.3f},{aw:.3f},{tw:.3f},{ev:.4f}\n"
                    )

                if first_window_avg is None and elapsed >= 30.0:
                    bl = [s.tflops for s in all_samples if s.t <= 30.0]
                    if bl:
                        first_window_avg = sum(bl) / len(bl)

                live.update(
                    make_panel(
                        dtype_label=dtype_label,
                        iters=iters,
                        elapsed=elapsed,
                        recent=recent,
                        spark_data=spark_data,
                        all_samples=all_samples,
                        first_window_avg=first_window_avg,
                        breakdown=bd,
                        power=power,
                    )
                )
    except KeyboardInterrupt:
        elapsed = time.perf_counter() - start
        console.print()
        console.print("[yellow]Stopping workers…[/]", end=" ")
    finally:
        cleanup()
        power.stop()
        if log_f:
            log_f.close()
        console.print("[green]done.[/]")
        if all_samples:
            avg_all = sum(s.tflops for s in all_samples) / len(all_samples)
            elapsed = time.perf_counter() - start
            head = [s.tflops for s in all_samples if s.t <= 30.0]
            tail = [s.tflops for s in all_samples if s.t >= elapsed - 30.0]
            console.print(f"[dim]ran {elapsed:.1f}s, {iters} windows[/]")
            if head and tail and elapsed > 60:
                h, ta = sum(head) / len(head), sum(tail) / len(tail)
                d = (ta - h) / h * 100
                console.print(
                    f"[dim]first 30s {h:.2f}  ·  last 30s {ta:.2f}  ({d:+.1f}%)[/]"
                )
            console.print(f"[bold]all-time avg: {avg_all:.2f} TFLOPS[/]")


if __name__ == "__main__":
    main()
