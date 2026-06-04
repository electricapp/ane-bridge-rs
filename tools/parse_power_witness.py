# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rich>=13",
#     "typer>=0.12",
# ]
# ///
"""Align powermetrics samples with witness phase markers.

Takes the raw powermetrics text log and the phases.txt produced by
power_witness, identifies the wall-clock start time of each
powermetrics sample, then buckets per-sample `(cpu_mW, ane_mW)` into
each phase by unix-time intersection. Prints a compact summary table.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path  # noqa: TC003 -- used at runtime, not just in annotations
from typing import Annotated, Final

import typer
from rich.console import Console
from rich.table import Table

SAMPLE_HEADER = re.compile(
    r"\*\*\* Sampled system activity \(([^)]+)\) \((\d+\.\d+)ms elapsed\)",
)

# Empirical thresholds for the verdict logic. The ANE power channel
# sits at the noise floor (a few mW) when the unit is idle and rises
# to hundreds of mW when actually running inference.
ANE_IDLE_NOISE_FLOOR_MW: Final[int] = 10
ANE_ENGAGED_MW_LOWER_BOUND: Final[int] = 50
ANE_RATIO_FOR_DISTINCT_UNITS: Final[int] = 2

console = Console()


def parse_powermetrics(path: Path) -> list[tuple[float, float, int, int]]:
    """Return list of (wall_unix_time_sec, elapsed_ms, cpu_mW, ane_mW)."""
    samples: list[tuple[float, float, int, int]] = []
    current_ts: float | None = None
    current_elapsed: float | None = None
    cpu: int | None = None
    ane: int | None = None
    for line in path.read_text().splitlines():
        match = SAMPLE_HEADER.match(line)
        if match:
            if (
                current_ts is not None
                and current_elapsed is not None
                and cpu is not None
                and ane is not None
            ):
                samples.append((current_ts, current_elapsed, cpu, ane))
            cpu = None
            ane = None
            try:
                current_ts = datetime.strptime(
                    match.group(1), "%a %b %d %H:%M:%S %Y %z",
                ).timestamp()
            except ValueError:
                current_ts = None
            current_elapsed = float(match.group(2))
            continue
        if line.startswith("CPU Power:"):
            cpu = int(line.split(":")[1].strip().split()[0])
        elif line.startswith("ANE Power:"):
            ane = int(line.split(":")[1].strip().split()[0])
    if (
        current_ts is not None
        and current_elapsed is not None
        and cpu is not None
        and ane is not None
    ):
        samples.append((current_ts, current_elapsed, cpu, ane))
    return samples


def parse_phases(path: Path) -> dict[str, float]:
    """Return mapping of phase marker name to unix timestamp."""
    phases: dict[str, float] = {}
    min_phase_tokens = 3
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < min_phase_tokens or parts[0] != "PHASE":
            continue
        try:
            phases[parts[1]] = float(parts[2])
        except ValueError:
            continue
    return phases


def window_stats(
    samples: list[tuple[float, float, int, int]], t0: float, t1: float,
) -> tuple[int, float, int, float, int]:
    """Return (count, cpu_avg_mW, cpu_max_mW, ane_avg_mW, ane_max_mW)."""
    in_window = [s for s in samples if t0 <= s[0] <= t1]
    if not in_window:
        return (0, 0.0, 0, 0.0, 0)
    cpu_vals = [s[2] for s in in_window]
    ane_vals = [s[3] for s in in_window]
    return (
        len(in_window),
        sum(cpu_vals) / len(cpu_vals),
        max(cpu_vals),
        sum(ane_vals) / len(ane_vals),
        max(ane_vals),
    )


def main(
    pm_log: Annotated[Path, typer.Argument(help="powermetrics text log")],
    phase_log: Annotated[Path, typer.Argument(help="phase markers stderr capture")],
) -> None:
    """Align powermetrics samples with witness phase markers and print verdict."""
    samples = parse_powermetrics(pm_log)
    phases = parse_phases(phase_log)
    if not samples:
        console.print("[red]no parseable powermetrics samples[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"[bold]aligned power summary[/bold]  "
        f"({len(samples)} samples, span {samples[-1][0] - samples[0][0]:.1f}s)",
    )

    windows = [
        ("IDLE", "IDLE_START", "IDLE_END"),
        ("CPU MATMUL", "CPU_MATMUL_START", "CPU_MATMUL_END"),
        ("GAP", "GAP_START", "GAP_END"),
        ("ANE", "ANE_START", "ANE_END"),
    ]
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("phase", style="bold")
    table.add_column("n", justify="right")
    table.add_column("CPU avg mW", justify="right")
    table.add_column("CPU max mW", justify="right")
    table.add_column("ANE avg mW", justify="right")
    table.add_column("ANE max mW", justify="right")
    cpu_stats: dict[str, tuple[int, float, int, float, int]] = {}
    for label, start_key, end_key in windows:
        t0 = phases.get(start_key)
        t1 = phases.get(end_key)
        if t0 is None or t1 is None:
            table.add_row(label, "—", "—", "—", "—", "—")
            continue
        stats = window_stats(samples, t0, t1)
        cpu_stats[label] = stats
        n, cavg, cmax, aavg, amax = stats
        table.add_row(
            label,
            str(n),
            f"{cavg:.0f}",
            str(cmax),
            f"{aavg:.1f}",
            str(amax),
        )
    console.print(table)

    cpu_mm = cpu_stats.get("CPU MATMUL")
    ane = cpu_stats.get("ANE")
    if cpu_mm is not None and ane is not None and cpu_mm[0] > 0 and ane[0] > 0:
        console.print()
        console.print("[bold]verdict[/bold]")
        cm_ane_avg = cpu_mm[3]
        ane_ane_avg = ane[3]
        if (
            cm_ane_avg < ANE_IDLE_NOISE_FLOOR_MW
            and ane_ane_avg > cm_ane_avg * ANE_RATIO_FOR_DISTINCT_UNITS
        ):
            console.print(
                "  [green]CPU matmul draws CPU power, no ANE power.[/green]\n"
                "  [green]ANE phase draws ANE power, CPU drops.[/green]\n"
                "  [bold green]→ CPU matmul does NOT use the Neural Engine. "
                "They are different units.[/bold green]",
            )
        elif cm_ane_avg > ANE_ENGAGED_MW_LOWER_BOUND:
            console.print(
                "  [yellow]ANE power present during CPU matmul — would support "
                "shared-silicon claim.[/yellow]",
            )
        else:
            console.print(
                "  [yellow]Mixed signal — extend phase durations and re-run.[/yellow]",
            )


if __name__ == "__main__":
    typer.run(main)
