#!/usr/bin/env bash
#
# Runs the ane-bridge benchmark and the matched CoreML benchmark
# against the same .mlmodelc bundle, then diffs the per-call latency.
#
# Usage:
#     ./scripts/bench_vs_coreml.sh                 # default: AVFuser
#     ./scripts/bench_vs_coreml.sh <mlmodelc_dir>
#
# Both runs use:
#   * the same warmup + iteration counts (set in the source files),
#   * a zero-filled input buffer reused across iterations,
#   * `cpuAndNeuralEngine` for the CoreML side, `QoS::UserInteractive`
#     for ane-bridge.
#
# Output: a side-by-side TSV summary with per-percentile delta.

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
DEFAULT_MODEL="/System/Library/Frameworks/SoundAnalysis.framework/Versions/A/Resources/SNLanguageAlignedAVFuserModel.mlmodelc"
MODEL="${1:-$DEFAULT_MODEL}"

if [ ! -d "$MODEL" ]; then
    echo "model bundle not found: $MODEL" >&2
    exit 1
fi
if [ ! -f "$MODEL/model.mil" ] || [ ! -f "$MODEL/weights/weight.bin" ]; then
    echo "missing model.mil or weights/weight.bin under $MODEL" >&2
    exit 1
fi

echo "== model: $MODEL"
echo

cd "$REPO/rust"
echo "== building bench_ane_bridge (release)"
cargo build --release --example bench_ane_bridge >/dev/null 2>&1

run_ane() {
    cargo run --release --example bench_ane_bridge -- "$MODEL" 2>/dev/null | tail -1
}
run_coreml() {
    local units=$1
    cd "$REPO"
    ANE_BENCH_UNITS="$units" swift scripts/bench_coreml.swift "$MODEL" 2>/dev/null | tail -1
}

ANE_LINE=$(run_ane)
ML_ANE_LINE=$(run_coreml ane)
ML_ALL_LINE=$(run_coreml all)
ML_CPU_LINE=$(run_coreml cpu)

echo
echo "== summary (microseconds per prediction)"
printf "%-14s %10s %10s %10s %10s %10s %10s %10s\n" \
    "path" "min" "p50" "p90" "p99" "max" "mean" "std"
parse() {
    awk -F'\t' '{ printf "%-14s %10s %10s %10s %10s %10s %10s %10s\n", $1, $4, $5, $6, $7, $8, $9, $10 }' \
        <<<"$1"
}
parse "$ANE_LINE"
parse "$ML_ANE_LINE"
parse "$ML_ALL_LINE"
parse "$ML_CPU_LINE"

# Compare ane-bridge against `coreml/.cpuAndNeuralEngine` — the most
# direct apples-to-apples comparison (both ask for ANE explicitly).
ane_p50=$(awk -F'\t' '{print $5}' <<<"$ANE_LINE")
ml_p50=$(awk -F'\t' '{print $5}' <<<"$ML_ANE_LINE")
ane_p99=$(awk -F'\t' '{print $7}' <<<"$ANE_LINE")
ml_p99=$(awk -F'\t' '{print $7}' <<<"$ML_ANE_LINE")
ane_mean=$(awk -F'\t' '{print $9}' <<<"$ANE_LINE")
ml_mean=$(awk -F'\t' '{print $9}' <<<"$ML_ANE_LINE")
cpu_p50=$(awk -F'\t' '{print $5}' <<<"$ML_CPU_LINE")

echo
awk -v ane_p50="$ane_p50" -v ml_p50="$ml_p50" \
    -v ane_p99="$ane_p99" -v ml_p99="$ml_p99" \
    -v ane_mean="$ane_mean" -v ml_mean="$ml_mean" \
    -v cpu_p50="$cpu_p50" '
BEGIN {
    printf "== delta vs coreml/.cpuAndNeuralEngine =="
    printf "\n  p50:  %+.3f us  (%.2fx)", ane_p50 - ml_p50, ml_p50 / ane_p50
    printf "\n  p99:  %+.3f us  (%.2fx)", ane_p99 - ml_p99, ml_p99 / ane_p99
    printf "\n  mean: %+.3f us  (%.2fx)", ane_mean - ml_mean, ml_mean / ane_mean
    printf "\n  ANE-vs-CPU speedup (p50, both via CoreML): %.2fx\n", cpu_p50 / ml_p50
}'
