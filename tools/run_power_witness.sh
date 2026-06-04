#!/usr/bin/env bash
# Captures ANE and CPU power channels via powermetrics while
# power_witness drives BNNS-INT8, idle, and real-ANE phases in
# sequence. Requires sudo for powermetrics.
set -euo pipefail

cd "$(dirname "$0")/.."

PM_LOG=/tmp/ane_power_witness.pm.txt
PHASE_LOG=/tmp/ane_power_witness.phases.txt

if [ ! -x build/bin/power_witness ]; then
    echo "build/bin/power_witness missing — run: make -C tools all"; exit 1
fi
if [ ! -f build/identity/model.mil ] || [ ! -f build/identity/weights.bin ]; then
    echo "missing identity fixture — run: uv run python tools/make_identity_model.py build/identity"
    exit 1
fi

sudo -v
# Start powermetrics covering longer than the witness so the
# witness can never outrun the sampler. Phases: 3s idle + 5s BNNS +
# 3s gap + 5s ANE = 16s, plus warmup ~1s and powermetrics warmup ~1s
# = ~18s total. Sample for 25s (100 samples at 250ms).
sudo powermetrics --samplers ane_power,cpu_power -i 250 -n 100 \
    --format text > "$PM_LOG" 2>&1 &
PM_PID=$!
sleep 1

./build/bin/power_witness build/identity/model.mil build/identity/weights.bin 2> "$PHASE_LOG" || true

# Wait for powermetrics to complete its own -n count rather than
# killing it. The previous version raced and cut off the ANE phase.
wait "$PM_PID" 2>/dev/null || true

echo
echo "==== phase markers ===="
cat "$PHASE_LOG"
echo
# Parse: pair each powermetrics sample's elapsed-since-start with its
# ANE / CPU power, then bucket into phase windows using the marker
# timestamps. The python script writes a compact summary table.
python3 tools/parse_power_witness.py "$PM_LOG" "$PHASE_LOG"
