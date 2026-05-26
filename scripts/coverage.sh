#!/usr/bin/env bash
#
# Generate an LLVM source-based coverage report for the C/Obj-C parser.
#
# Rebuilds `ane-bridge-sys` with `-fprofile-instr-generate
# -fcoverage-mapping` (via the `ANE_BRIDGE_COVERAGE=1` env var picked
# up by `build.rs`), runs the parser_fuzz + corpus tests with
# `LLVM_PROFILE_FILE` set, merges the raw .profraw files, and dumps
# both a text summary and an HTML report under `target/coverage/`.
#
# Usage:
#   ./scripts/coverage.sh                 # all parser-relevant tests
#   ./scripts/coverage.sh parser_fuzz     # just the named test binary
#
# The report focuses on `c/src/ane_bridge.m` (where the parser lives)
# but includes `ane_private.m` for completeness.

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
RUST_DIR="$REPO/rust"
OUT="$REPO/target/coverage"
PROF_DIR="$OUT/profraw"

rm -rf "$OUT"
mkdir -p "$PROF_DIR"

cd "$RUST_DIR"

# Force a clean rebuild of the sys crate so the coverage flags take
# effect. The Rust side does not need coverage instrumentation for
# this report — we're targeting the C parser.
ANE_BRIDGE_COVERAGE=1 cargo clean -p ane-bridge-sys
ANE_BRIDGE_COVERAGE=1 cargo build --release --workspace --tests

if [ $# -eq 0 ]; then
  set -- parser_fuzz corpus borrow_safety integration
fi
# IMPORTANT: keep `ANE_BRIDGE_COVERAGE=1` set during the test runs.
# Without it, cargo's build-script rerun logic re-links the test
# binaries without the profile runtime and no .profraw is emitted.
for t in "$@"; do
  echo "==> running test binary: $t"
  ANE_BRIDGE_COVERAGE=1 \
    LLVM_PROFILE_FILE="$PROF_DIR/$t-%p-%m.profraw" \
    cargo test --release --workspace --test "$t" -- --nocapture
done

# Merge raw profiles into a single .profdata.
echo "==> merging profiles"
xcrun llvm-profdata merge -sparse "$PROF_DIR"/*.profraw -o "$OUT/coverage.profdata"

# Find the static archive that holds our C objects. cargo writes it
# under target/release/build/<hash>/out/libane_bridge.a — pick the
# freshest one in case stale hashes from earlier builds linger.
LIB=$(find "$RUST_DIR/target/release/build" -name 'libane_bridge.a' \
       -print0 2>/dev/null \
       | xargs -0 stat -f '%m %N' \
       | sort -rn | head -n1 | cut -d' ' -f2-)
if [ -z "$LIB" ] || [ ! -f "$LIB" ]; then
  echo "could not find libane_bridge.a — did the build succeed?" >&2
  exit 1
fi

echo "==> generating reports against $LIB"
xcrun llvm-cov report \
  -instr-profile="$OUT/coverage.profdata" \
  "$LIB" \
  "$REPO/c/src/ane_bridge.m" \
  "$REPO/c/src/ane_private.m" \
  > "$OUT/report.txt"

xcrun llvm-cov show \
  -instr-profile="$OUT/coverage.profdata" \
  -format=html \
  -output-dir="$OUT/html" \
  "$LIB" \
  "$REPO/c/src/ane_bridge.m" \
  "$REPO/c/src/ane_private.m"

echo
echo "== summary =="
cat "$OUT/report.txt"
echo
echo "HTML report: $OUT/html/index.html"
