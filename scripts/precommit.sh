#!/usr/bin/env bash
# Mirrors the CI checks. Install as a git pre-commit hook with:
#   ln -sf ../../scripts/precommit.sh .git/hooks/pre-commit
set -euo pipefail

cd "$(dirname "$0")/.."

PY_TOOLS=(tools/parse_power_witness.py tools/make_identity_model.py tools/make_bench_model.py)

echo "→ ruff --select ALL  (tools/*.py)"
uv run --quiet --with ruff -- ruff check --select ALL "${PY_TOOLS[@]}"

echo "→ mypy --strict  (tools/*.py)"
uv run --quiet --with mypy --with typer --with rich --with torch --with coremltools \
    -- mypy --strict "${PY_TOOLS[@]}"

echo "→ cargo fmt --check"
cargo fmt --manifest-path rust/Cargo.toml --all -- --check

echo "→ cargo clippy --workspace --all-targets -- -D warnings"
cargo clippy --manifest-path rust/Cargo.toml --workspace --all-targets -- -D warnings

echo "→ cargo doc --no-deps --workspace (warnings as errors)"
RUSTDOCFLAGS="-D warnings" \
    cargo doc --manifest-path rust/Cargo.toml --no-deps --workspace --quiet

echo "→ cargo test --workspace (release-style, no slow examples)"
cargo test --manifest-path rust/Cargo.toml --workspace --quiet

echo "→ clang-format --dry-run --Werror  (c/)"
CLANG_FORMAT="$(xcrun -f clang-format 2>/dev/null || command -v clang-format || true)"
if [ -z "$CLANG_FORMAT" ]; then
    echo "clang-format not found — install the Xcode Command Line Tools" >&2
    exit 1
fi
"$CLANG_FORMAT" --dry-run --Werror \
    c/src/*.m c/src/*.h c/include/*.h c/examples/*.c c/examples/*.m

echo "→ clang-tidy  (c/src, max checks as errors)"
CLANG_TIDY="${CLANG_TIDY:-$(brew --prefix llvm 2>/dev/null)/bin/clang-tidy}"
if [ ! -x "$CLANG_TIDY" ]; then CLANG_TIDY="$(command -v clang-tidy || true)"; fi
if [ -z "$CLANG_TIDY" ] || [ ! -x "$CLANG_TIDY" ]; then
    echo "clang-tidy not found — install with: brew install llvm" >&2
    exit 1
fi
"$CLANG_TIDY" c/src/ane_bridge.m c/src/ane_private.m -- \
    -I c/include -fno-objc-arc -x objective-c -isysroot "$(xcrun --show-sdk-path)"

echo "→ make (C dylib + examples, -Weverything -Werror)"
make >/dev/null
make examples >/dev/null

echo "✔ all precommit checks passed"
