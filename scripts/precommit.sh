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

echo "→ make (C dylib + warnings as errors)"
make >/dev/null

echo "✔ all precommit checks passed"
