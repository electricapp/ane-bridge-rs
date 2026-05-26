# Contributing to ane-bridge

Thanks for your interest. This is a small project; the contribution
process matches that.

## Development setup

Requires a recent macOS with the Xcode command line tools, Python 3.11+,
and Rust 1.95+.

```bash
git clone https://github.com/anthropics/ane-bridge
cd ane-bridge

# Build the dylib + C examples
make
make examples

# Generate the identity-model fixture and run the C example
python3 tools/make_identity_model.py build/identity
./build/bin/identity build/identity/model.mil build/identity/weights.bin

# Build + test the Rust side
cd rust
cargo build --workspace --all-targets
cargo test --release --workspace
```

## Pre-commit checks

CI mirrors these — please run them before opening a PR:

```bash
cd rust
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --release --workspace
```

## Lint policy

Every `unsafe` block and `unsafe impl` carries a `// SAFETY:` comment.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the full lint config and
safety invariants. If a clippy lint fights the FFI boundary, add a
narrow `#[allow(clippy::name)]` with a one-line comment.

## Areas where help is welcome

- Additional MIL op coverage tests.
- A `MLState` / iOS 18 stateful-model adapter (a higher-level alternative
  to the manual KV-cache path).
- More example models in `c/examples/` and `rust/ane-bridge/examples/`.
- Documenting newly observed `_ANECompiler` constraints (op shape limits,
  dtype constraints).

## License

By contributing, you agree your contributions are licensed under the
MIT license, the same as the rest of the project. See `LICENSE`.
