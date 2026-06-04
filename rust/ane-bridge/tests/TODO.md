# Tests — incremental lint-suppression cleanup

Every `tests/*.rs` and `examples/*.rs` currently carries a large
file-level `#![allow(...)]` block that disables the panic-flavored
clippy restriction lints (`unwrap_used`, `expect_used`,
`indexing_slicing`, `as_conversions`, `panic`, `print_stdout`,
`undocumented_unsafe_blocks`, …). This was added when the workspace
moved to `restriction = deny` so the test/example crates could keep
compiling against the same strict lints as the library.

The library is the source of truth — it is clippy-clean *without*
these blanket allows. The blocks in tests + examples are a debt to
pay down, not a permanent policy.

## How to chip away

Pick a lint from the allow list. For that lint:

  1. Remove the lint from the `#![allow(...)]` block in one file.
  2. Run `cargo clippy -p ane-bridge --test <name>` (or
     `--example <name>`); look at the failures it surfaces.
  3. Fix at the source — replace `.unwrap()` with `?` against a
     `Result<(), Box<dyn Error>>` test signature, swap raw indexing
     for `.get(..).expect(...)` with a load-bearing message, switch
     `as` to `try_from`, etc.
  4. When that file is clean for the lint, drop it from that file's
     allow block and move on.

Suggested order (cheapest first):

  - `clippy::std_instead_of_core` / `std_instead_of_alloc` —
    mechanical: swap `std::fmt` for `core::fmt`, `std::sync::Arc`
    for `alloc::sync::Arc`, etc.
  - `clippy::separated_literal_suffix` / `unseparated_literal_suffix`
    — pick one style globally, drop the other.
  - `clippy::doc_paragraphs_missing_punctuation` — terminal `.`
    on module/item doc comments.
  - `clippy::tests_outside_test_module` — wrap top-level `#[test]`
    fns in `mod tests { ... }`; cheap but touches every file.
  - `clippy::cast_possible_wrap` / `cast_possible_truncation` /
    `cast_sign_loss` — replace `as` with `usize::try_from(...)?`
    or document with a justified `#[expect]`.
  - `clippy::indexing_slicing` — bounds-check or `.get()`.
  - `clippy::unwrap_used` / `expect_used` — the big one. Switching
    `#[test] fn foo()` to `#[test] fn foo() -> Result<(), Box<dyn
    Error>>` lets `?` replace nearly all of them; the residual
    handful become a small set of `#[expect]`s on lines where the
    panic IS the assertion.

## Done = these files no longer carry the bulk allow

When a file's allow block is back down to its task-specific lints
(the ones that existed *before* the bulk block was added — e.g.
`cast_precision_loss` for percentile math), delete the block
entirely and let the workspace lints take over.

The end state is symmetric with the library: every test/example
also lives under `restriction = deny`, with only narrow,
justified per-fn `#[expect(...)]`s where the lint genuinely
fights the test pattern.
