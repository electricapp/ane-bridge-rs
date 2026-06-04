//! Ad-hoc sanity check: open the same model twice in-process and print
//! wall time + `cache_hit` on each. Used to validate the warm-start cache
//! optimization end-to-end on a real (compiled) mlmodelc.
//!
//!     cargo run --release --example double_open_check -- <mil_path> <weights_path>

#![allow(
    clippy::allow_attributes,
    clippy::allow_attributes_without_reason,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::panic,
    clippy::print_stdout,
    clippy::dbg_macro,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::missing_assert_message,
    clippy::missing_docs_in_private_items,
    clippy::std_instead_of_core,
    clippy::std_instead_of_alloc,
    clippy::separated_literal_suffix,
    clippy::unseparated_literal_suffix,
    clippy::unreadable_literal,
    clippy::shadow_unrelated,
    clippy::shadow_reuse,
    clippy::shadow_same,
    clippy::min_ident_chars,
    clippy::float_arithmetic,
    clippy::float_cmp,
    clippy::arithmetic_side_effects,
    clippy::integer_division,
    clippy::default_numeric_fallback,
    clippy::pattern_type_mismatch,
    clippy::if_then_some_else_none,
    clippy::single_call_fn,
    clippy::needless_pass_by_value,
    clippy::let_underscore_must_use,
    clippy::let_underscore_untyped,
    clippy::redundant_pub_crate,
    clippy::semicolon_outside_block,
    clippy::semicolon_inside_block,
    clippy::semicolon_if_nothing_returned,
    clippy::cast_precision_loss,
    reason = "example uses idiomatic CLI patterns — `unwrap` / `println!` are intentional"
)]

use std::time::Instant;

use ane_bridge::{Model, OpenOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mil = std::env::args()
        .nth(1)
        .ok_or("usage: double_open_check <mil_path> <weights_path>")?;
    let wts = std::env::args()
        .nth(2)
        .ok_or("usage: double_open_check <mil_path> <weights_path>")?;
    let opts = OpenOptions::new(&mil, &wts);

    let t = Instant::now();
    let m1 = Model::open(&opts)?;
    let cold = t.elapsed();
    println!("cold:  {cold:?}  was_cached={}", m1.was_cached());

    let t = Instant::now();
    let m2 = Model::open(&opts)?;
    let warm = t.elapsed();
    println!("warm:  {warm:?}  was_cached={}", m2.was_cached());

    drop((m1, m2));
    Ok(())
}
