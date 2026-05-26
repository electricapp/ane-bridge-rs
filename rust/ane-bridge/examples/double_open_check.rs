//! Ad-hoc sanity check: open the same model twice in-process and print
//! wall time + `cache_hit` on each. Used to validate the warm-start cache
//! optimization end-to-end on a real (compiled) mlmodelc.
//!
//!     cargo run --release --example double_open_check -- <mil_path> <weights_path>

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
