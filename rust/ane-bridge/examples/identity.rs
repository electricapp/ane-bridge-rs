//! Rust mirror of `c/examples/identity.c` — verifies the safe wrapper
//! end-to-end against a tiny identity-via-cast MIL program.
#![allow(clippy::cast_precision_loss)] // benchmarking math only

use ane_bridge::{Model, OpenOptions, QoS};
use std::time::Instant;

const SHAPE: [i64; 4] = [1, 64, 1, 16];

fn n_elem() -> usize {
    SHAPE
        .iter()
        .copied()
        .map(|d| usize::try_from(d).expect("shape must be non-negative"))
        .product()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <mil_path> <weights_path>", args[0]);
        std::process::exit(1);
    }
    let mil_path = &args[1];
    let weights_path = &args[2];

    let opts = OpenOptions::new(mil_path, weights_path).qos(QoS::Default);

    let model = Model::open(&opts)?;
    println!(
        "[open] OK  inputs={} outputs={}  in_bytes={} out_bytes={}",
        model.num_inputs(),
        model.num_outputs(),
        model.input_nbytes(0),
        model.output_nbytes(0),
    );

    let mut req = model.request()?;
    let n = n_elem();
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();

    // SAFETY:
    // We reinterpret `&[f32]` as `&[u8]`. `f32` has no padding and any
    // bit pattern is valid `u8`, so reading those bytes is sound. The
    // borrow is shared and lives for the duration of the call into
    // `set_input_bytes`, which the library guarantees to memcpy out
    // synchronously before returning.
    let in_bytes: &[u8] =
        unsafe { core::slice::from_raw_parts(input.as_ptr().cast::<u8>(), input.len() * 4) };
    req.set_input_bytes(0, in_bytes)?;

    // Warm up
    for _ in 0..3 {
        req.run(QoS::Default)?;
    }

    let t0 = Instant::now();
    let iters = 10;
    for _ in 0..iters {
        req.run(QoS::Default)?;
    }
    let total = t0.elapsed();
    println!(
        "[run] {} iters in {:.2} ms (avg {:.3} ms/iter)",
        iters,
        total.as_secs_f64() * 1e3,
        total.as_secs_f64() * 1e3 / f64::from(iters),
    );

    let mut output = vec![0_f32; n];
    // SAFETY:
    // We reinterpret `&mut [f32]` as `&mut [u8]`. `f32` has no padding,
    // any bit pattern is a valid `u8`, and any bit pattern is also a
    // valid `f32` (modulo NaN normalization which we do not depend on).
    // The borrow is exclusive and lives for the call into
    // `get_output_bytes`, which memcpys into the slice synchronously.
    let out_bytes: &mut [u8] = unsafe {
        core::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<u8>(), output.len() * 4)
    };
    req.get_output_bytes(0, out_bytes)?;

    let ok = output
        .iter()
        .take(8)
        .enumerate()
        .all(|(i, &v)| (i as f32).mul_add(-0.01, v).abs() < 1e-2);
    println!(
        "[check] first 8 values: {}",
        if ok { "OK" } else { "MISMATCH" }
    );
    println!(
        "  in[0..3]:  {:.4} {:.4} {:.4} {:.4}",
        input[0], input[1], input[2], input[3]
    );
    println!(
        "  out[0..3]: {:.4} {:.4} {:.4} {:.4}",
        output[0], output[1], output[2], output[3]
    );

    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
