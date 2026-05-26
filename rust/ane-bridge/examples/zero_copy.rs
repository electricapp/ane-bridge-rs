//! Zero-copy bind path: allocate IOSurface-backed buffers once, write
//! directly into the mapped memory, then run the model repeatedly with
//! no host-side copy on the request boundary.
#![allow(clippy::cast_precision_loss)]

use ane_bridge::{BufferAccess, Model, OpenOptions, QoS};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <mil_path> <weights_path>", args[0]);
        std::process::exit(1);
    }

    let opts = OpenOptions::new(&args[1], &args[2]);
    let model = Model::open(&opts)?;

    // Allocate IOSurface-backed buffers sized from the schema.
    let mut in_buf = model.input_buffer(0)?;
    let out_buf = model.output_buffer(0)?;
    println!(
        "[buf] in IOSurfaceID={}  out IOSurfaceID={}  nbytes={}/{}",
        in_buf.iosurface_id(),
        out_buf.iosurface_id(),
        in_buf.nbytes(),
        out_buf.nbytes(),
    );

    // Fill the input directly through the mapped pointer.
    in_buf.with_locked(BufferAccess::Write, |bytes| {
        for (i, chunk) in bytes.chunks_exact_mut(4).enumerate() {
            let v = (i as f32) * 0.01;
            chunk.copy_from_slice(&v.to_le_bytes());
        }
    })?;

    // Bind: the Request now owns both buffers. Subsequent runs reuse
    // the same surfaces and we access them back via the Request.
    let mut req = model.request()?;
    req.bind_input(0, in_buf)?;
    req.bind_output(0, out_buf)?;

    // Warm up + bench.
    for _ in 0..3 {
        req.run(QoS::Default)?;
    }
    let t0 = Instant::now();
    let iters = 100;
    for _ in 0..iters {
        req.run(QoS::Default)?;
    }
    let total = t0.elapsed();
    println!(
        "[bench] zero-copy: {} iters in {:.2} ms (avg {:.3} ms/iter)",
        iters,
        total.as_secs_f64() * 1e3,
        total.as_secs_f64() * 1e3 / f64::from(iters),
    );

    // Read the output directly out of the mapped output IOSurface.
    let out_ref = req.output_buffer_mut(0).expect("output bound");
    let ok = out_ref.with_locked(BufferAccess::Read, |bytes| {
        let mut all_ok = true;
        for (i, chunk) in bytes.chunks_exact(4).take(8).enumerate() {
            let v = f32::from_le_bytes(chunk.try_into().expect("4-byte chunk"));
            let expected = (i as f32) * 0.01;
            if (v - expected).abs() > 1e-2 {
                all_ok = false;
            }
        }
        all_ok
    })?;
    println!(
        "[check] first 8 values: {}",
        if ok { "OK" } else { "MISMATCH" }
    );
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
