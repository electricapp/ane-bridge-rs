//! Demonstrates [`Request::submit_async`] under an executor-agnostic
//! `block_on` written in 20-ish lines on top of `std`. The example
//! pipelines two evaluations: while one is in flight on the ANE, the
//! host thread can do other work.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::time::Instant;

use ane_bridge::{Model, OpenOptions, QoS};

/// Minimal executor: park the current thread until the future wakes us.
fn block_on<F: Future>(f: F) -> F::Output {
    struct ThreadWaker(std::thread::Thread);
    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker: Waker = Arc::new(ThreadWaker(std::thread::current())).into();
    let mut ctx = Context::from_waker(&waker);
    let mut f = pin!(f);
    loop {
        match f.as_mut().poll(&mut ctx) {
            Poll::Ready(v) => return v,
            Poll::Pending => std::thread::park(),
        }
    }
}

const SHAPE: [i64; 4] = [1, 64, 1, 16];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <mil_path> <weights_path>", args[0]);
        std::process::exit(1);
    }

    let opts = OpenOptions::new(&args[1], &args[2]);
    let model = Model::open(&opts)?;
    let mut req = model.request()?;

    let n_elem = (SHAPE.iter().copied().product::<i64>()) as usize;
    let input: Vec<f32> = (0..n_elem).map(|i| (i as f32) * 0.01).collect();

    // SAFETY: f32 is plain bytes; reinterpreting as &[u8] is sound.
    let in_bytes: &[u8] =
        unsafe { core::slice::from_raw_parts(input.as_ptr().cast::<u8>(), input.len() * 4) };
    req.set_input_bytes(0, in_bytes)?;

    // Warm up.
    block_on(async { req.submit_async(QoS::Default)?.await })?;

    // Benchmark: do 32 awaited submissions.
    let t0 = Instant::now();
    block_on(async {
        for _ in 0..32 {
            req.submit_async(QoS::Default)?.await?;
        }
        Ok::<_, ane_bridge::Error>(())
    })?;
    let total = t0.elapsed();
    println!(
        "[async] 32 awaited submissions in {:.2} ms (avg {:.3} ms/iter)",
        total.as_secs_f64() * 1e3,
        total.as_secs_f64() * 1e3 / 32.0,
    );
    Ok(())
}
