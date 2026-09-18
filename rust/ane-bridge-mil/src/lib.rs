//! Write MIL programs for the Apple Neural Engine in Rust.
//!
//! [`ane-bridge`] takes MIL text plus a weight blob as its input contract, and
//! no converter ships with it. This crate writes both. It is a builder rather
//! than a compiler: each call emits one SSA statement, and the [`Val`] it
//! returns carries the static shape and dtype of the tensor that statement
//! produced.
//!
//! [`ane-bridge`]: https://docs.rs/ane-bridge
//!
//! ```no_run
//! use ane_bridge_mil::{ConvOpts, Dtype, Graph};
//!
//! let mut g = Graph::new("fir-demo");
//! let x = g.input("x", Dtype::Fp16, &[1, 1, 1, 4096]);
//! let taps = g.weight_fp16(&[1, 1, 1, 8], &[0.125; 8]);
//! let y = g.conv(&x, &taps, None, ConvOpts::decimating_fir(8, 1, 1));
//! let (mil_path, weights_path) = g.save("build/fir", &[&y])?;
//! # Ok::<(), std::io::Error>(())
//! ```
//!
//! # The builder rejects what the hardware mishandles
//!
//! The framework accepts less MIL than MIL describes, and one of its
//! restrictions is violated silently: an output width that is not a multiple of
//! 32 runs and returns mis-strided rows. The builder panics on that and on four
//! other cases rather than letting them reach the compiler.
//!
//! | rejected                              | because                                               |
//! | ------------------------------------- | ----------------------------------------------------- |
//! | boundary width off [`WIDTH_ALIGN`]    | rows come back mis-strided, with no error             |
//! | kernel wider than [`MAX_CONV_KERNEL`] | `CompilationFailure`                                  |
//! | fp32 arithmetic                       | the ANE computes in fp16; cast at the boundary        |
//! | a strided dense convolution           | runs at ~40% of the machine's bandwidth               |
//! | an op no output can reach             | dead subgraphs are compiled, and some fail to compile |
//!
//! Every panic names the call that caused it. Left to the framework, the same
//! mistakes surface as `InvalidMILProgram` against a generated identifier, or
//! as wrong numbers.
//!
//! # Pricing a graph before dispatching it
//!
//! Ops on this hardware do not fuse, so a graph's runtime is the bytes its
//! intermediates move. [`Graph::traffic`] counts them from the program text,
//! which makes two formulations comparable without building either.
//!
//! What the constraints are, where they come from, and what to do about a graph
//! that is too slow are in
//! [docs/MIL-CONSTRAINTS.md](https://github.com/electricapp/ane-bridge-rs/blob/master/docs/MIL-CONSTRAINTS.md).

pub mod program;
pub mod weights;

pub use program::{ConvOpts, Dtype, Graph, MAX_CONV_KERNEL, Pad, Traffic, Val, WIDTH_ALIGN};
pub use weights::{BlobDtype, BlobOffset, BlobWriter};
