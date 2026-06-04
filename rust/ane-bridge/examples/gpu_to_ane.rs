//! Rust dual of `c/examples/gpu_to_ane.m`.
//!
//! Drives Metal + IOSurface via the actively-maintained `objc2-*`
//! crate family — no hand-rolled `objc_msgSend`. End-to-end contract:
//! Metal blit-writes through an `MTLBuffer` aliased to the same
//! `IOSurface` that ANE adopts, ANE runs an identity model, and the
//! readback matches what Metal wrote. Also asserts that attaching
//! `SharedEvents` to a direct request returns `Unsupported` rather
//! than crashing the framework worker.
//!
//! Run:
//!   uv run python tools/make_identity_model.py build/identity
//!   cargo run --example gpu_to_ane -- \
//!     ../build/identity/model.mil ../build/identity/weights.bin

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
    clippy::print_stderr,
    clippy::clone_on_ref_ptr,
    clippy::integer_division_remainder_used,
    clippy::missing_const_for_fn,
    clippy::use_debug,
    clippy::little_endian_bytes,
    clippy::big_endian_bytes,
    clippy::deref_by_slicing,
    clippy::doc_paragraphs_missing_punctuation,
    clippy::doc_markdown,
    clippy::multiple_unsafe_ops_per_block,
    clippy::undocumented_unsafe_blocks,
    clippy::unused_trait_names,
    clippy::string_slice,
    clippy::cast_possible_wrap,
    clippy::cast_ptr_alignment,
    clippy::assertions_on_result_states,
    clippy::borrow_as_ptr,
    clippy::too_many_lines,
    clippy::unused_result_ok,
    clippy::map_with_unused_argument_over_ranges,
    clippy::ignored_unit_patterns,
    clippy::unreachable,
    clippy::decimal_literal_representation,
    clippy::single_char_pattern,
    clippy::cast_precision_loss,
    reason = "Metal interop example uses raw FFI patterns + idiomatic CLI conventions"
)]

use std::env;
use std::ffi::c_void;
use std::process::ExitCode;
use std::ptr::NonNull;

use ane_bridge::{BufferAccess, Model, OpenOptions, QoS, SharedEvents, sys};

use objc2::rc::Retained;
use objc2_core_foundation::{
    CFDictionary, CFNumber, CFRetained, CFString, kCFTypeDictionaryKeyCallBacks,
    kCFTypeDictionaryValueCallBacks,
};
use objc2_io_surface::{IOSurfaceLockOptions, IOSurfaceRef};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions,
};

fn make_iosurface_props(nbytes: usize) -> CFRetained<CFDictionary> {
    let (w, h): (i64, i64) = if nbytes <= 16384 {
        (nbytes.max(1) as i64, 1)
    } else {
        let mut w = 4096i64;
        while (nbytes as i64) % w != 0 {
            w /= 2;
        }
        (w, (nbytes as i64) / w)
    };

    let entries: [(CFRetained<CFString>, CFRetained<CFNumber>); 6] = [
        (
            CFString::from_static_str("IOSurfaceWidth"),
            CFNumber::new_i64(w),
        ),
        (
            CFString::from_static_str("IOSurfaceHeight"),
            CFNumber::new_i64(h),
        ),
        (
            CFString::from_static_str("IOSurfaceBytesPerElement"),
            CFNumber::new_i32(1),
        ),
        (
            CFString::from_static_str("IOSurfaceBytesPerRow"),
            CFNumber::new_i64(w),
        ),
        (
            CFString::from_static_str("IOSurfaceAllocSize"),
            CFNumber::new_i64(nbytes.max(1) as i64),
        ),
        (
            CFString::from_static_str("IOSurfacePixelFormat"),
            CFNumber::new_i32(0),
        ),
    ];
    let mut keys: [*const c_void; 6] = [core::ptr::null(); 6];
    let mut vals: [*const c_void; 6] = [core::ptr::null(); 6];
    for (i, (k, v)) in entries.iter().enumerate() {
        keys[i] = CFRetained::as_ptr(k).as_ptr().cast();
        vals[i] = CFRetained::as_ptr(v).as_ptr().cast();
    }
    // SAFETY: `entries` owns the keys/values for the duration of the
    // call; CFDictionaryCreate retains them via the type callbacks.
    unsafe {
        CFDictionary::new(
            None,
            keys.as_mut_ptr(),
            vals.as_mut_ptr(),
            keys.len() as isize,
            &kCFTypeDictionaryKeyCallBacks,
            &kCFTypeDictionaryValueCallBacks,
        )
        .expect("CFDictionaryCreate")
    }
}

fn fill_iosurface_with_pattern(s: &IOSurfaceRef, n_floats: usize) {
    // SAFETY: lock for write before touching CPU-mapped bytes; unlock
    // after. Seed is unused (pass null).
    unsafe {
        let kr = s.lock(IOSurfaceLockOptions::empty(), core::ptr::null_mut());
        assert_eq!(kr, 0, "IOSurfaceLock failed: {kr}");
    }
    let base = s.base_address().as_ptr().cast::<f32>();
    for i in 0..n_floats {
        unsafe { base.add(i).write(((i % 100) + 1) as f32) };
    }
    unsafe {
        let kr = s.unlock(IOSurfaceLockOptions::empty(), core::ptr::null_mut());
        assert_eq!(kr, 0, "IOSurfaceUnlock failed: {kr}");
    }
}

fn iosurface_base(s: &IOSurfaceRef) -> NonNull<c_void> {
    s.base_address()
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model.mil> <weights.bin>", args[0]);
        return ExitCode::from(2);
    }

    let model = Model::open(&OpenOptions::new(&args[1], &args[2])).expect("open");
    let nbytes_in = model.input_nbytes(0);
    let nbytes_out = model.output_nbytes(0);
    assert_eq!(nbytes_in, nbytes_out, "identity model expected");
    let n_floats = nbytes_in / 4;

    let device = MTLCreateSystemDefaultDevice().expect("Metal device");
    let queue = device.newCommandQueue().expect("command queue");

    // --- Source IOSurface, prefilled with fp16-exact integers ---
    let src_props = make_iosurface_props(nbytes_in);
    let src_surface = unsafe { IOSurfaceRef::new(&src_props) }.expect("IOSurfaceCreate src");
    fill_iosurface_with_pattern(&src_surface, n_floats);

    let src_mtl = unsafe {
        device.newBufferWithBytesNoCopy_length_options_deallocator(
            iosurface_base(&src_surface),
            nbytes_in,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }
    .expect("new src buffer");

    // --- ANE-side IOSurface adopted into AneBuffer, also wrapped as MTLBuffer ---
    let ane_in_props = make_iosurface_props(nbytes_in);
    let ane_in_surface = unsafe { IOSurfaceRef::new(&ane_in_props) }.expect("IOSurfaceCreate ane");
    let surface_raw: *mut c_void = CFRetained::as_ptr(&ane_in_surface).as_ptr().cast();
    let ane_in_buf =
        unsafe { ane_bridge::Buffer::adopt_iosurface(surface_raw, nbytes_in) }.expect("adopt");
    let back_ref = ane_in_buf.iosurface_ref();
    assert_eq!(back_ref, surface_raw, "iosurface_ref round-trip");

    let ane_in_mtl = unsafe {
        device.newBufferWithBytesNoCopy_length_options_deallocator(
            iosurface_base(&ane_in_surface),
            nbytes_in,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }
    .expect("new ane buffer");

    let ane_out_buf = model.output_buffer(0).expect("output buffer");

    let mut req = model.request().expect("request");
    req.bind_input(0, ane_in_buf).expect("bind_input");
    req.bind_output(0, ane_out_buf).expect("bind_output");

    // --- GPU writes through the shared IOSurface via Metal blit copy ---
    let cb = queue.commandBuffer().expect("command buffer");
    let blit = cb.blitCommandEncoder().expect("blit encoder");
    unsafe {
        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
            &src_mtl,
            0,
            &ane_in_mtl,
            0,
            nbytes_in,
        );
    }
    blit.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();

    req.run(QoS::Default).expect("ane run");

    let out_buf = req.output_buffer_mut(0).expect("output buffer bound");
    let mut matches = 0usize;
    let mut mismatches = 0usize;
    let mut max_abs_err = 0.0f32;
    out_buf
        .with_locked(BufferAccess::Read, |bytes| {
            let floats: &[f32] = unsafe {
                core::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4)
            };
            for (i, &v) in floats.iter().enumerate() {
                let expected = ((i % 100) + 1) as f32;
                let err = (v - expected).abs();
                if err > max_abs_err {
                    max_abs_err = err;
                }
                if err < 1e-3 {
                    matches += 1;
                } else {
                    mismatches += 1;
                }
            }
        })
        .expect("output lock");

    println!("=== GPU → ANE handoff via shared IOSurface (Rust) ===");
    println!("model           : {}", args[1]);
    println!("input bytes     : {nbytes_in}");
    println!("fp32 elements   : {n_floats}");
    println!("matches         : {matches} / {n_floats}");
    println!("mismatches      : {mismatches}");
    println!("max abs error   : {max_abs_err:.6}");
    let mut rc = u8::from(mismatches != 0);
    println!(
        "{}: GPU writes to a shared IOSurface are visible to ANE.",
        if rc == 0 { "PASS" } else { "FAIL" }
    );

    // --- Safe-rejection contract: shared events on a direct request ---
    {
        let event = device.newSharedEvent().expect("shared event");
        let mut ev = SharedEvents::new().expect("shared events create");
        let mtl_evt_ptr: *mut c_void = Retained::as_ptr(&event).cast::<c_void>().cast_mut();
        unsafe { ev.add_wait(1, mtl_evt_ptr, sys::AneEventType::Default) }.expect("add_wait");

        let mut req2 = model.request().expect("request 2");
        req2.set_shared_events(Some(ev)).expect("attach events");
        match req2.run(QoS::Default) {
            Err(e) if e.status == sys::AneStatus::Unsupported => {
                println!(
                    "PASS: shared events on a direct request reject with UNSUPPORTED\n      (use ane_chain_* for shared-event sync)."
                );
            }
            other => {
                println!("FAIL: expected UNSUPPORTED rejection, got {other:?}");
                rc = 1;
            }
        }
    }

    ExitCode::from(rc)
}
