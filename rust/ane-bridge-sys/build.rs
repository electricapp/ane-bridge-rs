//! build.rs — compile the C/Obj-C sources into a static archive that we link
//! into this crate. We don't depend on a prebuilt dylib so `cargo build`
//! works standalone.
#![allow(missing_docs)]

use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // Project layout: rust/ane-bridge-sys/ → ../../c/{include,src}
    let root = crate_dir
        .parent() // rust/
        .and_then(|p| p.parent()) // ane-bridge/
        .expect("expected crate to live under <root>/rust/ane-bridge-sys");
    let inc_dir = root.join("c/include");
    let src_dir = root.join("c/src");

    let mut build = cc::Build::new();
    build
        .file(src_dir.join("ane_private.m"))
        .file(src_dir.join("ane_bridge.m"))
        .include(&inc_dir)
        .include(&src_dir)
        .flag("-fobjc-arc-exceptions")
        .flag("-fno-objc-arc")
        .flag("-Wall")
        .flag("-Wextra")
        .flag("-Wno-unused-parameter")
        .opt_level(2);

    // Opt-in LLVM source-based coverage for the C/Obj-C side. Set
    // `ANE_BRIDGE_COVERAGE=1` before `cargo clean -p ane-bridge-sys &&
    // cargo build`. The maintainer script at `scripts/coverage.sh`
    // bundles this with `LLVM_PROFILE_FILE` + `llvm-profdata merge` +
    // `llvm-cov show` to produce a parser coverage report.
    if std::env::var("ANE_BRIDGE_COVERAGE").is_ok_and(|v| v == "1") {
        build
            .flag("-fprofile-instr-generate")
            .flag("-fcoverage-mapping")
            .flag("-g");
        // rustc passes `-fprofile-instr-generate` to the linker but doesn't
        // know to pull in `libclang_rt.profile_osx.a` — clang's normal
        // driver does that for us, but we're going through rustc. Link
        // it explicitly so the test binaries emit .profraw on exit.
        let runtime_dir = std::process::Command::new("clang")
            .args(["--print-runtime-dir"])
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| {
                "/Library/Developer/CommandLineTools/usr/lib/clang/21/lib/darwin".to_string()
            });
        println!("cargo:rustc-link-search=native={runtime_dir}");
        println!("cargo:rustc-link-lib=static=clang_rt.profile_osx");
        // Force the linker to pull `__llvm_profile_runtime` from the
        // archive — without this, nothing references the symbol and the
        // archive is dropped, leaving the binary without the writeout
        // machinery and no .profraw on exit. macOS prefixes external
        // symbols with `_`, so the linker sees `___llvm_profile_runtime`.
        println!("cargo:rustc-link-arg=-Wl,-u,___llvm_profile_runtime");
        println!("cargo:warning=ane-bridge-sys: built with LLVM coverage (C-side)");
    }
    println!("cargo:rerun-if-env-changed=ANE_BRIDGE_COVERAGE");

    // Opt-in ThreadSanitizer for the C/Obj-C side. Set `ANE_BRIDGE_TSAN=1`
    // before `cargo clean -p ane-bridge-sys && cargo build`. We link the
    // dynamic TSAN runtime explicitly because rustc passes `-nodefaultlibs`
    // and drops the runtime clang would normally add via `-fsanitize=thread`.
    if std::env::var("ANE_BRIDGE_TSAN").is_ok_and(|v| v == "1") {
        build.flag("-fsanitize=thread").flag("-g");
        // Locate `libclang_rt.tsan_osx_dynamic.dylib`. We ask the
        // toolchain via `clang --print-runtime-dir`; if that fails we
        // fall back to the well-known CLT path.
        let runtime_dir = std::process::Command::new("clang")
            .args(["--print-runtime-dir"])
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| {
                "/Library/Developer/CommandLineTools/usr/lib/clang/21/lib/darwin".to_string()
            });
        println!("cargo:rustc-link-search=native={runtime_dir}");
        println!("cargo:rustc-link-lib=dylib=clang_rt.tsan_osx_dynamic");
        // The runtime is shipped as `libclang_rt.tsan_osx_dynamic.dylib`
        // — pass an rpath so it can be found at runtime.
        println!("cargo:rustc-link-arg=-Wl,-rpath,{runtime_dir}");
        println!("cargo:warning=ane-bridge-sys: built with ThreadSanitizer (C-side only)");
    }
    println!("cargo:rerun-if-env-changed=ANE_BRIDGE_TSAN");

    // Register `loom` as a known cfg name so `#[cfg(loom)]` in test
    // files doesn't fire `unexpected_cfgs` on normal builds.
    println!("cargo:rustc-check-cfg=cfg(loom)");

    build.compile("ane_bridge");

    println!(
        "cargo:rerun-if-changed={}",
        inc_dir.join("ane_bridge.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        src_dir.join("ane_bridge.m").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        src_dir.join("ane_private.m").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        src_dir.join("ane_private.h").display()
    );

    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=IOSurface");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=dylib=dl");
}
