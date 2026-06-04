//! Surface-completeness invariant.
//!
//! Every public `ane_*` function in `ane_bridge.h` must appear in the
//! Rust FFI (`ane-bridge-sys`) AND be exported by the compiled dylib.
//! A drift on any side fails the build — keeping the three sources of
//! truth in lockstep is the whole point of the bridge.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

const HEADER_REL:  &str = "../../c/include/ane_bridge.h";
const SYS_LIB_REL: &str = "../ane-bridge-sys/src/lib.rs";

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    let path = manifest_dir().join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Pull function names matching `ane_<name>(` from a string, excluding the
/// `_ane_internal_*` fuzz hooks and macro-like uses.
fn extract_functions(src: &str, prefix: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for line in src.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") || trimmed.starts_with("*") || trimmed.starts_with("/*") {
            continue;
        }
        let mut rest = line;
        while let Some(idx) = rest.find(prefix) {
            let after = &rest[idx + prefix.len()..];
            let name_len = after
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .count();
            if name_len == 0 {
                rest = &after[1.min(after.len())..];
                continue;
            }
            let candidate = format!("{prefix}{}", &after[..name_len]);
            let post = &after[name_len..];
            // Only count when followed by `(` (possibly preceded by spaces)
            // — declarations / calls, not field names.
            let post_trimmed = post.trim_start_matches([' ', '\t']);
            if post_trimmed.starts_with('(')
                && !candidate.starts_with("_ane_internal_")
                && !candidate.starts_with("ane_internal_")
            {
                out.insert(candidate);
            }
            rest = &after[name_len..];
        }
    }
    out
}

/// Dylib path for the compiled C library.
fn dylib_path() -> Option<PathBuf> {
    // Walk up from the manifest dir to find `build/lib/libane_bridge.dylib`.
    let mut dir: &Path = &manifest_dir();
    for _ in 0..6 {
        let cand = dir.join("build/lib/libane_bridge.dylib");
        if cand.exists() {
            return Some(cand);
        }
        dir = dir.parent()?;
    }
    None
}

fn extract_dylib_exports(dylib: &Path) -> BTreeSet<String> {
    let out = Command::new("nm")
        .args(["-gU", dylib.to_str().expect("utf-8 dylib path")])
        .output()
        .expect("invoke nm");
    assert!(out.status.success(), "nm failed: {:?}", out.status);
    let mut set = BTreeSet::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let Some(sym) = line.split_whitespace().last() else { continue };
        // macOS prefixes external symbols with `_`.
        let Some(stripped) = sym.strip_prefix("_ane_") else { continue };
        if stripped.starts_with("internal_") {
            continue;
        }
        let full = format!("ane_{stripped}");
        if full.starts_with("ane_internal_") {
            continue;
        }
        set.insert(full);
    }
    set
}

#[test]
fn header_matches_rust_ffi() {
    let header_fns = extract_functions(&read(HEADER_REL), "ane_");
    let sys_fns    = extract_functions(&read(SYS_LIB_REL), "ane_");

    let only_in_header: Vec<_> = header_fns.difference(&sys_fns).collect();
    let only_in_sys:    Vec<_> = sys_fns.difference(&header_fns).collect();

    assert!(
        only_in_header.is_empty() && only_in_sys.is_empty(),
        "C↔Rust drift:\n  header-only: {only_in_header:#?}\n  sys-only:    {only_in_sys:#?}"
    );
    assert!(!header_fns.is_empty(), "header parse produced no symbols");
}

#[test]
fn dylib_exports_match_header() {
    let Some(dylib) = dylib_path() else {
        eprintln!("dylib not built; run `make` to enable this assertion. Skipping.");
        return;
    };
    let header_fns  = extract_functions(&read(HEADER_REL), "ane_");
    let dylib_syms  = extract_dylib_exports(&dylib);

    let header_missing: Vec<_> = header_fns.difference(&dylib_syms).collect();
    let dylib_extra:    Vec<_> = dylib_syms.difference(&header_fns).collect();

    assert!(
        header_missing.is_empty(),
        "header declares but dylib does not export: {header_missing:#?}"
    );
    // `dylib_extra` is allowed — the dylib may legitimately re-export
    // helpers from system frameworks. We only assert the header subset.
    let _ = dylib_extra;
}

#[test]
fn no_internal_fuzz_in_public_surface_diff() {
    // Internal fuzz hooks must not accidentally leak into the public
    // surface set we diff above.
    let header_fns = extract_functions(&read(HEADER_REL), "ane_");
    for name in &header_fns {
        assert!(
            !name.starts_with("ane_internal_"),
            "internal hook leaked into diff set: {name}"
        );
    }
}
