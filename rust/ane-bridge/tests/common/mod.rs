//! Shared test fixtures.
//!
//! Integration-test convention: items must be `pub` so they're reachable
//! from each `tests/*.rs` binary. The `unreachable_pub` lint is
//! suppressed here because nothing outside the test binaries consumes
//! these symbols.
#![allow(unreachable_pub)]
//!
//! These tests need a compilable MIL program on disk. To avoid the
//! framework's `hexStringIdentifier`-based temp dir colliding between
//! parallel tests (the hex is derived from the MIL bytes), each
//! [`Fixture`] embeds a unique sentinel in the MIL's `buildInfo`.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use tempfile::TempDir;

fn unique_token() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0_u128, |d| d.as_nanos());
    format!("{nanos}-{n}-{:?}", std::thread::current().id())
}

/// Identity MIL: y = cast(fp32, cast(fp16, x)). Shape [1, 64, 1, 16].
#[allow(dead_code)] // corpus.rs ships its own builder
fn identity_mil(token: &str) -> String {
    format!(
        r#"program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}, {{"ane-bridge-test", "{token}"}}}})]
{{
    func main<ios18>(tensor<fp32, [1, 64, 1, 16]> x) {{
        string td16 = const()[name = string("td16"), val = string("fp16")];
        tensor<fp16, [1, 64, 1, 16]> x16 = cast(dtype = td16, x = x)[name = string("x16")];
        string td32 = const()[name = string("td32"), val = string("fp32")];
        tensor<fp32, [1, 64, 1, 16]> y = cast(dtype = td32, x = x16)[name = string("y")];
    }} -> (y);
}}
"#
    )
}

/// Two-input / two-output identity: ya = id(a), yb = id(b).
#[allow(dead_code)] // used only by tests/integration.rs
fn dual_io_mil(token: &str) -> String {
    format!(
        r#"program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}, {{"ane-bridge-test", "{token}"}}}})]
{{
    func main<ios18>(tensor<fp32, [1, 64, 1, 16]> a, tensor<fp32, [1, 64, 1, 16]> b) {{
        string td16 = const()[name = string("td16"), val = string("fp16")];
        tensor<fp16, [1, 64, 1, 16]> a16 = cast(dtype = td16, x = a)[name = string("a16")];
        tensor<fp16, [1, 64, 1, 16]> b16 = cast(dtype = td16, x = b)[name = string("b16")];
        string td32 = const()[name = string("td32"), val = string("fp32")];
        tensor<fp32, [1, 64, 1, 16]> ya = cast(dtype = td32, x = a16)[name = string("ya")];
        tensor<fp32, [1, 64, 1, 16]> yb = cast(dtype = td32, x = b16)[name = string("yb")];
    }} -> (ya, yb);
}}
"#
    )
}

/// On-disk MIL + weights fixture. The `TempDir` is held inside so the
/// files survive until the fixture is dropped, but ane-bridge copies the
/// MIL/weights into its own temp dir at open time, so once `Model::open`
/// returns the fixture can be dropped freely.
pub struct Fixture {
    pub _dir: TempDir,
    pub mil: PathBuf,
    pub weights: PathBuf,
}

impl Fixture {
    fn from(mil: &str) -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let mil_path = dir.path().join("model.mil");
        let wts_path = dir.path().join("weights.bin");
        fs::write(&mil_path, mil).expect("write MIL");
        fs::write(&wts_path, b"\x00").expect("write weights placeholder");
        Self {
            _dir: dir,
            mil: mil_path,
            weights: wts_path,
        }
    }
    #[allow(dead_code)] // corpus.rs builds its own fixtures inline
    pub fn identity() -> Self {
        Self::from(&identity_mil(&unique_token()))
    }
    #[allow(dead_code)] // used only by tests/integration.rs
    pub fn dual_io() -> Self {
        Self::from(&dual_io_mil(&unique_token()))
    }
    pub fn mil_path(&self) -> &str {
        self.mil.to_str().expect("utf-8 path")
    }
    pub fn weights_path(&self) -> &str {
        self.weights.to_str().expect("utf-8 path")
    }
}
