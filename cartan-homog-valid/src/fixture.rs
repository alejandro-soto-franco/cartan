//! Fixture loader: NPZ files with companion meta.json describing each ECHOES test case.

use std::path::{Path, PathBuf};
use serde::Deserialize;

pub fn fixture_root() -> PathBuf {
    std::env::var("CARTAN_HOMOG_FIXTURES_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/basic"))
}

#[derive(Debug, Deserialize, Clone)]
pub struct FixtureMeta {
    pub case_id: String,
    pub tensor_order: u8,
    pub scheme: String,
    pub tolerance_tier: String,
    pub echoes_version: String,
    #[serde(default)]
    pub echoes_commit: String,
    #[serde(default)]
    pub provenance: serde_json::Value,
}

#[derive(Debug)]
pub struct Fixture {
    pub meta: FixtureMeta,
    pub npz_path: PathBuf,
}

impl Fixture {
    pub fn load_all(root: &Path) -> Vec<Fixture> {
        let mut out = Vec::new();
        if !root.exists() { return out; }
        let v1 = root.join("v1");
        if !v1.exists() { return out; }
        visit(&v1, &mut out);
        out.sort_by(|a, b| a.meta.case_id.cmp(&b.meta.case_id));
        out
    }

    pub fn tolerance(&self) -> f64 {
        match self.meta.tolerance_tier.as_str() {
            "exact"                => 1e-10,
            "tight"                => 1e-8,
            "iterative"            => 1e-6,
            "quadrature_sensitive" => 1e-4,
            _                      => 1e-6,
        }
    }
}

fn visit(dir: &Path, out: &mut Vec<Fixture>) {
    let rd = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return,
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() { visit(&path, out); continue; }
        if path.extension().and_then(|s| s.to_str()) != Some("npz") { continue; }
        let meta_path = path.with_extension("json");
        if !meta_path.exists() { continue; }
        let raw = match std::fs::read_to_string(&meta_path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let meta: FixtureMeta = match serde_json::from_str(&raw) {
            Ok(m) => m,
            Err(_) => continue,
        };
        out.push(Fixture { meta, npz_path: path });
    }
}
