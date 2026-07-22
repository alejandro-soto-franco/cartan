//! cartan side of the cross-language comparison.
//!
//! Reads the shared fixtures, computes the same operations as the Julia and
//! Python harnesses, and writes values alongside timings. Values first: a
//! speed comparison between implementations that disagree means nothing.
//!
//! Manifold dimensions are const generic, so the runtime dimension from the
//! fixture file is dispatched through a macro over the dimensions the fixtures
//! actually contain. A dimension with no arm is reported, never silently
//! skipped.
//!
//! ```bash
//! cargo run --release --bin cartan-bench-crosslang
//! ```

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use nalgebra::{SMatrix, SVector};
use serde_json::{json, Value};

use cartan_core::{Manifold, ParallelTransport};
use cartan_manifolds::{Spd, Sphere};

/// Discarded before timing, so the branch predictor and caches are warm.
const WARMUP: usize = 50;
/// Number of timed batches. The median is taken across these.
const SAMPLES: usize = 200;
/// A batch must run at least this long for the clock read to be negligible.
const MIN_BATCH_NS: u128 = 500_000;

/// Median and interquartile range of one call to `f`, in nanoseconds.
///
/// Calls are timed in batches rather than individually. `Instant::now` costs
/// on the order of 20 ns, which is comparable to the operations being measured
/// here: timing a 15 ns `exp` one call at a time reports roughly 40 ns, almost
/// all of it clock overhead. Julia's BenchmarkTools batches for the same
/// reason, so timing individually would have biased the comparison against
/// cartan on exactly the fastest operations.
///
/// The batch size is chosen so each batch runs for at least `MIN_BATCH_NS`,
/// which makes a single clock read irrelevant to the result.
fn time_ns<T, F: FnMut() -> T>(mut f: F) -> (f64, f64, f64) {
    for _ in 0..WARMUP {
        std::hint::black_box(f());
    }

    // Grow the batch until it runs long enough to time accurately.
    let mut batch = 1usize;
    loop {
        let t0 = Instant::now();
        for _ in 0..batch {
            std::hint::black_box(f());
        }
        if t0.elapsed().as_nanos() >= MIN_BATCH_NS || batch >= 1 << 22 {
            break;
        }
        batch *= 2;
    }

    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let t0 = Instant::now();
        for _ in 0..batch {
            std::hint::black_box(f());
        }
        samples.push(t0.elapsed().as_nanos() as f64 / batch as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |q: f64| samples[((samples.len() as f64 - 1.0) * q).round() as usize];
    (at(0.5), at(0.25), at(0.75))
}

/// `black_box` for an input.
///
/// Blackboxing only the result is not enough: with a loop-invariant argument
/// the optimiser hoists the entire call out of the timing batch and the
/// measurement collapses to zero. Every timed closure below launders its
/// inputs through this.
#[inline(always)]
fn bb<T>(x: &T) -> &T {
    std::hint::black_box(x)
}

fn record(manifold: &str, dim: usize, op: &str, value: Vec<f64>, t: (f64, f64, f64)) -> Value {
    json!({
        "lib": "cartan",
        "manifold": manifold,
        "dim": dim,
        "op": op,
        "value": value,
        "median_ns": t.0,
        "q1_ns": t.1,
        "q3_ns": t.2,
    })
}

fn vec_of(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("fixture field must be an array")
        .iter()
        .map(|x| x.as_f64().expect("fixture entry must be a number"))
        .collect()
}

/// Flatten a fixture's row-major nested list into a single vector.
fn mat_of(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("fixture matrix must be an array of rows")
        .iter()
        .flat_map(vec_of)
        .collect()
}

macro_rules! sphere_arm {
    ($n:literal, $case:expr, $out:expr) => {{
        let m = Sphere::<$n>;
        let p = SVector::<f64, $n>::from_column_slice(&vec_of(&$case["p"]));
        let v = SVector::<f64, $n>::from_column_slice(&vec_of(&$case["v"]));
        let q = SVector::<f64, $n>::from_column_slice(&vec_of(&$case["q"]));

        let e = m.exp(&p, &v);
        $out.push(record("sphere", $n, "exp", e.as_slice().to_vec(),
                         time_ns(|| m.exp(bb(&p), bb(&v)))));

        let l = m.log(&p, &q).expect("fixtures stay inside the injectivity radius");
        $out.push(record("sphere", $n, "log", l.as_slice().to_vec(),
                         time_ns(|| m.log(bb(&p), bb(&q)))));

        let d = m.dist(&p, &q).expect("fixtures stay inside the injectivity radius");
        $out.push(record("sphere", $n, "dist", vec![d],
                         time_ns(|| m.dist(bb(&p), bb(&q)))));

        let t = m.transport(&p, &q, &v).expect("transport along a minimising geodesic");
        $out.push(record("sphere", $n, "transport", t.as_slice().to_vec(),
                         time_ns(|| m.transport(bb(&p), bb(&q), bb(&v)))));
    }};
}

macro_rules! spd_arm {
    ($n:literal, $case:expr, $out:expr) => {{
        let m = Spd::<$n>;
        // Fixtures are row-major; SMatrix::from_row_slice matches that.
        let p = SMatrix::<f64, $n, $n>::from_row_slice(&mat_of(&$case["p"]));
        let v = SMatrix::<f64, $n, $n>::from_row_slice(&mat_of(&$case["v"]));
        let q = SMatrix::<f64, $n, $n>::from_row_slice(&mat_of(&$case["q"]));

        // Values are emitted row-major so every harness agrees on layout.
        let row_major = |x: &SMatrix<f64, $n, $n>| {
            let mut o = Vec::with_capacity($n * $n);
            for i in 0..$n {
                for j in 0..$n {
                    o.push(x[(i, j)]);
                }
            }
            o
        };

        let e = m.exp(&p, &v);
        $out.push(record("spd", $n, "exp", row_major(&e), time_ns(|| m.exp(bb(&p), bb(&v)))));

        let l = m.log(&p, &q).expect("the SPD cone is complete, so log always succeeds");
        $out.push(record("spd", $n, "log", row_major(&l), time_ns(|| m.log(bb(&p), bb(&q)))));

        let d = m.dist(&p, &q).expect("the SPD cone is complete");
        $out.push(record("spd", $n, "dist", vec![d], time_ns(|| m.dist(bb(&p), bb(&q)))));
    }};
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("rust/ sits inside benchmarks/")
        .to_path_buf();
    let fixtures = root.join("fixtures/geometry_cases.json");
    let out_path = root.join("results/cartan_geometry.jsonl");

    let data: Value = serde_json::from_str(
        &fs::read_to_string(&fixtures)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", fixtures.display())),
    )
    .expect("fixtures must be valid JSON");

    let mut out = Vec::new();
    for case in data["cases"].as_array().expect("cases must be an array") {
        let kind = case["manifold"].as_str().expect("manifold must be a string");
        let dim = case["dim"].as_u64().expect("dim must be an integer") as usize;
        eprintln!("benchmarking {kind} dim={dim}");

        match (kind, dim) {
            ("sphere", 3) => sphere_arm!(3, case, out),
            ("sphere", 10) => sphere_arm!(10, case, out),
            ("sphere", 50) => sphere_arm!(50, case, out),
            ("spd", 3) => spd_arm!(3, case, out),
            ("spd", 6) => spd_arm!(6, case, out),
            ("spd", 10) => spd_arm!(10, case, out),
            _ => eprintln!(
                "  no const-generic arm for {kind} dim={dim}; add one to crosslang_main.rs"
            ),
        }
    }

    fs::create_dir_all(out_path.parent().expect("results dir has a parent"))
        .expect("cannot create results directory");
    let body: String = out
        .iter()
        .map(|r| format!("{r}\n"))
        .collect();
    fs::write(&out_path, body).expect("cannot write results");

    eprintln!("wrote {} records to {}", out.len(), out_path.display());
}
