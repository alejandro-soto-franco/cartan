//! Native Rust timing binary for cartan benchmarks.
//!
//! Outputs JSON lines with median and IQR timings for each
//! (manifold, operation, dimension) combination.
//!
//! ```bash
//! cargo run --release -- --all
//! cargo run --release -- --manifold sphere --dims 2,3,5,10
//! ```

use std::time::Instant;

use clap::Parser;
use rand::SeedableRng;
use rand::rngs::StdRng;

use cartan_core::{Manifold, ParallelTransport, Real};
use cartan_manifolds::{Spd, Sphere};
use cartan_stochastic::{
    horizontal_velocity, random_frame_at, stochastic_development, stratonovich_step, wishart_step,
};

const SEED: u64 = 42;
const WARMUP: usize = 5;
const REPS: usize = 200;

#[derive(Parser)]
struct Args {
    /// Run all manifolds and dimensions.
    #[arg(long)]
    all: bool,

    /// Manifold to benchmark (sphere, euclidean).
    #[arg(long, default_value = "sphere")]
    manifold: String,

    /// Comma-separated dimensions.
    #[arg(long, default_value = "2,3,5,10,25,50,100,250,500,1000")]
    dims: String,
}

fn parse_dims(s: &str) -> Vec<usize> {
    s.split(',').filter_map(|d| d.trim().parse().ok()).collect()
}

macro_rules! bench_sphere {
    ($n:expr, $dims_filter:expr) => {{
        const N: usize = $n;
        if $dims_filter.contains(&(N - 1)) {
            let manifold = Sphere::<N>;
            let mut rng = StdRng::seed_from_u64(SEED);

            let p = manifold.random_point(&mut rng);
            let q = manifold.random_point(&mut rng);
            let v = manifold.random_tangent(&p, &mut rng);

            // exp
            let times = time_op(|| {
                manifold.exp(&p, &v);
            });
            emit("sphere", "exp", N - 1, &times);

            // log
            let times = time_op(|| {
                let _ = manifold.log(&p, &q);
            });
            emit("sphere", "log", N - 1, &times);

            // dist
            let times = time_op(|| {
                let _ = manifold.dist(&p, &q);
            });
            emit("sphere", "dist", N - 1, &times);

            // parallel_transport
            let times = time_op(|| {
                let _ = manifold.transport(&p, &q, &v);
            });
            emit("sphere", "parallel_transport", N - 1, &times);
        }
    }};
}

fn time_op<F: FnMut()>(mut f: F) -> Vec<u128> {
    for _ in 0..WARMUP {
        f();
    }
    let mut times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t0 = Instant::now();
        f();
        let elapsed = t0.elapsed().as_nanos();
        times.push(elapsed);
    }
    times
}

fn emit(manifold: &str, op: &str, dim: usize, times: &[u128]) {
    let mut sorted = times.to_vec();
    sorted.sort();
    let n = sorted.len();
    let median = sorted[n / 2];
    let q1 = sorted[n / 4];
    let q3 = sorted[3 * n / 4];
    println!(
        "{}",
        serde_json::json!({
            "manifold": manifold,
            "op": op,
            "dim": dim,
            "median_ns": median,
            "q1_ns": q1,
            "q3_ns": q3,
        })
    );
}

fn bench_spheres(dims: &[usize]) {
    // Const-generic dimensions must be known at compile time.
    // We enumerate the supported sizes explicitly.
    bench_sphere!(3, dims); // S^2
    bench_sphere!(4, dims); // S^3
    bench_sphere!(6, dims); // S^5
    bench_sphere!(11, dims); // S^10
    bench_sphere!(26, dims); // S^25
    bench_sphere!(51, dims); // S^50
    bench_sphere!(101, dims); // S^100
}

// ─── Stochastic benchmarks (L1) ─────────────────────────────────────────────

macro_rules! bench_sphere_stochastic {
    ($n:expr, $dims_filter:expr) => {{
        const N: usize = $n;
        if $dims_filter.contains(&(N - 1)) {
            let manifold = Sphere::<N>;
            let mut rng = StdRng::seed_from_u64(SEED);
            let p = manifold.random_point(&mut rng);
            let frame = random_frame_at(&manifold, &p, &mut rng).unwrap();
            let dw: Vec<Real> = (0..(N - 1)).map(|i| (i as Real) * 0.01).collect();

            // horizontal_velocity: linear combination of basis vectors.
            let times = time_op(|| {
                let _ = horizontal_velocity(&frame, &dw);
            });
            emit("sphere", "horizontal_velocity", N - 1, &times);

            // stratonovich_step: one Euler step on O(M).
            let times = time_op(|| {
                let _ = stratonovich_step(&manifold, &p, &frame, &dw, 0.01, 1e-10);
            });
            emit("sphere", "stratonovich_step", N - 1, &times);

            // stochastic_development: 32 steps end-to-end.
            let times = time_op(|| {
                let mut r = StdRng::seed_from_u64(SEED);
                let f = random_frame_at(&manifold, &p, &mut r).unwrap();
                let _ = stochastic_development(&manifold, &p, f, 32, 0.01, &mut r, 1e-10);
            });
            emit("sphere", "stochastic_development_32", N - 1, &times);
        }
    }};
}

fn bench_sphere_stochastic_all(dims: &[usize]) {
    bench_sphere_stochastic!(3, dims);
    bench_sphere_stochastic!(4, dims);
    bench_sphere_stochastic!(6, dims);
    bench_sphere_stochastic!(11, dims);
    bench_sphere_stochastic!(26, dims);
    bench_sphere_stochastic!(51, dims);
}

macro_rules! bench_spd_wishart {
    ($n:expr) => {{
        const N: usize = $n;
        let mut rng = StdRng::seed_from_u64(SEED);
        let m = Spd::<N>;
        let p = m.random_point(&mut rng);
        let times = time_op(|| {
            let _ = wishart_step(&p, N as Real, 0.01, &mut rng);
        });
        emit("spd", "wishart_step", N, &times);

        // Stochastic development on SPD with the affine-invariant metric:
        // exercises Lyapunov-like frame transport paths.
        let frame = random_frame_at(&m, &p, &mut rng).unwrap();
        let dw = [0.01_f64; $n * ($n + 1) / 2];
        let times = time_op(|| {
            let _ = stratonovich_step(&m, &p, &frame, &dw, 0.01, 1e-8);
        });
        emit("spd", "stratonovich_step", N, &times);
    }};
}

fn bench_spd_all() {
    bench_spd_wishart!(2);
    bench_spd_wishart!(3);
    bench_spd_wishart!(5);
}

fn main() {
    let args = Args::parse();
    let dims = parse_dims(&args.dims);

    if args.all || args.manifold == "sphere" {
        bench_spheres(&dims);
        bench_sphere_stochastic_all(&dims);
    }
    if args.all || args.manifold == "spd" || args.manifold == "stochastic" {
        bench_spd_all();
    }
}
