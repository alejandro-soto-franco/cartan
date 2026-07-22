//! Criterion benchmarks for the core manifold operations.
//!
//! These guard against regressions and supply the confidence intervals the
//! cross-language report quotes. The hand-rolled harness in `benchmarks/rust`
//! stays separate: it emits JSON lines shaped for the cross-language
//! comparison, which criterion's format does not suit.
//!
//! ```text
//! cargo bench -p cartan-manifolds
//! ```

use cartan_core::{Manifold, ParallelTransport};
use cartan_manifolds::{Spd, Sphere};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra::{SMatrix, SVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::hint::black_box;

/// Fixed so a run is reproducible and two runs are comparable.
const SEED: u64 = 42;

/// Benchmarks exp, log, dist and transport on `Sphere<N>` for several N.
///
/// The sphere has closed-form everything, so this measures the cost of the
/// abstraction rather than of an iterative solve.
fn sphere_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("sphere");

    macro_rules! bench_dim {
        ($n:literal) => {{
            let m = Sphere::<$n>;
            let mut rng = StdRng::seed_from_u64(SEED);
            let p = m.random_point(&mut rng);
            let v = m.random_tangent(&p, &mut rng);
            let q = m.exp(&p, &v);

            group.bench_with_input(BenchmarkId::new("exp", $n), &$n, |b, _| {
                b.iter(|| m.exp(black_box(&p), black_box(&v)))
            });
            group.bench_with_input(BenchmarkId::new("log", $n), &$n, |b, _| {
                b.iter(|| m.log(black_box(&p), black_box(&q)))
            });
            group.bench_with_input(BenchmarkId::new("dist", $n), &$n, |b, _| {
                b.iter(|| m.dist(black_box(&p), black_box(&q)))
            });
            group.bench_with_input(BenchmarkId::new("transport", $n), &$n, |b, _| {
                b.iter(|| m.transport(black_box(&p), black_box(&q), black_box(&v)))
            });
        }};
    }

    bench_dim!(3);
    bench_dim!(10);
    bench_dim!(50);

    group.finish();
}

/// Benchmarks the SPD cone, where exp and log need an eigen decomposition.
///
/// Cost grows sharply with N, so this is the case where the abstraction is
/// irrelevant and the linear algebra dominates.
fn spd_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("spd");

    macro_rules! bench_dim {
        ($n:literal) => {{
            let m = Spd::<$n>;
            let mut rng = StdRng::seed_from_u64(SEED);
            let p = m.random_point(&mut rng);
            let v = m.random_tangent(&p, &mut rng);
            let q = m.exp(&p, &v);

            group.bench_with_input(BenchmarkId::new("exp", $n), &$n, |b, _| {
                b.iter(|| m.exp(black_box(&p), black_box(&v)))
            });
            group.bench_with_input(BenchmarkId::new("log", $n), &$n, |b, _| {
                b.iter(|| m.log(black_box(&p), black_box(&q)))
            });
            group.bench_with_input(BenchmarkId::new("dist", $n), &$n, |b, _| {
                b.iter(|| m.dist(black_box(&p), black_box(&q)))
            });
        }};
    }

    bench_dim!(3);
    bench_dim!(6);
    bench_dim!(10);

    group.finish();
}

/// A round trip through the tangent space, which is the composite operation
/// most downstream code actually performs.
fn round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("round_trip");

    let s = Sphere::<10>;
    let mut rng = StdRng::seed_from_u64(SEED);
    let p: SVector<f64, 10> = s.random_point(&mut rng);
    let q: SVector<f64, 10> = s.random_point(&mut rng);

    group.bench_function("sphere_10_log_then_exp", |b| {
        b.iter(|| {
            let v = s.log(black_box(&p), black_box(&q)).unwrap();
            s.exp(black_box(&p), &v)
        })
    });

    let spd = Spd::<6>;
    let a: SMatrix<f64, 6, 6> = spd.random_point(&mut rng);
    let bb: SMatrix<f64, 6, 6> = spd.random_point(&mut rng);

    group.bench_function("spd_6_log_then_exp", |b| {
        b.iter(|| {
            let v = spd.log(black_box(&a), black_box(&bb)).unwrap();
            spd.exp(black_box(&a), &v)
        })
    });

    group.finish();
}

criterion_group!(benches, sphere_ops, spd_ops, round_trip);
criterion_main!(benches);
