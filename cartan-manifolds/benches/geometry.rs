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
use cartan_manifolds::{
    Corr, Grassmann, Spd, SpdBuresWasserstein, SpecialEuclidean, SpecialOrthogonal, Sphere,
};
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

/// The in-place forms against the value-returning ones.
///
/// The buffer is allocated once outside the loop, which is the pattern the
/// `_into` API exists for. Anything that allocates per iteration would be
/// measuring the allocator instead.
fn into_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("sphere_into");

    macro_rules! bench_dim {
        ($n:literal) => {{
            let m = Sphere::<$n>;
            let mut rng = StdRng::seed_from_u64(SEED);
            let p = m.random_point(&mut rng);
            let q = m.random_point(&mut rng);
            let v = m.random_tangent(&p, &mut rng);
            let mut out = SVector::<f64, $n>::zeros();

            group.bench_with_input(BenchmarkId::new("exp_into", $n), &$n, |b, _| {
                b.iter(|| m.exp_into(black_box(&p), black_box(&v), &mut out))
            });
            group.bench_with_input(BenchmarkId::new("log_into", $n), &$n, |b, _| {
                b.iter(|| m.log_into(black_box(&p), black_box(&q), &mut out))
            });
            group.bench_with_input(BenchmarkId::new("transport_into", $n), &$n, |b, _| {
                b.iter(|| m.transport_into(black_box(&p), black_box(&q), black_box(&v), &mut out))
            });
        }};
    }

    bench_dim!(3);
    bench_dim!(10);
    bench_dim!(50);

    group.finish();
}

/// Manifolds that carry quantitative-finance workloads, at sizes those
/// workloads actually use.
///
/// `Corr` is the correlation matrix of a multi-asset model. `Grassmann` is a
/// factor subspace, so `Grassmann<20, 5>` is five factors over twenty assets.
/// `SpdBuresWasserstein` is the optimal-transport geometry between Gaussians,
/// which is a different metric on the same cone as `Spd` and answers a
/// different question about covariance.
fn quant_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("quant");

    macro_rules! corr_dim {
        ($n:literal) => {{
            let m = Corr::<$n>;
            let mut rng = StdRng::seed_from_u64(SEED);
            let p = m.random_point(&mut rng);
            let q = m.random_point(&mut rng);
            let v = m.random_tangent(&p, &mut rng);

            group.bench_with_input(BenchmarkId::new("corr_exp", $n), &$n, |b, _| {
                b.iter(|| m.exp(black_box(&p), black_box(&v)))
            });
            group.bench_with_input(BenchmarkId::new("corr_log", $n), &$n, |b, _| {
                b.iter(|| m.log(black_box(&p), black_box(&q)))
            });
            group.bench_with_input(BenchmarkId::new("corr_dist", $n), &$n, |b, _| {
                b.iter(|| m.dist(black_box(&p), black_box(&q)))
            });
        }};
    }

    corr_dim!(5);
    corr_dim!(10);
    corr_dim!(20);

    macro_rules! bw_dim {
        ($n:literal) => {{
            let m = SpdBuresWasserstein::<$n>;
            let mut rng = StdRng::seed_from_u64(SEED);
            let p = m.random_point(&mut rng);
            let q = m.random_point(&mut rng);

            group.bench_with_input(BenchmarkId::new("bw_dist", $n), &$n, |b, _| {
                b.iter(|| m.dist(black_box(&p), black_box(&q)))
            });
            group.bench_with_input(BenchmarkId::new("bw_log", $n), &$n, |b, _| {
                b.iter(|| m.log(black_box(&p), black_box(&q)))
            });
        }};
    }

    bw_dim!(3);
    bw_dim!(10);

    macro_rules! grass {
        ($n:literal, $k:literal, $label:literal) => {{
            let m = Grassmann::<$n, $k>;
            let mut rng = StdRng::seed_from_u64(SEED);
            let p = m.random_point(&mut rng);
            let q = m.random_point(&mut rng);
            let v = m.random_tangent(&p, &mut rng);

            group.bench_function(concat!("grassmann_exp_", $label), |b| {
                b.iter(|| m.exp(black_box(&p), black_box(&v)))
            });
            group.bench_function(concat!("grassmann_log_", $label), |b| {
                b.iter(|| m.log(black_box(&p), black_box(&q)))
            });
            group.bench_function(concat!("grassmann_dist_", $label), |b| {
                b.iter(|| m.dist(black_box(&p), black_box(&q)))
            });
        }};
    }

    grass!(10, 3, "10_3");
    grass!(20, 5, "20_5");

    // Robotics: SO(3) attitude, SE(3) pose. SO(10) is there to show how the
    // inner product scales once it is no longer a matrix product.
    macro_rules! so_dim {
        ($n:literal) => {{
            let m = SpecialOrthogonal::<$n>;
            let mut rng = StdRng::seed_from_u64(SEED);
            let p = m.random_point(&mut rng);
            let v = m.random_tangent(&p, &mut rng);
            let q = m.exp(&p, &v);

            group.bench_with_input(BenchmarkId::new("so_exp", $n), &$n, |b, _| {
                b.iter(|| m.exp(black_box(&p), black_box(&v)))
            });
            group.bench_with_input(BenchmarkId::new("so_log", $n), &$n, |b, _| {
                b.iter(|| m.log(black_box(&p), black_box(&q)))
            });
            group.bench_with_input(BenchmarkId::new("so_dist", $n), &$n, |b, _| {
                b.iter(|| m.dist(black_box(&p), black_box(&q)))
            });
        }};
    }

    so_dim!(3);
    so_dim!(10);

    {
        let m = SpecialEuclidean::<3> { weight: 1.0 };
        let mut rng = StdRng::seed_from_u64(SEED);
        let p = m.random_point(&mut rng);
        let v = m.random_tangent(&p, &mut rng);
        let q = m.exp(&p, &v);

        group.bench_function("se3_exp", |b| {
            b.iter(|| m.exp(black_box(&p), black_box(&v)))
        });
        group.bench_function("se3_log", |b| {
            b.iter(|| m.log(black_box(&p), black_box(&q)))
        });
        group.bench_function("se3_dist", |b| {
            b.iter(|| m.dist(black_box(&p), black_box(&q)))
        });
    }

    group.finish();
}

criterion_group!(benches, sphere_ops, spd_ops, round_trip, into_variants, quant_ops);
criterion_main!(benches);
