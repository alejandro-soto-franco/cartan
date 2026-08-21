//! Criterion benchmarks for the geodesic and Jacobi field routines.
//!
//! These run on `Sphere<10>`, where every manifold call is closed form and the
//! loop overhead shows, and on `Spd<6>`, where each call needs an
//! eigendecomposition and the count of calls per step is what matters.
//!
//! ```text
//! cargo bench -p cartan-geo
//! ```

use cartan_core::Manifold;
use cartan_geo::{Geodesic, integrate_jacobi};
use cartan_manifolds::{Spd, Sphere};
use criterion::{Criterion, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::hint::black_box;

/// Fixed so a run is reproducible and two runs are comparable.
const SEED: u64 = 42;

/// Sampling a geodesic, which is one exponential map per sample.
fn geodesic_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("geodesic");

    let s = Sphere::<10>;
    let mut rng = StdRng::seed_from_u64(SEED);
    let p = s.random_point(&mut rng);
    let v = s.random_tangent(&p, &mut rng);
    let g = Geodesic::new(&s, p, v);

    group.bench_function("sphere_10_sample_64", |b| {
        b.iter(|| g.sample(black_box(64)))
    });

    let spd = Spd::<6>;
    let pp = spd.random_point(&mut rng);
    let vv = spd.random_tangent(&pp, &mut rng);
    let gg = Geodesic::new(&spd, pp, vv);

    group.bench_function("spd_6_sample_16", |b| b.iter(|| gg.sample(black_box(16))));

    group.finish();
}

/// Jacobi field integration, the composite the crate exists for.
///
/// Each RK4 step evaluates the curvature tensor four times, transports twice,
/// and needs the geodesic at the step endpoint.
fn jacobi(c: &mut Criterion) {
    let mut group = c.benchmark_group("jacobi");

    let s = Sphere::<10>;
    let mut rng = StdRng::seed_from_u64(SEED);
    let p = s.random_point(&mut rng);
    let v = s.random_tangent(&p, &mut rng);
    let j0 = s.random_tangent(&p, &mut rng);
    let j0_dot = s.random_tangent(&p, &mut rng);
    let g = Geodesic::new(&s, p, v);

    group.bench_function("sphere_10_steps_32", |b| {
        b.iter(|| integrate_jacobi(&g, j0, j0_dot, black_box(32)))
    });

    let spd = Spd::<6>;
    let pp = spd.random_point(&mut rng);
    let vv = spd.random_tangent(&pp, &mut rng);
    let jj0 = spd.random_tangent(&pp, &mut rng);
    let jj0_dot = spd.random_tangent(&pp, &mut rng);
    let gg = Geodesic::new(&spd, pp, vv);

    group.bench_function("spd_6_steps_16", |b| {
        b.iter(|| integrate_jacobi(&gg, jj0, jj0_dot, black_box(16)))
    });

    group.finish();
}

criterion_group!(benches, geodesic_sampling, jacobi);
criterion_main!(benches);
