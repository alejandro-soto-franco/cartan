//! Criterion benchmarks for the optimisers.
//!
//! The Frechet mean is the workload that reads the metric hardest: every
//! iteration takes one logarithm per point and one norm, so it is where a
//! cheaper `inner` on `Spd` shows up.
//!
//! ```text
//! cargo bench -p cartan-optim
//! ```

use cartan_core::Manifold;
use cartan_manifolds::{Spd, Sphere};
use cartan_optim::{FrechetConfig, frechet_mean};
use criterion::{Criterion, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::hint::black_box;

/// Fixed so a run is reproducible and two runs are comparable.
const SEED: u64 = 42;

fn frechet(c: &mut Criterion) {
    let mut group = c.benchmark_group("frechet");
    let config = FrechetConfig::default();

    let s = Sphere::<10>;
    let mut rng = StdRng::seed_from_u64(SEED);
    // Points in a small cap, so the mean is well defined and the flow
    // converges rather than wandering near the cut locus.
    let centre = s.random_point(&mut rng);
    let sphere_points: Vec<_> = (0..64)
        .map(|_| {
            let t = s.random_tangent(&centre, &mut rng);
            s.exp(&centre, &(t * 0.15))
        })
        .collect();

    group.bench_function("sphere_10_n64", |b| {
        b.iter(|| frechet_mean(&s, black_box(&sphere_points), None, &config))
    });

    let spd = Spd::<6>;
    let spd_centre = spd.random_point(&mut rng);
    let spd_points: Vec<_> = (0..32)
        .map(|_| {
            let t = spd.random_tangent(&spd_centre, &mut rng);
            spd.exp(&spd_centre, &(t * 0.15))
        })
        .collect();

    group.bench_function("spd_6_n32", |b| {
        b.iter(|| frechet_mean(&spd, black_box(&spd_points), None, &config))
    });

    group.finish();
}

criterion_group!(benches, frechet);
criterion_main!(benches);
