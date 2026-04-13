//! Bures-Wasserstein SPD: metric axioms, exp/log roundtrip, closed-form distance.

use cartan_core::{Manifold, Real, Retraction};
use cartan_manifolds::{bw_distance_sq, SpdBuresWasserstein};
use nalgebra::SMatrix;
use rand::SeedableRng;
use rand::rngs::SmallRng;
type StdRng = SmallRng;

type M2 = SpdBuresWasserstein<2>;
type M3 = SpdBuresWasserstein<3>;

fn identity<const N: usize>() -> SMatrix<Real, N, N> {
    SMatrix::<Real, N, N>::identity()
}

#[test]
fn inner_product_is_symmetric_and_positive_definite() {
    let m: M3 = SpdBuresWasserstein;
    let mut rng = StdRng::seed_from_u64(1);
    let p = m.random_point(&mut rng);
    let u = m.random_tangent(&p, &mut rng);
    let v = m.random_tangent(&p, &mut rng);
    let uv = m.inner(&p, &u, &v);
    let vu = m.inner(&p, &v, &u);
    assert!((uv - vu).abs() < 1e-8, "inner asymmetric: {uv} vs {vu}");
    let uu = m.inner(&p, &u, &u);
    assert!(uu > 0.0, "inner(u,u) = {uu} not positive");
}

#[test]
fn exp_log_roundtrip_at_identity() {
    let m: M2 = SpdBuresWasserstein;
    let p = identity::<2>();
    // Small symmetric tangent vector.
    let v = SMatrix::<Real, 2, 2>::new(0.1, 0.02, 0.02, -0.05);
    let q = m.exp(&p, &v);
    let v_recovered = m.log(&p, &q).unwrap();
    let diff = (v - v_recovered).norm();
    assert!(diff < 1e-8, "roundtrip error {diff}, v={v:?}, recovered={v_recovered:?}");
}

#[test]
fn exp_log_roundtrip_at_nontrivial_point() {
    let m: M2 = SpdBuresWasserstein;
    // Non-identity base point.
    let p = SMatrix::<Real, 2, 2>::new(2.0, 0.3, 0.3, 1.5);
    let v = SMatrix::<Real, 2, 2>::new(0.05, 0.01, 0.01, -0.02);
    let q = m.exp(&p, &v);
    let v_recovered = m.log(&p, &q).unwrap();
    let diff = (v - v_recovered).norm();
    assert!(diff < 1e-6, "roundtrip error {diff}");
}

#[test]
fn distance_matches_log_norm() {
    // d_{BW}(P, Q)^2 from the closed form should match ||Log_P(Q)||_P^2.
    let m: M2 = SpdBuresWasserstein;
    let p = SMatrix::<Real, 2, 2>::new(2.0, 0.3, 0.3, 1.5);
    let q = SMatrix::<Real, 2, 2>::new(1.2, -0.1, -0.1, 0.8);
    let closed = bw_distance_sq(&p, &q);
    let v = m.log(&p, &q).unwrap();
    let from_norm_sq = m.inner(&p, &v, &v);
    assert!(
        (closed - from_norm_sq).abs() < 1e-6,
        "closed={closed}, from_norm_sq={from_norm_sq}"
    );
}

#[test]
fn distance_vanishes_at_zero() {
    let m: M2 = SpdBuresWasserstein;
    let p = SMatrix::<Real, 2, 2>::new(1.7, 0.2, 0.2, 0.9);
    assert!(bw_distance_sq(&p, &p).abs() < 1e-10);
    let v_zero = m.zero_tangent(&p);
    let q = m.exp(&p, &v_zero);
    assert!((p - q).norm() < 1e-10);
}

#[test]
fn retraction_preserves_spd() {
    let m: M3 = SpdBuresWasserstein;
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..10 {
        let p = m.random_point(&mut rng);
        let v = m.random_tangent(&p, &mut rng) * 0.1;
        let q = m.retract(&p, &v);
        m.check_point(&q).unwrap();
    }
}
