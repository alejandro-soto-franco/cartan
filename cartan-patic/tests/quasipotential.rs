//! Freidlin-Wentzell action against exact reference answers.

use cartan_patic::quasipotential::{MorseDecomposition, action, minimum_action_path};
use nalgebra::{DMatrix, DVector};

/// Quartic double well `U(x) = (x^2 - 1)^2 / 4`, minima at +/-1, saddle at 0.
fn well_u(x: f64) -> f64 {
    (x * x - 1.0).powi(2) / 4.0
}
fn well_drift(x: &[f64]) -> Vec<f64> {
    vec![-(x[0] * x[0] - 1.0) * x[0]]
}

#[test]
fn following_the_drift_costs_nothing() {
    // The downhill path solves phi' = b exactly, so its action is zero.
    let n = 400;
    let dt = 1.0 / (n - 1) as f64;
    let mut path = vec![vec![0.9_f64]];
    for _ in 1..n {
        let x = path.last().unwrap().clone();
        let d = well_drift(&x)[0];
        path.push(vec![x[0] + dt * d]);
    }
    let s = action(&path, &well_drift, dt);
    assert!(s < 1e-6, "downhill action {s:e} should vanish");
}

/// Under a gradient drift the uphill quasipotential is `2 (U_1 - U_0)`.
#[test]
fn the_gradient_quasipotential_is_twice_the_barrier() {
    let (_, s) = minimum_action_path(&[1.0], &[0.0], &well_drift, 60, 4000, 1e-2);
    let expect = 2.0 * (well_u(0.0) - well_u(1.0));
    assert!(
        (s - expect).abs() < 0.02 * expect,
        "action {s:e} against 2 * barrier {expect:e}"
    );
}

/// The same statement in two dimensions, on a separable gradient field.
#[test]
fn the_gradient_result_holds_in_two_dimensions() {
    let drift = |x: &[f64]| vec![-(x[0] * x[0] - 1.0) * x[0], -2.0 * x[1]];
    let u = |x: &[f64]| (x[0] * x[0] - 1.0).powi(2) / 4.0 + x[1] * x[1];
    let (_, s) = minimum_action_path(&[1.0, 0.0], &[0.0, 0.0], &drift, 60, 4000, 1e-2);
    let expect = 2.0 * (u(&[0.0, 0.0]) - u(&[1.0, 0.0]));
    assert!(
        (s - expect).abs() < 0.03 * expect,
        "action {s:e} against {expect:e}"
    );
}

/// The case that separates a real implementation from a gradient-only one.
///
/// For `b = A x` with `A` stable and not symmetric, the stationary density is
/// Gaussian with covariance `Sigma` solving `A Sigma + Sigma A^T + 2 I = 0`,
/// and the quasipotential from the origin is `x^T Sigma^-1 x`.
#[test]
fn the_linear_non_gradient_quasipotential_matches_the_lyapunov_solution() {
    let a = DMatrix::<f64>::from_row_slice(2, 2, &[-1.0, 2.0, -0.5, -1.0]);
    assert!(
        (a[(0, 1)] - a[(1, 0)]).abs() > 1e-9_f64,
        "the fixture must be non-symmetric or it proves nothing"
    );
    let drift = |x: &[f64]| {
        vec![
            a[(0, 0)] * x[0] + a[(0, 1)] * x[1],
            a[(1, 0)] * x[0] + a[(1, 1)] * x[1],
        ]
    };

    // Solve A S + S A^T + 2 I = 0 by vectorising: (I kron A + A kron I) s = -2 vec(I).
    let mut m = DMatrix::<f64>::zeros(4, 4);
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                m[(i * 2 + j, k * 2 + j)] += a[(i, k)];
                m[(i * 2 + j, i * 2 + k)] += a[(j, k)];
            }
        }
    }
    let rhs = DVector::<f64>::from_row_slice(&[-2.0, 0.0, 0.0, -2.0]);
    let s = m.lu().solve(&rhs).expect("Lyapunov system is solvable");
    let sigma = DMatrix::<f64>::from_row_slice(2, 2, &[s[0], s[1], s[2], s[3]]);
    let sinv = sigma
        .clone()
        .try_inverse()
        .expect("Sigma is positive definite");

    let target = [0.35_f64, -0.2];
    let x = DVector::<f64>::from_row_slice(&target);
    let expect = (x.transpose() * &sinv * &x)[(0, 0)];

    let (_, got) = minimum_action_path(&[0.0, 0.0], &target, &drift, 80, 8000, 5e-3);
    assert!(
        (got - expect).abs() < 0.08 * expect,
        "action {got:e} against Lyapunov reference {expect:e}"
    );
}

/// More path points can only lower the infimum's discrete approximation.
#[test]
fn refining_the_path_lowers_the_action() {
    let coarse = minimum_action_path(&[1.0], &[0.0], &well_drift, 15, 3000, 1e-2).1;
    let fine = minimum_action_path(&[1.0], &[0.0], &well_drift, 90, 6000, 1e-2).1;
    assert!(
        fine <= coarse + 1e-9,
        "refining raised the action: {coarse:e} -> {fine:e}"
    );
}

#[test]
fn the_double_well_has_two_attractors() {
    let xs: Vec<f64> = (0..81).map(|i| -2.0 + i as f64 * 0.05).collect();
    let d = MorseDecomposition::of_line(&xs, &well_drift);
    assert_eq!(d.attractors.len(), 2, "a double well has two attractors");
}

#[test]
fn a_single_well_has_one_attractor() {
    let drift = |x: &[f64]| vec![-x[0]];
    let xs: Vec<f64> = (0..81).map(|i| -2.0 + i as f64 * 0.05).collect();
    let d = MorseDecomposition::of_line(&xs, &drift);
    assert_eq!(d.attractors.len(), 1);
}

/// A cycle is one recurrent component and is its own attractor, since nothing
/// leaves it.
#[test]
fn a_cycle_is_a_single_recurrent_component() {
    let edges = [(0usize, 1usize), (1, 2), (2, 0)];
    let d = MorseDecomposition::of_graph(3, &edges);
    assert_eq!(d.components.len(), 1);
    assert_eq!(d.attractors.len(), 1);
    assert_eq!(d.components[0].len(), 3);
}

/// A saddle flowing into two wells condenses to three components, of which
/// the two wells are the attractors.
#[test]
fn a_saddle_between_two_wells_condenses_correctly() {
    // 0 is the saddle, 1 and 2 are wells with self-loops.
    let edges = [(0usize, 1usize), (0, 2), (1, 1), (2, 2)];
    let d = MorseDecomposition::of_graph(3, &edges);
    assert_eq!(d.components.len(), 3);
    assert_eq!(d.attractors.len(), 2);
}
