//! The active stress and the force it drives.
//!
//! A rank-2 stress built linearly from a rank-`m` tensor needs exactly `m - 2`
//! derivatives to contract the surplus indices, so
//!
//! ```text
//! sigma^active = zeta grad^(x)(m-2) . T_m
//! ```
//!
//! and the force `f = div sigma` therefore needs `m - 1`.
//!
//! `m` is the harmonic degree of the invariant tensor, which is what the code
//! reads from `InvariantBasis::degree`, not the symmetry order. They agree for
//! the dihedral family, where a `p`-atic has `m = p`, and part elsewhere:
//! `Dicyclic<1>` has `m = 3`, `Cyclic<8>` has `m = 5` against an image order
//! of 4, and the polyhedral groups have no symmetry order while having
//! `m = 3, 4, 6`. Stating the counting in `m` covers every symmetry the crate
//! supports. The passive theory
//! uses the same counting: Krommydas, Carenza and Giomi write the reactive
//! stress for general `p` with the same `grad^(x)(p-2)` prefactor. The linear
//! coupling familiar from nematics does not generalise: Giomi, Toner and
//! Sarkar state that `p = 1, 2` are the only orders at which the strain rate
//! couples linearly to the order parameter at leading order in derivatives.
//!
//! **This active form is a construction, not a citation.** The literature
//! keeps `sigma^active[T]` unspecified for `p > 2`, saying only that it
//! comprises the symmetry-allowed contractions of `T`. What is cited is the
//! index counting; what is chosen is applying it to `Q_p` rather than to some
//! other allowed contraction.
//!
//! ## Why only degree 2 runs here
//!
//! Piecewise-linear vertex data supplies one derivative. The force needs
//! `m - 1`, so `m = 2` is exactly what this element space can express, and
//! higher degrees return [`KaticError::InsufficientRegularity`] rather than a
//! silently wrong number. Reaching `m = 3` needs a higher-order space or a
//! gradient-recovery step.

use nalgebra::DVector;

use crate::complex3::Complex3;
use crate::energy::{Energy, State};
use crate::error::KaticError;
use crate::geometry::Geometry3;

/// Assemble the active force as a one-cochain.
///
/// The stress is piecewise linear from the vertex order parameters, so its
/// divergence is constant on each tetrahedron, and the cochain entry is the
/// exact integral of that constant field against the Whitney one-form,
/// `integral w_ij = (V/4)(grad lambda_j - grad lambda_i)`.
pub fn active_force(
    c: &Complex3,
    g: &Geometry3,
    energy: &Energy,
    state: &State,
    zeta: f64,
) -> Result<DVector<f64>, KaticError> {
    let degree = energy.basis().degree();
    if degree != 2 {
        return Err(KaticError::InsufficientRegularity {
            degree,
            needed: degree.saturating_sub(1),
            available: 1,
        });
    }

    // The order parameter as a symmetric traceless matrix at every vertex.
    let q: Vec<[[f64; 3]; 3]> = (0..state.n_vertices())
        .map(|v| {
            let t = energy.tensor(&state.rotors[v], state.amps(v));
            energy
                .basis()
                .as_matrix3(&t)
                .expect("degree 2 checked above")
        })
        .collect();

    let mut f = DVector::zeros(c.n_edges());
    for tet in c.tets() {
        let d = g.tet_data(tet);
        // div sigma, constant on the tetrahedron:
        // (div Q)_i = sum_a sum_j Q^a_ij (grad lambda_a)_j
        let mut div = [0.0_f64; 3];
        for (a, &vtx) in tet.iter().enumerate() {
            for (i, dv) in div.iter_mut().enumerate() {
                for (j, &gj) in d.grads[a].iter().enumerate() {
                    *dv += q[vtx][i][j] * gj;
                }
            }
        }
        for dv in &mut div {
            *dv *= zeta;
        }
        // Test against each Whitney one-form of the tetrahedron.
        const LE: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
        for &[a, b] in LE.iter() {
            let w = [
                d.volume * (d.grads[b][0] - d.grads[a][0]) / 4.0,
                d.volume * (d.grads[b][1] - d.grads[a][1]) / 4.0,
                d.volume * (d.grads[b][2] - d.grads[a][2]) / 4.0,
            ];
            let e = c.edge_of(&[tet[a], tet[b]]);
            f[e] += div[0] * w[0] + div[1] * w[1] + div[2] * w[2];
        }
    }
    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::{AxialApolar, BinaryTetrahedral, SymmetryGroup};
    use cartan_core::rotor::Rotor3;
    use nalgebra::DMatrix;

    fn energy_for<H: SymmetryGroup>() -> Energy {
        let n = H::N_AMPLITUDES;
        Energy::new::<H>(
            DMatrix::from_diagonal_element(n, n, -1.0),
            vec![0.0; n * n * n],
            DMatrix::identity(n, n),
            0.5,
        )
        .expect("coercive")
    }

    fn rig(n: usize) -> (Complex3, Geometry3) {
        (Complex3::cube_grid(n), Geometry3::cube_grid(n))
    }

    /// A constant order parameter has zero divergence, so the force must
    /// vanish exactly. This is the test that catches a wrong derivative order:
    /// any term without a derivative survives a uniform field.
    #[test]
    fn a_uniform_field_drives_no_force() {
        let (c, g) = rig(2);
        let e = energy_for::<AxialApolar>();
        let state = State::uniform(
            c.n_vertices(),
            Rotor3 {
                w: 0.6,
                x: 0.8,
                y: 0.0,
                z: 0.0,
            },
            &[0.7],
        );
        let f = active_force(&c, &g, &e, &state, 1.3).expect("degree 2 runs");
        assert!(f.amax() < 1e-13, "uniform field drove force {:e}", f.amax());
    }

    /// The order parameter at degree 2 is symmetric and traceless, which is
    /// the Q-tensor structure the whole nematic literature assumes.
    #[test]
    fn the_degree_two_order_parameter_is_a_q_tensor() {
        let e = energy_for::<AxialApolar>();
        for r in [
            Rotor3::IDENTITY,
            Rotor3 {
                w: 0.6,
                x: 0.8,
                y: 0.0,
                z: 0.0,
            },
            Rotor3 {
                w: 0.5,
                x: 0.5,
                y: 0.5,
                z: 0.5,
            },
        ] {
            let t = e.tensor(&r, &[0.9]);
            let q = e.basis().as_matrix3(&t).expect("degree 2");
            let trace = q[0][0] + q[1][1] + q[2][2];
            assert!(trace.abs() < 1e-13, "trace {trace:e}");
            for (i, row) in q.iter().enumerate() {
                for (j, &qij) in row.iter().enumerate() {
                    assert!((qij - q[j][i]).abs() < 1e-13, "asymmetry at {i},{j}");
                }
            }
        }
    }

    /// A spatially varying field drives a force, and it is the divergence of
    /// the stress rather than an artefact of the assembly: scaling the
    /// activity scales the force exactly.
    #[test]
    fn a_varying_field_drives_a_force_linear_in_activity() {
        let (c, g) = rig(2);
        let e = energy_for::<AxialApolar>();
        let p = g.positions();
        let mut state = State::uniform(c.n_vertices(), Rotor3::IDENTITY, &[0.8]);
        for (v, pos) in p.iter().enumerate() {
            let theta = 1.7 * pos[0];
            let (s, cth) = (theta / 2.0).sin_cos();
            state.rotors[v] = Rotor3 {
                w: cth,
                x: 0.0,
                y: s,
                z: 0.0,
            };
        }
        let f1 = active_force(&c, &g, &e, &state, 1.0).expect("degree 2 runs");
        let f2 = active_force(&c, &g, &e, &state, 2.5).expect("degree 2 runs");
        assert!(f1.amax() > 1e-6, "a varying field must drive a force");
        assert!(
            (&f2 - &f1 * 2.5).amax() < 1e-12 * f1.amax(),
            "force is not linear in the activity"
        );
    }

    /// Higher symmetry orders need more derivatives than piecewise-linear
    /// elements supply, and say so rather than returning a wrong number.
    #[test]
    fn higher_orders_report_insufficient_regularity() {
        let (c, g) = rig(1);
        let e = energy_for::<BinaryTetrahedral>();
        let state = State::uniform(c.n_vertices(), Rotor3::IDENTITY, &[0.5]);
        match active_force(&c, &g, &e, &state, 1.0) {
            Err(KaticError::InsufficientRegularity {
                degree,
                needed,
                available,
            }) => {
                assert_eq!((degree, needed, available), (3, 2, 1));
            }
            other => panic!("expected InsufficientRegularity, got {other:?}"),
        }
    }

    /// The whole chain: an active force drives an incompressible flow.
    #[test]
    fn the_active_force_drives_an_incompressible_flow() {
        use crate::stokes::Stokes;
        let (c, g) = rig(2);
        let e = energy_for::<AxialApolar>();
        let p = g.positions();
        let mut state = State::uniform(c.n_vertices(), Rotor3::IDENTITY, &[0.8]);
        for (v, pos) in p.iter().enumerate() {
            let theta = 2.1 * pos[1];
            let (s, cth) = (theta / 2.0).sin_cos();
            state.rotors[v] = Rotor3 {
                w: cth,
                x: s,
                y: 0.0,
                z: 0.0,
            };
        }
        let f = active_force(&c, &g, &e, &state, 1.0).expect("degree 2 runs");
        let stokes = Stokes::assemble(&c, &g, 1.0);
        let (u, _) = stokes.solve(&f);
        assert!(stokes.velocity_norm(&u) > 1e-9, "activity must drive flow");
        assert!(
            stokes.divergence(&u).amax() < 1e-8 * f.norm(),
            "active flow is not divergence free: {:e}",
            stokes.divergence(&u).amax()
        );
    }
}
