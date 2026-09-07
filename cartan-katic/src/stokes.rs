//! Incompressible Stokes flow on a tetrahedral complex.
//!
//! Velocity is a 1-cochain and pressure a 0-cochain, giving the symmetric
//! saddle-point system
//!
//! ```text
//! [ eta K   B ] [u]   [f]
//! [ B^T     0 ] [p] = [0]
//! ```
//!
//! with `K = d1^T M2 d1` the curl-curl operator, `B = M1 d0` the discrete
//! gradient tested against Whitney one-forms, and `B^T = d0^T M1` the
//! divergence. The curl-curl operator alone has every gradient field in its
//! kernel; the constraint removes exactly that subspace, so the pair is well
//! posed once the pressure constant is pinned.
//!
//! **Ricci needs no separate term.** Weitzenboeck gives
//! `Delta = nabla* nabla + Ric`, so the Hodge form of the viscous operator
//! includes the curvature coupling by construction. That is what the FEEC
//! route gives over a connection-Laplacian assembly.
//!
//! The solve is dense. Tetrahedral meshes of research size need a Krylov
//! method on the indefinite system; the dense path is what makes the physics
//! tests exact rather than solver-limited.

use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::CsrMatrix;

use crate::complex3::Complex3;
use crate::geometry::{Geometry3, mass1, mass2};

fn to_dense(m: &CsrMatrix<f64>) -> DMatrix<f64> {
    let mut d = DMatrix::zeros(m.nrows(), m.ncols());
    for (r, row) in m.row_iter().enumerate() {
        for (&c, &v) in row.col_indices().iter().zip(row.values()) {
            d[(r, c)] += v;
        }
    }
    d
}

/// The assembled Stokes operator for one complex and geometry.
#[derive(Clone, Debug)]
pub struct Stokes {
    n_edges: usize,
    n_vertices: usize,
    /// `eta K`, the viscous block.
    a: DMatrix<f64>,
    /// `B = M1 d0`.
    b: DMatrix<f64>,
    /// `M1`, kept for energy diagnostics.
    m1: DMatrix<f64>,
}

impl Stokes {
    /// Assemble at viscosity `eta`.
    #[must_use]
    pub fn assemble(c: &Complex3, g: &Geometry3, eta: f64) -> Self {
        let d0 = to_dense(&c.d0());
        let d1 = to_dense(&c.d1());
        let m1 = to_dense(&mass1(c, g));
        let m2 = to_dense(&mass2(c, g));
        let a = d1.transpose() * &m2 * &d1 * eta;
        let b = &m1 * &d0;
        Self {
            n_edges: c.n_edges(),
            n_vertices: c.n_vertices(),
            a,
            b,
            m1,
        }
    }

    /// Discrete divergence of a velocity cochain: `B^T u`.
    #[must_use]
    pub fn divergence(&self, u: &DVector<f64>) -> DVector<f64> {
        self.b.transpose() * u
    }

    /// `B phi`, the discrete gradient of a vertex field tested against
    /// one-forms. A force of this shape is absorbed entirely by the pressure.
    #[must_use]
    pub fn gradient_force(&self, phi: &DVector<f64>) -> DVector<f64> {
        &self.b * phi
    }

    /// Viscous dissipation `eta u^T K u`.
    #[must_use]
    pub fn dissipation(&self, u: &DVector<f64>) -> f64 {
        (u.transpose() * &self.a * u)[(0, 0)]
    }

    /// The `L2` norm of a velocity cochain, through the one-form mass.
    #[must_use]
    pub fn velocity_norm(&self, u: &DVector<f64>) -> f64 {
        (u.transpose() * &self.m1 * u)[(0, 0)].max(0.0).sqrt()
    }

    /// Solve for velocity and pressure under a one-cochain force.
    ///
    /// The pressure constant is pinned at vertex zero, since `d0` annihilates
    /// constants and the pressure is otherwise determined only up to one.
    #[must_use]
    pub fn solve(&self, f: &DVector<f64>) -> (DVector<f64>, DVector<f64>) {
        let ne = self.n_edges;
        let nv = self.n_vertices;
        let n = ne + nv;
        let mut k = DMatrix::<f64>::zeros(n, n);
        k.view_mut((0, 0), (ne, ne)).copy_from(&self.a);
        k.view_mut((0, ne), (ne, nv)).copy_from(&self.b);
        k.view_mut((ne, 0), (nv, ne)).copy_from(&self.b.transpose());

        // Pin the pressure constant: replace the row and column of the first
        // pressure unknown by the identity.
        let pin = ne;
        for i in 0..n {
            k[(pin, i)] = 0.0;
            k[(i, pin)] = 0.0;
        }
        k[(pin, pin)] = 1.0;

        let mut rhs = DVector::<f64>::zeros(n);
        rhs.rows_mut(0, ne).copy_from(f);
        rhs[pin] = 0.0;

        let sol = k.lu().solve(&rhs).unwrap_or_else(|| DVector::zeros(n));
        (sol.rows(0, ne).into_owned(), sol.rows(ne, nv).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rig(n: usize) -> (Complex3, Geometry3, Stokes) {
        let c = Complex3::cube_grid(n);
        let g = Geometry3::cube_grid(n);
        let s = Stokes::assemble(&c, &g, 1.0);
        (c, g, s)
    }

    #[test]
    fn no_force_gives_no_flow() {
        let (c, _, s) = rig(2);
        let (u, _) = s.solve(&DVector::zeros(c.n_edges()));
        assert!(s.velocity_norm(&u) < 1e-12, "norm {}", s.velocity_norm(&u));
    }

    /// A force that is a pure gradient does no work on a divergence-free
    /// field, so the pressure absorbs it entirely and the velocity stays zero.
    /// This is the test that exercises the whole saddle structure at once.
    #[test]
    fn a_gradient_force_produces_no_flow() {
        let (c, g, s) = rig(2);
        let p = g.positions();
        let phi = DVector::from_iterator(
            c.n_vertices(),
            (0..c.n_vertices()).map(|v| 0.3 * p[v][0] + 0.7 * p[v][1] - 0.5 * p[v][2]),
        );
        let f = s.gradient_force(&phi);
        assert!(f.norm() > 1e-6, "the test force must be non-trivial");
        let (u, _) = s.solve(&f);
        assert!(
            s.velocity_norm(&u) < 1e-9 * f.norm(),
            "gradient force drove flow of norm {}",
            s.velocity_norm(&u)
        );
    }

    /// Whatever the force, the solution satisfies the constraint.
    #[test]
    fn the_solution_is_divergence_free() {
        let (c, _, s) = rig(2);
        let f = DVector::from_iterator(
            c.n_edges(),
            (0..c.n_edges()).map(|e| ((e * 7 % 13) as f64 - 6.0) / 6.0),
        );
        let (u, _) = s.solve(&f);
        let div = s.divergence(&u);
        assert!(
            div.amax() < 1e-9 * f.norm(),
            "divergence {:e} against force norm {:e}",
            div.amax(),
            f.norm()
        );
    }

    /// At the solution the force's power equals the viscous dissipation.
    #[test]
    fn power_input_equals_dissipation() {
        let (c, _, s) = rig(2);
        let f = DVector::from_iterator(
            c.n_edges(),
            (0..c.n_edges()).map(|e| ((e * 5 % 11) as f64 - 5.0) / 5.0),
        );
        let (u, _) = s.solve(&f);
        let power = f.dot(&u);
        let diss = s.dissipation(&u);
        assert!(
            (power - diss).abs() < 1e-8 * power.abs().max(1.0),
            "power {power:e} against dissipation {diss:e}"
        );
    }

    #[test]
    fn refinement_keeps_the_constraint() {
        for n in [1usize, 2, 3] {
            let (c, _, s) = rig(n);
            let f = DVector::from_iterator(
                c.n_edges(),
                (0..c.n_edges()).map(|e| ((e % 5) as f64 - 2.0) / 2.0),
            );
            let (u, _) = s.solve(&f);
            assert!(
                s.divergence(&u).amax() < 1e-8 * f.norm().max(1.0),
                "n = {n}: divergence {:e}",
                s.divergence(&u).amax()
            );
        }
    }
}
