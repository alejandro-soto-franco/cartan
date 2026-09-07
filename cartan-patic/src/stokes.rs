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
//! ## Factorise once
//!
//! The saddle matrix depends on the mesh, the viscosity and the constrained
//! edges, none of which change while a simulation runs. [`FactoredStokes`]
//! decomposes it once and every later solve is a pair of triangular solves,
//! `O(N^2)` rather than `O(N^3)`. A frame loop that re-factorised each step
//! was spending all its time rebuilding the same matrix.
//!
//! The solve is dense. Tetrahedral meshes past a few thousand edges need a
//! Krylov method on the indefinite system; the dense path is what makes the
//! physics tests exact rather than solver-limited.

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

    /// Solve with no-slip on the given edges: their velocity is constrained
    /// to zero exactly, by eliminating the unknowns rather than by penalty.
    #[must_use]
    pub fn solve_no_slip(
        &self,
        f: &DVector<f64>,
        fixed_edges: &[usize],
    ) -> (DVector<f64>, DVector<f64>) {
        self.factor(fixed_edges).solve(f)
    }

    /// Factorise the saddle system once for a fixed set of constrained edges.
    ///
    /// Every later solve reuses it, which is what makes a frame loop cheap.
    #[must_use]
    pub fn factor(&self, fixed_edges: &[usize]) -> FactoredStokes {
        let ne = self.n_edges;
        let nv = self.n_vertices;
        let n = ne + nv;
        let mut k = DMatrix::<f64>::zeros(n, n);
        k.view_mut((0, 0), (ne, ne)).copy_from(&self.a);
        k.view_mut((0, ne), (ne, nv)).copy_from(&self.b);
        k.view_mut((ne, 0), (nv, ne)).copy_from(&self.b.transpose());

        // Pin the pressure constant.
        let pin = ne;
        for i in 0..n {
            k[(pin, i)] = 0.0;
            k[(i, pin)] = 0.0;
        }
        k[(pin, pin)] = 1.0;

        // No-slip, by elimination.
        for &e in fixed_edges {
            for i in 0..n {
                k[(e, i)] = 0.0;
                k[(i, e)] = 0.0;
            }
            k[(e, e)] = 1.0;
        }

        FactoredStokes {
            lu: k.lu(),
            n_edges: ne,
            n_vertices: nv,
            fixed: fixed_edges.to_vec(),
        }
    }

    /// Solve for velocity and pressure under a one-cochain force.
    ///
    /// Factorises on the spot. A loop over frames should hold a
    /// [`FactoredStokes`] from [`Stokes::factor`] instead.
    #[must_use]
    pub fn solve(&self, f: &DVector<f64>) -> (DVector<f64>, DVector<f64>) {
        self.factor(&[]).solve(f)
    }
}

/// A factorised saddle system, reusable across right-hand sides.
pub struct FactoredStokes {
    lu: nalgebra::LU<f64, nalgebra::Dyn, nalgebra::Dyn>,
    n_edges: usize,
    n_vertices: usize,
    fixed: Vec<usize>,
}

impl FactoredStokes {
    /// Solve for a new force. Two triangular solves, no factorisation.
    #[must_use]
    pub fn solve(&self, f: &DVector<f64>) -> (DVector<f64>, DVector<f64>) {
        let (ne, nv) = (self.n_edges, self.n_vertices);
        let mut rhs = DVector::<f64>::zeros(ne + nv);
        rhs.rows_mut(0, ne).copy_from(f);
        for &e in &self.fixed {
            rhs[e] = 0.0;
        }
        rhs[ne] = 0.0;
        let sol = self
            .lu
            .solve(&rhs)
            .unwrap_or_else(|| DVector::zeros(ne + nv));
        (sol.rows(0, ne).into_owned(), sol.rows(ne, nv).into_owned())
    }

    /// The constrained edges.
    #[must_use]
    pub fn fixed_edges(&self) -> &[usize] {
        &self.fixed
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
