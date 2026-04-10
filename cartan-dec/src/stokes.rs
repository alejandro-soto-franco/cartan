//! Augmented Lagrangian Stokes solver on triangle meshes in R^3.
//!
//! Solves the incompressible Stokes equations on a surface:
//!
//!   2*mu*DIV^nabla(K*U) - GRAD*p + f = 0
//!   DIV*U = 0
//!
//! using the augmented Lagrangian (AL) method from Zhu et al. (2024),
//! Algorithm 3. The inner linear system is solved by conjugate gradient.
//!
//! ## References
//!
//! - Zhu, Saintillan, Chern (2024). "Active nematic fluids on Riemannian
//!   2-manifolds." arXiv:2405.06044. Section 3.6, Algorithm 3.

use nalgebra::SVector;
use sprs::{CsMat, TriMat};

use crate::extrinsic::ExtrinsicOperators;
use crate::mesh::Mesh;
use cartan_core::Manifold;

const D: usize = 3;

/// Augmented Lagrangian Stokes solver.
///
/// Precomputes the augmented system matrix A = L + k * GRAD * DIV
/// and the Killing vector basis (rigid motion kernel).
pub struct StokesSolverAL {
    /// Extrinsic operators (DIV, GRAD, L).
    ops: ExtrinsicOperators,
    /// Augmented system matrix: A = L + k * GRAD * DIV.
    augmented: CsMat<f64>,
    /// Killing vector basis (rigid motions to project out).
    killing_basis: Vec<Vec<f64>>,
    /// Penalty parameter for augmented Lagrangian.
    penalty: f64,
    /// Convergence tolerance for ||DIV u||.
    tolerance: f64,
    /// Maximum AL iterations.
    max_al_iterations: usize,
    /// Maximum CG iterations per inner solve.
    max_cg_iterations: usize,
    /// Number of vertices.
    n_vertices: usize,
}

/// Result of a Stokes solve.
#[derive(Debug, Clone)]
pub struct StokesResult {
    /// Velocity field: R^3-valued per vertex (length 3*nv).
    pub velocity: Vec<f64>,
    /// Pressure: scalar per vertex (length nv).
    pub pressure: Vec<f64>,
    /// Final divergence residual ||DIV u||.
    pub div_residual: f64,
    /// Number of AL iterations used.
    pub al_iterations: usize,
}

impl StokesSolverAL {
    /// Create a Stokes solver from a triangle mesh.
    ///
    /// # Parameters
    ///
    /// - `penalty`: augmented Lagrangian penalty parameter (k). Higher = faster
    ///   convergence but worse conditioning. Typical: 1e3 to 1e6.
    /// - `tolerance`: stop when ||DIV u|| / ||f|| < tolerance.
    /// - `max_al_iterations`: maximum outer AL iterations.
    /// - `max_cg_iterations`: maximum CG iterations per inner solve.
    pub fn new<M: Manifold<Point = SVector<f64, D>>>(
        mesh: &Mesh<M, 3, 2>,
        penalty: f64,
        tolerance: f64,
        max_al_iterations: usize,
        max_cg_iterations: usize,
    ) -> Self {
        let ops = ExtrinsicOperators::from_mesh(mesh);
        let nv = ops.n_vertices;

        // Build augmented matrix: A = -L + k * DIV^T * DIV.
        // -L is positive semi-definite (L = K^T A^{-1} K is PSD, so -L is NSD,
        // but we negate it to get PSD for the solver).
        // DIV^T * DIV is PSD.
        // So A is PSD (positive semi-definite), suitable for CG.
        let dtd = &ops.div.transpose_view().to_csc() * &ops.div;
        let neg_l = ops.viscosity_lap.map(|&v| -v);
        let augmented_base = &neg_l + &(dtd.map(|&v| v * penalty));

        // Add small regularisation to make strictly positive definite
        // (removes the Killing vector kernel numerically).
        let n = 3 * nv;
        let eps = 1e-8;
        let mut reg = TriMat::new((n, n));
        for i in 0..n {
            reg.add_triplet(i, i, eps);
        }
        let augmented = &augmented_base + &reg.to_csc();

        // Compute Killing vector basis: rigid motions of R^3.
        // 3 translations + 3 rotations = 6D kernel.
        let killing_basis = compute_killing_basis(mesh);

        Self {
            ops,
            augmented,
            killing_basis,
            penalty,
            tolerance,
            max_al_iterations,
            max_cg_iterations,
            n_vertices: nv,
        }
    }

    /// Solve the Stokes equations for a given body force.
    ///
    /// `force` is R^3-valued per vertex (length 3*nv), e.g. from active stress
    /// divergence: f = DIV^nabla(zeta * Q).
    pub fn solve(&self, force: &[f64]) -> StokesResult {
        let n = 3 * self.n_vertices;
        assert_eq!(force.len(), n);

        let mut pressure = vec![0.0; self.n_vertices];
        let mut velocity = vec![0.0; n];
        let mut div_residual = f64::MAX;

        let force_norm = force.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-15);

        for iter in 0..self.max_al_iterations {
            // RHS: f - GRAD * p
            let grad_p = self.ops.apply_grad(&pressure);
            let rhs: Vec<f64> = force.iter().zip(&grad_p).map(|(f, gp)| f - gp).collect();

            // Solve A * u = rhs via CG.
            velocity = self.cg_solve(&rhs, &velocity);

            // Project out Killing vectors.
            self.project_out_killing(&mut velocity);

            // Update pressure: p += k * DIV * u
            let div_u = self.ops.apply_div(&velocity);
            for i in 0..self.n_vertices {
                pressure[i] += self.penalty * div_u[i];
            }

            // Check convergence.
            div_residual = div_u.iter().map(|x| x * x).sum::<f64>().sqrt();
            if div_residual / force_norm < self.tolerance {
                return StokesResult {
                    velocity,
                    pressure,
                    div_residual,
                    al_iterations: iter + 1,
                };
            }
        }

        StokesResult {
            velocity,
            pressure,
            div_residual,
            al_iterations: self.max_al_iterations,
        }
    }

    /// Conjugate gradient solve: A * x = b.
    fn cg_solve(&self, b: &[f64], x0: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut x = x0.to_vec();
        let ax = sparse_matvec_real(&self.augmented, &x);
        let mut r: Vec<f64> = b.iter().zip(&ax).map(|(bi, ai)| bi - ai).collect();
        let mut p = r.clone();
        let mut rs_old: f64 = r.iter().map(|ri| ri * ri).sum();

        if rs_old.sqrt() < 1e-15 {
            return x;
        }

        for _ in 0..self.max_cg_iterations {
            let ap = sparse_matvec_real(&self.augmented, &p);
            let pap: f64 = p.iter().zip(&ap).map(|(pi, api)| pi * api).sum();
            if pap.abs() < 1e-30 {
                break;
            }
            let alpha = rs_old / pap;

            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }

            let rs_new: f64 = r.iter().map(|ri| ri * ri).sum();
            if rs_new.sqrt() < 1e-12 {
                break;
            }

            let beta = rs_new / rs_old;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            rs_old = rs_new;
        }

        x
    }

    /// Project out Killing vector fields (rigid motions) from a velocity field.
    fn project_out_killing(&self, u: &mut [f64]) {
        for basis in &self.killing_basis {
            let dot: f64 = u.iter().zip(basis).map(|(a, b)| a * b).sum();
            let norm_sq: f64 = basis.iter().map(|b| b * b).sum();
            if norm_sq > 1e-30 {
                let coeff = dot / norm_sq;
                for i in 0..u.len() {
                    u[i] -= coeff * basis[i];
                }
            }
        }
    }
}

/// Compute the 6 Killing vector fields (rigid motions) for a surface in R^3.
///
/// 3 translations: e_x, e_y, e_z at every vertex.
/// 3 rotations: omega x r at every vertex (cross product of rotation axis with position).
fn compute_killing_basis<M: Manifold<Point = SVector<f64, D>>>(
    mesh: &Mesh<M, 3, 2>,
) -> Vec<Vec<f64>> {
    let nv = mesh.n_vertices();
    let n = 3 * nv;
    let mut basis = Vec::with_capacity(6);

    // Translations.
    for axis in 0..3 {
        let mut b = vec![0.0; n];
        for v in 0..nv {
            b[v * 3 + axis] = 1.0;
        }
        basis.push(b);
    }

    // Rotations: omega_axis x r_vertex.
    for axis in 0..3 {
        let mut b = vec![0.0; n];
        for v in 0..nv {
            let r = mesh.vertices[v];
            // Cross product of unit axis vector with r.
            let cross = match axis {
                0 => SVector::<f64, 3>::new(0.0, -r[2], r[1]), // e_x x r
                1 => SVector::<f64, 3>::new(r[2], 0.0, -r[0]), // e_y x r
                2 => SVector::<f64, 3>::new(-r[1], r[0], 0.0), // e_z x r
                _ => unreachable!(),
            };
            b[v * 3] = cross[0];
            b[v * 3 + 1] = cross[1];
            b[v * 3 + 2] = cross[2];
        }
        basis.push(b);
    }

    // Gram-Schmidt orthogonalise.
    for i in 0..basis.len() {
        for j in 0..i {
            let dot: f64 = basis[i].iter().zip(&basis[j]).map(|(a, b)| a * b).sum();
            let norm_sq: f64 = basis[j].iter().map(|b| b * b).sum();
            if norm_sq > 1e-30 {
                let coeff = dot / norm_sq;
                let bj = basis[j].clone();
                for (k, bj_k) in bj.iter().enumerate() {
                    basis[i][k] -= coeff * bj_k;
                }
            }
        }
    }

    basis
}

/// Sparse matrix-vector multiply (real-valued).
fn sparse_matvec_real(mat: &CsMat<f64>, x: &[f64]) -> Vec<f64> {
    let nrows = mat.rows();
    let mut y = vec![0.0; nrows];
    for (col, col_view) in mat.outer_iterator().enumerate() {
        let xc = x[col];
        if xc.abs() < 1e-30 {
            continue;
        }
        for (row, &val) in col_view.iter() {
            y[row] += val * xc;
        }
    }
    y
}
