//! Mimetic Hodge stars for compatible discretisation.
//!
//! A compatible discretisation represents the exterior derivative exactly, the
//! coboundary being combinatorial and satisfying `d d = 0` identically, and puts
//! every metric quantity into the inner product on k-cochains. That inner
//! product is the only choice the method makes, since the codifferential is its
//! adjoint,
//!
//! ```text
//! delta_k = M_{k-1}^-1 d_{k-1}^T M_k        Laplacian = d delta + delta d
//! ```
//!
//! and the Laplacian follows. This crate builds `M_k` on one simplex.
//!
//! # What a star has to satisfy
//!
//! **Consistency**: the discrete inner product reproduces the exact one whenever
//! both arguments are constant k-forms on the cell. Writing `N` for the degrees
//! of freedom of a basis of those forms, one row per k-face, and `G` for their
//! exact Gram matrix, that is `N^T M N = G`.
//!
//! **Stability**: `M` is positive definite, and spectrally equivalent to its own
//! diagonal, so the conditioning does not degrade under refinement.
//!
//! # Why the diagonal answer is not always available
//!
//! A diagonal star has one unknown per k-face and consistency imposes one
//! equation per symmetric pair of constant forms:
//!
//! ```text
//! unknowns  = C(n+1, k+1)                      k-faces of an n-simplex
//! equations = C(n,k) (C(n,k) + 1) / 2          symmetric entries of G
//! ```
//!
//! which decides the question before any geometry enters:
//!
//! | n | k | unknowns | equations | diagonal star |
//! |---|---|----------|-----------|---------------|
//! | 2 | 0 | 3 | 1 | a 2-parameter family |
//! | 2 | 1 | 3 | 3 | unique, and equal to `cot/2` |
//! | 3 | 0 | 4 | 1 | a 3-parameter family |
//! | 3 | 1 | 6 | 6 | unique, sign not guaranteed |
//! | 3 | 2 | 4 | 6 | overdetermined, generically none |
//!
//! Two consequences, each measured in the tests beside this file. At
//! `n = 2, k = 1` the unique diagonal star turns negative exactly when the
//! triangle is obtuse, so consistency, diagonality and positivity cannot hold at
//! once. At `n = 3, k = 2` no diagonal star is consistent on a general
//! tetrahedron at all, and at `n = 3, k = 1` the unique one is negative on every
//! tetrahedron of a subdivided cube, which is the ordinary way to mesh a box.
//!
//! Giving up diagonality restores both properties, which is what this crate
//! does. [`local_star`] is consistent to machine precision and
//! positive definite on every simplex tested, slivers, needles and caps
//! included.

use nalgebra::{DMatrix, DVector};

mod assemble;
mod combinatorics;
mod local;

pub use assemble::{Mesh, assemble_star};
pub use combinatorics::{k_faces, n_choose_k};
pub use local::{DiagonalStar, Simplex, diagonal_star, local_star};

/// The consistency data of one simplex at one form degree.
///
/// `dofs` lists the degrees of freedom of a basis of constant k-forms, one row
/// per k-face and one column per basis form, and `gram` their exact inner
/// products. A star `M` is consistent exactly when `dofs^T M dofs == gram`.
#[derive(Debug, Clone)]
pub struct Consistency {
    /// One row per k-face of the simplex, one column per constant k-form.
    pub dofs: DMatrix<f64>,
    /// The exact Gram matrix of those forms over the simplex.
    pub gram: DMatrix<f64>,
}

impl Consistency {
    /// How far a candidate star is from consistent, relative to the cell volume.
    ///
    /// Zero to rounding is the only acceptable answer. This is the check to
    /// apply to any star from anywhere, including one this crate did not build.
    pub fn residual(&self, star: &DMatrix<f64>) -> f64 {
        let scale = self.gram.diagonal().iter().cloned().fold(0.0_f64, f64::max);
        if scale <= 0.0 {
            return 0.0;
        }
        (&self.dofs.transpose() * star * &self.dofs - &self.gram)
            .abs()
            .max()
            / scale
    }

    /// The least-squares diagonal star, and its consistency residual.
    ///
    /// The residual is zero where a consistent diagonal star exists and positive
    /// where the system is overdetermined, which is what the table above
    /// predicts and what makes this a usable test rather than a guess.
    pub fn best_diagonal(&self) -> (DVector<f64>, f64) {
        // Each equation is one symmetric entry of `dofs^T D dofs`.
        let m = self.gram.nrows();
        let faces = self.dofs.nrows();
        let mut rows = Vec::new();
        let mut rhs = Vec::new();
        for a in 0..m {
            for b in a..m {
                rows.push(
                    (0..faces)
                        .map(|f| self.dofs[(f, a)] * self.dofs[(f, b)])
                        .collect::<Vec<_>>(),
                );
                rhs.push(self.gram[(a, b)]);
            }
        }
        let lhs = DMatrix::from_fn(rows.len(), faces, |i, j| rows[i][j]);
        let rhs = DVector::from_vec(rhs);
        // Normal equations, which is enough at these sizes and keeps the
        // dependency surface to nalgebra's dense core.
        let ata = lhs.transpose() * &lhs;
        let atb = lhs.transpose() * &rhs;
        let d = ata
            .clone()
            .svd(true, true)
            .solve(&atb, 1e-12)
            .unwrap_or_else(|_| DVector::zeros(faces));
        let scale = self.gram.diagonal().iter().cloned().fold(0.0_f64, f64::max);
        let residual = if scale > 0.0 {
            (&lhs * &d - &rhs).abs().max() / scale
        } else {
            0.0
        };
        (d, residual)
    }
}
