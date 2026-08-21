// ~/cartan/cartan-manifolds/src/util/eig.rs

//! Symmetric eigendecomposition, shared by every manifold that needs one.
//!
//! Two solvers live here and one dispatcher chooses between them by matrix
//! size. The stack-based cyclic Jacobi solver runs anywhere, `no_std`
//! included. The `DMatrix` path calls nalgebra's tridiagonalise-then-QR
//! solver, which allocates and so needs `std`.
//!
//! The module exists because three consumers now need the same
//! decomposition: the SPD spectral functions in [`super::sym`], the nearest
//! correlation matrix iteration, and the orthogonal matrix logarithm in
//! [`super::matrix_log`]. Before it, the solver sat inside `sym`, which is
//! `std`-only, and `matrix_log` could not reach it.
//!
//! ## References
//!
//! - Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*, 4th ed.,
//!   Algorithm 8.4.3 (cyclic Jacobi).
//! - Demmel, J. & Veselic, K. (1992). "Jacobi's Method is More Accurate than
//!   QR." *SIAM J. Matrix Anal. Appl.*, 13(4), 1204-1245.

use cartan_core::Real;
#[cfg(not(feature = "std"))]
use nalgebra::ComplexField;
#[cfg(feature = "std")]
use nalgebra::{DMatrix, DVector};
use nalgebra::{SMatrix, SVector};

/// Ambient dimension at or below which the stack-based Jacobi solver is used.
///
/// Measured, not chosen. Cost of `Spd::exp`, Jacobi against the `DMatrix`
/// path:
///
/// ```text
///   N        Jacobi     DMatrix
///   3        428 ns      676 ns     Jacobi 1.58x
///   4       1182 ns     1037 ns
///   5       2656 ns     1661 ns
///   6       4083 ns     2336 ns
///  10      18495 ns     6496 ns     DMatrix 2.85x
/// ```
///
/// Jacobi is O(N^3) per sweep and needs several sweeps, so it loses to a
/// tridiagonalise-then-QR solver as soon as the matrix is large enough for
/// that solver's setup to amortise. It wins at N = 3, which is the size most
/// SPD work uses: covariance matrices, diffusion tensors, and the Order2
/// Kelvin-Mandel representation in `cartan-homog`.
#[cfg_attr(not(feature = "std"), allow(dead_code))]
const JACOBI_MAX_N: usize = 3;

/// Sweeps before giving up. Cyclic Jacobi converges quadratically and needs
/// six to ten sweeps in practice; this is a backstop, not an operating point.
const JACOBI_MAX_SWEEPS: usize = 30;

/// Symmetric eigendecomposition by cyclic Jacobi rotations, entirely on the
/// stack.
///
/// Returns `(V, d)` with `M = V diag(d) V^T`. Neither the order of the
/// eigenvalues nor the sign of the eigenvectors is canonicalised, because
/// every consumer here recomposes `V diag(f(d)) V^T`, which is invariant to
/// both.
///
/// This exists because the `DMatrix` path allocates on the heap for every
/// call and runs a general tridiagonalise-then-QR solver, which is the wrong
/// trade for the 3x3 to 10x10 matrices the SPD manifold actually uses. Jacobi
/// is also backward stable and tends to give better relative accuracy on small
/// eigenvalues than QR, which matters here since the results feed a logarithm.
pub(crate) fn jacobi_eigen<const N: usize>(
    m: &SMatrix<Real, N, N>,
) -> (SMatrix<Real, N, N>, SVector<Real, N>) {
    let mut a = *m;
    let mut v = SMatrix::<Real, N, N>::identity();

    for _ in 0..JACOBI_MAX_SWEEPS {
        // Sum of squared off-diagonals: the quantity Jacobi drives to zero.
        let mut off = 0.0;
        for i in 0..N {
            for j in (i + 1)..N {
                off += a[(i, j)] * a[(i, j)];
            }
        }
        if off <= Real::EPSILON * Real::EPSILON {
            break;
        }

        for pp in 0..N {
            for qq in (pp + 1)..N {
                let apq = a[(pp, qq)];
                if apq == 0.0 {
                    continue;
                }

                // t is the smaller root of t^2 + 2 theta t - 1 = 0, which is
                // the rotation that keeps the transformation well conditioned.
                let theta = (a[(qq, qq)] - a[(pp, pp)]) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    -1.0 / (-theta + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                let app = a[(pp, pp)];
                let aqq = a[(qq, qq)];
                a[(pp, pp)] = app - t * apq;
                a[(qq, qq)] = aqq + t * apq;
                a[(pp, qq)] = 0.0;
                a[(qq, pp)] = 0.0;

                for k in 0..N {
                    if k != pp && k != qq {
                        let akp = a[(k, pp)];
                        let akq = a[(k, qq)];
                        let np = c * akp - s * akq;
                        let nq = s * akp + c * akq;
                        a[(k, pp)] = np;
                        a[(pp, k)] = np;
                        a[(k, qq)] = nq;
                        a[(qq, k)] = nq;
                    }
                    let vkp = v[(k, pp)];
                    let vkq = v[(k, qq)];
                    v[(k, pp)] = c * vkp - s * vkq;
                    v[(k, qq)] = s * vkp + c * vkq;
                }
            }
        }
    }

    let d = SVector::<Real, N>::from_fn(|i, _| a[(i, i)]);
    (v, d)
}

/// Eigenvalues by cyclic Jacobi, without accumulating the eigenvectors.
///
/// Same rotations as [`jacobi_eigen`], minus the rotation of `V`. That update
/// touches `2N` entries per rotation, the same order as the update to `A`
/// itself, so dropping it is worth roughly a third of the work. The
/// eigenvalues-only consumers, the affine-invariant distance among them, were
/// paying for eigenvectors they discarded.
#[cfg(feature = "std")]
fn jacobi_eigenvalues<const N: usize>(m: &SMatrix<Real, N, N>) -> SVector<Real, N> {
    let mut a = *m;

    for _ in 0..JACOBI_MAX_SWEEPS {
        let mut off = 0.0;
        for i in 0..N {
            for j in (i + 1)..N {
                off += a[(i, j)] * a[(i, j)];
            }
        }
        if off <= Real::EPSILON * Real::EPSILON {
            break;
        }

        for pp in 0..N {
            for qq in (pp + 1)..N {
                let apq = a[(pp, qq)];
                if apq == 0.0 {
                    continue;
                }

                let theta = (a[(qq, qq)] - a[(pp, pp)]) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    -1.0 / (-theta + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                let app = a[(pp, pp)];
                let aqq = a[(qq, qq)];
                a[(pp, pp)] = app - t * apq;
                a[(qq, qq)] = aqq + t * apq;
                a[(pp, qq)] = 0.0;
                a[(qq, pp)] = 0.0;

                for k in 0..N {
                    if k != pp && k != qq {
                        let akp = a[(k, pp)];
                        let akq = a[(k, qq)];
                        let np = c * akp - s * akq;
                        let nq = s * akp + c * akq;
                        a[(k, pp)] = np;
                        a[(pp, k)] = np;
                        a[(k, qq)] = nq;
                        a[(qq, k)] = nq;
                    }
                }
            }
        }
    }

    SVector::<Real, N>::from_fn(|i, _| a[(i, i)])
}

/// Rebuild `V diag(fd) V^T` from an eigendecomposition.
///
/// Column i of `V diag(fd)` is column i of `V` scaled by `fd[i]`, which is
/// O(N^2). Materialising the diagonal matrix and multiplying by it instead
/// costs a second O(N^3) product for the same answer.
#[inline]
pub(crate) fn recompose<const N: usize>(
    v: &SMatrix<Real, N, N>,
    fd: &SVector<Real, N>,
) -> SMatrix<Real, N, N> {
    let mut scaled = *v;
    for i in 0..N {
        let f = fd[i];
        for j in 0..N {
            scaled[(j, i)] *= f;
        }
    }
    scaled * v.transpose()
}

/// Eigendecomposition of a symmetric matrix: returns (V, d) where M = V diag(d) V^T.
///
/// Internal helper shared by all sym_* functions. Avoids repeating the
/// DMatrix conversion and eigen call at every call site.
#[inline]
#[cfg(feature = "std")]
fn sym_eigen<const N: usize>(m: &SMatrix<Real, N, N>) -> (DMatrix<Real>, DVector<Real>) {
    let dm = DMatrix::from_column_slice(N, N, m.as_slice());
    let eigen = dm.symmetric_eigen();
    (eigen.eigenvectors, eigen.eigenvalues)
}

/// Symmetric eigendecomposition, on the stack for the sizes that benefit.
///
/// `N` is a compile-time constant, so the branch folds away entirely.
#[inline]
pub(crate) fn sym_eigen_s<const N: usize>(
    m: &SMatrix<Real, N, N>,
) -> (SMatrix<Real, N, N>, SVector<Real, N>) {
    #[cfg(feature = "std")]
    {
        if N <= JACOBI_MAX_N {
            jacobi_eigen(m)
        } else {
            let (v, d) = sym_eigen(m);
            (
                SMatrix::from_column_slice(v.as_slice()),
                SVector::from_column_slice(d.as_slice()),
            )
        }
    }
    // Without `std` there is no heap, so Jacobi runs at every size. It is
    // slower than QR for a large matrix and correct at any size.
    #[cfg(not(feature = "std"))]
    {
        jacobi_eigen(m)
    }
}

/// Eigenvalues of a symmetric matrix, without the eigenvectors.
///
/// `symmetric_eigenvalues` skips the eigenvector accumulation that
/// `symmetric_eigen` performs, which is the bulk of the work. Used where only
/// the spectrum reaches the answer, as in the affine-invariant distance.
///
/// Goes through `DMatrix` for the same reason `sym_eigen` does: the const
/// generic `SMatrix<Real, N, N>` does not satisfy nalgebra's `DimSub<U1>`
/// bound for the symmetric eigensolver.
#[cfg(feature = "std")]
#[inline]
pub(crate) fn sym_eigenvalues<const N: usize>(m: &SMatrix<Real, N, N>) -> DVector<Real> {
    if N <= JACOBI_MAX_N {
        DVector::from_column_slice(jacobi_eigenvalues(m).as_slice())
    } else {
        DMatrix::from_column_slice(N, N, m.as_slice()).symmetric_eigenvalues()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The Jacobi path is only reached at N = 3, so a regression there would
    /// be invisible to the larger-N tests. This pins the decomposition itself:
    /// V must be orthogonal and V diag(d) V^T must reproduce the input.
    #[test]
    fn test_jacobi_eigen_reconstructs_and_is_orthogonal() {
        let cases = [
            SMatrix::<Real, 3, 3>::from_row_slice(&[4.0, 2.0, 1.0, 2.0, 3.0, 0.5, 1.0, 0.5, 2.0]),
            // Degenerate spectrum: repeated eigenvalues are where a rotation
            // based solver can stall if the pivot choice is wrong.
            SMatrix::<Real, 3, 3>::identity() * 2.5,
            // Nearly singular, which is what the eigenvalue floors exist for.
            SMatrix::<Real, 3, 3>::from_row_slice(&[1.0, 0.0, 0.0, 0.0, 1e-13, 0.0, 0.0, 0.0, 1.0]),
            // Already diagonal: zero rotations needed.
            SMatrix::<Real, 3, 3>::from_row_slice(&[3.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 1.0]),
        ];

        for m in cases {
            let (v, d) = jacobi_eigen(&m);

            let recon = recompose::<3>(&v, &d);
            assert!(
                (recon - m).norm() < 1e-12,
                "V diag(d) V^T does not reproduce the input: {:.3e}",
                (recon - m).norm()
            );

            let vtv = v.transpose() * v;
            assert!(
                (vtv - SMatrix::<Real, 3, 3>::identity()).norm() < 1e-12,
                "eigenvectors are not orthonormal"
            );
        }
    }

    /// The eigenvalues-only Jacobi is a separate loop from the full one, so a
    /// divergence between them would be silent: the distance would drift while
    /// every decomposition-based result stayed correct.
    #[test]
    fn test_jacobi_eigenvalues_matches_full_decomposition() {
        let cases = [
            SMatrix::<Real, 3, 3>::from_row_slice(&[4.0, 2.0, 1.0, 2.0, 3.0, 0.5, 1.0, 0.5, 2.0]),
            SMatrix::<Real, 3, 3>::identity() * 2.5,
            SMatrix::<Real, 3, 3>::from_row_slice(&[1.0, 0.0, 0.0, 0.0, 1e-13, 0.0, 0.0, 0.0, 1.0]),
            SMatrix::<Real, 3, 3>::from_row_slice(&[3.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 1.0]),
        ];

        for m in cases {
            let (_, full) = jacobi_eigen(&m);
            let only = jacobi_eigenvalues(&m);

            // Same rotations in the same order, so these should agree exactly;
            // the tolerance is there only to survive a future reordering.
            let mut a: Vec<Real> = full.iter().copied().collect();
            let mut b: Vec<Real> = only.iter().copied().collect();
            a.sort_by(|x, y| x.partial_cmp(y).unwrap());
            b.sort_by(|x, y| x.partial_cmp(y).unwrap());
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-14, "eigenvalues diverge: {x} vs {y}");
            }
        }
    }
}
