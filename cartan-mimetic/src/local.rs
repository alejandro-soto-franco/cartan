//! The star on one simplex.

use nalgebra::{DMatrix, DVector};

use crate::combinatorics::{form_indices, k_faces};
use crate::Consistency;

/// A simplex, by the coordinates of its vertices in whatever space it sits in.
///
/// The ambient dimension may exceed the simplex's own, which is the surface
/// case: a triangle in three dimensions has a two-dimensional tangent space and
/// the constant 1-forms on it span two dimensions rather than three. Everything
/// below works in that tangent space, so the ambient dimension never enters the
/// answer.
#[derive(Debug, Clone)]
pub struct Simplex {
    /// One row per vertex.
    pub vertices: DMatrix<f64>,
}

impl Simplex {
    /// From a slice of points, each of the same length.
    pub fn new(points: &[Vec<f64>]) -> Self {
        let rows = points.len();
        let cols = points.first().map_or(0, Vec::len);
        assert!(
            points.iter().all(|p| p.len() == cols),
            "every vertex needs the same number of coordinates"
        );
        Self {
            vertices: DMatrix::from_fn(rows, cols, |i, j| points[i][j]),
        }
    }

    /// The simplex's own dimension: one less than its vertex count.
    pub fn dim(&self) -> usize {
        self.vertices.nrows() - 1
    }

    /// An orthonormal basis of the tangent space, and the edge vectors from
    /// vertex zero expressed in it.
    ///
    /// Gram-Schmidt on the edges out of the first vertex. A degenerate simplex,
    /// whose edges do not span, returns `None` rather than a basis that silently
    /// omits a direction.
    fn tangent_frame(&self) -> Option<DMatrix<f64>> {
        let n = self.dim();
        let ambient = self.vertices.ncols();
        let mut basis: Vec<DVector<f64>> = Vec::with_capacity(n);
        for i in 1..=n {
            let mut v = DVector::from_fn(ambient, |c, _| {
                self.vertices[(i, c)] - self.vertices[(0, c)]
            });
            let scale = v.norm();
            for b in &basis {
                let proj = v.dot(b);
                v -= proj * b;
            }
            let norm = v.norm();
            if norm <= 1e-12 * scale.max(1.0) {
                return None;
            }
            basis.push(v / norm);
        }
        Some(DMatrix::from_fn(ambient, n, |r, c| basis[c][r]))
    }

    /// Vertex coordinates in the tangent frame, relative to vertex zero.
    fn local_coordinates(&self) -> Option<DMatrix<f64>> {
        let frame = self.tangent_frame()?;
        let n = self.dim();
        Some(DMatrix::from_fn(n + 1, n, |i, j| {
            (0..self.vertices.ncols())
                .map(|c| (self.vertices[(i, c)] - self.vertices[(0, c)]) * frame[(c, j)])
                .sum()
        }))
    }

    /// The unsigned volume of the simplex.
    pub fn volume(&self) -> f64 {
        let Some(local) = self.local_coordinates() else {
            return 0.0;
        };
        let n = self.dim();
        if n == 0 {
            return 1.0;
        }
        let m = DMatrix::from_fn(n, n, |r, c| local[(r + 1, c)]);
        m.determinant().abs() / factorial(n)
    }

    /// The consistency data at form degree `k`.
    ///
    /// Row `f`, column `I` is the integral over the `f`-th k-face of the basis
    /// k-form `dx_I`, which for a constant form is its value on the face's
    /// k-vector divided by `k!`. The k-vector's components are the `k x k`
    /// minors of the face's edge matrix, so this is exact rather than quadrature.
    pub fn consistency(&self, k: usize) -> Option<Consistency> {
        let n = self.dim();
        if k > n {
            return None;
        }
        let local = self.local_coordinates()?;
        let faces = k_faces(n + 1, k);
        let forms = form_indices(n, k);
        let scale = factorial(k);

        let dofs = DMatrix::from_fn(faces.len(), forms.len(), |f, i| {
            if k == 0 {
                return 1.0;
            }
            let face = &faces[f];
            let index = &forms[i];
            // The k edge vectors of the face, out of its first vertex, and the
            // minor on the rows this basis form names.
            let minor = DMatrix::from_fn(k, k, |r, c| {
                local[(face[c + 1], index[r])] - local[(face[0], index[r])]
            });
            minor.determinant() / scale
        });

        let gram = self.volume() * DMatrix::identity(forms.len(), forms.len());
        Some(Consistency { dofs, gram })
    }
}

fn factorial(n: usize) -> f64 {
    (1..=n).map(|i| i as f64).product::<f64>().max(1.0)
}

/// What a diagonal star turned out to be on a given simplex.
#[derive(Debug, Clone)]
pub struct DiagonalStar {
    /// One entry per k-face.
    pub entries: DVector<f64>,
    /// How far it is from consistent, relative to the cell volume. Nonzero means
    /// the system was overdetermined and no diagonal star is consistent here.
    pub residual: f64,
}

impl DiagonalStar {
    /// Whether a consistent diagonal star exists on this simplex at this degree.
    pub fn is_consistent(&self) -> bool {
        self.residual < 1e-10
    }

    /// Whether every entry is positive, which a stable star needs.
    pub fn is_positive(&self) -> bool {
        self.entries.iter().all(|&x| x > 0.0)
    }

    /// Usable as a Hodge star: consistent and positive at once.
    ///
    /// The two fail independently and for different reasons, so a caller
    /// deciding whether the cheap diagonal path is available on a given mesh
    /// wants this rather than either alone.
    pub fn is_usable(&self) -> bool {
        self.is_consistent() && self.is_positive()
    }
}

/// The best diagonal star on a simplex, and what is wrong with it.
///
/// Returns `None` only for a degenerate simplex. A non-degenerate one always
/// returns something; whether it may be used is [`DiagonalStar::is_usable`].
pub fn diagonal_star(simplex: &Simplex, k: usize) -> Option<DiagonalStar> {
    let c = simplex.consistency(k)?;
    let (entries, residual) = c.best_diagonal();
    Some(DiagonalStar { entries, residual })
}

/// The mimetic star: consistent by construction and positive definite.
///
/// Consistency fixes the star on the range of the degrees of freedom,
///
/// ```text
/// M_c = N (N^T N)^-1 G (N^T N)^-1 N^T,     so   N^T M_c N = G
/// ```
///
/// and anything annihilated by `N^T` may be added without disturbing that. The
/// stabilisation is the projector onto that complement, scaled to the mean
/// diagonal of the consistent part so the two terms sit at one magnitude, which
/// is what keeps the star spectrally equivalent to its diagonal.
///
/// Returns `None` for a degenerate simplex, where the tangent space is not
/// spanned and no inner product is defined.
pub fn local_star(simplex: &Simplex, k: usize) -> Option<DMatrix<f64>> {
    let c = simplex.consistency(k)?;
    let n = c.dofs.nrows();
    let ntn = (c.dofs.transpose() * &c.dofs).try_inverse()?;
    let consistent = &c.dofs * &ntn * &c.gram * &ntn * c.dofs.transpose();
    let projector = &c.dofs * &ntn * c.dofs.transpose();
    let gamma = consistent.trace() / c.gram.nrows() as f64;
    Some(consistent + gamma * (DMatrix::identity(n, n) - projector))
}
