// ~/cartan/cartan-py/src/manifolds/so.rs

//! Python wrapper for `cartan_manifolds::SpecialOrthogonal<N>`.
//!
//! SO(n) is the Lie group of n x n orthogonal matrices with determinant +1.
//! It is a compact Riemannian manifold with nonnegative sectional curvature
//! under the bi-invariant metric. Points are n x n matrices passed as
//! 2-D numpy arrays. Supported sizes: 2 through 4.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

#[pyclass(name = "SO")]
#[derive(Debug, Clone)]
pub struct PySo {
    pub(crate) n: usize,
}

#[pymethods]
impl PySo {
    /// Construct an SO manifold of size `n` (2 <= n <= 4).
    ///
    /// Points on SO(n) are n x n orthogonal matrices with determinant +1,
    /// passed as 2-D numpy arrays of shape (n, n).
    #[new]
    fn new(n: usize) -> PyResult<Self> {
        if n < 2 || n > 4 {
            return Err(PyValueError::new_err(
                format!("SO: unsupported size {n}, need 2 <= n <= 4")
            ));
        }
        Ok(Self { n })
    }

    /// Intrinsic dimension: n*(n-1)/2 (dimension of the Lie algebra so(n)).
    fn dim(&self) -> usize { self.n * (self.n - 1) / 2 }

    /// Ambient dimension: n*n (total matrix entries).
    fn ambient_dim(&self) -> usize { self.n * self.n }

    fn __repr__(&self) -> String {
        format!("SO(n={})", self.n)
    }
}

// Generate all trait method wrappers as a separate #[pymethods] impl block.
// The macro is invoked at module level so pyo3 can process it correctly.
crate::impl_matrix_manifold_methods!(PySo, SpecialOrthogonal, n, [2, 3, 4]);
