// ~/cartan/cartan-py/src/manifolds/spd.rs

//! Python wrapper for `cartan_manifolds::Spd<N>`.
//!
//! SPD(n) is the manifold of n x n symmetric positive-definite matrices.
//! It is a Cartan-Hadamard manifold with nonpositive sectional curvature
//! and infinite injectivity radius. Points are n x n matrices passed as
//! 2-D numpy arrays. Supported sizes: 2 through 8.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

#[pyclass(name = "SPD")]
#[derive(Debug, Clone)]
pub struct PySpd {
    pub(crate) n: usize,
}

#[pymethods]
impl PySpd {
    /// Construct an SPD manifold of size `n` (2 <= n <= 8).
    ///
    /// Points on SPD(n) are n x n symmetric positive-definite matrices
    /// passed as 2-D numpy arrays of shape (n, n).
    #[new]
    fn new(n: usize) -> PyResult<Self> {
        if n < 2 || n > 8 {
            return Err(PyValueError::new_err(
                format!("SPD: unsupported size {n}, need 2 <= n <= 8")
            ));
        }
        Ok(Self { n })
    }

    /// Intrinsic dimension: n*(n+1)/2 (number of independent entries).
    fn dim(&self) -> usize { self.n * (self.n + 1) / 2 }

    /// Ambient dimension: n*n (total matrix entries).
    fn ambient_dim(&self) -> usize { self.n * self.n }

    fn __repr__(&self) -> String {
        format!("SPD(n={})", self.n)
    }
}

// Generate all trait method wrappers as a separate #[pymethods] impl block.
// The macro is invoked at module level so pyo3 can process it correctly.
crate::impl_matrix_manifold_methods!(PySpd, Spd, n, [2, 3, 4, 5, 6, 7, 8]);
