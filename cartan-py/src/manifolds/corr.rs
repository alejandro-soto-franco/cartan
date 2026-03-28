// ~/cartan/cartan-py/src/manifolds/corr.rs

//! Python wrapper for `cartan_manifolds::Corr<N>`.
//!
//! Corr(n) is the manifold of n x n correlation matrices: symmetric
//! positive definite matrices with unit diagonal entries. It is a flat
//! submanifold of Sym(n) with the Frobenius metric. Supported sizes: 2 to 8.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

#[pyclass(name = "Corr")]
#[derive(Debug, Clone)]
pub struct PyCorr {
    pub(crate) n: usize,
}

#[pymethods]
impl PyCorr {
    /// Construct a Corr manifold of size `n` (2 <= n <= 8).
    ///
    /// Points on Corr(n) are n x n symmetric positive definite matrices
    /// with unit diagonal entries, passed as 2-D numpy arrays of shape (n, n).
    #[new]
    fn new(n: usize) -> PyResult<Self> {
        if n < 2 || n > 8 {
            return Err(PyValueError::new_err(
                format!("Corr: unsupported size {n}, need 2 <= n <= 8")
            ));
        }
        Ok(Self { n })
    }

    /// Intrinsic dimension: n*(n-1)/2 (off-diagonal independent entries).
    fn dim(&self) -> usize { self.n * (self.n - 1) / 2 }

    /// Ambient dimension: n*n (total matrix entries).
    fn ambient_dim(&self) -> usize { self.n * self.n }

    fn __repr__(&self) -> String {
        format!("Corr(n={})", self.n)
    }
}

// Generate all trait method wrappers as a separate #[pymethods] impl block.
// The macro is invoked at module level so pyo3 can process it correctly.
crate::impl_matrix_manifold_methods!(PyCorr, Corr, n, [2, 3, 4, 5, 6, 7, 8]);
