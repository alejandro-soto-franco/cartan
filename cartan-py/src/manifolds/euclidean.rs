// ~/cartan/cartan-py/src/manifolds/euclidean.rs

//! Python wrapper for `cartan_manifolds::Euclidean<N>`.
//!
//! Euclidean space R^n is the simplest manifold: its exponential map is
//! vector addition, logarithm is subtraction, and all curvatures are zero.
//! Supported dimensions: 1 through 10.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

#[pyclass(name = "Euclidean")]
#[derive(Debug, Clone)]
pub struct PyEuclidean {
    pub(crate) n: usize,
}

#[pymethods]
impl PyEuclidean {
    /// Construct a Euclidean manifold of dimension `n` (1 <= n <= 10).
    #[new]
    fn new(n: usize) -> PyResult<Self> {
        if n == 0 || n > 10 {
            return Err(PyValueError::new_err(
                format!("Euclidean: unsupported dimension {n}, need 1 <= n <= 10")
            ));
        }
        Ok(Self { n })
    }

    /// Intrinsic dimension of the manifold.
    fn dim(&self) -> usize { self.n }

    /// Ambient dimension (same as intrinsic for Euclidean space).
    fn ambient_dim(&self) -> usize { self.n }

    fn __repr__(&self) -> String {
        format!("Euclidean(n={})", self.n)
    }
}

// Generate all trait method wrappers as a separate #[pymethods] impl block.
// The macro is invoked at module level so pyo3 can process it correctly.
crate::impl_vector_manifold_methods!(PyEuclidean, Euclidean, n, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
