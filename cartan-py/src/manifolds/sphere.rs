// ~/cartan/cartan-py/src/manifolds/sphere.rs

//! Python wrapper for `cartan_manifolds::Sphere<N>`.
//!
//! The unit sphere S^n sits in R^(n+1) as the set of vectors with unit norm.
//! The Python constructor takes the intrinsic dimension (e.g. 2 for S^2 in R^3),
//! and the ambient dimension is stored as `ambient_n = intrinsic_dim + 1`.
//! Supported intrinsic dimensions: 1 through 9 (ambient 2 through 10).

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

#[pyclass(name = "Sphere")]
#[derive(Debug, Clone)]
pub struct PySphere {
    pub(crate) ambient_n: usize,
}

#[pymethods]
impl PySphere {
    /// Construct a sphere S^n with the given intrinsic dimension (1 <= n <= 9).
    ///
    /// `Sphere(2)` creates S^2 embedded in R^3.
    #[new]
    fn new(intrinsic_dim: usize) -> PyResult<Self> {
        let ambient_n = intrinsic_dim + 1;
        if ambient_n < 2 || ambient_n > 10 {
            return Err(PyValueError::new_err(
                format!("Sphere: unsupported intrinsic dimension {intrinsic_dim}, need 1 <= dim <= 9")
            ));
        }
        Ok(Self { ambient_n })
    }

    /// Intrinsic dimension of the sphere.
    fn dim(&self) -> usize { self.ambient_n - 1 }

    /// Ambient dimension (intrinsic + 1).
    fn ambient_dim(&self) -> usize { self.ambient_n }

    fn __repr__(&self) -> String {
        format!("Sphere(dim={}, ambient={})", self.ambient_n - 1, self.ambient_n)
    }
}

// Generate all trait method wrappers as a separate #[pymethods] impl block.
// Ambient dimensions 2-10 correspond to intrinsic dimensions 1-9.
crate::impl_vector_manifold_methods!(PySphere, Sphere, ambient_n, [2, 3, 4, 5, 6, 7, 8, 9, 10]);
