// ~/cartan/cartan-py/src/dec/mod.rs

//! DEC (Discrete Exterior Calculus) Python bindings.
//!
//! Exposes:
//! - `Mesh` — triangulated flat 2D mesh (wraps `FlatMesh`)
//! - `ExteriorDerivative` — d0 and d1 matrices
//! - `HodgeStar` — diagonal star0, star1, star2 operators
//! - `Operators` — assembled Laplacians
//! - `apply_scalar_advection`, `apply_vector_advection` — upwind advection
//! - `apply_divergence`, `apply_tensor_divergence` — codifferential divergence

pub mod mesh;
pub mod operators;

use pyo3::types::PyModuleMethods;

/// Register all DEC classes and free functions into the cartan Python module.
pub fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    m.add_class::<mesh::PyMesh>()?;
    operators::register(m)?;
    Ok(())
}
