// ~/cartan/cartan-py/src/manifolds/mod.rs

//! Python-facing manifold wrappers.
//!
//! Each submodule wraps a cartan-manifolds type with numpy I/O and
//! dimension dispatch. The macros in `macros.rs` generate all trait
//! method wrappers so each manifold file is minimal boilerplate.

#[macro_use]
pub mod macros;

pub mod euclidean;

/// Register all manifold classes on the Python module.
pub fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::types::PyModuleMethods;
    m.add_class::<euclidean::PyEuclidean>()?;
    Ok(())
}
