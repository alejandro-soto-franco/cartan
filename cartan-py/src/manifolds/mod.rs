// ~/cartan/cartan-py/src/manifolds/mod.rs

//! Python-facing manifold wrappers.
//!
//! Each submodule wraps a cartan-manifolds type with numpy I/O and
//! dimension dispatch. The macros in `macros.rs` generate all trait
//! method wrappers so each manifold file is minimal boilerplate.

#[macro_use]
pub mod macros;

// Will be extended with pub mod euclidean; etc. in later tasks.

/// Register all manifold classes on the Python module.
pub fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    // Will add classes here as manifolds are added.
    let _ = m;
    Ok(())
}
