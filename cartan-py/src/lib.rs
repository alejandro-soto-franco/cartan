use pyo3::prelude::*;

/// cartan: Riemannian geometry, manifold optimization, and geodesic computation.
#[pymodule]
fn cartan(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    Ok(())
}
