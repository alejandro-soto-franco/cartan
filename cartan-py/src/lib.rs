use pyo3::prelude::*;

pub(crate) mod convert;
mod error;
mod manifolds;
mod optim;

/// cartan: Riemannian geometry, manifold optimization, and geodesic computation.
#[pymodule]
fn cartan(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    error::register(m)?;
    manifolds::register(m)?;
    optim::register(m)?;
    Ok(())
}
