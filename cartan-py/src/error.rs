// ~/cartan/cartan-py/src/error.rs

//! Python exception hierarchy for cartan errors.
//!
//! Maps CartanError (cartan-core) and DecError (cartan-dec) to typed
//! Python exceptions so callers can catch specific failure modes.

use pyo3::exceptions::{PyException, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::{create_exception, PyErr};

use cartan_core::CartanError;
use cartan_dec::DecError;

// ---------------------------------------------------------------------------
// Exception class definitions
// ---------------------------------------------------------------------------

// Base exception for all cartan errors.
create_exception!(cartan, CartanPyError, PyException);

// Raised when the logarithmic map fails near the cut locus.
create_exception!(cartan, CutLocusError, PyValueError);

// Raised when a matrix decomposition or numerical computation fails.
create_exception!(cartan, NumericalError, PyRuntimeError);

// Raised when a point or tangent vector fails a manifold constraint.
create_exception!(cartan, ValidationError, PyValueError);

// Raised when an optimizer does not converge or line search fails.
create_exception!(cartan, ConvergenceError, PyRuntimeError);

// Raised for errors from discrete exterior calculus operations.
create_exception!(cartan, DecPyError, PyRuntimeError);

// ---------------------------------------------------------------------------
// Converters
// ---------------------------------------------------------------------------

/// Convert a CartanError into a PyErr with the appropriate Python type.
pub fn cartan_err_to_py(e: CartanError) -> PyErr {
    match e {
        CartanError::CutLocus { message } => {
            CutLocusError::new_err(format!("cut locus: {message}"))
        }
        CartanError::NumericalFailure { operation, message } => {
            NumericalError::new_err(format!(
                "numerical failure in {operation}: {message}"
            ))
        }
        CartanError::NotOnManifold {
            constraint,
            violation,
        } => ValidationError::new_err(format!(
            "point not on manifold: {constraint} violated by {violation}"
        )),
        CartanError::NotInTangentSpace {
            constraint,
            violation,
        } => ValidationError::new_err(format!(
            "tangent vector not in tangent space: {constraint} violated by {violation}"
        )),
        CartanError::LineSearchFailed { steps_tried } => {
            ConvergenceError::new_err(format!(
                "line search failed after {steps_tried} steps"
            ))
        }
        CartanError::ConvergenceFailure {
            iterations,
            gradient_norm,
        } => ConvergenceError::new_err(format!(
            "optimizer did not converge after {iterations} iterations \
             (gradient norm: {gradient_norm:.2e})"
        )),
    }
}

/// Convert a DecError into a PyErr with the DecPyError Python type.
#[allow(dead_code)]
pub fn dec_err_to_py(e: DecError) -> PyErr {
    DecPyError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Add all exception classes to the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CartanPyError", m.py().get_type::<CartanPyError>())?;
    m.add("CutLocusError", m.py().get_type::<CutLocusError>())?;
    m.add("NumericalError", m.py().get_type::<NumericalError>())?;
    m.add("ValidationError", m.py().get_type::<ValidationError>())?;
    m.add("ConvergenceError", m.py().get_type::<ConvergenceError>())?;
    m.add("DecPyError", m.py().get_type::<DecPyError>())?;
    Ok(())
}
