// ~/cartan/cartan-py/src/optim.rs

//! Riemannian optimizer dispatch for Python.
//!
//! Exposes `minimize_rgd` as a Python function that accepts any supported
//! manifold wrapper and two Python callables (cost, gradient). Dispatch is
//! resolved at runtime via PyO3 downcasting to the concrete manifold type.
//!
//! ## Supported manifolds
//!
//! - `Euclidean(n)`  -- n = 1..10 (vector points)
//! - `Sphere(dim)`   -- intrinsic dim = 1..9 (vector points in ambient space)
//! - `SPD(n)`        -- n = 2..8 (matrix points)
//! - `SO(n)`         -- n = 2..4 (matrix points)
//! - `Corr(n)`       -- n = 2..8 (matrix points)
//! - `QTensor3()`    -- fixed 3x3 (matrix points)
//!
//! SE and Grassmann are not yet supported (complex point types / two-param dispatch).

use pyo3::prelude::*;
use pyo3::exceptions::{PyTypeError, PyValueError};
use numpy::PyReadonlyArrayDyn;

use crate::manifolds::euclidean::PyEuclidean;
use crate::manifolds::sphere::PySphere;
use crate::manifolds::spd::PySpd;
use crate::manifolds::so::PySo;
use crate::manifolds::corr::PyCorr;
use crate::manifolds::qtensor::PyQTensor3;

// ---------------------------------------------------------------------------
// Python-visible result type
// ---------------------------------------------------------------------------

/// Result returned by Riemannian optimizers.
///
/// All fields are read-only attributes accessible from Python.
#[pyclass(name = "OptResult")]
#[derive(Debug)]
pub struct PyOptResult {
    /// Final iterate on the manifold (numpy array).
    #[pyo3(get)]
    pub point: PyObject,
    /// Cost function value at the final iterate.
    #[pyo3(get)]
    pub value: f64,
    /// Riemannian gradient norm at the final iterate.
    #[pyo3(get)]
    pub grad_norm: f64,
    /// Total number of iterations executed.
    #[pyo3(get)]
    pub iterations: usize,
    /// Whether the solver reached the gradient tolerance before max iterations.
    #[pyo3(get)]
    pub converged: bool,
}

#[pymethods]
impl PyOptResult {
    fn __repr__(&self) -> String {
        format!(
            "OptResult(value={:.6e}, grad_norm={:.6e}, iterations={}, converged={})",
            self.value, self.grad_norm, self.iterations, self.converged
        )
    }
}

// ---------------------------------------------------------------------------
// Dispatch macros
// ---------------------------------------------------------------------------

/// Dispatch RGD for vector-point manifolds (SVector<f64, N>).
///
/// Matches `$dim` against the list of supported const-generic values `[$($N:literal),+]`,
/// constructs the Rust manifold value `cartan_manifolds::$mtype::<$N>`, converts
/// `$x0` from numpy, runs `minimize_rgd`, and returns `PyOptResult`.
macro_rules! dispatch_rgd_vector {
    ($py:expr, $cost:expr, $grad:expr, $x0:expr, $config:expr, $mtype:ident, $dim:expr, [$($N:literal),+]) => {
        match $dim {
            $($N => {
                let mf = cartan_manifolds::$mtype::<$N>;
                let x0_arr: PyReadonlyArrayDyn<f64> = $x0.extract()?;
                let x0_pt = $crate::convert::arr_to_svector::<$N>(x0_arr, "x0")?;

                // Capture py, cost, grad by shared reference. The GIL is held
                // for the entire duration of minimize_rgd (we are inside a
                // #[pyfunction]), so these borrows are valid.
                let cost_fn = |p: &nalgebra::SVector<f64, $N>| -> f64 {
                    let p_py = $crate::convert::svector_to_pyarray($py, p);
                    $cost
                        .call1((p_py,))
                        .and_then(|r| r.extract::<f64>())
                        .expect("cost function must return a float")
                };

                let grad_fn = |p: &nalgebra::SVector<f64, $N>| -> nalgebra::SVector<f64, $N> {
                    let p_py = $crate::convert::svector_to_pyarray($py, p);
                    let result = $grad.call1((p_py,)).expect("grad function failed");
                    let arr: PyReadonlyArrayDyn<f64> =
                        result.extract().expect("grad must return a numpy array");
                    $crate::convert::arr_to_svector::<$N>(arr, "grad_result")
                        .expect("grad output shape mismatch")
                };

                let res = cartan_optim::minimize_rgd(&mf, cost_fn, grad_fn, x0_pt, $config);
                Ok(PyOptResult {
                    point: $crate::convert::svector_to_pyarray($py, &res.point)
                        .into_any()
                        .unbind(),
                    value: res.value,
                    grad_norm: res.grad_norm,
                    iterations: res.iterations,
                    converged: res.converged,
                })
            },)+
            _ => Err(PyValueError::new_err(format!(
                "{}: unsupported dimension {}",
                stringify!($mtype),
                $dim
            ))),
        }
    };
}

/// Dispatch RGD for matrix-point manifolds (SMatrix<f64, N, N>).
macro_rules! dispatch_rgd_matrix {
    ($py:expr, $cost:expr, $grad:expr, $x0:expr, $config:expr, $mtype:ident, $dim:expr, [$($N:literal),+]) => {
        match $dim {
            $($N => {
                let mf = cartan_manifolds::$mtype::<$N>;
                let x0_arr: PyReadonlyArrayDyn<f64> = $x0.extract()?;
                let x0_pt = $crate::convert::arr_to_smatrix::<$N, $N>(x0_arr, "x0")?;

                let cost_fn = |p: &nalgebra::SMatrix<f64, $N, $N>| -> f64 {
                    let p_py = $crate::convert::smatrix_to_pyarray($py, p);
                    $cost
                        .call1((p_py,))
                        .and_then(|r| r.extract::<f64>())
                        .expect("cost function must return a float")
                };

                let grad_fn =
                    |p: &nalgebra::SMatrix<f64, $N, $N>| -> nalgebra::SMatrix<f64, $N, $N> {
                        let p_py = $crate::convert::smatrix_to_pyarray($py, p);
                        let result = $grad.call1((p_py,)).expect("grad function failed");
                        let arr: PyReadonlyArrayDyn<f64> =
                            result.extract().expect("grad must return a numpy array");
                        $crate::convert::arr_to_smatrix::<$N, $N>(arr, "grad_result")
                            .expect("grad output shape mismatch")
                    };

                let res = cartan_optim::minimize_rgd(&mf, cost_fn, grad_fn, x0_pt, $config);
                Ok(PyOptResult {
                    point: $crate::convert::smatrix_to_pyarray($py, &res.point)
                        .into_any()
                        .unbind(),
                    value: res.value,
                    grad_norm: res.grad_norm,
                    iterations: res.iterations,
                    converged: res.converged,
                })
            },)+
            _ => Err(PyValueError::new_err(format!(
                "{}: unsupported dimension {}",
                stringify!($mtype),
                $dim
            ))),
        }
    };
}

// ---------------------------------------------------------------------------
// Public pyfunction
// ---------------------------------------------------------------------------

/// Run Riemannian Gradient Descent on the given manifold.
///
/// Parameters
/// ----------
/// manifold : cartan manifold object
///     Any of: `Euclidean`, `Sphere`, `SPD`, `SO`, `Corr`, `QTensor3`.
/// cost : callable
///     `cost(p) -> float` — the objective function.
/// grad : callable
///     `grad(p) -> array` — the Riemannian gradient at `p`.
///     Must return an array with the same shape as `p`.
/// x0 : array_like
///     Initial point on the manifold (numpy array).
/// max_iters : int, optional
///     Maximum number of iterations (default 1000).
/// grad_tol : float, optional
///     Stop when the Riemannian gradient norm is below this value (default 1e-6).
/// init_step : float, optional
///     Initial Armijo step size (default 1.0).
/// armijo_c : float, optional
///     Sufficient decrease constant (default 1e-4).
/// armijo_beta : float, optional
///     Backtracking factor, must be in (0, 1) (default 0.5).
/// max_ls_iters : int, optional
///     Maximum backtracking steps per iteration (default 50).
///
/// Returns
/// -------
/// OptResult
///     Object with attributes: `point`, `value`, `grad_norm`, `iterations`, `converged`.
#[pyfunction]
#[pyo3(
    signature = (
        manifold,
        cost,
        grad,
        x0,
        max_iters = 1000,
        grad_tol = 1e-6,
        init_step = 1.0,
        armijo_c = 1e-4,
        armijo_beta = 0.5,
        max_ls_iters = 50,
    )
)]
pub fn minimize_rgd(
    py: Python<'_>,
    manifold: &Bound<'_, PyAny>,
    cost: &Bound<'_, PyAny>,
    grad: &Bound<'_, PyAny>,
    x0: &Bound<'_, PyAny>,
    max_iters: usize,
    grad_tol: f64,
    init_step: f64,
    armijo_c: f64,
    armijo_beta: f64,
    max_ls_iters: usize,
) -> PyResult<PyOptResult> {
    let config = cartan_optim::RGDConfig {
        max_iters,
        grad_tol,
        init_step,
        armijo_c,
        armijo_beta,
        max_ls_iters,
    };

    // --- Euclidean ---
    if let Ok(m) = manifold.downcast::<PyEuclidean>() {
        let dim = m.borrow().n;
        return dispatch_rgd_vector!(
            py, cost, grad, x0, &config,
            Euclidean, dim,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        );
    }

    // --- Sphere (field is ambient_n, i.e. the const generic) ---
    if let Ok(m) = manifold.downcast::<PySphere>() {
        let dim = m.borrow().ambient_n;
        return dispatch_rgd_vector!(
            py, cost, grad, x0, &config,
            Sphere, dim,
            [2, 3, 4, 5, 6, 7, 8, 9, 10]
        );
    }

    // --- SPD ---
    if let Ok(m) = manifold.downcast::<PySpd>() {
        let dim = m.borrow().n;
        return dispatch_rgd_matrix!(
            py, cost, grad, x0, &config,
            Spd, dim,
            [2, 3, 4, 5, 6, 7, 8]
        );
    }

    // --- SO ---
    if let Ok(m) = manifold.downcast::<PySo>() {
        let dim = m.borrow().n;
        return dispatch_rgd_matrix!(
            py, cost, grad, x0, &config,
            SpecialOrthogonal, dim,
            [2, 3, 4]
        );
    }

    // --- Corr ---
    if let Ok(m) = manifold.downcast::<PyCorr>() {
        let dim = m.borrow().n;
        return dispatch_rgd_matrix!(
            py, cost, grad, x0, &config,
            Corr, dim,
            [2, 3, 4, 5, 6, 7, 8]
        );
    }

    // --- QTensor3 (fixed 3x3, no dispatch needed) ---
    if manifold.downcast::<PyQTensor3>().is_ok() {
        use cartan_manifolds::qtensor::QTensor3;

        let mf = QTensor3;
        let x0_arr: PyReadonlyArrayDyn<f64> = x0.extract()?;
        let x0_pt = crate::convert::arr_to_smatrix::<3, 3>(x0_arr, "x0")?;

        let cost_fn = |p: &nalgebra::SMatrix<f64, 3, 3>| -> f64 {
            let p_py = crate::convert::smatrix_to_pyarray(py, p);
            cost.call1((p_py,))
                .and_then(|r| r.extract::<f64>())
                .expect("cost function must return a float")
        };

        let grad_fn =
            |p: &nalgebra::SMatrix<f64, 3, 3>| -> nalgebra::SMatrix<f64, 3, 3> {
                let p_py = crate::convert::smatrix_to_pyarray(py, p);
                let result = grad.call1((p_py,)).expect("grad function failed");
                let arr: PyReadonlyArrayDyn<f64> =
                    result.extract().expect("grad must return a numpy array");
                crate::convert::arr_to_smatrix::<3, 3>(arr, "grad_result")
                    .expect("grad output shape mismatch")
            };

        let res = cartan_optim::minimize_rgd(&mf, cost_fn, grad_fn, x0_pt, &config);
        return Ok(PyOptResult {
            point: crate::convert::smatrix_to_pyarray(py, &res.point)
                .into_any()
                .unbind(),
            value: res.value,
            grad_norm: res.grad_norm,
            iterations: res.iterations,
            converged: res.converged,
        });
    }

    Err(PyTypeError::new_err(
        "minimize_rgd: unsupported manifold type. \
         Supported: Euclidean, Sphere, SPD, SO, Corr, QTensor3.",
    ))
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register the optim submodule items onto the root `cartan` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOptResult>()?;
    m.add_function(wrap_pyfunction!(minimize_rgd, m)?)?;
    Ok(())
}
