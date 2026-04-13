// ~/cartan/cartan-py/src/stochastic.rs

//! Python bindings for `cartan-stochastic`.
//!
//! Exposes three primitives:
//!
//! - `stochastic_bm_on_sphere(dim, p0, n_steps, dt, seed)`: Brownian motion
//!   on `S^{dim}` via the Eells-Elworthy-Malliavin construction.
//! - `stochastic_bm_on_spd(n, p0, n_steps, dt, seed)`: Brownian motion on
//!   `SPD(n)` with the affine-invariant metric.
//! - `wishart_step(x, shape, dt, seed)`: one Itô-Euler step of the Wishart
//!   SDE.
//!
//! Runtime dim / n arguments are dispatched to const-generic cartan types
//! via per-size match arms. Supported intrinsic sphere dimensions: 1..=9
//! (ambient 2..=10). Supported SPD sizes: 2..=5.

use cartan_core::Real;
use cartan_manifolds::{Sphere, Spd};
use cartan_stochastic::{random_frame_at, stochastic_development, wishart_step as ws_step};
use nalgebra::{SMatrix, SVector};
use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Brownian motion on the unit sphere `S^{intrinsic_dim}`.
///
/// Returns an `(n_steps + 1, intrinsic_dim + 1)` ndarray. Row `i` is the
/// path point at time `i * dt`. The starting point `p0` must have shape
/// `(intrinsic_dim + 1,)` and unit norm.
#[pyfunction]
#[pyo3(signature = (intrinsic_dim, p0, n_steps, dt, seed=0xC0FFEE))]
fn stochastic_bm_on_sphere<'py>(
    py: Python<'py>,
    intrinsic_dim: usize,
    p0: PyReadonlyArray1<'py, Real>,
    n_steps: usize,
    dt: Real,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<Real>>> {
    let ambient = intrinsic_dim + 1;
    let p0_slice = p0.as_slice()?;
    if p0_slice.len() != ambient {
        return Err(PyValueError::new_err(format!(
            "p0 must have length {ambient} (intrinsic_dim + 1), got {}",
            p0_slice.len()
        )));
    }
    macro_rules! dispatch_sphere {
        ($($n:literal),+) => {
            match ambient {
                $(
                    $n => sphere_bm_impl::<$n>(p0_slice, n_steps, dt, seed),
                )+
                _ => Err(PyValueError::new_err(format!(
                    "unsupported sphere ambient dimension {ambient}; supported: 2..=10"
                ))),
            }
        };
    }
    let path = dispatch_sphere!(2, 3, 4, 5, 6, 7, 8, 9, 10)?;
    // Flatten path to a 2D array.
    let rows = path.len();
    let cols = ambient;
    let mut flat = Vec::with_capacity(rows * cols);
    for row in &path {
        flat.extend_from_slice(row);
    }
    let arr = Array2::from_shape_vec((rows, cols), flat).map_err(|e| {
        PyValueError::new_err(format!("reshape failed: {e}"))
    })?;
    Ok(arr.into_pyarray(py))
}

fn sphere_bm_impl<const N: usize>(
    p0: &[Real],
    n_steps: usize,
    dt: Real,
    seed: u64,
) -> PyResult<Vec<Vec<Real>>> {
    let m: Sphere<N> = Sphere::<N>;
    let mut rng = StdRng::seed_from_u64(seed);
    let p0_vec: SVector<Real, N> = SVector::from_column_slice(p0);
    // Sanity: the starting point should be on the manifold to tolerance.
    if (p0_vec.norm() - 1.0).abs() > 1e-6 {
        return Err(PyValueError::new_err(format!(
            "p0 is not on the sphere: ||p0|| = {}, expected 1.0",
            p0_vec.norm()
        )));
    }
    let frame = random_frame_at(&m, &p0_vec, &mut rng)
        .map_err(|e| PyValueError::new_err(format!("frame construction failed: {e:?}")))?;
    let result = stochastic_development(&m, &p0_vec, frame, n_steps, dt, &mut rng, 1e-10)
        .map_err(|e| PyValueError::new_err(format!("development failed: {e:?}")))?;
    Ok(result.path.into_iter().map(|v| v.iter().copied().collect()).collect())
}

/// Brownian motion on `SPD(n)` with the affine-invariant metric.
///
/// Returns an `(n_steps + 1, n, n)` ndarray of matrices.
#[pyfunction]
#[pyo3(signature = (n, p0, n_steps, dt, seed=0xC0FFEE))]
fn stochastic_bm_on_spd<'py>(
    py: Python<'py>,
    n: usize,
    p0: PyReadonlyArray2<'py, Real>,
    n_steps: usize,
    dt: Real,
    seed: u64,
) -> PyResult<Bound<'py, PyArray3<Real>>> {
    let shape = p0.shape();
    if shape != [n, n] {
        return Err(PyValueError::new_err(format!(
            "p0 must be {n}x{n}, got {:?}",
            shape
        )));
    }
    let p0_slice = p0.as_slice()?;
    macro_rules! dispatch_spd {
        ($($nn:literal),+) => {
            match n {
                $(
                    $nn => spd_bm_impl::<$nn>(p0_slice, n_steps, dt, seed),
                )+
                _ => Err(PyValueError::new_err(format!(
                    "unsupported SPD dimension {n}; supported: 2..=5"
                ))),
            }
        };
    }
    let path = dispatch_spd!(2, 3, 4, 5)?;
    let rows = path.len();
    let mut flat = Vec::with_capacity(rows * n * n);
    for mat in &path {
        flat.extend_from_slice(mat);
    }
    let arr =
        Array3::from_shape_vec((rows, n, n), flat).map_err(|e| {
            PyValueError::new_err(format!("reshape failed: {e}"))
        })?;
    Ok(arr.into_pyarray(py))
}

fn spd_bm_impl<const N: usize>(
    p0_flat: &[Real],
    n_steps: usize,
    dt: Real,
    seed: u64,
) -> PyResult<Vec<Vec<Real>>> {
    let m: Spd<N> = Spd::<N>;
    let mut rng = StdRng::seed_from_u64(seed);
    let p0 = SMatrix::<Real, N, N>::from_row_slice(p0_flat);
    let frame = random_frame_at(&m, &p0, &mut rng)
        .map_err(|e| PyValueError::new_err(format!("frame construction failed: {e:?}")))?;
    let result = stochastic_development(&m, &p0, frame, n_steps, dt, &mut rng, 1e-8)
        .map_err(|e| PyValueError::new_err(format!("development failed: {e:?}")))?;
    Ok(result
        .path
        .into_iter()
        .map(|mat| {
            // Row-major flatten to match PyArray3 layout.
            let mut row = Vec::with_capacity(N * N);
            for i in 0..N {
                for j in 0..N {
                    row.push(mat[(i, j)]);
                }
            }
            row
        })
        .collect())
}

/// One Itô-Euler step of the Wishart process on `SPD(n)`.
///
/// `x` is an `n×n` SPD matrix. Returns the next state.
#[pyfunction]
#[pyo3(signature = (x, shape_param, dt, seed=0xC0FFEE))]
fn wishart_step<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, Real>,
    shape_param: Real,
    dt: Real,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<Real>>> {
    let sh = x.shape();
    if sh.len() != 2 || sh[0] != sh[1] {
        return Err(PyValueError::new_err(format!("x must be square, got {sh:?}")));
    }
    let n = sh[0];
    let x_slice = x.as_slice()?;
    macro_rules! dispatch_ws {
        ($($nn:literal),+) => {
            match n {
                $(
                    $nn => {
                        let mut rng = StdRng::seed_from_u64(seed);
                        let xm = SMatrix::<Real, $nn, $nn>::from_row_slice(x_slice);
                        let next = ws_step(&xm, shape_param, dt, &mut rng);
                        let mut flat = Vec::with_capacity($nn * $nn);
                        for i in 0..$nn { for j in 0..$nn { flat.push(next[(i, j)]); } }
                        Ok(flat)
                    },
                )+
                _ => Err(PyValueError::new_err(format!(
                    "unsupported SPD dimension {n}; supported: 2..=5"
                ))),
            }
        };
    }
    let flat: Vec<Real> = dispatch_ws!(2, 3, 4, 5)?;
    let arr = Array2::from_shape_vec((n, n), flat).map_err(|e| {
        PyValueError::new_err(format!("reshape failed: {e}"))
    })?;
    Ok(arr.into_pyarray(py))
}

pub(crate) fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::types::PyModuleMethods;
    m.add_function(wrap_pyfunction!(stochastic_bm_on_sphere, m)?)?;
    m.add_function(wrap_pyfunction!(stochastic_bm_on_spd, m)?)?;
    m.add_function(wrap_pyfunction!(wishart_step, m)?)?;
    Ok(())
}
