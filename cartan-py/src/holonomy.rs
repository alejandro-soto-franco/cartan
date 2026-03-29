// ~/cartan/cartan-py/src/holonomy.rs

//! Python bindings for holonomy-based nematic defect detection.
//!
//! All functions operate on SO(3) frames represented as 3x3 rotation matrices
//! (numpy arrays of shape (3, 3) and dtype float64).

use nalgebra::SMatrix;
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use cartan_core::Real;

use crate::convert::{arr_to_smatrix, smatrix_to_pyarray};

// ---------------------------------------------------------------------------
// Edge transition
// ---------------------------------------------------------------------------

/// Compute the D2 gauge-fixed edge transition matrix between two SO(3) frames.
///
/// Parameters
/// ----------
/// r_src : array of shape (3, 3)
///     Source SO(3) frame.
/// r_dst : array of shape (3, 3)
///     Destination SO(3) frame.
///
/// Returns
/// -------
/// T : array of shape (3, 3)
///     The transition matrix T = r_src^T * (r_dst * g), where g in D2 minimises
///     the distance from r_src to r_dst * g.
#[pyfunction]
fn edge_transition<'py>(
    py: Python<'py>,
    r_src: PyReadonlyArrayDyn<'py, f64>,
    r_dst: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<PyObject> {
    let src = arr_to_smatrix::<3, 3>(r_src, "r_src")?;
    let dst = arr_to_smatrix::<3, 3>(r_dst, "r_dst")?;
    let result = cartan_geo::edge_transition(&src, &dst);
    Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
}

// ---------------------------------------------------------------------------
// Loop holonomy
// ---------------------------------------------------------------------------

/// Compute holonomy around an ordered loop of SO(3) frames.
///
/// Parameters
/// ----------
/// frames : list of arrays, each of shape (3, 3)
///     Ordered list of SO(3) frames forming a closed loop. Must have at least
///     2 frames. The loop is closed implicitly: the last edge goes from
///     frames[-1] back to frames[0].
///
/// Returns
/// -------
/// H : array of shape (3, 3)
///     The holonomy matrix (product of all edge transition matrices around the loop).
///
/// Raises
/// ------
/// ValueError
///     If fewer than 2 frames are provided.
#[pyfunction]
fn loop_holonomy<'py>(
    py: Python<'py>,
    frames: Vec<PyReadonlyArrayDyn<'py, f64>>,
) -> PyResult<PyObject> {
    if frames.len() < 2 {
        return Err(PyValueError::new_err(
            "loop_holonomy requires at least 2 frames",
        ));
    }
    let mats: Vec<SMatrix<Real, 3, 3>> = frames
        .into_iter()
        .enumerate()
        .map(|(i, arr)| arr_to_smatrix::<3, 3>(arr, &format!("frames[{i}]")))
        .collect::<PyResult<_>>()?;
    let result = cartan_geo::loop_holonomy(&mats);
    Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
}

// ---------------------------------------------------------------------------
// Scalar measures
// ---------------------------------------------------------------------------

/// Frobenius deviation from identity: ||H - I||_F.
///
/// Parameters
/// ----------
/// hol : array of shape (3, 3)
///     An SO(3) holonomy matrix.
///
/// Returns
/// -------
/// float
///     The Frobenius norm of (H - I). Zero for the identity; approximately 2*sqrt(2)
///     for a pi-rotation (half-integer disclination).
#[pyfunction]
fn holonomy_deviation(hol: PyReadonlyArrayDyn<'_, f64>) -> PyResult<f64> {
    let h = arr_to_smatrix::<3, 3>(hol, "hol")?;
    Ok(cartan_geo::holonomy_deviation(&h))
}

/// Rotation angle of a 3x3 SO(3) matrix, in [0, pi].
///
/// Uses the formula theta = arccos((tr(H) - 1) / 2), clamped to avoid
/// floating-point domain errors.
///
/// Parameters
/// ----------
/// hol : array of shape (3, 3)
///     An SO(3) matrix.
///
/// Returns
/// -------
/// float
///     Rotation angle in radians, in [0, pi].
#[pyfunction]
fn rotation_angle(hol: PyReadonlyArrayDyn<'_, f64>) -> PyResult<f64> {
    let h = arr_to_smatrix::<3, 3>(hol, "hol")?;
    Ok(cartan_geo::rotation_angle(&h))
}

/// Check if a holonomy matrix indicates a half-integer disclination.
///
/// Returns True if the rotation angle of `hol` exceeds `threshold`.
/// The default threshold is pi/2, which is the midpoint between "no defect"
/// (angle = 0) and "half-integer disclination" (angle = pi).
///
/// Parameters
/// ----------
/// hol : array of shape (3, 3)
///     An SO(3) holonomy matrix.
/// threshold : float, optional
///     Rotation angle threshold in radians (default pi/2 = 1.5707963...).
///
/// Returns
/// -------
/// bool
///     True if the holonomy angle exceeds the threshold.
#[pyfunction]
#[pyo3(signature = (hol, threshold = 1.5707963267948966))]
fn is_half_disclination(hol: PyReadonlyArrayDyn<'_, f64>, threshold: f64) -> PyResult<bool> {
    let h = arr_to_smatrix::<3, 3>(hol, "hol")?;
    Ok(cartan_geo::is_half_disclination(&h, threshold))
}

// ---------------------------------------------------------------------------
// Disclination type and scan
// ---------------------------------------------------------------------------

/// A detected disclination in a 2D frame field.
///
/// Attributes
/// ----------
/// plaquette : tuple of (int, int)
///     Grid coordinates (i, j) of the plaquette lower-left corner.
/// angle : float
///     Rotation angle of the plaquette holonomy in radians.
/// holonomy : array of shape (3, 3)
///     The holonomy matrix of the plaquette loop.
#[pyclass(name = "Disclination")]
#[derive(Debug, Clone)]
pub struct PyDisclination {
    #[pyo3(get)]
    plaquette: (usize, usize),
    #[pyo3(get)]
    angle: f64,
    /// Column-major storage of the 3x3 holonomy matrix (nalgebra's internal layout).
    holonomy_data: Vec<f64>,
}

#[pymethods]
impl PyDisclination {
    /// The holonomy matrix of this disclination as a (3, 3) numpy array.
    #[getter]
    fn holonomy<'py>(&self, py: Python<'py>) -> PyObject {
        let h = SMatrix::<Real, 3, 3>::from_column_slice(&self.holonomy_data);
        smatrix_to_pyarray(py, &h).into_any().unbind()
    }

    fn __repr__(&self) -> String {
        format!(
            "Disclination(plaquette=({}, {}), angle={:.4})",
            self.plaquette.0, self.plaquette.1, self.angle
        )
    }
}

/// Scan a 2D grid of SO(3) frames for topological disclinations.
///
/// For each elementary plaquette (i, j), the counter-clockwise loop
/// (i,j) -> (i+1,j) -> (i+1,j+1) -> (i,j+1) -> (i,j) is traversed and the
/// holonomy computed. Plaquettes whose holonomy rotation angle exceeds
/// `threshold` are returned as `Disclination` objects.
///
/// Parameters
/// ----------
/// frames : list of arrays, each of shape (3, 3)
///     Ordered frame field with row-major indexing: frames[i * ny + j] is the
///     SO(3) frame at grid vertex (i, j). Must have exactly nx * ny elements.
/// nx : int
///     Number of grid points in the x direction.
/// ny : int
///     Number of grid points in the y direction.
/// threshold : float, optional
///     Rotation angle threshold for disclination detection (default pi/2).
///
/// Returns
/// -------
/// list of Disclination
///     All plaquettes whose holonomy angle exceeds the threshold.
///
/// Raises
/// ------
/// ValueError
///     If len(frames) != nx * ny.
#[pyfunction]
#[pyo3(name = "scan_disclinations", signature = (frames, nx, ny, threshold = 1.5707963267948966))]
fn scan_disclinations_py(
    frames: Vec<PyReadonlyArrayDyn<'_, f64>>,
    nx: usize,
    ny: usize,
    threshold: f64,
) -> PyResult<Vec<PyDisclination>> {
    let n = nx * ny;
    if frames.len() != n {
        return Err(PyValueError::new_err(format!(
            "scan_disclinations: expected {} frames ({}*{}), got {}",
            n,
            nx,
            ny,
            frames.len()
        )));
    }
    let mats: Vec<SMatrix<Real, 3, 3>> = frames
        .into_iter()
        .enumerate()
        .map(|(i, arr)| arr_to_smatrix::<3, 3>(arr, &format!("frames[{i}]")))
        .collect::<PyResult<_>>()?;

    let results = cartan_geo::scan_disclinations(&mats, nx, ny, threshold);
    Ok(results
        .into_iter()
        .map(|d| {
            // Store holonomy in column-major order (nalgebra iterator order)
            // for round-trip via SMatrix::from_column_slice.
            let holonomy_data: Vec<f64> = d.holonomy.iter().copied().collect();
            PyDisclination {
                plaquette: d.plaquette,
                angle: d.angle,
                holonomy_data,
            }
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDisclination>()?;
    m.add_function(wrap_pyfunction!(edge_transition, m)?)?;
    m.add_function(wrap_pyfunction!(loop_holonomy, m)?)?;
    m.add_function(wrap_pyfunction!(holonomy_deviation, m)?)?;
    m.add_function(wrap_pyfunction!(rotation_angle, m)?)?;
    m.add_function(wrap_pyfunction!(is_half_disclination, m)?)?;
    m.add_function(wrap_pyfunction!(scan_disclinations_py, m)?)?;
    Ok(())
}
