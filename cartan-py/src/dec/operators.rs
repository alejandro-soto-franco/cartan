// ~/cartan/cartan-py/src/dec/operators.rs

//! PyExteriorDerivative, PyHodgeStar, PyOperators, and free DEC functions.
//!
//! Wraps the cartan-dec operator types and free functions for the Python API.
//! All numpy arrays use row-major layout consistent with the rest of cartan-py.

use nalgebra::DVector;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use cartan_dec::{
    apply_divergence as dec_apply_divergence,
    apply_scalar_advection as dec_apply_scalar_advection,
    apply_tensor_divergence as dec_apply_tensor_divergence,
    apply_vector_advection as dec_apply_vector_advection,
    ExteriorDerivative, HodgeStar, Operators,
};
use cartan_manifolds::euclidean::Euclidean;

use crate::convert::{dmatrix_to_pyarray, dvector_to_pyarray};
use crate::dec::mesh::PyMesh;

// ---------------------------------------------------------------------------
// Helper: numpy 1D array -> DVector<f64>
// ---------------------------------------------------------------------------

fn array1_to_dvector(arr: &PyReadonlyArray1<'_, f64>, name: &str) -> PyResult<DVector<f64>> {
    let slice = arr
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name}: array must be contiguous")))?;
    Ok(DVector::from_column_slice(slice))
}

// ---------------------------------------------------------------------------
// PyExteriorDerivative
// ---------------------------------------------------------------------------

/// Discrete exterior derivative operators d0 and d1 for a triangular mesh.
///
/// The exterior derivative is a purely topological operator (metric-free).
/// It satisfies d1 @ d0 = 0 (exactness / nilpotency).
///
/// Attributes
/// ----------
/// d0 : ndarray, shape (n_boundaries, n_vertices)
///     Maps 0-forms (vertex scalars) to 1-forms (edge scalars).
/// d1 : ndarray, shape (n_simplices, n_boundaries)
///     Maps 1-forms (edge scalars) to 2-forms (triangle scalars).
///
/// # Examples
///
/// ```python
/// mesh = cartan.Mesh.unit_square_grid(5)
/// ext  = cartan.ExteriorDerivative(mesh)
/// assert ext.check_exactness() < 1e-14
/// ```
#[pyclass(name = "ExteriorDerivative")]
pub struct PyExteriorDerivative {
    pub(crate) inner: ExteriorDerivative,
}

#[pymethods]
impl PyExteriorDerivative {
    /// Construct the exterior derivative operators from a mesh.
    ///
    /// Parameters
    /// ----------
    /// mesh : Mesh
    ///     The triangular mesh.
    #[new]
    pub fn new(mesh: &PyMesh) -> Self {
        Self {
            inner: ExteriorDerivative::from_mesh(&mesh.inner),
        }
    }

    /// d0 matrix as a (n_boundaries, n_vertices) float64 numpy array.
    #[getter]
    pub fn d0<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        dmatrix_to_pyarray(py, &self.inner.d0)
    }

    /// d1 matrix as a (n_simplices, n_boundaries) float64 numpy array.
    #[getter]
    pub fn d1<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        dmatrix_to_pyarray(py, &self.inner.d1)
    }

    /// Check exactness: return max |d1 @ d0|.
    ///
    /// For a correct implementation this should be < 1e-14 (machine epsilon).
    pub fn check_exactness(&self) -> f64 {
        self.inner.check_exactness()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "ExteriorDerivative(d0={:?}, d1={:?})",
            self.inner.d0.shape(),
            self.inner.d1.shape()
        )
    }
}

// ---------------------------------------------------------------------------
// PyHodgeStar
// ---------------------------------------------------------------------------

/// Diagonal Hodge star operators for a 2D triangular mesh.
///
/// The Hodge star encodes the metric via primal/dual volume ratios.
/// All three stars are stored as 1D arrays (diagonal entries only).
///
/// Attributes
/// ----------
/// star0 : ndarray, shape (n_vertices,)
///     Diagonal of ⋆0 (dual cell areas, barycentric).
/// star1 : ndarray, shape (n_boundaries,)
///     Diagonal of ⋆1 (dual/primal edge length ratio).
/// star2 : ndarray, shape (n_simplices,)
///     Diagonal of ⋆2 (= 1 / triangle area).
///
/// # Examples
///
/// ```python
/// mesh  = cartan.Mesh.unit_square_grid(5)
/// hodge = cartan.HodgeStar(mesh)
/// assert np.all(hodge.star0 > 0)
/// ```
#[pyclass(name = "HodgeStar")]
pub struct PyHodgeStar {
    pub(crate) inner: HodgeStar,
}

#[pymethods]
impl PyHodgeStar {
    /// Construct the Hodge star diagonals from a flat mesh.
    ///
    /// Parameters
    /// ----------
    /// mesh : Mesh
    ///     The triangular mesh.
    #[new]
    pub fn new(mesh: &PyMesh) -> Self {
        Self {
            inner: HodgeStar::from_mesh(&mesh.inner, &Euclidean::<2>),
        }
    }

    /// Diagonal of ⋆0 as a (n_vertices,) float64 array.
    #[getter]
    pub fn star0<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector_to_pyarray(py, &self.inner.star0)
    }

    /// Diagonal of ⋆1 as a (n_boundaries,) float64 array.
    #[getter]
    pub fn star1<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector_to_pyarray(py, &self.inner.star1)
    }

    /// Diagonal of ⋆2 as a (n_simplices,) float64 array.
    #[getter]
    pub fn star2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector_to_pyarray(py, &self.inner.star2)
    }

    /// Inverse Hodge star ⋆0⁻¹ as a (n_vertices,) float64 array.
    #[getter]
    pub fn star0_inv<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector_to_pyarray(py, &self.inner.star0_inv())
    }

    /// Inverse Hodge star ⋆1⁻¹ as a (n_boundaries,) float64 array.
    #[getter]
    pub fn star1_inv<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector_to_pyarray(py, &self.inner.star1_inv())
    }

    /// Inverse Hodge star ⋆2⁻¹ as a (n_simplices,) float64 array.
    #[getter]
    pub fn star2_inv<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector_to_pyarray(py, &self.inner.star2_inv())
    }

    pub fn __repr__(&self) -> String {
        format!(
            "HodgeStar(star0.len={}, star1.len={}, star2.len={})",
            self.inner.star0.len(),
            self.inner.star1.len(),
            self.inner.star2.len()
        )
    }
}

// ---------------------------------------------------------------------------
// PyOperators
// ---------------------------------------------------------------------------

/// Assembled discrete differential operators for a flat 2D mesh.
///
/// Contains the Laplace-Beltrami, Bochner, and Lichnerowicz Laplacians,
/// plus the underlying ExteriorDerivative and HodgeStar for lower-level use.
///
/// Attributes
/// ----------
/// laplace_beltrami : ndarray, shape (n_vertices, n_vertices)
///     Scalar Laplace-Beltrami matrix (cotangent-weight Laplacian).
///
/// # Examples
///
/// ```python
/// mesh = cartan.Mesh.unit_square_grid(10)
/// ops  = cartan.Operators(mesh)
/// f    = np.ones(mesh.n_vertices())
/// Lf   = ops.apply_laplace_beltrami(f)  # should be ~0 for constant f
/// ```
#[pyclass(name = "Operators")]
pub struct PyOperators {
    pub(crate) inner: Operators<Euclidean<2>>,
}

#[pymethods]
impl PyOperators {
    /// Assemble all DEC operators from a flat mesh.
    ///
    /// Parameters
    /// ----------
    /// mesh : Mesh
    ///     The triangular mesh.
    #[new]
    pub fn new(mesh: &PyMesh) -> Self {
        Self {
            inner: Operators::from_mesh(&mesh.inner, &Euclidean::<2>),
        }
    }

    /// Scalar Laplace-Beltrami matrix as a (n_vertices, n_vertices) float64 array.
    #[getter]
    pub fn laplace_beltrami<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        dmatrix_to_pyarray(py, &self.inner.laplace_beltrami)
    }

    /// Apply the scalar Laplace-Beltrami operator to a 0-form.
    ///
    /// Parameters
    /// ----------
    /// f : ndarray, shape (n_vertices,)
    ///     Scalar field at vertices.
    ///
    /// Returns
    /// -------
    /// ndarray, shape (n_vertices,)
    ///     Δf at each vertex.
    pub fn apply_laplace_beltrami<'py>(
        &self,
        py: Python<'py>,
        f: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fv = array1_to_dvector(&f, "f")?;
        let result = self.inner.apply_laplace_beltrami(&fv);
        Ok(dvector_to_pyarray(py, &result))
    }

    /// Apply the Bochner (connection) Laplacian to a vector field.
    ///
    /// Input layout: [u_x[0..n_v], u_y[0..n_v]] (structure-of-arrays, 2*n_v).
    ///
    /// Parameters
    /// ----------
    /// u : ndarray, shape (2 * n_vertices,)
    ///     Vector field at vertices (x-components first, then y-components).
    ///
    /// Returns
    /// -------
    /// ndarray, shape (2 * n_vertices,)
    ///     (∇*∇) u at each vertex, same SoA layout.
    pub fn apply_bochner_laplacian<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let uv = array1_to_dvector(&u, "u")?;
        let result = self.inner.apply_bochner_laplacian(&uv, None);
        Ok(dvector_to_pyarray(py, &result))
    }

    /// Apply the Lichnerowicz Laplacian to a symmetric 2-tensor field.
    ///
    /// Input layout: [Q_xx[0..n_v], Q_xy[0..n_v], Q_yy[0..n_v]] (3*n_v).
    ///
    /// Parameters
    /// ----------
    /// q : ndarray, shape (3 * n_vertices,)
    ///     Symmetric 2-tensor field at vertices (SoA layout: xx, xy, yy).
    ///
    /// Returns
    /// -------
    /// ndarray, shape (3 * n_vertices,)
    ///     ΔL q at each vertex, same SoA layout.
    pub fn apply_lichnerowicz_laplacian<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let qv = array1_to_dvector(&q, "q")?;
        let result = self.inner.apply_lichnerowicz_laplacian(&qv, None);
        Ok(dvector_to_pyarray(py, &result))
    }

    pub fn __repr__(&self) -> String {
        let n = self.inner.laplace_beltrami.nrows();
        format!("Operators(n_vertices={})", n)
    }
}

// ---------------------------------------------------------------------------
// Free functions: advection and divergence
// ---------------------------------------------------------------------------

/// Apply the upwind scalar advection operator: (u · ∇) f.
///
/// Parameters
/// ----------
/// mesh : Mesh
///     The triangular mesh.
/// f : ndarray, shape (n_vertices,)
///     Scalar field at vertices.
/// u : ndarray, shape (2 * n_vertices,)
///     Velocity field at vertices, SoA layout [u_x..., u_y...].
///
/// Returns
/// -------
/// ndarray, shape (n_vertices,)
///     Advective time-derivative (u · ∇) f at each vertex.
#[pyfunction]
pub fn apply_scalar_advection<'py>(
    py: Python<'py>,
    mesh: &PyMesh,
    f: PyReadonlyArray1<'_, f64>,
    u: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let fv = array1_to_dvector(&f, "f")?;
    let uv = array1_to_dvector(&u, "u")?;
    let result = dec_apply_scalar_advection(&mesh.inner, &fv, &uv);
    Ok(dvector_to_pyarray(py, &result))
}

/// Apply the upwind vector advection operator: (u · ∇) q.
///
/// Parameters
/// ----------
/// mesh : Mesh
///     The triangular mesh.
/// q : ndarray, shape (2 * n_vertices,)
///     Vector field at vertices, SoA layout [q_x..., q_y...].
/// u : ndarray, shape (2 * n_vertices,)
///     Velocity field at vertices, SoA layout [u_x..., u_y...].
///
/// Returns
/// -------
/// ndarray, shape (2 * n_vertices,)
///     Advective time-derivative (u · ∇) q at each vertex, same SoA layout.
#[pyfunction]
pub fn apply_vector_advection<'py>(
    py: Python<'py>,
    mesh: &PyMesh,
    q: PyReadonlyArray1<'_, f64>,
    u: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let qv = array1_to_dvector(&q, "q")?;
    let uv = array1_to_dvector(&u, "u")?;
    let result = dec_apply_vector_advection(&mesh.inner, &qv, &uv);
    Ok(dvector_to_pyarray(py, &result))
}

/// Compute the discrete divergence of a vertex-based vector field.
///
/// Uses the DEC codifferential: div(u) = ⋆0⁻¹ d0ᵀ ⋆1 û.
///
/// Parameters
/// ----------
/// mesh : Mesh
///     The triangular mesh.
/// ops : Operators
///     Precomputed DEC operators (ext and hodge are extracted internally).
/// u : ndarray, shape (2 * n_vertices,)
///     Vector field at vertices, SoA layout [u_x..., u_y...].
///
/// Returns
/// -------
/// ndarray, shape (n_vertices,)
///     div(u) at each vertex.
#[pyfunction]
pub fn apply_divergence<'py>(
    py: Python<'py>,
    mesh: &PyMesh,
    ops: &PyOperators,
    u: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let uv = array1_to_dvector(&u, "u")?;
    let result = dec_apply_divergence(&mesh.inner, &ops.inner.ext, &ops.inner.hodge, &uv);
    Ok(dvector_to_pyarray(py, &result))
}

/// Compute the discrete divergence of a symmetric 2-tensor field.
///
/// Treats each column of the tensor as a vector field and takes divergence.
///
/// Parameters
/// ----------
/// mesh : Mesh
///     The triangular mesh.
/// ops : Operators
///     Precomputed DEC operators (ext and hodge are extracted internally).
/// t : ndarray, shape (3 * n_vertices,)
///     Symmetric 2-tensor at vertices, SoA layout [T_xx..., T_xy..., T_yy...].
///
/// Returns
/// -------
/// ndarray, shape (2 * n_vertices,)
///     div(T) at each vertex, SoA layout [div_x..., div_y...].
#[pyfunction]
pub fn apply_tensor_divergence<'py>(
    py: Python<'py>,
    mesh: &PyMesh,
    ops: &PyOperators,
    t: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let tv = array1_to_dvector(&t, "t")?;
    let result = dec_apply_tensor_divergence(&mesh.inner, &ops.inner.ext, &ops.inner.hodge, &tv);
    Ok(dvector_to_pyarray(py, &result))
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    m.add_class::<PyExteriorDerivative>()?;
    m.add_class::<PyHodgeStar>()?;
    m.add_class::<PyOperators>()?;
    m.add_function(pyo3::wrap_pyfunction!(apply_scalar_advection, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(apply_vector_advection, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(apply_divergence, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(apply_tensor_divergence, m)?)?;
    Ok(())
}
