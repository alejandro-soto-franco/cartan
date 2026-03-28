// ~/cartan/cartan-py/src/geo.rs

//! Python bindings for `cartan_geo`: Geodesic, CurvatureQuery, and Jacobi field integration.
//!
//! ## Design: type-erased ManifoldTag
//!
//! `cartan_geo::Geodesic<'a, M>` and `CurvatureQuery<'a, M>` both borrow the manifold
//! by reference. PyO3 `#[pyclass]` types cannot hold borrows, so instead we store:
//!
//! - A `ManifoldTag` enum that captures the manifold kind and dimension.
//! - `base_data` / `velocity_data` / `point_data` as `Vec<f64>` in nalgebra
//!   column-major order (the natural layout from `.iter().copied().collect()`).
//!
//! On every method call we reconstruct the concrete Rust type, build the geodesic
//! or curvature query on the stack, and dispatch. This is a small allocation per
//! call but avoids all lifetime headaches.
//!
//! ## Storage convention for matrix manifolds
//!
//! Numpy arrays arrive in row-major order. We convert to nalgebra using
//! `SMatrix::from_row_slice` (see `convert::arr_to_smatrix`), then flatten
//! via `.iter().copied().collect()` which traverses column-major order.
//! On reconstruction we use `SMatrix::from_column_slice` to recover the same matrix.

use pyo3::prelude::*;
use pyo3::exceptions::{PyTypeError, PyValueError};
use numpy::PyReadonlyArrayDyn;

use cartan_core::Real;

use crate::manifolds::euclidean::PyEuclidean;
use crate::manifolds::sphere::PySphere;
use crate::manifolds::spd::PySpd;
use crate::manifolds::so::PySo;
use crate::manifolds::corr::PyCorr;
use crate::manifolds::qtensor::PyQTensor3;

// ---------------------------------------------------------------------------
// ManifoldTag: carries type identity at runtime
// ---------------------------------------------------------------------------

/// Runtime tag that identifies a manifold type and its dimension.
///
/// Vector manifolds store the ambient or intrinsic size. Matrix manifolds
/// store the matrix side-length N so that points are N x N.
#[derive(Debug, Clone)]
enum ManifoldTag {
    /// R^N (Euclidean): n = ambient = intrinsic dimension.
    EuclideanVec(usize),
    /// S^(ambient-1) (Sphere): ambient_n is the embedding dimension.
    SphereVec(usize),
    /// SPD(N) (Symmetric Positive Definite): n x n matrices, n = 2..8.
    SpdMat(usize),
    /// SO(N) (Special Orthogonal): n x n matrices, n = 2..4.
    SoMat(usize),
    /// Corr(N) (Correlation): n x n matrices, n = 2..8.
    CorrMat(usize),
    /// QTensor3: fixed 3x3 symmetric traceless matrices.
    QTensor3Mat,
}

/// Identify the manifold type from a Python object by downcasting.
fn identify_manifold(manifold: &Bound<'_, PyAny>) -> PyResult<ManifoldTag> {
    if let Ok(m) = manifold.downcast::<PyEuclidean>() {
        return Ok(ManifoldTag::EuclideanVec(m.borrow().n));
    }
    if let Ok(m) = manifold.downcast::<PySphere>() {
        return Ok(ManifoldTag::SphereVec(m.borrow().ambient_n));
    }
    if let Ok(m) = manifold.downcast::<PySpd>() {
        return Ok(ManifoldTag::SpdMat(m.borrow().n));
    }
    if let Ok(m) = manifold.downcast::<PySo>() {
        return Ok(ManifoldTag::SoMat(m.borrow().n));
    }
    if let Ok(m) = manifold.downcast::<PyCorr>() {
        return Ok(ManifoldTag::CorrMat(m.borrow().n));
    }
    if manifold.downcast::<PyQTensor3>().is_ok() {
        return Ok(ManifoldTag::QTensor3Mat);
    }
    Err(PyTypeError::new_err(
        "unsupported manifold type; expected Euclidean, Sphere, SPD, SO, Corr, or QTensor3",
    ))
}

// ---------------------------------------------------------------------------
// Point/tangent extraction helpers
// ---------------------------------------------------------------------------

/// Extract a raw Vec<f64> from a numpy array (contiguity check, no length check).
fn extract_raw(arr: PyReadonlyArrayDyn<'_, f64>) -> PyResult<Vec<f64>> {
    let slice = arr
        .as_slice()
        .map_err(|_| PyValueError::new_err("array must be contiguous"))?;
    Ok(slice.to_vec())
}

/// For vector manifolds: extract Vec<f64> of exactly `n` elements from a 1-D array.
fn extract_vec_n(arr: PyReadonlyArrayDyn<'_, f64>, n: usize, name: &str) -> PyResult<Vec<f64>> {
    crate::convert::arr_to_vec(arr, n, name)
}

/// For matrix manifolds: extract an N x N numpy array and flatten to Vec<f64>
/// in nalgebra column-major order so that `SMatrix::from_column_slice` reconstructs correctly.
fn extract_mat_n(
    arr: PyReadonlyArrayDyn<'_, f64>,
    n: usize,
    name: &str,
) -> PyResult<Vec<f64>> {
    // Validate element count.
    let slice = arr
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name}: array must be contiguous")))?;
    if slice.len() != n * n {
        return Err(PyValueError::new_err(format!(
            "{name}: expected {}x{} matrix ({} elements), got {}",
            n, n, n * n, slice.len()
        )));
    }
    // Numpy is row-major; convert to nalgebra column-major for storage.
    // We do: SMatrix::from_row_slice (numpy order) then .iter() (column-major order).
    // We dispatch on n to build an SMatrix statically.
    macro_rules! to_col_major {
        ($($N:literal),+) => {
            match n {
                $($N => {
                    let m = nalgebra::SMatrix::<f64, $N, $N>::from_row_slice(slice);
                    Ok(m.iter().copied().collect())
                },)+
                _ => Err(PyValueError::new_err(format!("{name}: unsupported matrix size {n}")))
            }
        };
    }
    to_col_major!(2, 3, 4, 5, 6, 7, 8)
}

// ---------------------------------------------------------------------------
// PyGeodesic
// ---------------------------------------------------------------------------

/// A parameterized geodesic on a Riemannian manifold: gamma(t) = Exp_p(t * v).
///
/// Construct from a base point and velocity, or from two endpoints:
///
///     geo = cartan.Geodesic(manifold, p, v)
///     geo = cartan.Geodesic.from_two_points(manifold, p, q)
///
/// The velocity is NOT normalized. Setting `v = manifold.log(p, q)` produces
/// a geodesic with `geo.eval(0) == p` and `geo.eval(1) == q`.
#[pyclass(name = "Geodesic")]
#[derive(Debug, Clone)]
pub struct PyGeodesic {
    manifold_tag: ManifoldTag,
    /// Base point data in nalgebra storage order (column-major for matrices).
    base_data: Vec<f64>,
    /// Velocity data in the same storage order as base_data.
    velocity_data: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Dispatch macros for PyGeodesic methods
// ---------------------------------------------------------------------------

/// Dispatch a geodesic operation for vector-point manifolds.
///
/// Reconstructs `SVector<f64, N>` from stored data, builds `Geodesic`, and calls `$body`.
/// `$body` is a closure `|py, geo: &Geodesic<mtype<N>>| -> PyResult<PyObject>`.
macro_rules! dispatch_geo_vector {
    ($self:expr, $py:expr, $mtype:ident, $dim:expr, [$($N:literal),+], $body:expr) => {
        match $dim {
            $($N => {
                let mf = cartan_manifolds::$mtype::<$N>;
                let base = nalgebra::SVector::<f64, $N>::from_column_slice(&$self.base_data);
                let vel = nalgebra::SVector::<f64, $N>::from_column_slice(&$self.velocity_data);
                let geo = cartan_geo::Geodesic::new(&mf, base, vel);
                $body($py, &mf, &geo)
            },)+
            _ => Err(PyValueError::new_err(format!("unsupported dimension {}", $dim))),
        }
    };
}

/// Dispatch a geodesic operation for matrix-point manifolds.
macro_rules! dispatch_geo_matrix {
    ($self:expr, $py:expr, $mtype:ident, $dim:expr, [$($N:literal),+], $body:expr) => {
        match $dim {
            $($N => {
                let mf = cartan_manifolds::$mtype::<$N>;
                let base = nalgebra::SMatrix::<f64, $N, $N>::from_column_slice(&$self.base_data);
                let vel = nalgebra::SMatrix::<f64, $N, $N>::from_column_slice(&$self.velocity_data);
                let geo = cartan_geo::Geodesic::new(&mf, base, vel);
                $body($py, &mf, &geo)
            },)+
            _ => Err(PyValueError::new_err(format!("unsupported dimension {}", $dim))),
        }
    };
}

#[pymethods]
impl PyGeodesic {
    /// Construct a geodesic from a base point `p` and initial velocity `v`.
    ///
    /// `gamma(t) = Exp_p(t * v)`. At `t=0` returns `p`, at `t=1` returns `Exp_p(v)`.
    #[new]
    fn new(
        manifold: &Bound<'_, PyAny>,
        p: PyReadonlyArrayDyn<'_, f64>,
        v: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<Self> {
        let tag = identify_manifold(manifold)?;
        let (base_data, velocity_data) = extract_base_vel(&tag, p, v)?;
        Ok(Self {
            manifold_tag: tag,
            base_data,
            velocity_data,
        })
    }

    /// Construct a geodesic from two points: `gamma(0) = p`, `gamma(1) = q`.
    ///
    /// Uses `Log_p(q)` as the velocity. Fails if `p` and `q` are antipodal
    /// (at the cut locus).
    #[staticmethod]
    fn from_two_points(
        manifold: &Bound<'_, PyAny>,
        p: PyReadonlyArrayDyn<'_, f64>,
        q: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<Self> {
        let tag = identify_manifold(manifold)?;
        let (p_data, q_data) = extract_base_vel(&tag, p, q)?;
        // Compute velocity = log(p, q) by re-dispatching on tag.
        let velocity_data = compute_log_data(&tag, &p_data, &q_data)?;
        Ok(Self {
            manifold_tag: tag,
            base_data: p_data,
            velocity_data,
        })
    }

    /// Evaluate the geodesic at parameter `t`: `Exp_p(t * v)`.
    fn eval<'py>(&self, py: Python<'py>, t: f64) -> PyResult<PyObject> {
        match &self.manifold_tag {
            ManifoldTag::EuclideanVec(n) => {
                dispatch_geo_vector!(self, py, Euclidean, *n, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    |py, _mf, geo: &cartan_geo::Geodesic<'_, cartan_manifolds::Euclidean<_>>| {
                        let pt = geo.eval(t as Real);
                        Ok(crate::convert::svector_to_pyarray(py, &pt).into_any().unbind())
                    })
            }
            ManifoldTag::SphereVec(n) => {
                dispatch_geo_vector!(self, py, Sphere, *n, [2, 3, 4, 5, 6, 7, 8, 9, 10],
                    |py, _mf, geo: &cartan_geo::Geodesic<'_, cartan_manifolds::Sphere<_>>| {
                        let pt = geo.eval(t as Real);
                        Ok(crate::convert::svector_to_pyarray(py, &pt).into_any().unbind())
                    })
            }
            ManifoldTag::SpdMat(n) => {
                dispatch_geo_matrix!(self, py, Spd, *n, [2, 3, 4, 5, 6, 7, 8],
                    |py, _mf, geo: &cartan_geo::Geodesic<'_, cartan_manifolds::Spd<_>>| {
                        let pt = geo.eval(t as Real);
                        Ok(crate::convert::smatrix_to_pyarray(py, &pt).into_any().unbind())
                    })
            }
            ManifoldTag::SoMat(n) => {
                dispatch_geo_matrix!(self, py, SpecialOrthogonal, *n, [2, 3, 4],
                    |py, _mf, geo: &cartan_geo::Geodesic<'_, cartan_manifolds::SpecialOrthogonal<_>>| {
                        let pt = geo.eval(t as Real);
                        Ok(crate::convert::smatrix_to_pyarray(py, &pt).into_any().unbind())
                    })
            }
            ManifoldTag::CorrMat(n) => {
                dispatch_geo_matrix!(self, py, Corr, *n, [2, 3, 4, 5, 6, 7, 8],
                    |py, _mf, geo: &cartan_geo::Geodesic<'_, cartan_manifolds::Corr<_>>| {
                        let pt = geo.eval(t as Real);
                        Ok(crate::convert::smatrix_to_pyarray(py, &pt).into_any().unbind())
                    })
            }
            ManifoldTag::QTensor3Mat => {
                let mf = cartan_manifolds::qtensor::QTensor3;
                let base = nalgebra::SMatrix::<f64, 3, 3>::from_column_slice(&self.base_data);
                let vel = nalgebra::SMatrix::<f64, 3, 3>::from_column_slice(&self.velocity_data);
                let geo = cartan_geo::Geodesic::new(&mf, base, vel);
                let pt = geo.eval(t as Real);
                Ok(crate::convert::smatrix_to_pyarray(py, &pt).into_any().unbind())
            }
        }
    }

    /// Arc length of the geodesic on [0, 1]: the Riemannian norm of the velocity.
    fn length(&self) -> PyResult<f64> {
        geo_length_impl(self)
    }

    /// Midpoint of the geodesic: `gamma(0.5) = Exp_p(0.5 * v)`.
    fn midpoint<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        self.eval(py, 0.5)
    }

    /// Sample `n` evenly-spaced points along the geodesic on [0, 1].
    ///
    /// Returns a Python list of numpy arrays. Panics if `n == 0`.
    fn sample<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Vec<PyObject>> {
        if n == 0 {
            return Err(PyValueError::new_err("sample: n must be >= 1"));
        }
        let step = if n == 1 { 0.0 } else { 1.0 / (n - 1) as f64 };
        (0..n)
            .map(|i| self.eval(py, i as f64 * step))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Geodesic(manifold={:?}, base_len={}, vel_len={})",
            self.manifold_tag,
            self.base_data.len(),
            self.velocity_data.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Helper: geo_length_impl (no Python token needed)
// ---------------------------------------------------------------------------

/// Compute geodesic length without requiring a Python token.
/// Reconstructs the concrete Geodesic on the stack and calls `.length()`.
fn geo_length_impl(g: &PyGeodesic) -> PyResult<f64> {
    macro_rules! length_vector {
        ($mtype:ident, $n:expr, [$($N:literal),+]) => {
            match $n {
                $($N => {
                    let mf = cartan_manifolds::$mtype::<$N>;
                    let base = nalgebra::SVector::<f64, $N>::from_column_slice(&g.base_data);
                    let vel = nalgebra::SVector::<f64, $N>::from_column_slice(&g.velocity_data);
                    let geo = cartan_geo::Geodesic::new(&mf, base, vel);
                    Ok(geo.length())
                },)+
                _ => Err(PyValueError::new_err(format!("unsupported dimension {}", $n))),
            }
        };
    }
    macro_rules! length_matrix {
        ($mtype:ident, $n:expr, [$($N:literal),+]) => {
            match $n {
                $($N => {
                    let mf = cartan_manifolds::$mtype::<$N>;
                    let base = nalgebra::SMatrix::<f64, $N, $N>::from_column_slice(&g.base_data);
                    let vel = nalgebra::SMatrix::<f64, $N, $N>::from_column_slice(&g.velocity_data);
                    let geo = cartan_geo::Geodesic::new(&mf, base, vel);
                    Ok(geo.length())
                },)+
                _ => Err(PyValueError::new_err(format!("unsupported dimension {}", $n))),
            }
        };
    }
    match &g.manifold_tag {
        ManifoldTag::EuclideanVec(n) => length_vector!(Euclidean, *n, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ManifoldTag::SphereVec(n) => length_vector!(Sphere, *n, [2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ManifoldTag::SpdMat(n) => length_matrix!(Spd, *n, [2, 3, 4, 5, 6, 7, 8]),
        ManifoldTag::SoMat(n) => length_matrix!(SpecialOrthogonal, *n, [2, 3, 4]),
        ManifoldTag::CorrMat(n) => length_matrix!(Corr, *n, [2, 3, 4, 5, 6, 7, 8]),
        ManifoldTag::QTensor3Mat => {
            let mf = cartan_manifolds::qtensor::QTensor3;
            let base = nalgebra::SMatrix::<f64, 3, 3>::from_column_slice(&g.base_data);
            let vel = nalgebra::SMatrix::<f64, 3, 3>::from_column_slice(&g.velocity_data);
            let geo = cartan_geo::Geodesic::new(&mf, base, vel);
            Ok(geo.length())
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: extract base + velocity data for construction
// ---------------------------------------------------------------------------

fn extract_base_vel(
    tag: &ManifoldTag,
    p: PyReadonlyArrayDyn<'_, f64>,
    v: PyReadonlyArrayDyn<'_, f64>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    match tag {
        ManifoldTag::EuclideanVec(n) => {
            let pd = extract_vec_n(p, *n, "p")?;
            let vd = extract_vec_n(v, *n, "v")?;
            Ok((pd, vd))
        }
        ManifoldTag::SphereVec(n) => {
            let pd = extract_vec_n(p, *n, "p")?;
            let vd = extract_vec_n(v, *n, "v")?;
            Ok((pd, vd))
        }
        ManifoldTag::SpdMat(n) => {
            let pd = extract_mat_n(p, *n, "p")?;
            let vd = extract_mat_n(v, *n, "v")?;
            Ok((pd, vd))
        }
        ManifoldTag::SoMat(n) => {
            let pd = extract_mat_n(p, *n, "p")?;
            let vd = extract_mat_n(v, *n, "v")?;
            Ok((pd, vd))
        }
        ManifoldTag::CorrMat(n) => {
            let pd = extract_mat_n(p, *n, "p")?;
            let vd = extract_mat_n(v, *n, "v")?;
            Ok((pd, vd))
        }
        ManifoldTag::QTensor3Mat => {
            let pd = extract_mat_n(p, 3, "p")?;
            let vd = extract_mat_n(v, 3, "v")?;
            Ok((pd, vd))
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: compute log(p, q) in column-major Vec<f64> form
// ---------------------------------------------------------------------------

fn compute_log_data(tag: &ManifoldTag, p_data: &[f64], q_data: &[f64]) -> PyResult<Vec<f64>> {
    use cartan_core::Manifold;
    macro_rules! log_vector {
        ($mtype:ident, $n:expr, [$($N:literal),+]) => {
            match $n {
                $($N => {
                    let mf = cartan_manifolds::$mtype::<$N>;
                    let p = nalgebra::SVector::<f64, $N>::from_column_slice(p_data);
                    let q = nalgebra::SVector::<f64, $N>::from_column_slice(q_data);
                    let v = Manifold::log(&mf, &p, &q)
                        .map_err(crate::error::cartan_err_to_py)?;
                    Ok(v.iter().copied().collect())
                },)+
                _ => Err(PyValueError::new_err(format!("unsupported dimension {}", $n))),
            }
        };
    }
    macro_rules! log_matrix {
        ($mtype:ident, $n:expr, [$($N:literal),+]) => {
            match $n {
                $($N => {
                    let mf = cartan_manifolds::$mtype::<$N>;
                    let p = nalgebra::SMatrix::<f64, $N, $N>::from_column_slice(p_data);
                    let q = nalgebra::SMatrix::<f64, $N, $N>::from_column_slice(q_data);
                    let v = Manifold::log(&mf, &p, &q)
                        .map_err(crate::error::cartan_err_to_py)?;
                    Ok(v.iter().copied().collect())
                },)+
                _ => Err(PyValueError::new_err(format!("unsupported dimension {}", $n))),
            }
        };
    }
    match tag {
        ManifoldTag::EuclideanVec(n) => log_vector!(Euclidean, *n, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ManifoldTag::SphereVec(n) => log_vector!(Sphere, *n, [2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ManifoldTag::SpdMat(n) => log_matrix!(Spd, *n, [2, 3, 4, 5, 6, 7, 8]),
        ManifoldTag::SoMat(n) => log_matrix!(SpecialOrthogonal, *n, [2, 3, 4]),
        ManifoldTag::CorrMat(n) => log_matrix!(Corr, *n, [2, 3, 4, 5, 6, 7, 8]),
        ManifoldTag::QTensor3Mat => {
            use cartan_manifolds::qtensor::QTensor3;
            let mf = QTensor3;
            let p = nalgebra::SMatrix::<f64, 3, 3>::from_column_slice(p_data);
            let q = nalgebra::SMatrix::<f64, 3, 3>::from_column_slice(q_data);
            let v = Manifold::log(&mf, &p, &q)
                .map_err(crate::error::cartan_err_to_py)?;
            Ok(v.iter().copied().collect())
        }
    }
}

// ---------------------------------------------------------------------------
// PyCurvatureQuery
// ---------------------------------------------------------------------------

/// Curvature queries at a fixed point on a Riemannian manifold.
///
///     curv = cartan.CurvatureQuery(manifold, p)
///     curv.scalar()                 # scalar curvature
///     curv.sectional(u, v)          # sectional curvature of plane spanned by u, v
///     curv.ricci(u, v)              # Ricci curvature Ric(u, v)
///     curv.riemann(u, v, w)         # Riemann tensor R(u, v)w
#[pyclass(name = "CurvatureQuery")]
#[derive(Debug, Clone)]
pub struct PyCurvatureQuery {
    manifold_tag: ManifoldTag,
    /// Point data in nalgebra column-major order.
    point_data: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Dispatch macros for PyCurvatureQuery methods
// ---------------------------------------------------------------------------

macro_rules! dispatch_curv_vector {
    ($self:expr, $py:expr, $mtype:ident, $dim:expr, [$($N:literal),+], $body:expr) => {
        match $dim {
            $($N => {
                let mf = cartan_manifolds::$mtype::<$N>;
                let pt = nalgebra::SVector::<f64, $N>::from_column_slice(&$self.point_data);
                let cq = cartan_geo::CurvatureQuery::new(&mf, pt);
                $body($py, &mf, &cq)
            },)+
            _ => Err(PyValueError::new_err(format!("unsupported dimension {}", $dim))),
        }
    };
}

macro_rules! dispatch_curv_matrix {
    ($self:expr, $py:expr, $mtype:ident, $dim:expr, [$($N:literal),+], $body:expr) => {
        match $dim {
            $($N => {
                let mf = cartan_manifolds::$mtype::<$N>;
                let pt = nalgebra::SMatrix::<f64, $N, $N>::from_column_slice(&$self.point_data);
                let cq = cartan_geo::CurvatureQuery::new(&mf, pt);
                $body($py, &mf, &cq)
            },)+
            _ => Err(PyValueError::new_err(format!("unsupported dimension {}", $dim))),
        }
    };
}

#[pymethods]
impl PyCurvatureQuery {
    /// Construct a curvature query at point `p` on the given manifold.
    #[new]
    fn new(manifold: &Bound<'_, PyAny>, p: PyReadonlyArrayDyn<'_, f64>) -> PyResult<Self> {
        let tag = identify_manifold(manifold)?;
        let point_data = extract_point_data(&tag, p)?;
        Ok(Self {
            manifold_tag: tag,
            point_data,
        })
    }

    /// Scalar curvature at the stored point.
    fn scalar(&self) -> PyResult<f64> {
        let py_dummy = ();
        match &self.manifold_tag {
            ManifoldTag::EuclideanVec(n) => {
                dispatch_curv_vector!(self, py_dummy, Euclidean, *n,
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    |_py, _mf, cq: &cartan_geo::CurvatureQuery<'_, cartan_manifolds::Euclidean<_>>| {
                        Ok::<f64, PyErr>(cq.scalar())
                    })
            }
            ManifoldTag::SphereVec(n) => {
                dispatch_curv_vector!(self, py_dummy, Sphere, *n,
                    [2, 3, 4, 5, 6, 7, 8, 9, 10],
                    |_py, _mf, cq: &cartan_geo::CurvatureQuery<'_, cartan_manifolds::Sphere<_>>| {
                        Ok::<f64, PyErr>(cq.scalar())
                    })
            }
            ManifoldTag::SpdMat(n) => {
                dispatch_curv_matrix!(self, py_dummy, Spd, *n,
                    [2, 3, 4, 5, 6, 7, 8],
                    |_py, _mf, cq: &cartan_geo::CurvatureQuery<'_, cartan_manifolds::Spd<_>>| {
                        Ok::<f64, PyErr>(cq.scalar())
                    })
            }
            ManifoldTag::SoMat(n) => {
                dispatch_curv_matrix!(self, py_dummy, SpecialOrthogonal, *n,
                    [2, 3, 4],
                    |_py, _mf, cq: &cartan_geo::CurvatureQuery<'_, cartan_manifolds::SpecialOrthogonal<_>>| {
                        Ok::<f64, PyErr>(cq.scalar())
                    })
            }
            ManifoldTag::CorrMat(n) => {
                dispatch_curv_matrix!(self, py_dummy, Corr, *n,
                    [2, 3, 4, 5, 6, 7, 8],
                    |_py, _mf, cq: &cartan_geo::CurvatureQuery<'_, cartan_manifolds::Corr<_>>| {
                        Ok::<f64, PyErr>(cq.scalar())
                    })
            }
            ManifoldTag::QTensor3Mat => {
                let mf = cartan_manifolds::qtensor::QTensor3;
                let pt = nalgebra::SMatrix::<f64, 3, 3>::from_column_slice(&self.point_data);
                let cq = cartan_geo::CurvatureQuery::new(&mf, pt);
                Ok(cq.scalar())
            }
        }
    }

    /// Sectional curvature K(u, v) of the 2-plane spanned by tangent vectors `u` and `v`.
    fn sectional(
        &self,
        u: PyReadonlyArrayDyn<'_, f64>,
        v: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<f64> {
        let py_dummy = ();
        match &self.manifold_tag {
            ManifoldTag::EuclideanVec(n) => {
                let u_d = extract_vec_n(u, *n, "u")?;
                let v_d = extract_vec_n(v, *n, "v")?;
                dispatch_curv_vector!(self, py_dummy, Euclidean, *n,
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    |_py, _mf, cq: &cartan_geo::CurvatureQuery<'_, cartan_manifolds::Euclidean<_>>| {
                        let uu = nalgebra::SVector::<f64, _>::from_column_slice(&u_d);
                        let vv = nalgebra::SVector::<f64, _>::from_column_slice(&v_d);
                        Ok::<f64, PyErr>(cq.sectional(&uu, &vv))
                    })
            }
            ManifoldTag::SphereVec(n) => {
                let u_d = extract_vec_n(u, *n, "u")?;
                let v_d = extract_vec_n(v, *n, "v")?;
                dispatch_curv_vector!(self, py_dummy, Sphere, *n,
                    [2, 3, 4, 5, 6, 7, 8, 9, 10],
                    |_py, _mf, cq: &cartan_geo::CurvatureQuery<'_, cartan_manifolds::Sphere<_>>| {
                        let uu = nalgebra::SVector::<f64, _>::from_column_slice(&u_d);
                        let vv = nalgebra::SVector::<f64, _>::from_column_slice(&v_d);
                        Ok::<f64, PyErr>(cq.sectional(&uu, &vv))
                    })
            }
            ManifoldTag::SpdMat(n) => {
                let u_d = extract_mat_n(u, *n, "u")?;
                let v_d = extract_mat_n(v, *n, "v")?;
                dispatch_curv_matrix!(self, py_dummy, Spd, *n,
                    [2, 3, 4, 5, 6, 7, 8],
                    |_py, _mf, cq: &cartan_geo::CurvatureQuery<'_, cartan_manifolds::Spd<_>>| {
                        let uu = nalgebra::SMatrix::<f64, _, _>::from_column_slice(&u_d);
                        let vv = nalgebra::SMatrix::<f64, _, _>::from_column_slice(&v_d);
                        Ok::<f64, PyErr>(cq.sectional(&uu, &vv))
                    })
            }
            ManifoldTag::SoMat(n) => {
                let u_d = extract_mat_n(u, *n, "u")?;
                let v_d = extract_mat_n(v, *n, "v")?;
                dispatch_curv_matrix!(self, py_dummy, SpecialOrthogonal, *n,
                    [2, 3, 4],
                    |_py, _mf, cq: &cartan_geo::CurvatureQuery<'_, cartan_manifolds::SpecialOrthogonal<_>>| {
                        let uu = nalgebra::SMatrix::<f64, _, _>::from_column_slice(&u_d);
                        let vv = nalgebra::SMatrix::<f64, _, _>::from_column_slice(&v_d);
                        Ok::<f64, PyErr>(cq.sectional(&uu, &vv))
                    })
            }
            ManifoldTag::CorrMat(n) => {
                let u_d = extract_mat_n(u, *n, "u")?;
                let v_d = extract_mat_n(v, *n, "v")?;
                dispatch_curv_matrix!(self, py_dummy, Corr, *n,
                    [2, 3, 4, 5, 6, 7, 8],
                    |_py, _mf, cq: &cartan_geo::CurvatureQuery<'_, cartan_manifolds::Corr<_>>| {
                        let uu = nalgebra::SMatrix::<f64, _, _>::from_column_slice(&u_d);
                        let vv = nalgebra::SMatrix::<f64, _, _>::from_column_slice(&v_d);
                        Ok::<f64, PyErr>(cq.sectional(&uu, &vv))
                    })
            }
            ManifoldTag::QTensor3Mat => {
                let u_d = extract_mat_n(u, 3, "u")?;
                let v_d = extract_mat_n(v, 3, "v")?;
                let mf = cartan_manifolds::qtensor::QTensor3;
                let pt = nalgebra::SMatrix::<f64, 3, 3>::from_column_slice(&self.point_data);
                let cq = cartan_geo::CurvatureQuery::new(&mf, pt);
                let uu = nalgebra::SMatrix::<f64, 3, 3>::from_column_slice(&u_d);
                let vv = nalgebra::SMatrix::<f64, 3, 3>::from_column_slice(&v_d);
                Ok(cq.sectional(&uu, &vv))
            }
        }
    }

    /// Ricci curvature Ric(u, v) at the stored point.
    fn ricci(
        &self,
        u: PyReadonlyArrayDyn<'_, f64>,
        v: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<f64> {
        let py_dummy = ();
        match &self.manifold_tag {
            ManifoldTag::EuclideanVec(n) => {
                let u_d = extract_vec_n(u, *n, "u")?;
                let v_d = extract_vec_n(v, *n, "v")?;
                dispatch_curv_vector!(self, py_dummy, Euclidean, *n,
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    |_py, _mf, cq: &cartan_geo::CurvatureQuery<'_, cartan_manifolds::Euclidean<_>>| {
                        let uu = nalgebra::SVector::<f64, _>::from_column_slice(&u_d);
                        let vv = nalgebra::SVector::<f64, _>::from_column_slice(&v_d);
                        Ok::<f64, PyErr>(cq.ricci(&uu, &vv))
                    })
            }
            ManifoldTag::SphereVec(n) => {
                let u_d = extract_vec_n(u, *n, "u")?;
                let v_d = extract_vec_n(v, *n, "v")?;
                dispatch_curv_vector!(self, py_dummy, Sphere, *n,
                    [2, 3, 4, 5, 6, 7, 8, 9, 10],
                    |_py, _mf, cq: &cartan_geo::CurvatureQuery<'_, cartan_manifolds::Sphere<_>>| {
                        let uu = nalgebra::SVector::<f64, _>::from_column_slice(&u_d);
                        let vv = nalgebra::SVector::<f64, _>::from_column_slice(&v_d);
                        Ok::<f64, PyErr>(cq.ricci(&uu, &vv))
                    })
            }
            ManifoldTag::SpdMat(n) => {
                let u_d = extract_mat_n(u, *n, "u")?;
                let v_d = extract_mat_n(v, *n, "v")?;
                dispatch_curv_matrix!(self, py_dummy, Spd, *n,
                    [2, 3, 4, 5, 6, 7, 8],
                    |_py, _mf, cq: &cartan_geo::CurvatureQuery<'_, cartan_manifolds::Spd<_>>| {
                        let uu = nalgebra::SMatrix::<f64, _, _>::from_column_slice(&u_d);
                        let vv = nalgebra::SMatrix::<f64, _, _>::from_column_slice(&v_d);
                        Ok::<f64, PyErr>(cq.ricci(&uu, &vv))
                    })
            }
            ManifoldTag::SoMat(n) => {
                let u_d = extract_mat_n(u, *n, "u")?;
                let v_d = extract_mat_n(v, *n, "v")?;
                dispatch_curv_matrix!(self, py_dummy, SpecialOrthogonal, *n,
                    [2, 3, 4],
                    |_py, _mf, cq: &cartan_geo::CurvatureQuery<'_, cartan_manifolds::SpecialOrthogonal<_>>| {
                        let uu = nalgebra::SMatrix::<f64, _, _>::from_column_slice(&u_d);
                        let vv = nalgebra::SMatrix::<f64, _, _>::from_column_slice(&v_d);
                        Ok::<f64, PyErr>(cq.ricci(&uu, &vv))
                    })
            }
            ManifoldTag::CorrMat(n) => {
                let u_d = extract_mat_n(u, *n, "u")?;
                let v_d = extract_mat_n(v, *n, "v")?;
                dispatch_curv_matrix!(self, py_dummy, Corr, *n,
                    [2, 3, 4, 5, 6, 7, 8],
                    |_py, _mf, cq: &cartan_geo::CurvatureQuery<'_, cartan_manifolds::Corr<_>>| {
                        let uu = nalgebra::SMatrix::<f64, _, _>::from_column_slice(&u_d);
                        let vv = nalgebra::SMatrix::<f64, _, _>::from_column_slice(&v_d);
                        Ok::<f64, PyErr>(cq.ricci(&uu, &vv))
                    })
            }
            ManifoldTag::QTensor3Mat => {
                let u_d = extract_mat_n(u, 3, "u")?;
                let v_d = extract_mat_n(v, 3, "v")?;
                let mf = cartan_manifolds::qtensor::QTensor3;
                let pt = nalgebra::SMatrix::<f64, 3, 3>::from_column_slice(&self.point_data);
                let cq = cartan_geo::CurvatureQuery::new(&mf, pt);
                let uu = nalgebra::SMatrix::<f64, 3, 3>::from_column_slice(&u_d);
                let vv = nalgebra::SMatrix::<f64, 3, 3>::from_column_slice(&v_d);
                Ok(cq.ricci(&uu, &vv))
            }
        }
    }

    /// Riemann curvature tensor R(u, v)w at the stored point.
    ///
    /// Returns a tangent vector (same shape as u, v, w).
    fn riemann<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArrayDyn<'_, f64>,
        v: PyReadonlyArrayDyn<'_, f64>,
        w: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<PyObject> {
        let py_dummy = ();
        match &self.manifold_tag {
            ManifoldTag::EuclideanVec(n) => {
                let u_d = extract_vec_n(u, *n, "u")?;
                let v_d = extract_vec_n(v, *n, "v")?;
                let w_d = extract_vec_n(w, *n, "w")?;
                dispatch_curv_vector!(self, py_dummy, Euclidean, *n,
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    |_py, _mf, cq: &cartan_geo::CurvatureQuery<'_, cartan_manifolds::Euclidean<_>>| {
                        let uu = nalgebra::SVector::<f64, _>::from_column_slice(&u_d);
                        let vv = nalgebra::SVector::<f64, _>::from_column_slice(&v_d);
                        let ww = nalgebra::SVector::<f64, _>::from_column_slice(&w_d);
                        let res = cq.riemann(&uu, &vv, &ww);
                        Ok::<PyObject, PyErr>(crate::convert::svector_to_pyarray(py, &res).into_any().unbind())
                    })
            }
            ManifoldTag::SphereVec(n) => {
                let u_d = extract_vec_n(u, *n, "u")?;
                let v_d = extract_vec_n(v, *n, "v")?;
                let w_d = extract_vec_n(w, *n, "w")?;
                dispatch_curv_vector!(self, py_dummy, Sphere, *n,
                    [2, 3, 4, 5, 6, 7, 8, 9, 10],
                    |_py, _mf, cq: &cartan_geo::CurvatureQuery<'_, cartan_manifolds::Sphere<_>>| {
                        let uu = nalgebra::SVector::<f64, _>::from_column_slice(&u_d);
                        let vv = nalgebra::SVector::<f64, _>::from_column_slice(&v_d);
                        let ww = nalgebra::SVector::<f64, _>::from_column_slice(&w_d);
                        let res = cq.riemann(&uu, &vv, &ww);
                        Ok::<PyObject, PyErr>(crate::convert::svector_to_pyarray(py, &res).into_any().unbind())
                    })
            }
            ManifoldTag::SpdMat(n) => {
                let u_d = extract_mat_n(u, *n, "u")?;
                let v_d = extract_mat_n(v, *n, "v")?;
                let w_d = extract_mat_n(w, *n, "w")?;
                dispatch_curv_matrix!(self, py_dummy, Spd, *n,
                    [2, 3, 4, 5, 6, 7, 8],
                    |_py, _mf, cq: &cartan_geo::CurvatureQuery<'_, cartan_manifolds::Spd<_>>| {
                        let uu = nalgebra::SMatrix::<f64, _, _>::from_column_slice(&u_d);
                        let vv = nalgebra::SMatrix::<f64, _, _>::from_column_slice(&v_d);
                        let ww = nalgebra::SMatrix::<f64, _, _>::from_column_slice(&w_d);
                        let res = cq.riemann(&uu, &vv, &ww);
                        Ok::<PyObject, PyErr>(crate::convert::smatrix_to_pyarray(py, &res).into_any().unbind())
                    })
            }
            ManifoldTag::SoMat(n) => {
                let u_d = extract_mat_n(u, *n, "u")?;
                let v_d = extract_mat_n(v, *n, "v")?;
                let w_d = extract_mat_n(w, *n, "w")?;
                dispatch_curv_matrix!(self, py_dummy, SpecialOrthogonal, *n,
                    [2, 3, 4],
                    |_py, _mf, cq: &cartan_geo::CurvatureQuery<'_, cartan_manifolds::SpecialOrthogonal<_>>| {
                        let uu = nalgebra::SMatrix::<f64, _, _>::from_column_slice(&u_d);
                        let vv = nalgebra::SMatrix::<f64, _, _>::from_column_slice(&v_d);
                        let ww = nalgebra::SMatrix::<f64, _, _>::from_column_slice(&w_d);
                        let res = cq.riemann(&uu, &vv, &ww);
                        Ok::<PyObject, PyErr>(crate::convert::smatrix_to_pyarray(py, &res).into_any().unbind())
                    })
            }
            ManifoldTag::CorrMat(n) => {
                let u_d = extract_mat_n(u, *n, "u")?;
                let v_d = extract_mat_n(v, *n, "v")?;
                let w_d = extract_mat_n(w, *n, "w")?;
                dispatch_curv_matrix!(self, py_dummy, Corr, *n,
                    [2, 3, 4, 5, 6, 7, 8],
                    |_py, _mf, cq: &cartan_geo::CurvatureQuery<'_, cartan_manifolds::Corr<_>>| {
                        let uu = nalgebra::SMatrix::<f64, _, _>::from_column_slice(&u_d);
                        let vv = nalgebra::SMatrix::<f64, _, _>::from_column_slice(&v_d);
                        let ww = nalgebra::SMatrix::<f64, _, _>::from_column_slice(&w_d);
                        let res = cq.riemann(&uu, &vv, &ww);
                        Ok::<PyObject, PyErr>(crate::convert::smatrix_to_pyarray(py, &res).into_any().unbind())
                    })
            }
            ManifoldTag::QTensor3Mat => {
                let u_d = extract_mat_n(u, 3, "u")?;
                let v_d = extract_mat_n(v, 3, "v")?;
                let w_d = extract_mat_n(w, 3, "w")?;
                let mf = cartan_manifolds::qtensor::QTensor3;
                let pt = nalgebra::SMatrix::<f64, 3, 3>::from_column_slice(&self.point_data);
                let cq = cartan_geo::CurvatureQuery::new(&mf, pt);
                let uu = nalgebra::SMatrix::<f64, 3, 3>::from_column_slice(&u_d);
                let vv = nalgebra::SMatrix::<f64, 3, 3>::from_column_slice(&v_d);
                let ww = nalgebra::SMatrix::<f64, 3, 3>::from_column_slice(&w_d);
                let res = cq.riemann(&uu, &vv, &ww);
                Ok(crate::convert::smatrix_to_pyarray(py, &res).into_any().unbind())
            }
        }
    }

    fn __repr__(&self) -> String {
        format!("CurvatureQuery(manifold={:?})", self.manifold_tag)
    }
}

fn extract_point_data(tag: &ManifoldTag, p: PyReadonlyArrayDyn<'_, f64>) -> PyResult<Vec<f64>> {
    match tag {
        ManifoldTag::EuclideanVec(n) => extract_vec_n(p, *n, "p"),
        ManifoldTag::SphereVec(n) => extract_vec_n(p, *n, "p"),
        ManifoldTag::SpdMat(n) => extract_mat_n(p, *n, "p"),
        ManifoldTag::SoMat(n) => extract_mat_n(p, *n, "p"),
        ManifoldTag::CorrMat(n) => extract_mat_n(p, *n, "p"),
        ManifoldTag::QTensor3Mat => extract_mat_n(p, 3, "p"),
    }
}

// ---------------------------------------------------------------------------
// Dispatch macros for integrate_jacobi  (must be defined before first use)
// ---------------------------------------------------------------------------

macro_rules! dispatch_jacobi_vector {
    ($py:expr, $geo:expr, $j0_d:expr, $j0_dot_d:expr, $n_steps:expr, $mtype:ident, $dim:expr, [$($N:literal),+]) => {
        match $dim {
            $($N => {
                let mf = cartan_manifolds::$mtype::<$N>;
                let base = nalgebra::SVector::<f64, $N>::from_column_slice(&$geo.base_data);
                let vel = nalgebra::SVector::<f64, $N>::from_column_slice(&$geo.velocity_data);
                let geo = cartan_geo::Geodesic::new(&mf, base, vel);
                let j0_v = nalgebra::SVector::<f64, $N>::from_column_slice(&$j0_d);
                let j0_dot_v = nalgebra::SVector::<f64, $N>::from_column_slice(&$j0_dot_d);
                let res = cartan_geo::integrate_jacobi(&geo, j0_v, j0_dot_v, $n_steps);
                let field: Vec<PyObject> = res.field.iter()
                    .map(|v| crate::convert::svector_to_pyarray($py, v).into_any().unbind())
                    .collect();
                let velocity: Vec<PyObject> = res.velocity.iter()
                    .map(|v| crate::convert::svector_to_pyarray($py, v).into_any().unbind())
                    .collect();
                Ok(PyJacobiResult { params: res.params, field, velocity })
            },)+
            _ => Err(PyValueError::new_err(format!("unsupported dimension {}", $dim))),
        }
    };
}

macro_rules! dispatch_jacobi_matrix {
    ($py:expr, $geo:expr, $j0_d:expr, $j0_dot_d:expr, $n_steps:expr, $mtype:ident, $dim:expr, [$($N:literal),+]) => {
        match $dim {
            $($N => {
                let mf = cartan_manifolds::$mtype::<$N>;
                let base = nalgebra::SMatrix::<f64, $N, $N>::from_column_slice(&$geo.base_data);
                let vel = nalgebra::SMatrix::<f64, $N, $N>::from_column_slice(&$geo.velocity_data);
                let geo = cartan_geo::Geodesic::new(&mf, base, vel);
                let j0_v = nalgebra::SMatrix::<f64, $N, $N>::from_column_slice(&$j0_d);
                let j0_dot_v = nalgebra::SMatrix::<f64, $N, $N>::from_column_slice(&$j0_dot_d);
                let res = cartan_geo::integrate_jacobi(&geo, j0_v, j0_dot_v, $n_steps);
                let field: Vec<PyObject> = res.field.iter()
                    .map(|m| crate::convert::smatrix_to_pyarray($py, m).into_any().unbind())
                    .collect();
                let velocity: Vec<PyObject> = res.velocity.iter()
                    .map(|m| crate::convert::smatrix_to_pyarray($py, m).into_any().unbind())
                    .collect();
                Ok(PyJacobiResult { params: res.params, field, velocity })
            },)+
            _ => Err(PyValueError::new_err(format!("unsupported dimension {}", $dim))),
        }
    };
}

// ---------------------------------------------------------------------------
// PyJacobiResult
// ---------------------------------------------------------------------------

/// Result of a Jacobi field integration along a geodesic.
///
/// Attributes:
///
///     result.params    # list of float: parameter values t_0, ..., t_n
///     result.field     # list of numpy array: J(t_i) at each parameter
///     result.velocity  # list of numpy array: J'(t_i) at each parameter
#[pyclass(name = "JacobiResult")]
pub struct PyJacobiResult {
    /// Parameter values t_0, ..., t_n (length n_steps + 1).
    #[pyo3(get)]
    pub params: Vec<f64>,
    /// Jacobi field values J(t_i) as numpy arrays.
    #[pyo3(get)]
    pub field: Vec<PyObject>,
    /// Jacobi field velocities J'(t_i) as numpy arrays.
    #[pyo3(get)]
    pub velocity: Vec<PyObject>,
}

#[pymethods]
impl PyJacobiResult {
    fn __repr__(&self) -> String {
        format!("JacobiResult(n_steps={}, t_max={:.3})", self.params.len() - 1, self.params.last().copied().unwrap_or(0.0))
    }
}

// ---------------------------------------------------------------------------
// integrate_jacobi pyfunction
// ---------------------------------------------------------------------------

/// Integrate a Jacobi field along a geodesic using 4th-order Runge-Kutta.
///
/// A Jacobi field J(t) satisfies the geodesic deviation equation:
///
///     D²J/dt² + R(J, γ') γ' = 0
///
/// Parameters
/// ----------
/// geodesic : cartan.Geodesic
///     The base geodesic gamma(t) = Exp_p(t * v).
/// j0 : numpy array
///     Initial Jacobi field J(0) -- a tangent vector at gamma(0).
/// j0_dot : numpy array
///     Initial covariant velocity J'(0) -- a tangent vector at gamma(0).
/// n_steps : int
///     Number of RK4 integration steps on [0, 1]. More steps = higher accuracy.
///
/// Returns
/// -------
/// JacobiResult with `.params`, `.field`, `.velocity` lists.
#[pyfunction]
pub fn integrate_jacobi<'py>(
    py: Python<'py>,
    geodesic: &PyGeodesic,
    j0: PyReadonlyArrayDyn<'_, f64>,
    j0_dot: PyReadonlyArrayDyn<'_, f64>,
    n_steps: usize,
) -> PyResult<PyJacobiResult> {
    if n_steps == 0 {
        return Err(PyValueError::new_err("integrate_jacobi: n_steps must be >= 1"));
    }
    match &geodesic.manifold_tag {
        ManifoldTag::EuclideanVec(n) => {
            let n = *n;
            let j0_d = extract_vec_n(j0, n, "j0")?;
            let j0_dot_d = extract_vec_n(j0_dot, n, "j0_dot")?;
            dispatch_jacobi_vector!(py, geodesic, j0_d, j0_dot_d, n_steps, Euclidean, n,
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        }
        ManifoldTag::SphereVec(n) => {
            let n = *n;
            let j0_d = extract_vec_n(j0, n, "j0")?;
            let j0_dot_d = extract_vec_n(j0_dot, n, "j0_dot")?;
            dispatch_jacobi_vector!(py, geodesic, j0_d, j0_dot_d, n_steps, Sphere, n,
                [2, 3, 4, 5, 6, 7, 8, 9, 10])
        }
        ManifoldTag::SpdMat(n) => {
            let n = *n;
            let j0_d = extract_mat_n(j0, n, "j0")?;
            let j0_dot_d = extract_mat_n(j0_dot, n, "j0_dot")?;
            dispatch_jacobi_matrix!(py, geodesic, j0_d, j0_dot_d, n_steps, Spd, n,
                [2, 3, 4, 5, 6, 7, 8])
        }
        ManifoldTag::SoMat(n) => {
            let n = *n;
            let j0_d = extract_mat_n(j0, n, "j0")?;
            let j0_dot_d = extract_mat_n(j0_dot, n, "j0_dot")?;
            dispatch_jacobi_matrix!(py, geodesic, j0_d, j0_dot_d, n_steps, SpecialOrthogonal, n,
                [2, 3, 4])
        }
        ManifoldTag::CorrMat(n) => {
            let n = *n;
            let j0_d = extract_mat_n(j0, n, "j0")?;
            let j0_dot_d = extract_mat_n(j0_dot, n, "j0_dot")?;
            dispatch_jacobi_matrix!(py, geodesic, j0_d, j0_dot_d, n_steps, Corr, n,
                [2, 3, 4, 5, 6, 7, 8])
        }
        ManifoldTag::QTensor3Mat => {
            let j0_d = extract_mat_n(j0, 3, "j0")?;
            let j0_dot_d = extract_mat_n(j0_dot, 3, "j0_dot")?;
            let mf = cartan_manifolds::qtensor::QTensor3;
            let base = nalgebra::SMatrix::<f64, 3, 3>::from_column_slice(&geodesic.base_data);
            let vel = nalgebra::SMatrix::<f64, 3, 3>::from_column_slice(&geodesic.velocity_data);
            let geo = cartan_geo::Geodesic::new(&mf, base, vel);
            let j0_v = nalgebra::SMatrix::<f64, 3, 3>::from_column_slice(&j0_d);
            let j0_dot_v = nalgebra::SMatrix::<f64, 3, 3>::from_column_slice(&j0_dot_d);
            let res = cartan_geo::integrate_jacobi(&geo, j0_v, j0_dot_v, n_steps);
            let field: Vec<PyObject> = res.field.iter()
                .map(|m| crate::convert::smatrix_to_pyarray(py, m).into_any().unbind())
                .collect();
            let velocity: Vec<PyObject> = res.velocity.iter()
                .map(|m| crate::convert::smatrix_to_pyarray(py, m).into_any().unbind())
                .collect();
            Ok(PyJacobiResult { params: res.params, field, velocity })
        }
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register geo classes and functions on the Python module.
pub fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::types::PyModuleMethods;
    m.add_class::<PyGeodesic>()?;
    m.add_class::<PyCurvatureQuery>()?;
    m.add_class::<PyJacobiResult>()?;
    m.add_function(pyo3::wrap_pyfunction!(integrate_jacobi, m)?)?;
    Ok(())
}
