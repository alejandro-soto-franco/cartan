// ~/cartan/cartan-py/src/manifolds/se.rs

//! Python wrapper for `cartan_manifolds::SpecialEuclidean<N>`.
//!
//! SE(N) = SO(N) x R^N is the special Euclidean group of rigid-body motions.
//! Points are (rotation, translation) tuples, where rotation is an N x N
//! orthogonal matrix with det +1 and translation is an N-vector. Tangent
//! vectors follow the same tuple structure. Supported sizes: 2 and 3.
//!
//! Unlike the vector and matrix manifolds, SE(N) uses tuple-based I/O,
//! so this module is hand-written rather than macro-generated.

use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use cartan_core::{GeodesicInterpolation, Manifold, ParallelTransport, Retraction, VectorTransport};
use cartan_manifolds::{SEPoint, SETangent, SpecialEuclidean};

use crate::convert::{arr_to_smatrix, arr_to_svector, smatrix_to_pyarray, svector_to_pyarray};
use crate::error::cartan_err_to_py;

// ---------------------------------------------------------------------------
// Python class
// ---------------------------------------------------------------------------

#[pyclass(name = "SE")]
#[derive(Debug, Clone)]
pub struct PySe {
    pub(crate) n: usize,
    weight: f64,
}

// ---------------------------------------------------------------------------
// Helpers: Python tuple <-> Rust SE types
// ---------------------------------------------------------------------------

/// Extract a (rotation, translation) tuple from Python into SEPoint<N>.
fn extract_se_point<const N: usize>(obj: &Bound<'_, PyAny>) -> PyResult<SEPoint<N>> {
    let tuple: &Bound<'_, PyTuple> = obj.downcast()?;
    if tuple.len() != 2 {
        return Err(PyValueError::new_err(
            "SE point must be a (rotation, translation) tuple",
        ));
    }
    let r_arr: PyReadonlyArrayDyn<f64> = tuple.get_item(0)?.extract()?;
    let t_arr: PyReadonlyArrayDyn<f64> = tuple.get_item(1)?.extract()?;
    let rotation = arr_to_smatrix::<N, N>(r_arr, "rotation")?;
    let translation = arr_to_svector::<N>(t_arr, "translation")?;
    Ok(SEPoint {
        rotation,
        translation,
    })
}

/// Extract a (rotation, translation) tuple from Python into SETangent<N>.
fn extract_se_tangent<const N: usize>(obj: &Bound<'_, PyAny>) -> PyResult<SETangent<N>> {
    let tuple: &Bound<'_, PyTuple> = obj.downcast()?;
    if tuple.len() != 2 {
        return Err(PyValueError::new_err(
            "SE tangent must be a (rotation, translation) tuple",
        ));
    }
    let r_arr: PyReadonlyArrayDyn<f64> = tuple.get_item(0)?.extract()?;
    let t_arr: PyReadonlyArrayDyn<f64> = tuple.get_item(1)?.extract()?;
    let rotation = arr_to_smatrix::<N, N>(r_arr, "rotation")?;
    let translation = arr_to_svector::<N>(t_arr, "translation")?;
    Ok(SETangent {
        rotation,
        translation,
    })
}

/// Convert SEPoint<N> to a Python (R, t) tuple.
fn se_point_to_py<const N: usize>(py: Python<'_>, p: &SEPoint<N>) -> PyObject {
    let r = smatrix_to_pyarray(py, &p.rotation);
    let t = svector_to_pyarray(py, &p.translation);
    PyTuple::new(py, [r.into_any().unbind(), t.into_any().unbind()])
        .unwrap()
        .unbind()
        .into()
}

/// Convert SETangent<N> to a Python (omega, v) tuple.
fn se_tangent_to_py<const N: usize>(py: Python<'_>, v: &SETangent<N>) -> PyObject {
    let r = smatrix_to_pyarray(py, &v.rotation);
    let t = svector_to_pyarray(py, &v.translation);
    PyTuple::new(py, [r.into_any().unbind(), t.into_any().unbind()])
        .unwrap()
        .unbind()
        .into()
}

/// Build a SpecialEuclidean<N> instance with the given weight.
fn make_se<const N: usize>(weight: f64) -> SpecialEuclidean<N> {
    SpecialEuclidean::<N> { weight }
}

// ---------------------------------------------------------------------------
// Constructor, dim, repr
// ---------------------------------------------------------------------------

#[pymethods]
impl PySe {
    /// Construct an SE manifold of size `n` (2 or 3).
    ///
    /// Points on SE(n) are (R, t) tuples where R is an n x n rotation matrix
    /// (element of SO(n)) and t is an n-vector (translation in R^n).
    ///
    /// The optional `weight` parameter controls the relative importance of
    /// rotation vs. translation in the metric (default: 1.0).
    #[new]
    #[pyo3(signature = (n, weight=1.0))]
    fn new(n: usize, weight: f64) -> PyResult<Self> {
        if n < 2 || n > 3 {
            return Err(PyValueError::new_err(format!(
                "SE: unsupported size {n}, need 2 or 3"
            )));
        }
        Ok(Self { n, weight })
    }

    /// Intrinsic dimension: n*(n+1)/2.
    fn dim(&self) -> usize {
        self.n * (self.n + 1) / 2
    }

    /// Ambient dimension: n*n + n (rotation matrix entries + translation vector).
    fn ambient_dim(&self) -> usize {
        self.n * self.n + self.n
    }

    fn __repr__(&self) -> String {
        format!("SE(n={}, weight={})", self.n, self.weight)
    }
}

// ---------------------------------------------------------------------------
// All trait method wrappers (hand-written, not macro-generated)
// ---------------------------------------------------------------------------

#[pymethods]
impl PySe {
    /// Exponential map: Exp_p(v).
    fn exp<'py>(
        &self,
        py: Python<'py>,
        p: &Bound<'py, PyAny>,
        v: &Bound<'py, PyAny>,
    ) -> PyResult<PyObject> {
        macro_rules! do_it {
            ($N:literal) => {{
                let mf = make_se::<$N>(self.weight);
                let pp = extract_se_point::<$N>(p)?;
                let vv = extract_se_tangent::<$N>(v)?;
                let result = Manifold::exp(&mf, &pp, &vv);
                Ok(se_point_to_py::<$N>(py, &result))
            }};
        }
        match self.n {
            2 => do_it!(2),
            3 => do_it!(3),
            _ => Err(PyValueError::new_err("unreachable")),
        }
    }

    /// Logarithmic map: Log_p(q).
    fn log<'py>(
        &self,
        py: Python<'py>,
        p: &Bound<'py, PyAny>,
        q: &Bound<'py, PyAny>,
    ) -> PyResult<PyObject> {
        macro_rules! do_it {
            ($N:literal) => {{
                let mf = make_se::<$N>(self.weight);
                let pp = extract_se_point::<$N>(p)?;
                let qq = extract_se_point::<$N>(q)?;
                let result =
                    Manifold::log(&mf, &pp, &qq).map_err(cartan_err_to_py)?;
                Ok(se_tangent_to_py::<$N>(py, &result))
            }};
        }
        match self.n {
            2 => do_it!(2),
            3 => do_it!(3),
            _ => Err(PyValueError::new_err("unreachable")),
        }
    }

    /// Geodesic distance d(p, q).
    fn dist(&self, p: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>) -> PyResult<f64> {
        macro_rules! do_it {
            ($N:literal) => {{
                let mf = make_se::<$N>(self.weight);
                let pp = extract_se_point::<$N>(p)?;
                let qq = extract_se_point::<$N>(q)?;
                Manifold::dist(&mf, &pp, &qq).map_err(cartan_err_to_py)
            }};
        }
        match self.n {
            2 => do_it!(2),
            3 => do_it!(3),
            _ => Err(PyValueError::new_err("unreachable")),
        }
    }

    /// Riemannian inner product <u, v>_p.
    fn inner(
        &self,
        p: &Bound<'_, PyAny>,
        u: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        macro_rules! do_it {
            ($N:literal) => {{
                let mf = make_se::<$N>(self.weight);
                let pp = extract_se_point::<$N>(p)?;
                let uu = extract_se_tangent::<$N>(u)?;
                let vv = extract_se_tangent::<$N>(v)?;
                Ok(Manifold::inner(&mf, &pp, &uu, &vv))
            }};
        }
        match self.n {
            2 => do_it!(2),
            3 => do_it!(3),
            _ => Err(PyValueError::new_err("unreachable")),
        }
    }

    /// Induced norm ||v||_p.
    fn norm(&self, p: &Bound<'_, PyAny>, v: &Bound<'_, PyAny>) -> PyResult<f64> {
        macro_rules! do_it {
            ($N:literal) => {{
                let mf = make_se::<$N>(self.weight);
                let pp = extract_se_point::<$N>(p)?;
                let vv = extract_se_tangent::<$N>(v)?;
                Ok(Manifold::norm(&mf, &pp, &vv))
            }};
        }
        match self.n {
            2 => do_it!(2),
            3 => do_it!(3),
            _ => Err(PyValueError::new_err("unreachable")),
        }
    }

    /// Project an ambient point onto the manifold.
    fn project_point<'py>(
        &self,
        py: Python<'py>,
        p: &Bound<'py, PyAny>,
    ) -> PyResult<PyObject> {
        macro_rules! do_it {
            ($N:literal) => {{
                let mf = make_se::<$N>(self.weight);
                let pp = extract_se_point::<$N>(p)?;
                let result = Manifold::project_point(&mf, &pp);
                Ok(se_point_to_py::<$N>(py, &result))
            }};
        }
        match self.n {
            2 => do_it!(2),
            3 => do_it!(3),
            _ => Err(PyValueError::new_err("unreachable")),
        }
    }

    /// Project an ambient vector onto T_p M.
    fn project_tangent<'py>(
        &self,
        py: Python<'py>,
        p: &Bound<'py, PyAny>,
        v: &Bound<'py, PyAny>,
    ) -> PyResult<PyObject> {
        macro_rules! do_it {
            ($N:literal) => {{
                let mf = make_se::<$N>(self.weight);
                let pp = extract_se_point::<$N>(p)?;
                let vv = extract_se_tangent::<$N>(v)?;
                let result = Manifold::project_tangent(&mf, &pp, &vv);
                Ok(se_tangent_to_py::<$N>(py, &result))
            }};
        }
        match self.n {
            2 => do_it!(2),
            3 => do_it!(3),
            _ => Err(PyValueError::new_err("unreachable")),
        }
    }

    /// The zero tangent vector at p.
    fn zero_tangent<'py>(
        &self,
        py: Python<'py>,
        p: &Bound<'py, PyAny>,
    ) -> PyResult<PyObject> {
        macro_rules! do_it {
            ($N:literal) => {{
                let mf = make_se::<$N>(self.weight);
                let pp = extract_se_point::<$N>(p)?;
                let result = Manifold::zero_tangent(&mf, &pp);
                Ok(se_tangent_to_py::<$N>(py, &result))
            }};
        }
        match self.n {
            2 => do_it!(2),
            3 => do_it!(3),
            _ => Err(PyValueError::new_err("unreachable")),
        }
    }

    /// Validate that a point lies on the manifold.
    fn check_point(&self, p: &Bound<'_, PyAny>) -> PyResult<()> {
        macro_rules! do_it {
            ($N:literal) => {{
                let mf = make_se::<$N>(self.weight);
                let pp = extract_se_point::<$N>(p)?;
                Manifold::check_point(&mf, &pp).map_err(cartan_err_to_py)
            }};
        }
        match self.n {
            2 => do_it!(2),
            3 => do_it!(3),
            _ => Err(PyValueError::new_err("unreachable")),
        }
    }

    /// Validate that a tangent vector lies in T_p M.
    fn check_tangent(
        &self,
        p: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        macro_rules! do_it {
            ($N:literal) => {{
                let mf = make_se::<$N>(self.weight);
                let pp = extract_se_point::<$N>(p)?;
                let vv = extract_se_tangent::<$N>(v)?;
                Manifold::check_tangent(&mf, &pp, &vv).map_err(cartan_err_to_py)
            }};
        }
        match self.n {
            2 => do_it!(2),
            3 => do_it!(3),
            _ => Err(PyValueError::new_err("unreachable")),
        }
    }

    /// Random point on the manifold.
    #[pyo3(signature = (seed=None))]
    fn random_point<'py>(
        &self,
        py: Python<'py>,
        seed: Option<u64>,
    ) -> PyResult<PyObject> {
        use rand::SeedableRng;
        macro_rules! do_it {
            ($N:literal) => {{
                let mf = make_se::<$N>(self.weight);
                let result = match seed {
                    Some(s) => {
                        let mut rng = rand::rngs::StdRng::seed_from_u64(s);
                        Manifold::random_point(&mf, &mut rng)
                    }
                    None => Manifold::random_point(&mf, &mut rand::rng()),
                };
                Ok(se_point_to_py::<$N>(py, &result))
            }};
        }
        match self.n {
            2 => do_it!(2),
            3 => do_it!(3),
            _ => Err(PyValueError::new_err("unreachable")),
        }
    }

    /// Random tangent vector at p.
    #[pyo3(signature = (p, seed=None))]
    fn random_tangent<'py>(
        &self,
        py: Python<'py>,
        p: &Bound<'py, PyAny>,
        seed: Option<u64>,
    ) -> PyResult<PyObject> {
        use rand::SeedableRng;
        macro_rules! do_it {
            ($N:literal) => {{
                let mf = make_se::<$N>(self.weight);
                let pp = extract_se_point::<$N>(p)?;
                let result = match seed {
                    Some(s) => {
                        let mut rng = rand::rngs::StdRng::seed_from_u64(s);
                        Manifold::random_tangent(&mf, &pp, &mut rng)
                    }
                    None => Manifold::random_tangent(&mf, &pp, &mut rand::rng()),
                };
                Ok(se_tangent_to_py::<$N>(py, &result))
            }};
        }
        match self.n {
            2 => do_it!(2),
            3 => do_it!(3),
            _ => Err(PyValueError::new_err("unreachable")),
        }
    }

    /// Injectivity radius at p.
    fn injectivity_radius(&self, p: &Bound<'_, PyAny>) -> PyResult<f64> {
        macro_rules! do_it {
            ($N:literal) => {{
                let mf = make_se::<$N>(self.weight);
                let pp = extract_se_point::<$N>(p)?;
                Ok(Manifold::injectivity_radius(&mf, &pp))
            }};
        }
        match self.n {
            2 => do_it!(2),
            3 => do_it!(3),
            _ => Err(PyValueError::new_err("unreachable")),
        }
    }

    /// Retraction: cheap approximation to exp.
    fn retract<'py>(
        &self,
        py: Python<'py>,
        p: &Bound<'py, PyAny>,
        v: &Bound<'py, PyAny>,
    ) -> PyResult<PyObject> {
        macro_rules! do_it {
            ($N:literal) => {{
                let mf = make_se::<$N>(self.weight);
                let pp = extract_se_point::<$N>(p)?;
                let vv = extract_se_tangent::<$N>(v)?;
                let result = Retraction::retract(&mf, &pp, &vv);
                Ok(se_point_to_py::<$N>(py, &result))
            }};
        }
        match self.n {
            2 => do_it!(2),
            3 => do_it!(3),
            _ => Err(PyValueError::new_err("unreachable")),
        }
    }

    /// Inverse retraction: approximate log.
    fn inverse_retract<'py>(
        &self,
        py: Python<'py>,
        p: &Bound<'py, PyAny>,
        q: &Bound<'py, PyAny>,
    ) -> PyResult<PyObject> {
        macro_rules! do_it {
            ($N:literal) => {{
                let mf = make_se::<$N>(self.weight);
                let pp = extract_se_point::<$N>(p)?;
                let qq = extract_se_point::<$N>(q)?;
                let result = Retraction::inverse_retract(&mf, &pp, &qq)
                    .map_err(cartan_err_to_py)?;
                Ok(se_tangent_to_py::<$N>(py, &result))
            }};
        }
        match self.n {
            2 => do_it!(2),
            3 => do_it!(3),
            _ => Err(PyValueError::new_err("unreachable")),
        }
    }

    /// Parallel transport of v from p to q along the geodesic.
    fn parallel_transport<'py>(
        &self,
        py: Python<'py>,
        p: &Bound<'py, PyAny>,
        q: &Bound<'py, PyAny>,
        v: &Bound<'py, PyAny>,
    ) -> PyResult<PyObject> {
        macro_rules! do_it {
            ($N:literal) => {{
                let mf = make_se::<$N>(self.weight);
                let pp = extract_se_point::<$N>(p)?;
                let qq = extract_se_point::<$N>(q)?;
                let vv = extract_se_tangent::<$N>(v)?;
                let result = ParallelTransport::transport(&mf, &pp, &qq, &vv)
                    .map_err(cartan_err_to_py)?;
                Ok(se_tangent_to_py::<$N>(py, &result))
            }};
        }
        match self.n {
            2 => do_it!(2),
            3 => do_it!(3),
            _ => Err(PyValueError::new_err("unreachable")),
        }
    }

    /// Vector transport of v at p in direction u.
    fn vector_transport<'py>(
        &self,
        py: Python<'py>,
        p: &Bound<'py, PyAny>,
        direction: &Bound<'py, PyAny>,
        v: &Bound<'py, PyAny>,
    ) -> PyResult<PyObject> {
        macro_rules! do_it {
            ($N:literal) => {{
                let mf = make_se::<$N>(self.weight);
                let pp = extract_se_point::<$N>(p)?;
                let dd = extract_se_tangent::<$N>(direction)?;
                let vv = extract_se_tangent::<$N>(v)?;
                let result = VectorTransport::vector_transport(&mf, &pp, &dd, &vv)
                    .map_err(cartan_err_to_py)?;
                Ok(se_tangent_to_py::<$N>(py, &result))
            }};
        }
        match self.n {
            2 => do_it!(2),
            3 => do_it!(3),
            _ => Err(PyValueError::new_err("unreachable")),
        }
    }

    /// Sectional curvature of the 2-plane spanned by u and v at p.
    ///
    /// Not implemented for SE(N). SE(N) implements Connection but not the
    /// full Curvature trait, because the Riemann tensor computation for
    /// semidirect products requires additional infrastructure.
    fn sectional_curvature(
        &self,
        _p: &Bound<'_, PyAny>,
        _u: &Bound<'_, PyAny>,
        _v: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        Err(PyNotImplementedError::new_err(
            "SE(N) does not implement the Curvature trait",
        ))
    }

    /// Ricci curvature Ric(u, v) at p.
    ///
    /// Not implemented for SE(N).
    fn ricci_curvature(
        &self,
        _p: &Bound<'_, PyAny>,
        _u: &Bound<'_, PyAny>,
        _v: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        Err(PyNotImplementedError::new_err(
            "SE(N) does not implement the Curvature trait",
        ))
    }

    /// Scalar curvature at p.
    ///
    /// Not implemented for SE(N).
    fn scalar_curvature(&self, _p: &Bound<'_, PyAny>) -> PyResult<f64> {
        Err(PyNotImplementedError::new_err(
            "SE(N) does not implement the Curvature trait",
        ))
    }

    /// Geodesic interpolation: gamma(p, q, t).
    fn geodesic<'py>(
        &self,
        py: Python<'py>,
        p: &Bound<'py, PyAny>,
        q: &Bound<'py, PyAny>,
        t: f64,
    ) -> PyResult<PyObject> {
        macro_rules! do_it {
            ($N:literal) => {{
                let mf = make_se::<$N>(self.weight);
                let pp = extract_se_point::<$N>(p)?;
                let qq = extract_se_point::<$N>(q)?;
                let result = GeodesicInterpolation::geodesic(&mf, &pp, &qq, t)
                    .map_err(cartan_err_to_py)?;
                Ok(se_point_to_py::<$N>(py, &result))
            }};
        }
        match self.n {
            2 => do_it!(2),
            3 => do_it!(3),
            _ => Err(PyValueError::new_err("unreachable")),
        }
    }
}
