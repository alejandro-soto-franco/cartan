// ~/cartan/cartan-py/src/manifolds/qtensor.rs

//! Python wrapper for `cartan_manifolds::qtensor::QTensor3`.
//!
//! QTensor3 is the space of 3x3 symmetric traceless real matrices (Sym_0(R^3)),
//! which is the Q-tensor order parameter space for nematic liquid crystals. It is
//! a 5-dimensional flat Riemannian manifold with the Frobenius metric.
//!
//! Being a flat manifold, all curvatures are zero, parallel transport is the
//! identity, and geodesics are straight lines (exp = addition, log = subtraction).

use pyo3::prelude::*;
use numpy::{self, PyReadonlyArrayDyn};

use cartan_core::{
    Manifold, Retraction, ParallelTransport, Curvature, GeodesicInterpolation, Real,
};
use cartan_manifolds::qtensor::QTensor3;

use crate::convert::{arr_to_smatrix, smatrix_to_pyarray};
use crate::error::cartan_err_to_py;

/// Python wrapper for the Q-tensor manifold: 3x3 symmetric traceless matrices.
///
/// QTensor3 is the 5-dimensional flat Riemannian manifold Sym_0(R^3) equipped
/// with the Frobenius inner product. Points are 3x3 numpy arrays satisfying
/// symmetry, tracelessness, and eigenvalue bounds [-1/3, 2/3].
///
/// This manifold models the Q-tensor order parameter of nematic liquid crystals.
#[pyclass(name = "QTensor3")]
#[derive(Debug, Clone)]
pub struct PyQTensor3;

#[pymethods]
impl PyQTensor3 {
    /// Construct the Q-tensor manifold (no parameters).
    ///
    /// Points are 3x3 numpy arrays representing symmetric traceless matrices
    /// with eigenvalues in [-1/3, 2/3].
    #[new]
    fn new() -> Self {
        Self
    }

    /// Intrinsic dimension of Sym_0(R^3): 5.
    fn dim(&self) -> usize {
        5
    }

    /// Ambient dimension: 9 (all entries of a 3x3 matrix).
    fn ambient_dim(&self) -> usize {
        9
    }

    /// Exponential map: Exp_p(v) = p + v (straight-line in flat Q-space).
    fn exp<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
        v: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyObject> {
        let mf = QTensor3;
        let pp = arr_to_smatrix::<3, 3>(p, "p")?;
        let vv = arr_to_smatrix::<3, 3>(v, "v")?;
        let result = Manifold::exp(&mf, &pp, &vv);
        Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
    }

    /// Logarithmic map: Log_p(q) = q - p.
    fn log<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
        q: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyObject> {
        let mf = QTensor3;
        let pp = arr_to_smatrix::<3, 3>(p, "p")?;
        let qq = arr_to_smatrix::<3, 3>(q, "q")?;
        let result = Manifold::log(&mf, &pp, &qq).map_err(cartan_err_to_py)?;
        Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
    }

    /// Geodesic distance d(p, q) = ||q - p||_F.
    fn dist(
        &self,
        p: PyReadonlyArrayDyn<'_, f64>,
        q: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<f64> {
        let mf = QTensor3;
        let pp = arr_to_smatrix::<3, 3>(p, "p")?;
        let qq = arr_to_smatrix::<3, 3>(q, "q")?;
        Manifold::dist(&mf, &pp, &qq).map_err(cartan_err_to_py)
    }

    /// Frobenius inner product <u, v>_p = tr(u v) (independent of base point).
    fn inner(
        &self,
        p: PyReadonlyArrayDyn<'_, f64>,
        u: PyReadonlyArrayDyn<'_, f64>,
        v: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<f64> {
        let mf = QTensor3;
        let pp = arr_to_smatrix::<3, 3>(p, "p")?;
        let uu = arr_to_smatrix::<3, 3>(u, "u")?;
        let vv = arr_to_smatrix::<3, 3>(v, "v")?;
        Ok(Manifold::inner(&mf, &pp, &uu, &vv))
    }

    /// Frobenius norm ||v||_p = sqrt(tr(v^2)).
    fn norm(
        &self,
        p: PyReadonlyArrayDyn<'_, f64>,
        v: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<f64> {
        let mf = QTensor3;
        let pp = arr_to_smatrix::<3, 3>(p, "p")?;
        let vv = arr_to_smatrix::<3, 3>(v, "v")?;
        Ok(Manifold::norm(&mf, &pp, &vv))
    }

    /// Project an arbitrary 3x3 matrix onto the physical Q-manifold.
    ///
    /// Symmetrizes, removes trace, then clamps eigenvalues to [-1/3, 2/3].
    fn project_point<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyObject> {
        let mf = QTensor3;
        let pp = arr_to_smatrix::<3, 3>(p, "p")?;
        let result = Manifold::project_point(&mf, &pp);
        Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
    }

    /// Project an ambient matrix onto the sym-traceless tangent space.
    fn project_tangent<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
        v: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyObject> {
        let mf = QTensor3;
        let pp = arr_to_smatrix::<3, 3>(p, "p")?;
        let vv = arr_to_smatrix::<3, 3>(v, "v")?;
        let result = Manifold::project_tangent(&mf, &pp, &vv);
        Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
    }

    /// Zero tangent vector at p: the 3x3 zero matrix.
    fn zero_tangent<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyObject> {
        let mf = QTensor3;
        let pp = arr_to_smatrix::<3, 3>(p, "p")?;
        let result = Manifold::zero_tangent(&mf, &pp);
        Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
    }

    /// Validate that a matrix is a physical Q-tensor (symmetric, traceless, eigenvalues in [-1/3, 2/3]).
    fn check_point(
        &self,
        p: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<()> {
        let mf = QTensor3;
        let pp = arr_to_smatrix::<3, 3>(p, "p")?;
        Manifold::check_point(&mf, &pp).map_err(cartan_err_to_py)
    }

    /// Validate that a matrix is a valid tangent vector (symmetric, traceless).
    fn check_tangent(
        &self,
        p: PyReadonlyArrayDyn<'_, f64>,
        v: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<()> {
        let mf = QTensor3;
        let pp = arr_to_smatrix::<3, 3>(p, "p")?;
        let vv = arr_to_smatrix::<3, 3>(v, "v")?;
        Manifold::check_tangent(&mf, &pp, &vv).map_err(cartan_err_to_py)
    }

    /// Sample a random Q-tensor (weakly ordered, Frobenius norm ~ 0.05).
    #[pyo3(signature = (seed=None))]
    fn random_point<'py>(
        &self,
        py: Python<'py>,
        seed: Option<u64>,
    ) -> PyResult<PyObject> {
        let mf = QTensor3;
        use rand::SeedableRng;
        let result = match seed {
            Some(s) => {
                let mut rng = rand::rngs::StdRng::seed_from_u64(s);
                Manifold::random_point(&mf, &mut rng)
            }
            None => {
                Manifold::random_point(&mf, &mut rand::rng())
            }
        };
        Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
    }

    /// Sample a random unit-norm sym-traceless tangent vector at p.
    #[pyo3(signature = (p, seed=None))]
    fn random_tangent<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
        seed: Option<u64>,
    ) -> PyResult<PyObject> {
        let mf = QTensor3;
        let pp = arr_to_smatrix::<3, 3>(p, "p")?;
        use rand::SeedableRng;
        let result = match seed {
            Some(s) => {
                let mut rng = rand::rngs::StdRng::seed_from_u64(s);
                Manifold::random_tangent(&mf, &pp, &mut rng)
            }
            None => {
                Manifold::random_tangent(&mf, &pp, &mut rand::rng())
            }
        };
        Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
    }

    /// Injectivity radius: infinity (flat manifold, no cut locus).
    fn injectivity_radius(
        &self,
        p: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<f64> {
        let mf = QTensor3;
        let pp = arr_to_smatrix::<3, 3>(p, "p")?;
        Ok(Manifold::injectivity_radius(&mf, &pp))
    }

    /// Retraction: p + v (same as exp for flat Q-space).
    fn retract<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
        v: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyObject> {
        let mf = QTensor3;
        let pp = arr_to_smatrix::<3, 3>(p, "p")?;
        let vv = arr_to_smatrix::<3, 3>(v, "v")?;
        let result = Retraction::retract(&mf, &pp, &vv);
        Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
    }

    /// Inverse retraction: q - p (same as log for flat Q-space).
    fn inverse_retract<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
        q: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyObject> {
        let mf = QTensor3;
        let pp = arr_to_smatrix::<3, 3>(p, "p")?;
        let qq = arr_to_smatrix::<3, 3>(q, "q")?;
        let result = Retraction::inverse_retract(&mf, &pp, &qq).map_err(cartan_err_to_py)?;
        Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
    }

    /// Parallel transport of v from p to q: identity (flat manifold).
    fn parallel_transport<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
        q: PyReadonlyArrayDyn<'py, f64>,
        v: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyObject> {
        let mf = QTensor3;
        let pp = arr_to_smatrix::<3, 3>(p, "p")?;
        let qq = arr_to_smatrix::<3, 3>(q, "q")?;
        let vv = arr_to_smatrix::<3, 3>(v, "v")?;
        let result = ParallelTransport::transport(&mf, &pp, &qq, &vv).map_err(cartan_err_to_py)?;
        Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
    }

    /// Sectional curvature: 0.0 (flat manifold).
    fn sectional_curvature(
        &self,
        p: PyReadonlyArrayDyn<'_, f64>,
        u: PyReadonlyArrayDyn<'_, f64>,
        v: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<f64> {
        let mf = QTensor3;
        let pp = arr_to_smatrix::<3, 3>(p, "p")?;
        let uu = arr_to_smatrix::<3, 3>(u, "u")?;
        let vv = arr_to_smatrix::<3, 3>(v, "v")?;
        Ok(Curvature::sectional_curvature(&mf, &pp, &uu, &vv))
    }

    /// Ricci curvature: 0.0 (flat manifold).
    fn ricci_curvature(
        &self,
        p: PyReadonlyArrayDyn<'_, f64>,
        u: PyReadonlyArrayDyn<'_, f64>,
        v: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<f64> {
        let mf = QTensor3;
        let pp = arr_to_smatrix::<3, 3>(p, "p")?;
        let uu = arr_to_smatrix::<3, 3>(u, "u")?;
        let vv = arr_to_smatrix::<3, 3>(v, "v")?;
        Ok(Curvature::ricci_curvature(&mf, &pp, &uu, &vv))
    }

    /// Scalar curvature: 0.0 (flat manifold).
    fn scalar_curvature(
        &self,
        p: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<f64> {
        let mf = QTensor3;
        let pp = arr_to_smatrix::<3, 3>(p, "p")?;
        Ok(Curvature::scalar_curvature(&mf, &pp))
    }

    /// Geodesic interpolation: (1-t)*p + t*q (straight line in flat Q-space).
    fn geodesic<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
        q: PyReadonlyArrayDyn<'py, f64>,
        t: f64,
    ) -> PyResult<PyObject> {
        let mf = QTensor3;
        let pp = arr_to_smatrix::<3, 3>(p, "p")?;
        let qq = arr_to_smatrix::<3, 3>(q, "q")?;
        let result = GeodesicInterpolation::geodesic(&mf, &pp, &qq, t as Real)
            .map_err(cartan_err_to_py)?;
        Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
    }

    /// Pairwise distance matrix D[i,j] = dist(points[i], points[j]).
    fn dist_matrix<'py>(
        &self,
        py: Python<'py>,
        points: Vec<PyReadonlyArrayDyn<'py, f64>>,
    ) -> PyResult<PyObject> {
        let mf = QTensor3;
        let pts: Vec<_> = points.into_iter()
            .enumerate()
            .map(|(i, arr)| arr_to_smatrix::<3, 3>(arr, &format!("points[{i}]")))
            .collect::<PyResult<_>>()?;
        let n = pts.len();
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = vec![0.0f64; n];
            for j in (i + 1)..n {
                let d = Manifold::dist(&mf, &pts[i], &pts[j])
                    .map_err(cartan_err_to_py)?;
                row[j] = d;
            }
            rows.push(row);
        }
        for i in 0..n {
            for j in 0..i {
                rows[i][j] = rows[j][i];
            }
        }
        let arr = numpy::PyArray2::from_vec2(py, &rows).expect("valid shape");
        Ok(arr.into_any().unbind())
    }

    /// Apply exp_p to a batch of tangent vectors, returning a list of points.
    fn exp_batch<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
        vs: Vec<PyReadonlyArrayDyn<'py, f64>>,
    ) -> PyResult<Vec<PyObject>> {
        let mf = QTensor3;
        let pp = arr_to_smatrix::<3, 3>(p, "p")?;
        vs.into_iter()
            .enumerate()
            .map(|(i, arr)| {
                let vv = arr_to_smatrix::<3, 3>(arr, &format!("vs[{i}]"))?;
                let result = Manifold::exp(&mf, &pp, &vv);
                Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        "QTensor3()".to_string()
    }
}
