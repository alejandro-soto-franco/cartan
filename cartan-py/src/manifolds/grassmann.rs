// ~/cartan/cartan-py/src/manifolds/grassmann.rs

//! Python wrapper for `cartan_manifolds::Grassmann<N, K>`.
//!
//! The Grassmann manifold Gr(N, K) is the set of K-dimensional subspaces
//! of R^N. Points are represented as N x K matrices with orthonormal columns
//! (Q^T Q = I_K). Tangent vectors are N x K matrices V satisfying Q^T V = 0
//! (horizontal condition).
//!
//! Two const-generic parameters (N, K) require dispatch over all valid pairs
//! with 2 <= N <= 8, 1 <= K < N (28 combinations). Each method uses an inner
//! macro to avoid repeating the body.

use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use cartan_core::{
    Curvature, GeodesicInterpolation, Manifold, ParallelTransport, Retraction, VectorTransport,
};

use crate::convert::{arr_to_smatrix, smatrix_to_pyarray};
use crate::error::cartan_err_to_py;

// ---------------------------------------------------------------------------
// Dispatch helper: expands a do_it!($N, $K) call for every valid (N,K) pair.
// ---------------------------------------------------------------------------

/// Expand `$do_it!($N, $K)` for all 28 pairs (2<=N<=8, 1<=K<N), matching on
/// `($self_n, $self_k)`. The caller defines a local `do_it!` macro before
/// invoking this.
macro_rules! dispatch_nk {
    ($self_n:expr, $self_k:expr) => {
        match ($self_n, $self_k) {
            (2, 1) => do_it!(2, 1),
            (3, 1) => do_it!(3, 1),
            (3, 2) => do_it!(3, 2),
            (4, 1) => do_it!(4, 1),
            (4, 2) => do_it!(4, 2),
            (4, 3) => do_it!(4, 3),
            (5, 1) => do_it!(5, 1),
            (5, 2) => do_it!(5, 2),
            (5, 3) => do_it!(5, 3),
            (5, 4) => do_it!(5, 4),
            (6, 1) => do_it!(6, 1),
            (6, 2) => do_it!(6, 2),
            (6, 3) => do_it!(6, 3),
            (6, 4) => do_it!(6, 4),
            (6, 5) => do_it!(6, 5),
            (7, 1) => do_it!(7, 1),
            (7, 2) => do_it!(7, 2),
            (7, 3) => do_it!(7, 3),
            (7, 4) => do_it!(7, 4),
            (7, 5) => do_it!(7, 5),
            (7, 6) => do_it!(7, 6),
            (8, 1) => do_it!(8, 1),
            (8, 2) => do_it!(8, 2),
            (8, 3) => do_it!(8, 3),
            (8, 4) => do_it!(8, 4),
            (8, 5) => do_it!(8, 5),
            (8, 6) => do_it!(8, 6),
            (8, 7) => do_it!(8, 7),
            _ => Err(PyValueError::new_err("unsupported (n, k)")),
        }
    };
}

// ---------------------------------------------------------------------------
// Python class
// ---------------------------------------------------------------------------

/// The Grassmann manifold Gr(n, k) of k-dimensional subspaces of R^n.
///
/// Points are n x k matrices with orthonormal columns.
/// Tangent vectors are n x k matrices orthogonal to the base point.
/// Supported sizes: 2 <= n <= 8, 1 <= k < n.
#[pyclass(name = "Grassmann")]
#[derive(Debug, Clone)]
pub struct PyGrassmann {
    pub(crate) n: usize,
    pub(crate) k: usize,
}

// ---------------------------------------------------------------------------
// Constructor, dim, repr
// ---------------------------------------------------------------------------

#[pymethods]
impl PyGrassmann {
    /// Construct a Grassmann manifold Gr(n, k).
    ///
    /// Parameters: n (ambient dimension), k (subspace dimension).
    /// Constraint: 2 <= n <= 8, 1 <= k < n.
    #[new]
    fn new(n: usize, k: usize) -> PyResult<Self> {
        if n < 2 || n > 8 || k < 1 || k >= n {
            return Err(PyValueError::new_err(format!(
                "Grassmann: unsupported (n={n}, k={k}), need 2<=n<=8, 1<=k<n"
            )));
        }
        Ok(Self { n, k })
    }

    /// Intrinsic dimension: k * (n - k).
    fn dim(&self) -> usize {
        self.k * (self.n - self.k)
    }

    /// Ambient dimension (number of entries in the N x K matrix): n * k.
    fn ambient_dim(&self) -> usize {
        self.n * self.k
    }

    fn __repr__(&self) -> String {
        format!("Grassmann(n={}, k={})", self.n, self.k)
    }
}

// ---------------------------------------------------------------------------
// Trait method wrappers
// ---------------------------------------------------------------------------

#[pymethods]
impl PyGrassmann {
    /// Exponential map: Exp_p(v).
    fn exp<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
        v: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyObject> {
        macro_rules! do_it {
            ($N:literal, $K:literal) => {{
                let mf = cartan_manifolds::Grassmann::<$N, $K>;
                let pp = arr_to_smatrix::<$N, $K>(p, "p")?;
                let vv = arr_to_smatrix::<$N, $K>(v, "v")?;
                let result = Manifold::exp(&mf, &pp, &vv);
                Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
            }};
        }
        dispatch_nk!(self.n, self.k)
    }

    /// Logarithmic map: Log_p(q).
    fn log<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
        q: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyObject> {
        macro_rules! do_it {
            ($N:literal, $K:literal) => {{
                let mf = cartan_manifolds::Grassmann::<$N, $K>;
                let pp = arr_to_smatrix::<$N, $K>(p, "p")?;
                let qq = arr_to_smatrix::<$N, $K>(q, "q")?;
                let result = Manifold::log(&mf, &pp, &qq).map_err(cartan_err_to_py)?;
                Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
            }};
        }
        dispatch_nk!(self.n, self.k)
    }

    /// Geodesic distance d(p, q).
    fn dist(
        &self,
        p: PyReadonlyArrayDyn<'_, f64>,
        q: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<f64> {
        macro_rules! do_it {
            ($N:literal, $K:literal) => {{
                let mf = cartan_manifolds::Grassmann::<$N, $K>;
                let pp = arr_to_smatrix::<$N, $K>(p, "p")?;
                let qq = arr_to_smatrix::<$N, $K>(q, "q")?;
                Manifold::dist(&mf, &pp, &qq).map_err(cartan_err_to_py)
            }};
        }
        dispatch_nk!(self.n, self.k)
    }

    /// Riemannian inner product <u, v>_p.
    fn inner(
        &self,
        p: PyReadonlyArrayDyn<'_, f64>,
        u: PyReadonlyArrayDyn<'_, f64>,
        v: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<f64> {
        macro_rules! do_it {
            ($N:literal, $K:literal) => {{
                let mf = cartan_manifolds::Grassmann::<$N, $K>;
                let pp = arr_to_smatrix::<$N, $K>(p, "p")?;
                let uu = arr_to_smatrix::<$N, $K>(u, "u")?;
                let vv = arr_to_smatrix::<$N, $K>(v, "v")?;
                Ok(Manifold::inner(&mf, &pp, &uu, &vv))
            }};
        }
        dispatch_nk!(self.n, self.k)
    }

    /// Induced norm ||v||_p.
    fn norm(
        &self,
        p: PyReadonlyArrayDyn<'_, f64>,
        v: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<f64> {
        macro_rules! do_it {
            ($N:literal, $K:literal) => {{
                let mf = cartan_manifolds::Grassmann::<$N, $K>;
                let pp = arr_to_smatrix::<$N, $K>(p, "p")?;
                let vv = arr_to_smatrix::<$N, $K>(v, "v")?;
                Ok(Manifold::norm(&mf, &pp, &vv))
            }};
        }
        dispatch_nk!(self.n, self.k)
    }

    /// Project an ambient point onto the manifold.
    fn project_point<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyObject> {
        macro_rules! do_it {
            ($N:literal, $K:literal) => {{
                let mf = cartan_manifolds::Grassmann::<$N, $K>;
                let pp = arr_to_smatrix::<$N, $K>(p, "p")?;
                let result = Manifold::project_point(&mf, &pp);
                Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
            }};
        }
        dispatch_nk!(self.n, self.k)
    }

    /// Project an ambient vector onto T_p M.
    fn project_tangent<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
        v: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyObject> {
        macro_rules! do_it {
            ($N:literal, $K:literal) => {{
                let mf = cartan_manifolds::Grassmann::<$N, $K>;
                let pp = arr_to_smatrix::<$N, $K>(p, "p")?;
                let vv = arr_to_smatrix::<$N, $K>(v, "v")?;
                let result = Manifold::project_tangent(&mf, &pp, &vv);
                Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
            }};
        }
        dispatch_nk!(self.n, self.k)
    }

    /// The zero tangent vector at p.
    fn zero_tangent<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyObject> {
        macro_rules! do_it {
            ($N:literal, $K:literal) => {{
                let mf = cartan_manifolds::Grassmann::<$N, $K>;
                let pp = arr_to_smatrix::<$N, $K>(p, "p")?;
                let result = Manifold::zero_tangent(&mf, &pp);
                Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
            }};
        }
        dispatch_nk!(self.n, self.k)
    }

    /// Validate that a point lies on the manifold.
    fn check_point(&self, p: PyReadonlyArrayDyn<'_, f64>) -> PyResult<()> {
        macro_rules! do_it {
            ($N:literal, $K:literal) => {{
                let mf = cartan_manifolds::Grassmann::<$N, $K>;
                let pp = arr_to_smatrix::<$N, $K>(p, "p")?;
                Manifold::check_point(&mf, &pp).map_err(cartan_err_to_py)
            }};
        }
        dispatch_nk!(self.n, self.k)
    }

    /// Validate that a tangent vector lies in T_p M.
    fn check_tangent(
        &self,
        p: PyReadonlyArrayDyn<'_, f64>,
        v: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<()> {
        macro_rules! do_it {
            ($N:literal, $K:literal) => {{
                let mf = cartan_manifolds::Grassmann::<$N, $K>;
                let pp = arr_to_smatrix::<$N, $K>(p, "p")?;
                let vv = arr_to_smatrix::<$N, $K>(v, "v")?;
                Manifold::check_tangent(&mf, &pp, &vv).map_err(cartan_err_to_py)
            }};
        }
        dispatch_nk!(self.n, self.k)
    }

    /// Random point on the manifold.
    #[pyo3(signature = (seed=None))]
    fn random_point<'py>(&self, py: Python<'py>, seed: Option<u64>) -> PyResult<PyObject> {
        use rand::SeedableRng;
        macro_rules! do_it {
            ($N:literal, $K:literal) => {{
                let mf = cartan_manifolds::Grassmann::<$N, $K>;
                let result = match seed {
                    Some(s) => {
                        let mut rng = rand::rngs::StdRng::seed_from_u64(s);
                        Manifold::random_point(&mf, &mut rng)
                    }
                    None => Manifold::random_point(&mf, &mut rand::rng()),
                };
                Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
            }};
        }
        dispatch_nk!(self.n, self.k)
    }

    /// Random tangent vector at p.
    #[pyo3(signature = (p, seed=None))]
    fn random_tangent<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
        seed: Option<u64>,
    ) -> PyResult<PyObject> {
        use rand::SeedableRng;
        macro_rules! do_it {
            ($N:literal, $K:literal) => {{
                let mf = cartan_manifolds::Grassmann::<$N, $K>;
                let pp = arr_to_smatrix::<$N, $K>(p, "p")?;
                let result = match seed {
                    Some(s) => {
                        let mut rng = rand::rngs::StdRng::seed_from_u64(s);
                        Manifold::random_tangent(&mf, &pp, &mut rng)
                    }
                    None => Manifold::random_tangent(&mf, &pp, &mut rand::rng()),
                };
                Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
            }};
        }
        dispatch_nk!(self.n, self.k)
    }

    /// Injectivity radius at p.
    fn injectivity_radius(&self, p: PyReadonlyArrayDyn<'_, f64>) -> PyResult<f64> {
        macro_rules! do_it {
            ($N:literal, $K:literal) => {{
                let mf = cartan_manifolds::Grassmann::<$N, $K>;
                let pp = arr_to_smatrix::<$N, $K>(p, "p")?;
                Ok(Manifold::injectivity_radius(&mf, &pp))
            }};
        }
        dispatch_nk!(self.n, self.k)
    }

    /// Retraction: cheap approximation to exp.
    fn retract<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
        v: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyObject> {
        macro_rules! do_it {
            ($N:literal, $K:literal) => {{
                let mf = cartan_manifolds::Grassmann::<$N, $K>;
                let pp = arr_to_smatrix::<$N, $K>(p, "p")?;
                let vv = arr_to_smatrix::<$N, $K>(v, "v")?;
                let result = Retraction::retract(&mf, &pp, &vv);
                Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
            }};
        }
        dispatch_nk!(self.n, self.k)
    }

    /// Inverse retraction: approximate log.
    fn inverse_retract<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
        q: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyObject> {
        macro_rules! do_it {
            ($N:literal, $K:literal) => {{
                let mf = cartan_manifolds::Grassmann::<$N, $K>;
                let pp = arr_to_smatrix::<$N, $K>(p, "p")?;
                let qq = arr_to_smatrix::<$N, $K>(q, "q")?;
                let result =
                    Retraction::inverse_retract(&mf, &pp, &qq).map_err(cartan_err_to_py)?;
                Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
            }};
        }
        dispatch_nk!(self.n, self.k)
    }

    /// Parallel transport of v from p to q along the geodesic.
    fn parallel_transport<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
        q: PyReadonlyArrayDyn<'py, f64>,
        v: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyObject> {
        macro_rules! do_it {
            ($N:literal, $K:literal) => {{
                let mf = cartan_manifolds::Grassmann::<$N, $K>;
                let pp = arr_to_smatrix::<$N, $K>(p, "p")?;
                let qq = arr_to_smatrix::<$N, $K>(q, "q")?;
                let vv = arr_to_smatrix::<$N, $K>(v, "v")?;
                let result =
                    ParallelTransport::transport(&mf, &pp, &qq, &vv).map_err(cartan_err_to_py)?;
                Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
            }};
        }
        dispatch_nk!(self.n, self.k)
    }

    /// Vector transport of v at p in direction u.
    fn vector_transport<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
        direction: PyReadonlyArrayDyn<'py, f64>,
        v: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyObject> {
        macro_rules! do_it {
            ($N:literal, $K:literal) => {{
                let mf = cartan_manifolds::Grassmann::<$N, $K>;
                let pp = arr_to_smatrix::<$N, $K>(p, "p")?;
                let dd = arr_to_smatrix::<$N, $K>(direction, "direction")?;
                let vv = arr_to_smatrix::<$N, $K>(v, "v")?;
                let result = VectorTransport::vector_transport(&mf, &pp, &dd, &vv)
                    .map_err(cartan_err_to_py)?;
                Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
            }};
        }
        dispatch_nk!(self.n, self.k)
    }

    /// Sectional curvature of the 2-plane spanned by u and v at p.
    fn sectional_curvature(
        &self,
        p: PyReadonlyArrayDyn<'_, f64>,
        u: PyReadonlyArrayDyn<'_, f64>,
        v: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<f64> {
        macro_rules! do_it {
            ($N:literal, $K:literal) => {{
                let mf = cartan_manifolds::Grassmann::<$N, $K>;
                let pp = arr_to_smatrix::<$N, $K>(p, "p")?;
                let uu = arr_to_smatrix::<$N, $K>(u, "u")?;
                let vv = arr_to_smatrix::<$N, $K>(v, "v")?;
                Ok(Curvature::sectional_curvature(&mf, &pp, &uu, &vv))
            }};
        }
        dispatch_nk!(self.n, self.k)
    }

    /// Ricci curvature Ric(u, v) at p.
    fn ricci_curvature(
        &self,
        p: PyReadonlyArrayDyn<'_, f64>,
        u: PyReadonlyArrayDyn<'_, f64>,
        v: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<f64> {
        macro_rules! do_it {
            ($N:literal, $K:literal) => {{
                let mf = cartan_manifolds::Grassmann::<$N, $K>;
                let pp = arr_to_smatrix::<$N, $K>(p, "p")?;
                let uu = arr_to_smatrix::<$N, $K>(u, "u")?;
                let vv = arr_to_smatrix::<$N, $K>(v, "v")?;
                Ok(Curvature::ricci_curvature(&mf, &pp, &uu, &vv))
            }};
        }
        dispatch_nk!(self.n, self.k)
    }

    /// Scalar curvature at p.
    fn scalar_curvature(&self, p: PyReadonlyArrayDyn<'_, f64>) -> PyResult<f64> {
        macro_rules! do_it {
            ($N:literal, $K:literal) => {{
                let mf = cartan_manifolds::Grassmann::<$N, $K>;
                let pp = arr_to_smatrix::<$N, $K>(p, "p")?;
                Ok(Curvature::scalar_curvature(&mf, &pp))
            }};
        }
        dispatch_nk!(self.n, self.k)
    }

    /// Geodesic interpolation: gamma(p, q, t).
    fn geodesic<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArrayDyn<'py, f64>,
        q: PyReadonlyArrayDyn<'py, f64>,
        t: f64,
    ) -> PyResult<PyObject> {
        macro_rules! do_it {
            ($N:literal, $K:literal) => {{
                let mf = cartan_manifolds::Grassmann::<$N, $K>;
                let pp = arr_to_smatrix::<$N, $K>(p, "p")?;
                let qq = arr_to_smatrix::<$N, $K>(q, "q")?;
                let result =
                    GeodesicInterpolation::geodesic(&mf, &pp, &qq, t).map_err(cartan_err_to_py)?;
                Ok(smatrix_to_pyarray(py, &result).into_any().unbind())
            }};
        }
        dispatch_nk!(self.n, self.k)
    }
}
