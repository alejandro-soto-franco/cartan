// ~/cartan/cartan-py/src/manifolds/macros.rs

//! Dimension-dispatch macros for manifold method generation.
//!
//! These macros generate complete `#[pymethods] impl` blocks for Python
//! wrapper structs. Each manifold file invokes one macro at module level
//! (outside any `#[pymethods]` block) to avoid duplicating the conversion
//! and dispatch boilerplate.
//!
//! pyo3's `#[pymethods]` proc-macro does not permit macro invocations as
//! items inside the impl block, so all methods must be emitted by a macro
//! that wraps the entire `#[pymethods] impl $pytype { ... }` form.

/// Generate all Manifold trait method wrappers for manifolds whose Point
/// and Tangent types are `SVector<Real, N>` (e.g. Euclidean, Sphere).
///
/// Usage (at module level, outside any impl block):
/// ```ignore
/// impl_vector_manifold_methods!(PyEuclidean, Euclidean, n, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
/// ```
///
/// - `$pytype`    : the `#[pyclass]` struct name
/// - `$mtype`     : the `cartan_manifolds` type name
/// - `$dim_field` : ident of the runtime-dimension field on `$pytype`
/// - `[$($N),+]`  : list of supported const-generic dimensions
#[macro_export]
macro_rules! impl_vector_manifold_methods {
    ($pytype:ident, $mtype:ident, $dim_field:ident, [$($N:literal),+ $(,)?]) => {

        #[pyo3::pymethods]
        impl $pytype {

            /// Exponential map: Exp_p(v).
            fn exp<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
                v: numpy::PyReadonlyArrayDyn<'py, f64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_svector::<$N>(p, "p")?;
                        let vv = $crate::convert::arr_to_svector::<$N>(v, "v")?;
                        let result = cartan_core::Manifold::exp(&mf, &pp, &vv);
                        Ok($crate::convert::svector_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Logarithmic map: Log_p(q).
            fn log<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
                q: numpy::PyReadonlyArrayDyn<'py, f64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_svector::<$N>(p, "p")?;
                        let qq = $crate::convert::arr_to_svector::<$N>(q, "q")?;
                        let result = cartan_core::Manifold::log(&mf, &pp, &qq)
                            .map_err($crate::error::cartan_err_to_py)?;
                        Ok($crate::convert::svector_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Geodesic distance d(p, q).
            fn dist(
                &self,
                p: numpy::PyReadonlyArrayDyn<'_, f64>,
                q: numpy::PyReadonlyArrayDyn<'_, f64>,
            ) -> pyo3::PyResult<f64> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_svector::<$N>(p, "p")?;
                        let qq = $crate::convert::arr_to_svector::<$N>(q, "q")?;
                        cartan_core::Manifold::dist(&mf, &pp, &qq)
                            .map_err($crate::error::cartan_err_to_py)
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Riemannian inner product <u, v>_p.
            fn inner(
                &self,
                p: numpy::PyReadonlyArrayDyn<'_, f64>,
                u: numpy::PyReadonlyArrayDyn<'_, f64>,
                v: numpy::PyReadonlyArrayDyn<'_, f64>,
            ) -> pyo3::PyResult<f64> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_svector::<$N>(p, "p")?;
                        let uu = $crate::convert::arr_to_svector::<$N>(u, "u")?;
                        let vv = $crate::convert::arr_to_svector::<$N>(v, "v")?;
                        Ok(cartan_core::Manifold::inner(&mf, &pp, &uu, &vv))
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Induced norm ||v||_p.
            fn norm(
                &self,
                p: numpy::PyReadonlyArrayDyn<'_, f64>,
                v: numpy::PyReadonlyArrayDyn<'_, f64>,
            ) -> pyo3::PyResult<f64> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_svector::<$N>(p, "p")?;
                        let vv = $crate::convert::arr_to_svector::<$N>(v, "v")?;
                        Ok(cartan_core::Manifold::norm(&mf, &pp, &vv))
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Project an ambient point onto the manifold.
            fn project_point<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_svector::<$N>(p, "p")?;
                        let result = cartan_core::Manifold::project_point(&mf, &pp);
                        Ok($crate::convert::svector_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Project an ambient vector onto T_p M.
            fn project_tangent<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
                v: numpy::PyReadonlyArrayDyn<'py, f64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_svector::<$N>(p, "p")?;
                        let vv = $crate::convert::arr_to_svector::<$N>(v, "v")?;
                        let result = cartan_core::Manifold::project_tangent(&mf, &pp, &vv);
                        Ok($crate::convert::svector_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// The zero tangent vector at p.
            fn zero_tangent<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_svector::<$N>(p, "p")?;
                        let result = cartan_core::Manifold::zero_tangent(&mf, &pp);
                        Ok($crate::convert::svector_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Validate that a point lies on the manifold.
            fn check_point(
                &self,
                p: numpy::PyReadonlyArrayDyn<'_, f64>,
            ) -> pyo3::PyResult<()> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_svector::<$N>(p, "p")?;
                        cartan_core::Manifold::check_point(&mf, &pp)
                            .map_err($crate::error::cartan_err_to_py)
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Validate that a tangent vector lies in T_p M.
            fn check_tangent(
                &self,
                p: numpy::PyReadonlyArrayDyn<'_, f64>,
                v: numpy::PyReadonlyArrayDyn<'_, f64>,
            ) -> pyo3::PyResult<()> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_svector::<$N>(p, "p")?;
                        let vv = $crate::convert::arr_to_svector::<$N>(v, "v")?;
                        cartan_core::Manifold::check_tangent(&mf, &pp, &vv)
                            .map_err($crate::error::cartan_err_to_py)
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Random point on the manifold.
            #[pyo3(signature = (seed=None))]
            fn random_point<'py>(
                &self,
                py: pyo3::Python<'py>,
                seed: Option<u64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                use rand::SeedableRng;
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let result = match seed {
                            Some(s) => {
                                let mut rng = rand::rngs::StdRng::seed_from_u64(s);
                                cartan_core::Manifold::random_point(&mf, &mut rng)
                            }
                            None => {
                                cartan_core::Manifold::random_point(&mf, &mut rand::rng())
                            }
                        };
                        Ok($crate::convert::svector_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Random tangent vector at p.
            #[pyo3(signature = (p, seed=None))]
            fn random_tangent<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
                seed: Option<u64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                use rand::SeedableRng;
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_svector::<$N>(p, "p")?;
                        let result = match seed {
                            Some(s) => {
                                let mut rng = rand::rngs::StdRng::seed_from_u64(s);
                                cartan_core::Manifold::random_tangent(&mf, &pp, &mut rng)
                            }
                            None => {
                                cartan_core::Manifold::random_tangent(&mf, &pp, &mut rand::rng())
                            }
                        };
                        Ok($crate::convert::svector_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Injectivity radius at p.
            fn injectivity_radius(
                &self,
                p: numpy::PyReadonlyArrayDyn<'_, f64>,
            ) -> pyo3::PyResult<f64> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_svector::<$N>(p, "p")?;
                        Ok(cartan_core::Manifold::injectivity_radius(&mf, &pp))
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Retraction: cheap approximation to exp.
            fn retract<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
                v: numpy::PyReadonlyArrayDyn<'py, f64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_svector::<$N>(p, "p")?;
                        let vv = $crate::convert::arr_to_svector::<$N>(v, "v")?;
                        let result = cartan_core::Retraction::retract(&mf, &pp, &vv);
                        Ok($crate::convert::svector_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Inverse retraction: approximate log.
            fn inverse_retract<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
                q: numpy::PyReadonlyArrayDyn<'py, f64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_svector::<$N>(p, "p")?;
                        let qq = $crate::convert::arr_to_svector::<$N>(q, "q")?;
                        let result = cartan_core::Retraction::inverse_retract(&mf, &pp, &qq)
                            .map_err($crate::error::cartan_err_to_py)?;
                        Ok($crate::convert::svector_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Parallel transport of v from p to q along the geodesic.
            fn parallel_transport<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
                q: numpy::PyReadonlyArrayDyn<'py, f64>,
                v: numpy::PyReadonlyArrayDyn<'py, f64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_svector::<$N>(p, "p")?;
                        let qq = $crate::convert::arr_to_svector::<$N>(q, "q")?;
                        let vv = $crate::convert::arr_to_svector::<$N>(v, "v")?;
                        let result = cartan_core::ParallelTransport::transport(&mf, &pp, &qq, &vv)
                            .map_err($crate::error::cartan_err_to_py)?;
                        Ok($crate::convert::svector_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Vector transport of v at p in direction u.
            fn vector_transport<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
                direction: numpy::PyReadonlyArrayDyn<'py, f64>,
                v: numpy::PyReadonlyArrayDyn<'py, f64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_svector::<$N>(p, "p")?;
                        let dd = $crate::convert::arr_to_svector::<$N>(direction, "direction")?;
                        let vv = $crate::convert::arr_to_svector::<$N>(v, "v")?;
                        let result = cartan_core::VectorTransport::vector_transport(&mf, &pp, &dd, &vv)
                            .map_err($crate::error::cartan_err_to_py)?;
                        Ok($crate::convert::svector_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Sectional curvature of the 2-plane spanned by u and v at p.
            fn sectional_curvature(
                &self,
                p: numpy::PyReadonlyArrayDyn<'_, f64>,
                u: numpy::PyReadonlyArrayDyn<'_, f64>,
                v: numpy::PyReadonlyArrayDyn<'_, f64>,
            ) -> pyo3::PyResult<f64> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_svector::<$N>(p, "p")?;
                        let uu = $crate::convert::arr_to_svector::<$N>(u, "u")?;
                        let vv = $crate::convert::arr_to_svector::<$N>(v, "v")?;
                        Ok(cartan_core::Curvature::sectional_curvature(&mf, &pp, &uu, &vv))
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Ricci curvature Ric(u, v) at p.
            fn ricci_curvature(
                &self,
                p: numpy::PyReadonlyArrayDyn<'_, f64>,
                u: numpy::PyReadonlyArrayDyn<'_, f64>,
                v: numpy::PyReadonlyArrayDyn<'_, f64>,
            ) -> pyo3::PyResult<f64> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_svector::<$N>(p, "p")?;
                        let uu = $crate::convert::arr_to_svector::<$N>(u, "u")?;
                        let vv = $crate::convert::arr_to_svector::<$N>(v, "v")?;
                        Ok(cartan_core::Curvature::ricci_curvature(&mf, &pp, &uu, &vv))
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Scalar curvature at p.
            fn scalar_curvature(
                &self,
                p: numpy::PyReadonlyArrayDyn<'_, f64>,
            ) -> pyo3::PyResult<f64> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_svector::<$N>(p, "p")?;
                        Ok(cartan_core::Curvature::scalar_curvature(&mf, &pp))
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Geodesic interpolation: gamma(p, q, t).
            fn geodesic<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
                q: numpy::PyReadonlyArrayDyn<'py, f64>,
                t: f64,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_svector::<$N>(p, "p")?;
                        let qq = $crate::convert::arr_to_svector::<$N>(q, "q")?;
                        let result = cartan_core::GeodesicInterpolation::geodesic(&mf, &pp, &qq, t)
                            .map_err($crate::error::cartan_err_to_py)?;
                        Ok($crate::convert::svector_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }
        }
    };
}

/// Generate all Manifold trait method wrappers for manifolds whose Point
/// and Tangent types are `SMatrix<Real, N, N>` (e.g. SPD, SO, Corr).
///
/// Usage (at module level, outside any impl block):
/// ```ignore
/// impl_matrix_manifold_methods!(PySpd, Spd, n, [2, 3, 4, 5, 6, 7, 8]);
/// ```
///
/// - `$pytype`    : the `#[pyclass]` struct name
/// - `$mtype`     : the `cartan_manifolds` type name
/// - `$dim_field` : ident of the runtime-dimension field on `$pytype`
/// - `[$($N),+]`  : list of supported const-generic dimensions
#[macro_export]
macro_rules! impl_matrix_manifold_methods {
    ($pytype:ident, $mtype:ident, $dim_field:ident, [$($N:literal),+ $(,)?]) => {

        #[pyo3::pymethods]
        impl $pytype {

            /// Exponential map: Exp_p(v).
            fn exp<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
                v: numpy::PyReadonlyArrayDyn<'py, f64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_smatrix::<$N, $N>(p, "p")?;
                        let vv = $crate::convert::arr_to_smatrix::<$N, $N>(v, "v")?;
                        let result = cartan_core::Manifold::exp(&mf, &pp, &vv);
                        Ok($crate::convert::smatrix_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Logarithmic map: Log_p(q).
            fn log<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
                q: numpy::PyReadonlyArrayDyn<'py, f64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_smatrix::<$N, $N>(p, "p")?;
                        let qq = $crate::convert::arr_to_smatrix::<$N, $N>(q, "q")?;
                        let result = cartan_core::Manifold::log(&mf, &pp, &qq)
                            .map_err($crate::error::cartan_err_to_py)?;
                        Ok($crate::convert::smatrix_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Geodesic distance d(p, q).
            fn dist(
                &self,
                p: numpy::PyReadonlyArrayDyn<'_, f64>,
                q: numpy::PyReadonlyArrayDyn<'_, f64>,
            ) -> pyo3::PyResult<f64> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_smatrix::<$N, $N>(p, "p")?;
                        let qq = $crate::convert::arr_to_smatrix::<$N, $N>(q, "q")?;
                        cartan_core::Manifold::dist(&mf, &pp, &qq)
                            .map_err($crate::error::cartan_err_to_py)
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Riemannian inner product <u, v>_p.
            fn inner(
                &self,
                p: numpy::PyReadonlyArrayDyn<'_, f64>,
                u: numpy::PyReadonlyArrayDyn<'_, f64>,
                v: numpy::PyReadonlyArrayDyn<'_, f64>,
            ) -> pyo3::PyResult<f64> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_smatrix::<$N, $N>(p, "p")?;
                        let uu = $crate::convert::arr_to_smatrix::<$N, $N>(u, "u")?;
                        let vv = $crate::convert::arr_to_smatrix::<$N, $N>(v, "v")?;
                        Ok(cartan_core::Manifold::inner(&mf, &pp, &uu, &vv))
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Induced norm ||v||_p.
            fn norm(
                &self,
                p: numpy::PyReadonlyArrayDyn<'_, f64>,
                v: numpy::PyReadonlyArrayDyn<'_, f64>,
            ) -> pyo3::PyResult<f64> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_smatrix::<$N, $N>(p, "p")?;
                        let vv = $crate::convert::arr_to_smatrix::<$N, $N>(v, "v")?;
                        Ok(cartan_core::Manifold::norm(&mf, &pp, &vv))
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Project an ambient point onto the manifold.
            fn project_point<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_smatrix::<$N, $N>(p, "p")?;
                        let result = cartan_core::Manifold::project_point(&mf, &pp);
                        Ok($crate::convert::smatrix_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Project an ambient vector onto T_p M.
            fn project_tangent<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
                v: numpy::PyReadonlyArrayDyn<'py, f64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_smatrix::<$N, $N>(p, "p")?;
                        let vv = $crate::convert::arr_to_smatrix::<$N, $N>(v, "v")?;
                        let result = cartan_core::Manifold::project_tangent(&mf, &pp, &vv);
                        Ok($crate::convert::smatrix_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// The zero tangent vector at p.
            fn zero_tangent<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_smatrix::<$N, $N>(p, "p")?;
                        let result = cartan_core::Manifold::zero_tangent(&mf, &pp);
                        Ok($crate::convert::smatrix_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Validate that a point lies on the manifold.
            fn check_point(
                &self,
                p: numpy::PyReadonlyArrayDyn<'_, f64>,
            ) -> pyo3::PyResult<()> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_smatrix::<$N, $N>(p, "p")?;
                        cartan_core::Manifold::check_point(&mf, &pp)
                            .map_err($crate::error::cartan_err_to_py)
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Validate that a tangent vector lies in T_p M.
            fn check_tangent(
                &self,
                p: numpy::PyReadonlyArrayDyn<'_, f64>,
                v: numpy::PyReadonlyArrayDyn<'_, f64>,
            ) -> pyo3::PyResult<()> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_smatrix::<$N, $N>(p, "p")?;
                        let vv = $crate::convert::arr_to_smatrix::<$N, $N>(v, "v")?;
                        cartan_core::Manifold::check_tangent(&mf, &pp, &vv)
                            .map_err($crate::error::cartan_err_to_py)
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Random point on the manifold.
            #[pyo3(signature = (seed=None))]
            fn random_point<'py>(
                &self,
                py: pyo3::Python<'py>,
                seed: Option<u64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                use rand::SeedableRng;
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let result = match seed {
                            Some(s) => {
                                let mut rng = rand::rngs::StdRng::seed_from_u64(s);
                                cartan_core::Manifold::random_point(&mf, &mut rng)
                            }
                            None => {
                                cartan_core::Manifold::random_point(&mf, &mut rand::rng())
                            }
                        };
                        Ok($crate::convert::smatrix_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Random tangent vector at p.
            #[pyo3(signature = (p, seed=None))]
            fn random_tangent<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
                seed: Option<u64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                use rand::SeedableRng;
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_smatrix::<$N, $N>(p, "p")?;
                        let result = match seed {
                            Some(s) => {
                                let mut rng = rand::rngs::StdRng::seed_from_u64(s);
                                cartan_core::Manifold::random_tangent(&mf, &pp, &mut rng)
                            }
                            None => {
                                cartan_core::Manifold::random_tangent(&mf, &pp, &mut rand::rng())
                            }
                        };
                        Ok($crate::convert::smatrix_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Injectivity radius at p.
            fn injectivity_radius(
                &self,
                p: numpy::PyReadonlyArrayDyn<'_, f64>,
            ) -> pyo3::PyResult<f64> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_smatrix::<$N, $N>(p, "p")?;
                        Ok(cartan_core::Manifold::injectivity_radius(&mf, &pp))
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Retraction: cheap approximation to exp.
            fn retract<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
                v: numpy::PyReadonlyArrayDyn<'py, f64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_smatrix::<$N, $N>(p, "p")?;
                        let vv = $crate::convert::arr_to_smatrix::<$N, $N>(v, "v")?;
                        let result = cartan_core::Retraction::retract(&mf, &pp, &vv);
                        Ok($crate::convert::smatrix_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Inverse retraction: approximate log.
            fn inverse_retract<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
                q: numpy::PyReadonlyArrayDyn<'py, f64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_smatrix::<$N, $N>(p, "p")?;
                        let qq = $crate::convert::arr_to_smatrix::<$N, $N>(q, "q")?;
                        let result = cartan_core::Retraction::inverse_retract(&mf, &pp, &qq)
                            .map_err($crate::error::cartan_err_to_py)?;
                        Ok($crate::convert::smatrix_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Parallel transport of v from p to q along the geodesic.
            fn parallel_transport<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
                q: numpy::PyReadonlyArrayDyn<'py, f64>,
                v: numpy::PyReadonlyArrayDyn<'py, f64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_smatrix::<$N, $N>(p, "p")?;
                        let qq = $crate::convert::arr_to_smatrix::<$N, $N>(q, "q")?;
                        let vv = $crate::convert::arr_to_smatrix::<$N, $N>(v, "v")?;
                        let result = cartan_core::ParallelTransport::transport(&mf, &pp, &qq, &vv)
                            .map_err($crate::error::cartan_err_to_py)?;
                        Ok($crate::convert::smatrix_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Vector transport of v at p in direction u.
            fn vector_transport<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
                direction: numpy::PyReadonlyArrayDyn<'py, f64>,
                v: numpy::PyReadonlyArrayDyn<'py, f64>,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_smatrix::<$N, $N>(p, "p")?;
                        let dd = $crate::convert::arr_to_smatrix::<$N, $N>(direction, "direction")?;
                        let vv = $crate::convert::arr_to_smatrix::<$N, $N>(v, "v")?;
                        let result = cartan_core::VectorTransport::vector_transport(&mf, &pp, &dd, &vv)
                            .map_err($crate::error::cartan_err_to_py)?;
                        Ok($crate::convert::smatrix_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Sectional curvature of the 2-plane spanned by u and v at p.
            fn sectional_curvature(
                &self,
                p: numpy::PyReadonlyArrayDyn<'_, f64>,
                u: numpy::PyReadonlyArrayDyn<'_, f64>,
                v: numpy::PyReadonlyArrayDyn<'_, f64>,
            ) -> pyo3::PyResult<f64> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_smatrix::<$N, $N>(p, "p")?;
                        let uu = $crate::convert::arr_to_smatrix::<$N, $N>(u, "u")?;
                        let vv = $crate::convert::arr_to_smatrix::<$N, $N>(v, "v")?;
                        Ok(cartan_core::Curvature::sectional_curvature(&mf, &pp, &uu, &vv))
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Ricci curvature Ric(u, v) at p.
            fn ricci_curvature(
                &self,
                p: numpy::PyReadonlyArrayDyn<'_, f64>,
                u: numpy::PyReadonlyArrayDyn<'_, f64>,
                v: numpy::PyReadonlyArrayDyn<'_, f64>,
            ) -> pyo3::PyResult<f64> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_smatrix::<$N, $N>(p, "p")?;
                        let uu = $crate::convert::arr_to_smatrix::<$N, $N>(u, "u")?;
                        let vv = $crate::convert::arr_to_smatrix::<$N, $N>(v, "v")?;
                        Ok(cartan_core::Curvature::ricci_curvature(&mf, &pp, &uu, &vv))
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Scalar curvature at p.
            fn scalar_curvature(
                &self,
                p: numpy::PyReadonlyArrayDyn<'_, f64>,
            ) -> pyo3::PyResult<f64> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_smatrix::<$N, $N>(p, "p")?;
                        Ok(cartan_core::Curvature::scalar_curvature(&mf, &pp))
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }

            /// Geodesic interpolation: gamma(p, q, t).
            fn geodesic<'py>(
                &self,
                py: pyo3::Python<'py>,
                p: numpy::PyReadonlyArrayDyn<'py, f64>,
                q: numpy::PyReadonlyArrayDyn<'py, f64>,
                t: f64,
            ) -> pyo3::PyResult<pyo3::PyObject> {
                match self.$dim_field {
                    $($N => {
                        let mf = cartan_manifolds::$mtype::<$N>;
                        let pp = $crate::convert::arr_to_smatrix::<$N, $N>(p, "p")?;
                        let qq = $crate::convert::arr_to_smatrix::<$N, $N>(q, "q")?;
                        let result = cartan_core::GeodesicInterpolation::geodesic(&mf, &pp, &qq, t)
                            .map_err($crate::error::cartan_err_to_py)?;
                        Ok($crate::convert::smatrix_to_pyarray(py, &result).into_any().unbind())
                    },)+
                    _ => Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unsupported dimension: {}", self.$dim_field),
                    )),
                }
            }
        }
    };
}

pub(crate) use impl_matrix_manifold_methods;
pub(crate) use impl_vector_manifold_methods;
