//! Python bindings for the k-atic complex line-bundle sections of `cartan_dec`.
//!
//! Exposes the nematic (K=2) and hexatic (K=6) `Section<K>`, the Levi-Civita
//! `ConnectionAngles`, the covariant `BochnerLaplacian<K>`, and the per-face
//! topological-charge computation `defect_charges<K>`. Together these let a
//! caller build a hexatic or nematic texture on a flat triangle mesh, relax it
//! covariantly, and read its disclination charges (the discrete Poincare-Hopf
//! spectrum: +/-1/6 hexatic, +/-1/2 nematic).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use cartan_dec::line_bundle::{defect_charges, BochnerLaplacian, ConnectionAngles, Section};
use cartan_manifolds::euclidean::Euclidean;

use super::mesh::PyMesh;
use super::operators::PyHodgeStar;

/// Discrete Levi-Civita connection angles on a flat triangle mesh.
#[pyclass(name = "ConnectionAngles")]
pub struct PyConnectionAngles {
    pub(crate) inner: ConnectionAngles,
}

#[pymethods]
impl PyConnectionAngles {
    /// Build the connection angles from a flat mesh.
    #[staticmethod]
    pub fn from_mesh(mesh: &PyMesh) -> Self {
        Self {
            inner: ConnectionAngles::from_mesh(&mesh.inner, &Euclidean::<2>),
        }
    }

    pub fn __repr__(&self) -> String {
        format!("ConnectionAngles(n_primal={})", self.inner.primal.len())
    }
}

// Generate the K-specialised Section, Bochner Laplacian, and charge function.
macro_rules! katic_binding {
    ($sec:ident, $lap:ident, $charges:ident, $sec_name:literal, $lap_name:literal, $K:literal) => {
        #[doc = concat!("Complex line-bundle section of k-atic order K=", stringify!($K), ".")]
        #[pyclass(name = $sec_name)]
        pub struct $sec {
            pub(crate) inner: Section<$K>,
        }

        #[pymethods]
        impl $sec {
            /// Build from real components `z = q1 + i q2` (one complex per vertex).
            #[staticmethod]
            pub fn from_real_components(
                q1: PyReadonlyArray1<'_, f64>,
                q2: PyReadonlyArray1<'_, f64>,
            ) -> PyResult<Self> {
                let a = q1
                    .as_slice()
                    .map_err(|_| PyValueError::new_err("q1: array must be contiguous"))?;
                let b = q2
                    .as_slice()
                    .map_err(|_| PyValueError::new_err("q2: array must be contiguous"))?;
                if a.len() != b.len() {
                    return Err(PyValueError::new_err(format!(
                        "q1, q2 length mismatch: {} vs {}",
                        a.len(),
                        b.len()
                    )));
                }
                Ok(Self {
                    inner: Section::<$K>::from_real_components(a, b),
                })
            }

            /// Real and imaginary parts as a pair of (n_vertices,) arrays.
            pub fn components<'py>(
                &self,
                py: Python<'py>,
            ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
                let (q1, q2) = self.inner.to_real_components();
                (q1.into_pyarray(py), q2.into_pyarray(py))
            }

            /// Pointwise order-parameter magnitude `|z|` at each vertex.
            pub fn norms<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
                self.inner.norms().into_pyarray(py)
            }

            /// Mean order-parameter magnitude over the mesh.
            pub fn mean_norm(&self) -> f64 {
                self.inner.mean_norm()
            }

            pub fn n_vertices(&self) -> usize {
                self.inner.n_vertices()
            }

            pub fn __repr__(&self) -> String {
                format!(
                    concat!($sec_name, "(n_vertices={}, mean_norm={:.4})"),
                    self.inner.n_vertices(),
                    self.inner.mean_norm()
                )
            }
        }

        #[doc = concat!("Covariant Bochner Laplacian on the K=", stringify!($K), " line bundle.")]
        #[pyclass(name = $lap_name)]
        pub struct $lap {
            inner: BochnerLaplacian<$K>,
        }

        #[pymethods]
        impl $lap {
            /// Assemble from a mesh, its Hodge star, and the connection angles.
            #[staticmethod]
            pub fn from_mesh(mesh: &PyMesh, hodge: &PyHodgeStar, conn: &PyConnectionAngles) -> Self {
                Self {
                    inner: BochnerLaplacian::<$K>::from_mesh_data(
                        &mesh.inner,
                        &hodge.inner,
                        &conn.inner,
                    ),
                }
            }

            /// Apply the Laplacian to a section: `result = Delta z`.
            pub fn apply(&self, section: &$sec) -> $sec {
                $sec {
                    inner: self.inner.apply(&section.inner),
                }
            }
        }

        #[doc = concat!("Per-face topological charge of a K=", stringify!($K), " section.")]
        #[pyfunction]
        pub fn $charges<'py>(
            py: Python<'py>,
            section: &$sec,
            conn: &PyConnectionAngles,
            mesh: &PyMesh,
            gaussian_curvature_per_face: PyReadonlyArray1<'_, f64>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let g = gaussian_curvature_per_face
                .as_slice()
                .map_err(|_| PyValueError::new_err("gauss: array must be contiguous"))?;
            let charges = defect_charges::<$K>(&section.inner, &conn.inner, &mesh.inner, g);
            Ok(charges.into_pyarray(py))
        }
    };
}

katic_binding!(PySection2, PyBochnerLaplacian2, defect_charges_2, "Section2", "BochnerLaplacian2", 2);
katic_binding!(PySection6, PyBochnerLaplacian6, defect_charges_6, "Section6", "BochnerLaplacian6", 6);

pub fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    m.add_class::<PyConnectionAngles>()?;
    m.add_class::<PySection2>()?;
    m.add_class::<PySection6>()?;
    m.add_class::<PyBochnerLaplacian2>()?;
    m.add_class::<PyBochnerLaplacian6>()?;
    m.add_function(pyo3::wrap_pyfunction!(defect_charges_2, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(defect_charges_6, m)?)?;
    Ok(())
}
