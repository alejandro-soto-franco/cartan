// ~/cartan/cartan-py/src/cloud.rs

//! Python bindings for `cartan-holonomy`: the Levi-Civita connection of a point
//! cloud, its plaquette curvature and covariant jets.
//!
//! Inputs are cloud-sized, so they come in as `PyReadonlyArray2` (one borrow
//! per call) and the heavy work runs with the GIL released.

use cartan_holonomy::{
    Connection, JetOptions, curvature_operator, fit_jet, local_jet, triangle_geometry,
};
use numpy::{
    IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn rows(a: &PyReadonlyArray2<'_, f64>, name: &str) -> PyResult<(Vec<f64>, usize, usize)> {
    let s = a.shape();
    let v = a
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name}: array must be C-contiguous")))?
        .to_vec();
    Ok((v, s[0], s[1]))
}

fn shaped<'py>(py: Python<'py>, v: Vec<f64>, shape: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    Ok(v.into_pyarray(py).reshape(shape.to_vec())?.into_any())
}

/// The discrete Levi-Civita connection of a point cloud.
///
/// Parameters
/// ----------
/// points : array of shape (n, dim)
/// d : int
///     Frame dimension.
/// k_frame : int
///     Neighbours used for each tangent frame (local PCA).
/// k_graph : int
///     The graph is the mutual k_graph-nearest-neighbour graph.
#[pyclass(name = "CloudConnection", frozen)]
pub struct PyCloudConnection {
    c: Connection,
    x: Vec<f64>,
}

impl PyCloudConnection {
    fn field(&self, v: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<f64>> {
        let (vv, n, dim) = rows(v, "velocity")?;
        if n != self.c.n || dim != self.c.dim {
            return Err(PyValueError::new_err(
                "velocity must have the shape of the points",
            ));
        }
        Ok(vv)
    }
}

#[pymethods]
impl PyCloudConnection {
    #[new]
    fn new(
        py: Python<'_>,
        points: PyReadonlyArray2<'_, f64>,
        d: usize,
        k_frame: usize,
        k_graph: usize,
    ) -> PyResult<Self> {
        let (x, n, dim) = rows(&points, "points")?;
        if d == 0 || d > dim || k_frame >= n || k_graph >= n {
            return Err(PyValueError::new_err(
                "need 0 < d <= dim and k_frame, k_graph < n",
            ));
        }
        let c = py.detach(|| Connection::new(&x, n, dim, d, k_frame, k_graph));
        Ok(Self { c, x })
    }

    /// Tangent frames, shape (n, dim, d).
    #[getter]
    fn frames<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        shaped(py, self.c.frames.clone(), &[self.c.n, self.c.dim, self.c.d])
    }

    /// Undirected edges (i, j), i < j, shape (e, 2).
    #[getter]
    fn edges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let e: Vec<u64> = self
            .c
            .graph
            .edges
            .iter()
            .flat_map(|p| [p[0] as u64, p[1] as u64])
            .collect();
        Ok(e.into_pyarray(py)
            .reshape(vec![self.c.graph.edges.len(), 2])?
            .into_any())
    }

    /// Triangles with their holonomy angle and projected area.
    ///
    /// Returns a dict with ``triangles`` (t, 3), ``angle`` (t,), ``area`` (t,)
    /// and ``curvature``, the total angle over the total area.
    fn triangle_geometry<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let g = py.detach(|| triangle_geometry(&self.c, &self.x));
        let out = PyDict::new(py);
        let t: Vec<u64> = g
            .triangles
            .iter()
            .flat_map(|t| t.iter().map(|&v| v as u64))
            .collect();
        out.set_item(
            "triangles",
            t.into_pyarray(py).reshape(vec![g.triangles.len(), 3])?,
        )?;
        out.set_item("curvature", g.curvature())?;
        out.set_item("angle", PyArray1::from_vec(py, g.angle))?;
        out.set_item("area", PyArray1::from_vec(py, g.area))?;
        Ok(out)
    }

    /// Per-point curvature operator: array (n, 4) of scalar curvature,
    /// Ric(v, v)/|v|^2, fit residual and readings used.
    #[pyo3(signature = (velocity=None))]
    fn curvature_operator<'py>(
        &self,
        py: Python<'py>,
        velocity: Option<PyReadonlyArray2<'py, f64>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = velocity.map(|v| self.field(&v)).transpose()?;
        let ops = py.detach(|| {
            let g = triangle_geometry(&self.c, &self.x);
            curvature_operator(&self.c, &g, v.as_deref())
        });
        let flat: Vec<f64> = ops
            .iter()
            .flat_map(|o| [o.scalar, o.ric_vv, o.residual, o.count as f64])
            .collect();
        shaped(py, flat, &[self.c.n, 4])
    }

    /// Pointwise least-squares jet: array (n, 4) of expansion, shear^2,
    /// vorticity^2 and edges used.
    fn local_jet<'py>(
        &self,
        py: Python<'py>,
        velocity: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = self.field(&velocity)?;
        let j = py.detach(|| local_jet(&self.c, &self.x, &v));
        let flat: Vec<f64> = j
            .iter()
            .flat_map(|j| [j.theta, j.shear2, j.vort2, j.edges as f64])
            .collect();
        shaped(py, flat, &[self.c.n, 4])
    }

    /// Variational jet, one section of End(TM) regularised by the connection
    /// Laplacian.
    ///
    /// ``weights`` has one entry per directed edge in the order of ``edges``
    /// taken (a, b), then (b, a); zero-weight edges are held out and scored.
    /// Returns ``(A, info)`` with ``A`` of shape (n, d, d) in each point's frame.
    #[pyo3(signature = (velocity, alpha, weights=None, tol=1e-8, max_iter=5000))]
    fn fit_jet<'py>(
        &self,
        py: Python<'py>,
        velocity: PyReadonlyArray2<'py, f64>,
        alpha: f64,
        weights: Option<PyReadonlyArray1<'py, f64>>,
        tol: f64,
        max_iter: usize,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyDict>)> {
        let v = self.field(&velocity)?;
        let m = self.c.graph.edges.len();
        let w = match weights {
            Some(w) => {
                let w = w.as_slice()?.to_vec();
                if w.len() != 2 * m {
                    return Err(PyValueError::new_err(
                        "weights: one entry per directed edge",
                    ));
                }
                let mut out = vec![1.0; self.c.directed_edges()];
                for (e, &[a, b]) in self.c.graph.edges.iter().enumerate() {
                    out[self.c.slot(a, b)] = w[e];
                    out[self.c.slot(b, a)] = w[m + e];
                }
                Some(out)
            }
            None => None,
        };
        let opts = JetOptions {
            alpha,
            tol,
            max_iter,
            ..JetOptions::default()
        };
        let fit = py.detach(|| fit_jet(&self.c, &self.x, &v, w.as_deref(), &opts));
        let info = PyDict::new(py);
        info.set_item("cg_iters", fit.report.iterations)?;
        info.set_item("converged", fit.report.converged)?;
        info.set_item("heldout_rel_err", fit.heldout)?;
        info.set_item("train_rel_err", fit.train)?;
        info.set_item("dirichlet_rel", fit.dirichlet)?;
        info.set_item("scale", fit.scale)?;
        let d = self.c.d;
        Ok((shaped(py, fit.a, &[self.c.n, d, d])?, info))
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCloudConnection>()?;
    Ok(())
}
