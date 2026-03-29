// ~/cartan/cartan-py/src/dec/mesh.rs

//! PyMesh: Python wrapper for `FlatMesh` (= `Mesh<Euclidean<2>, 3, 2>`).
//!
//! Exposes mesh construction, topology queries, and vertex/simplex access
//! via numpy arrays. Serves as the entry point for all DEC computations
//! in the Python API.

use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use cartan_dec::FlatMesh;

/// A triangulated 2D flat mesh for discrete exterior calculus (DEC).
///
/// Wraps `cartan_dec::FlatMesh` (`Mesh<Euclidean<2>, 3, 2>`).
/// All DEC operators (ExteriorDerivative, HodgeStar, Operators) take
/// a `Mesh` as input and derive their structure from it.
///
/// # Examples
///
/// ```python
/// import numpy as np
/// import cartan
///
/// verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
/// tris  = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
/// mesh  = cartan.Mesh(verts, tris)
/// assert mesh.n_vertices() == 4
/// assert mesh.n_simplices() == 2
/// ```
#[pyclass(name = "Mesh")]
pub struct PyMesh {
    pub(crate) inner: FlatMesh,
}

#[pymethods]
impl PyMesh {
    /// Construct a flat triangular mesh from vertex positions and triangle indices.
    ///
    /// Parameters
    /// ----------
    /// vertices : ndarray, shape (N, 2), dtype float64
    ///     2D vertex positions.
    /// simplices : ndarray, shape (M, 3), dtype int64
    ///     Triangle vertex indices (0-based, counter-clockwise orientation).
    #[new]
    pub fn new(
        vertices: PyReadonlyArray2<'_, f64>,
        simplices: PyReadonlyArray2<'_, i64>,
    ) -> PyResult<Self> {
        let vshape = vertices.shape();
        if vshape.len() != 2 || vshape[1] != 2 {
            return Err(PyValueError::new_err(
                "vertices must have shape (N, 2)",
            ));
        }
        let sshape = simplices.shape();
        if sshape.len() != 2 || sshape[1] != 3 {
            return Err(PyValueError::new_err(
                "simplices must have shape (M, 3)",
            ));
        }

        let n_v = vshape[0];
        let n_t = sshape[0];

        // Convert to Vec<[f64; 2]> (row-major, contiguous expected)
        let vdata = vertices
            .as_slice()
            .map_err(|_| PyValueError::new_err("vertices array must be contiguous"))?;
        let verts: Vec<[f64; 2]> = (0..n_v)
            .map(|i| [vdata[2 * i], vdata[2 * i + 1]])
            .collect();

        // Convert to Vec<[usize; 3]>
        let sdata = simplices
            .as_slice()
            .map_err(|_| PyValueError::new_err("simplices array must be contiguous"))?;
        let tris: Vec<[usize; 3]> = (0..n_t)
            .map(|i| {
                let a = sdata[3 * i] as usize;
                let b = sdata[3 * i + 1] as usize;
                let c = sdata[3 * i + 2] as usize;
                [a, b, c]
            })
            .collect();

        let inner = FlatMesh::from_triangles(verts, tris);
        Ok(Self { inner })
    }

    /// Build a uniform triangulated grid on [0,1]^2 with `n` divisions per side.
    ///
    /// Produces `2*n^2` right triangles and `(n+1)^2` vertices.
    ///
    /// Parameters
    /// ----------
    /// n : int
    ///     Number of grid divisions per axis (>= 1).
    ///
    /// Returns
    /// -------
    /// Mesh
    ///     The constructed mesh.
    #[staticmethod]
    pub fn unit_square_grid(n: usize) -> PyResult<Self> {
        if n < 1 {
            return Err(PyValueError::new_err("n must be >= 1"));
        }
        Ok(Self {
            inner: FlatMesh::unit_square_grid(n),
        })
    }

    /// Number of vertices in the mesh.
    pub fn n_vertices(&self) -> usize {
        self.inner.n_vertices()
    }

    /// Number of triangles (2-simplices) in the mesh.
    pub fn n_simplices(&self) -> usize {
        self.inner.n_simplices()
    }

    /// Number of boundary faces (edges) in the mesh.
    pub fn n_boundaries(&self) -> usize {
        self.inner.n_boundaries()
    }

    /// Euler characteristic: V - E + F.
    pub fn euler_characteristic(&self) -> i32 {
        self.inner.euler_characteristic()
    }

    /// Vertex positions as a (N, 2) float64 numpy array.
    #[getter]
    pub fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let nv = self.inner.n_vertices();
        let rows: Vec<Vec<f64>> = (0..nv)
            .map(|i| {
                let v = self.inner.vertex(i);
                vec![v.x, v.y]
            })
            .collect();
        PyArray2::from_vec2(py, &rows).expect("uniform row lengths")
    }

    /// Triangle indices as an (M, 3) int64 numpy array.
    #[getter]
    pub fn simplices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i64>> {
        let nt = self.inner.n_simplices();
        // Build a DMatrix<i64> in row-major order
        let mut data = Vec::with_capacity(nt * 3);
        for t in 0..nt {
            let [a, b, c] = self.inner.simplices[t];
            data.push(a as i64);
            data.push(b as i64);
            data.push(c as i64);
        }
        // Use from_vec2 approach
        let rows: Vec<Vec<i64>> = (0..nt)
            .map(|t| {
                let [a, b, c] = self.inner.simplices[t];
                vec![a as i64, b as i64, c as i64]
            })
            .collect();
        drop(data);
        PyArray2::from_vec2(py, &rows).expect("uniform row lengths")
    }

    /// Boundary edge indices as an (E, 2) int64 numpy array.
    #[getter]
    pub fn boundaries<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i64>> {
        let ne = self.inner.n_boundaries();
        let rows: Vec<Vec<i64>> = (0..ne)
            .map(|e| {
                let [a, b] = self.inner.boundaries[e];
                vec![a as i64, b as i64]
            })
            .collect();
        PyArray2::from_vec2(py, &rows).expect("uniform row lengths")
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Mesh(n_vertices={}, n_simplices={}, n_boundaries={})",
            self.inner.n_vertices(),
            self.inner.n_simplices(),
            self.inner.n_boundaries()
        )
    }
}
