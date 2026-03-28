// ~/cartan/cartan-py/src/manifolds/frame_field.rs

//! Python wrapper for `cartan_manifolds::frame_field::FrameField3D`.
//!
//! FrameField3D is a data container holding an ordered list of orthonormal frames
//! F in SO(3), one per grid vertex, derived from Q-tensor eigendecomposition.
//! It supports D2 gauge fixing to make the frame field as smooth as possible
//! along a 1D chain.

use pyo3::prelude::*;
use pyo3::exceptions::PyIndexError;
use numpy::PyReadonlyArrayDyn;

use cartan_manifolds::frame_field::FrameField3D;

use crate::convert::{arr_to_smatrix, smatrix_to_pyarray};

/// Python wrapper for a frame field: an orthonormal frame F in SO(3) at each grid vertex.
///
/// Frames are constructed from Q-tensor eigendecompositions. Column 2 of each
/// frame is the director (principal eigenvector of the Q-tensor).
///
/// Supports D2 gauge fixing along a 1D chain to produce a smooth frame field.
#[pyclass(name = "FrameField3D")]
pub struct PyFrameField3D {
    // FrameField3D does not derive Clone/Debug, so we skip those derives here.
    pub(crate) inner: FrameField3D,
}

impl std::fmt::Debug for PyFrameField3D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PyFrameField3D(len={})", self.inner.len())
    }
}

#[pymethods]
impl PyFrameField3D {
    /// Construct a FrameField3D from a list of Q-tensors (3x3 numpy arrays).
    ///
    /// Each Q-tensor is eigendecomposed to extract the orthonormal director frame.
    /// No gauge fixing is applied; call gauge_fix_chain() afterward if needed.
    #[new]
    fn new(q_values: Vec<PyReadonlyArrayDyn<'_, f64>>) -> PyResult<Self> {
        let mut mats = Vec::with_capacity(q_values.len());
        for (i, arr) in q_values.into_iter().enumerate() {
            let name = format!("q_values[{i}]");
            let m = arr_to_smatrix::<3, 3>(arr, &name)?;
            mats.push(m);
        }
        let inner = FrameField3D::from_q_field(&mats);
        Ok(Self { inner })
    }

    /// Return the number of frames (grid vertices).
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Return the orthonormal frame at vertex i as a (3, 3) numpy array.
    ///
    /// Raises IndexError if i >= len().
    fn frame_at<'py>(&self, py: Python<'py>, i: usize) -> PyResult<PyObject> {
        if i >= self.inner.len() {
            return Err(PyIndexError::new_err(format!(
                "FrameField3D: index {i} out of range for field of length {}",
                self.inner.len()
            )));
        }
        let frame = self.inner.frame_at(i);
        Ok(smatrix_to_pyarray(py, frame).into_any().unbind())
    }

    /// Apply D2 gauge fixing along the chain (vertex order 0, 1, ..., n-1).
    ///
    /// Returns a new FrameField3D where each frame is the D2 representative
    /// closest to its predecessor, minimizing frame-to-frame jumps.
    fn gauge_fix_chain(&self) -> PyResult<PyFrameField3D> {
        let fixed = self.inner.gauge_fix_chain();
        Ok(PyFrameField3D { inner: fixed })
    }

    fn __repr__(&self) -> String {
        format!("FrameField3D(len={})", self.inner.len())
    }
}
