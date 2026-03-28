// ~/cartan/cartan-py/src/convert.rs

//! Numpy-to-nalgebra and nalgebra-to-numpy conversion utilities.
//!
//! These are the core bridge functions used by every manifold wrapper.
//! All functions validate shapes and produce clear Python-facing error
//! messages on mismatch.

use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use cartan_core::Real;

/// Extract a flat f64 slice from a numpy array, validating contiguity and length.
///
/// Returns a `Vec<f64>` of exactly `expected_len` elements, or a `PyValueError`
/// if the array is non-contiguous or has the wrong total number of elements.
pub fn arr_to_vec(
    arr: PyReadonlyArrayDyn<'_, f64>,
    expected_len: usize,
    name: &str,
) -> PyResult<Vec<f64>> {
    let slice = arr
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name}: array must be contiguous")))?;
    if slice.len() != expected_len {
        return Err(PyValueError::new_err(format!(
            "{name}: expected {expected_len} elements, got {}",
            slice.len()
        )));
    }
    Ok(slice.to_vec())
}

/// Convert a numpy array to `SVector<Real, N>`.
///
/// Flattens the array and checks that it has exactly N elements.
pub fn arr_to_svector<const N: usize>(
    arr: PyReadonlyArrayDyn<'_, f64>,
    name: &str,
) -> PyResult<SVector<Real, N>> {
    let data = arr_to_vec(arr, N, name)?;
    Ok(SVector::<Real, N>::from_column_slice(&data))
}

/// Convert a numpy array to `SMatrix<Real, R, C>`.
///
/// Numpy stores data in row-major order; `SMatrix::from_row_slice` handles
/// the conversion to nalgebra's column-major internal layout.
pub fn arr_to_smatrix<const R: usize, const C: usize>(
    arr: PyReadonlyArrayDyn<'_, f64>,
    name: &str,
) -> PyResult<SMatrix<Real, R, C>> {
    let data = arr_to_vec(arr, R * C, name)?;
    Ok(SMatrix::<Real, R, C>::from_row_slice(&data))
}

/// Convert `SVector<Real, N>` to a 1D numpy array.
pub fn svector_to_pyarray<'py, const N: usize>(
    py: Python<'py>,
    v: &SVector<Real, N>,
) -> Bound<'py, PyArray1<f64>> {
    let data: Vec<f64> = v.iter().copied().collect();
    data.into_pyarray(py)
}

/// Convert `SMatrix<Real, R, C>` to a 2D numpy array in row-major order.
///
/// Nalgebra stores column-major, numpy expects row-major. We iterate
/// row-by-row to produce the correct layout.
pub fn smatrix_to_pyarray<'py, const R: usize, const C: usize>(
    py: Python<'py>,
    m: &SMatrix<Real, R, C>,
) -> Bound<'py, PyArray2<f64>> {
    let rows: Vec<Vec<f64>> = (0..R)
        .map(|i| (0..C).map(|j| m[(i, j)]).collect())
        .collect();
    PyArray2::from_vec2(py, &rows).expect("row lengths are uniform by construction")
}

/// Convert `DVector<f64>` to a 1D numpy array.
pub fn dvector_to_pyarray<'py>(py: Python<'py>, v: &DVector<f64>) -> Bound<'py, PyArray1<f64>> {
    let data: Vec<f64> = v.iter().copied().collect();
    data.into_pyarray(py)
}

/// Convert `DMatrix<f64>` to a 2D numpy array in row-major order.
pub fn dmatrix_to_pyarray<'py>(py: Python<'py>, m: &DMatrix<f64>) -> Bound<'py, PyArray2<f64>> {
    let rows: Vec<Vec<f64>> = (0..m.nrows())
        .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect())
        .collect();
    PyArray2::from_vec2(py, &rows).expect("row lengths are uniform by construction")
}

// ---------------------------------------------------------------------------
// Unit tests (nalgebra only, no Python GIL needed)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svector_roundtrip() {
        let v = SVector::<f64, 3>::new(1.0, 2.0, 3.0);
        let data: Vec<f64> = v.iter().copied().collect();
        let reconstructed = SVector::<f64, 3>::from_column_slice(&data);
        assert_eq!(v, reconstructed);
    }

    #[test]
    fn smatrix_from_row_slice_roundtrip() {
        // Row-major data: [[1, 2], [3, 4]]
        let row_data = [1.0, 2.0, 3.0, 4.0];
        let m = SMatrix::<f64, 2, 2>::from_row_slice(&row_data);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(1, 0)], 3.0);
        assert_eq!(m[(1, 1)], 4.0);
    }

    #[test]
    fn smatrix_to_row_major_data() {
        // Build a matrix and verify row-major extraction matches input.
        let m = SMatrix::<f64, 2, 3>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rows: Vec<Vec<f64>> = (0..2)
            .map(|i| (0..3).map(|j| m[(i, j)]).collect())
            .collect();
        assert_eq!(rows, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    }

    #[test]
    fn dvector_iter_preserves_order() {
        let v = DVector::from_vec(vec![10.0, 20.0, 30.0]);
        let data: Vec<f64> = v.iter().copied().collect();
        assert_eq!(data, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn dmatrix_row_major_extraction() {
        let m = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rows: Vec<Vec<f64>> = (0..m.nrows())
            .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect())
            .collect();
        assert_eq!(
            rows,
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]
        );
    }
}
