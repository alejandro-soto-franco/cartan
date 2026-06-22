//! Staggered-leapfrog Maxwell evolver on an evolving Regge background.

use cartan_exterior::ExteriorGrade;
use cartan_simplicial::geometry::metric::mesh::MeshLengths;
use cartan_simplicial::topology::complex::Complex;
use sprs::CsMat;

/// The coboundary operator d_k: C^k -> C^{k+1} as a sparse matrix of shape
/// `nsimplices(k+1) x nsimplices(k)`. It is the transpose of the boundary
/// operator and is purely combinatorial (metric-free).
pub fn coboundary_matrix(complex: &Complex, k: ExteriorGrade) -> CsMat<f64> {
    // boundary_chain()[k] is the boundary d_{k+1}: shape nsimplices(k) x nsimplices(k+1).
    let boundary = &complex.boundary_chain()[k];
    boundary.transpose_view().to_csc()
}

/// A conservative CFL time-step estimate: a fraction of the smallest edge length.
pub fn cfl_dt(geometry: &MeshLengths) -> f64 {
    let mut hmin = f64::INFINITY;
    for i in 0..geometry.nedges() {
        hmin = hmin.min(geometry.length(i));
    }
    0.3 * hmin
}

/// Placeholder evolver struct: filled in Task 3.
pub struct MaxwellEvolver;
