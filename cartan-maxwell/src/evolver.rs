//! Staggered-leapfrog Maxwell evolver on an evolving Regge background.

use cartan_exterior::ExteriorGrade;
use cartan_simplicial::geometry::metric::mesh::MeshLengths;
use cartan_simplicial::topology::complex::Complex;
use nalgebra::DVector;
use sprs::CsMat;

use crate::driver::MetricDriver;
use crate::state::MaxwellState;

/// Sparse matrix times dense column vector (manual, since sprs does not impl Mul<DVector>).
pub(crate) fn spmv(m: &CsMat<f64>, v: &DVector<f64>) -> DVector<f64> {
    let mut result = DVector::zeros(m.rows());
    for (&val, (row, col)) in m.iter() {
        result[row] += val * v[col];
    }
    result
}

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

/// Staggered-leapfrog Maxwell evolver. E lives at integer steps, B at half steps.
pub struct MaxwellEvolver<'d, D: MetricDriver> {
    driver: &'d D,
    d1: CsMat<f64>,         // metric-free coboundary 1 -> 2
    d2: Option<CsMat<f64>>, // metric-free coboundary 2 -> 3 (None in 2D)
    dt: f64,
    t: f64,
}

impl<'d, D: MetricDriver> MaxwellEvolver<'d, D> {
    pub fn new(driver: &'d D, dt: f64) -> Self {
        let complex = driver.complex();
        let d1 = coboundary_matrix(complex, 1);
        let d2 = if complex.dim() >= 3 {
            Some(coboundary_matrix(complex, 2))
        } else {
            None
        };
        Self { driver, d1, d2, dt, t: 0.0 }
    }

    pub fn time(&self) -> f64 {
        self.t
    }

    /// Faraday half-step: B <- B - dt (d1 E). Metric-free and exact.
    /// Because d2 d1 = 0, this preserves the discrete magnetic Gauss law
    /// d2 B = 0 to machine precision for all time.
    pub fn faraday_step(&self, state: &mut MaxwellState) {
        let curl_e = spmv(&self.d1, state.e.coeffs());
        let new_b = state.b.coeffs() - self.dt * curl_e;
        state.b = cartan_feec::cochain::Cochain::new(2, new_b);
    }

    /// The discrete magnetic Gauss-law residual ||d2 B||_inf. Zero (to machine
    /// precision) means no magnetic monopoles appeared. In 2D there is no
    /// grade-3 space, so the residual is defined as 0.
    pub fn magnetic_gauss_residual(&self, state: &MaxwellState) -> f64 {
        match &self.d2 {
            Some(d2) => {
                let r = spmv(d2, state.b.coeffs());
                r.iter().fold(0.0f64, |m, &v| m.max(v.abs()))
            }
            None => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::FlrwDriver;
    use crate::state::MaxwellState;
    use cartan_feec::cochain::Cochain;
    use cartan_simplicial::r#gen::cartesian::CartesianMeshInfo;

    #[test]
    fn faraday_preserves_magnetic_gauss_law_exactly_3d() {
        let (complex, coords) = CartesianMeshInfo::new_unit(3, 2).compute_coord_complex();
        let base = coords.to_edge_lengths(&complex);
        let driver = FlrwDriver::static_metric(complex.clone(), base);
        let evolver = MaxwellEvolver::new(&driver, 0.01);

        // Start with B closed (d2 B = 0): take B = d1 of an arbitrary 1-cochain.
        let d1 = coboundary_matrix(&complex, 1);
        let seed = nalgebra::DVector::from_fn(complex.nsimplices(1), |i, _| (i as f64).sin());
        let b0 = spmv(&d1, &seed);
        let e0 = nalgebra::DVector::from_fn(complex.nsimplices(1), |i, _| (i as f64).cos());
        let mut state = MaxwellState::new(Cochain::new(1, e0), Cochain::new(2, b0));

        assert!(evolver.magnetic_gauss_residual(&state) < 1e-10, "seed B not closed");
        for _ in 0..200 {
            evolver.faraday_step(&mut state);
            assert!(
                evolver.magnetic_gauss_residual(&state) < 1e-10,
                "magnetic monopole appeared"
            );
        }
    }
}
