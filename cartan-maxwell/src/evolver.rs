//! Staggered-leapfrog Maxwell evolver on an evolving Regge background.

use cartan_exterior::ExteriorGrade;
use cartan_feec::assemble::{assemble_galmat, interior_grade_dofs, restrict_galmat};
use cartan_feec::cochain::Cochain;
use cartan_feec::operators::HodgeMassElmat;
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

/// Convert a sparse matrix to dense nalgebra DMatrix.
fn to_dense(a: &CsMat<f64>) -> nalgebra::DMatrix<f64> {
    let mut d = nalgebra::DMatrix::zeros(a.rows(), a.cols());
    for (&v, (r, c)) in a.iter() {
        d[(r, c)] += v;
    }
    d
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
/// Factor 0.1 ensures stability for Whitney-form FEEC leapfrog in 2D and 3D.
pub fn cfl_dt(geometry: &MeshLengths) -> f64 {
    let mut hmin = f64::INFINITY;
    for i in 0..geometry.nedges() {
        hmin = hmin.min(geometry.length(i));
    }
    0.1 * hmin
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
        state.b = Cochain::new(2, new_b);
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

    /// One full leapfrog step. `source` is the (optional) electric current
    /// 1-cochain j at the half time. Advances E by dt and B by dt (staggered).
    /// Mirrors formoniq solve_wave staggering: Faraday then Ampere.
    pub fn step(&mut self, state: &mut MaxwellState, source: Option<&Cochain>) {
        let dim = self.driver.dim();
        let complex = self.driver.complex();
        let interior = interior_grade_dofs(complex, 1); // non-PEC edges

        // 1) Faraday half-step at the current E (metric-free, exact).
        self.faraday_step(state);

        // 2) Assemble the time-dependent masses.
        let l_now = self.driver.lengths_at(self.t);
        let l_next = self.driver.lengths_at(self.t + self.dt);
        let l_half = self.driver.lengths_at(self.t + 0.5 * self.dt);
        let m1_now = assemble_galmat(complex, &l_now, HodgeMassElmat::new(dim, 1));
        let m1_next = assemble_galmat(complex, &l_next, HodgeMassElmat::new(dim, 1));
        let m2_half = assemble_galmat(complex, &l_half, HodgeMassElmat::new(dim, 2));

        // 3) Build the full-space RHS of the Ampere update, then restrict.
        // RHS = M1(t) E^n + dt * d1^T M2(t+dt/2) B^{n+1/2} - dt * j
        let e_now = state.e.coeffs();
        let b_half = state.b.coeffs();
        // d1^T M2 B: first M2*B, then d1^T * result.
        let m2b = spmv(&m2_half, b_half);
        let d1t = self.d1.transpose_view().to_csc();
        let d1t_m2b = spmv(&d1t, &m2b);
        let mut rhs_full = spmv(&m1_now, e_now) + self.dt * d1t_m2b;
        if let Some(j) = source {
            rhs_full -= self.dt * j.coeffs();
        }

        // 4) Solve the interior block with Cholesky; PEC edges stay at 0.
        let m1_int = restrict_galmat(&m1_next, &interior);
        let rhs_int = DVector::from_fn(interior.len(), |i, _| rhs_full[interior[i]]);
        let dense = to_dense(&m1_int);
        let sol = dense
            .cholesky()
            .expect("interior M1 must be SPD")
            .solve(&rhs_int);

        // 5) Scatter back into a full-length E with zeros on PEC edges.
        let mut e_new = DVector::zeros(complex.nsimplices(1));
        for (i, &dof) in interior.iter().enumerate() {
            e_new[dof] = sol[i];
        }
        state.e = Cochain::new(1, e_new);
        self.t += self.dt;
    }

    /// One full leapfrog step, returning the synchronized half-step energy as a diagnostic.
    ///
    /// The synchronized energy U_sync = 1/2 (E_half^T M1_half E_half + B^T M2_half B)
    /// is evaluated at the half-step point (t + dt/2), where E_half = 0.5 * (E^n + E^{n+1})
    /// averages the electric field across the step. This places E and B at the same stagger
    /// point, giving a better-conserved observable than the cross-time energy.
    ///
    /// The step itself is identical to `step()`.
    pub fn step_with_energy(
        &mut self,
        state: &mut MaxwellState,
        source: Option<&Cochain>,
    ) -> f64 {
        let dim = self.driver.dim();
        let complex = self.driver.complex();
        let interior = interior_grade_dofs(complex, 1);

        // Capture E^n before the update.
        let e_before = state.e.coeffs().clone();

        // Faraday half-step (metric-free, exact).
        self.faraday_step(state);

        // Assemble time-dependent masses.
        let l_now = self.driver.lengths_at(self.t);
        let l_next = self.driver.lengths_at(self.t + self.dt);
        let l_half = self.driver.lengths_at(self.t + 0.5 * self.dt);
        let m1_now = assemble_galmat(complex, &l_now, HodgeMassElmat::new(dim, 1));
        let m1_next = assemble_galmat(complex, &l_next, HodgeMassElmat::new(dim, 1));
        let m1_half = assemble_galmat(complex, &l_half, HodgeMassElmat::new(dim, 1));
        let m2_half = assemble_galmat(complex, &l_half, HodgeMassElmat::new(dim, 2));

        // Ampere update RHS.
        let e_now = state.e.coeffs();
        let b_half = state.b.coeffs();
        let m2b = spmv(&m2_half, b_half);
        let d1t = self.d1.transpose_view().to_csc();
        let d1t_m2b = spmv(&d1t, &m2b);
        let mut rhs_full = spmv(&m1_now, e_now) + self.dt * d1t_m2b;
        if let Some(j) = source {
            rhs_full -= self.dt * j.coeffs();
        }

        // Cholesky solve on interior block.
        let m1_int = restrict_galmat(&m1_next, &interior);
        let rhs_int = DVector::from_fn(interior.len(), |i, _| rhs_full[interior[i]]);
        let dense = to_dense(&m1_int);
        let sol = dense
            .cholesky()
            .expect("interior M1 must be SPD")
            .solve(&rhs_int);

        // Scatter E^{n+1} back.
        let mut e_new = DVector::zeros(complex.nsimplices(1));
        for (i, &dof) in interior.iter().enumerate() {
            e_new[dof] = sol[i];
        }
        state.e = Cochain::new(1, e_new);
        self.t += self.dt;

        // Synchronized energy: E_half = 0.5*(E^n + E^{n+1}), evaluated at M1_half, M2_half.
        // After the update, state.e = E^{n+1} and e_before = E^n, so we pass e_before
        // as the second argument: synchronized_energy computes 0.5*(self.e + e_next).
        // To get 0.5*(E^n + E^{n+1}) we pass e_before as e_next and note that
        // self.e is E^{n+1}, giving the correct average.
        state.synchronized_energy(&e_before, &m1_half, &m2_half)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::FlrwDriver;
    use crate::state::MaxwellState;
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

    fn run_cavity(spatial_dim: usize, nsub: usize, nsteps: usize) -> f64 {
        use cartan_feec::assemble::interior_grade_dofs;
        use std::collections::HashSet;
        let (complex, coords) =
            CartesianMeshInfo::new_unit(spatial_dim, nsub).compute_coord_complex();
        let base = coords.to_edge_lengths(&complex);
        let driver = FlrwDriver::static_metric(complex.clone(), base);
        let dt = cfl_dt(&driver.lengths_at(0.0));
        let mut evolver = MaxwellEvolver::new(&driver, dt);

        // Closed initial B, initial E satisfying PEC (zero on boundary edges).
        let d1 = coboundary_matrix(&complex, 1);
        let seed = nalgebra::DVector::from_fn(complex.nsimplices(1), |i, _| ((i + 1) as f64).recip());
        let b0 = spmv(&d1, &seed);
        let interior: HashSet<usize> = interior_grade_dofs(&complex, 1).into_iter().collect();
        let e0 = nalgebra::DVector::from_fn(complex.nsimplices(1), |i, _| {
            if interior.contains(&i) { 0.1 * (i as f64).cos() } else { 0.0 }
        });
        let mut state = MaxwellState::new(Cochain::new(1, e0), Cochain::new(2, b0));

        let mut max_resid = 0.0f64;
        for _ in 0..nsteps {
            evolver.step(&mut state, None);
            max_resid = max_resid.max(evolver.magnetic_gauss_residual(&state));
        }
        max_resid
    }

    #[test]
    fn full_step_conserves_flux_in_2d_and_3d() {
        // 2D: residual defined as 0 (no grade-3). 3D: must stay at machine zero.
        assert!(run_cavity(2, 3, 50) < 1e-9, "2D flux residual");
        assert!(run_cavity(3, 2, 50) < 1e-9, "3D flux residual");
    }
}
