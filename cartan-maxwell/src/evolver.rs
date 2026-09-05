//! Staggered-leapfrog Maxwell evolver on an evolving Regge background.

use cartan_matfree::{pcg, HostMass, Interior, MassBackend};
use derham::cochain::Cochain;
use exterior::ExteriorGrade;
use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;
use simplicial::geometry::metric::mesh::MeshLengthsSq;
use simplicial::topology::complex::Complex;

use crate::driver::MetricDriver;
use crate::state::MaxwellState;

/// Relative residual the Ampere solve converges to by default.
const DEFAULT_CG_TOL: f64 = 1e-12;
/// Iteration ceiling for the Ampere solve. Reaching it is a bug, not a budget.
const DEFAULT_CG_MAX_ITER: usize = 500;

/// The coboundary operator d_k: C^k -> C^{k+1} as a sparse matrix of shape
/// `nsimplices(k+1) x nsimplices(k)`. It is the transpose of the boundary
/// operator and is purely combinatorial (metric-free).
pub fn coboundary_matrix(complex: &Complex, k: ExteriorGrade) -> CsrMatrix<f64> {
    CsrMatrix::from(&complex.coboundary_operator(k))
}

/// A conservative CFL time-step estimate: a fraction of the smallest edge length.
/// Factor 0.1 ensures stability for Whitney-form FEEC leapfrog in 2D and 3D.
pub fn cfl_dt(geometry: &MeshLengthsSq) -> f64 {
    let mut hmin = f64::INFINITY;
    for i in 0..geometry.nedges() {
        hmin = hmin.min(geometry.length(i));
    }
    0.1 * hmin
}

/// The grade-1 and grade-2 Hodge masses at the half step, which the
/// synchronized energy is measured against. Kept in element form, so the
/// diagnostic costs a pair of matrix-free applications and no assembly.
struct HalfMasses {
    m1: HostMass,
    m2: HostMass,
}

/// What one Ampere update hands back for diagnostics.
struct AmpereOutcome {
    /// E^n, captured before the update overwrote it with E^{n+1}.
    e_before: DVector<f64>,
    /// Present only when the caller asked for it, since assembling M1 at the
    /// half step costs a full Galerkin pass that `step()` does not need.
    half_masses: Option<HalfMasses>,
}

/// Staggered-leapfrog Maxwell evolver. E lives at integer steps, B at half steps.
pub struct MaxwellEvolver<'d, D: MetricDriver> {
    driver: &'d D,
    d1: CsrMatrix<f64>,         // metric-free coboundary 1 -> 2
    d1t: CsrMatrix<f64>,        // its transpose, cached
    d2: Option<CsrMatrix<f64>>, // metric-free coboundary 2 -> 3 (None in 2D)
    /// The unconstrained grade-1 degrees of freedom. Purely topological, so it
    /// survives every metric change and is built once.
    interior: Interior,
    dt: f64,
    t: f64,
    cg_tol: f64,
    cg_max_iter: usize,
}

impl<'d, D: MetricDriver> MaxwellEvolver<'d, D> {
    pub fn new(driver: &'d D, dt: f64) -> Self {
        let complex = driver.complex();
        let d1 = coboundary_matrix(complex, 1);
        let d1t = d1.transpose();
        let d2 = if complex.dim() >= 3 {
            Some(coboundary_matrix(complex, 2))
        } else {
            None
        };
        let interior = Interior::boundary_constrained(complex, 1);
        Self {
            driver,
            d1,
            d1t,
            d2,
            interior,
            dt,
            t: 0.0,
            cg_tol: DEFAULT_CG_TOL,
            cg_max_iter: DEFAULT_CG_MAX_ITER,
        }
    }

    /// Set the relative residual the Ampere solve converges to, and the
    /// iteration ceiling. The default is `1e-12` in at most 500 iterations,
    /// which is tight enough that the solve does not show up in the energy
    /// drift and loose enough to reach in about 25 iterations on the meshes
    /// this evolver targets.
    pub fn with_cg(mut self, tol: f64, max_iter: usize) -> Self {
        self.cg_tol = tol;
        self.cg_max_iter = max_iter;
        self
    }

    pub fn time(&self) -> f64 {
        self.t
    }

    /// Faraday half-step: B <- B - dt (d1 E). Metric-free and exact.
    /// Because d2 d1 = 0, this preserves the discrete magnetic Gauss law
    /// d2 B = 0 to machine precision for all time.
    pub fn faraday_step(&self, state: &mut MaxwellState) {
        let curl_e = &self.d1 * state.e.coeffs();
        let new_b = state.b.coeffs() - self.dt * curl_e;
        state.b = Cochain::new(2, new_b);
    }

    /// The discrete magnetic Gauss-law residual ||d2 B||_inf. Zero (to machine
    /// precision) means no magnetic monopoles appeared. In 2D there is no
    /// grade-3 space, so the residual is defined as 0.
    pub fn magnetic_gauss_residual(&self, state: &MaxwellState) -> f64 {
        match &self.d2 {
            Some(d2) => {
                let r = d2 * state.b.coeffs();
                r.iter().fold(0.0f64, |m, &v| m.max(v.abs()))
            }
            None => 0.0,
        }
    }

    /// The shared Ampere update. Advances E by one step given the already
    /// Faraday-advanced B, and returns the pre-update E^n alongside the
    /// half-step masses so callers can form diagnostics.
    ///
    /// RHS = M1(t) E^n + dt d1^T M2(t + dt/2) B^{n+1/2} - dt j
    fn ampere_update(
        &mut self,
        state: &mut MaxwellState,
        source: Option<&Cochain>,
        want_half_mass: bool,
    ) -> AmpereOutcome {
        let complex = self.driver.complex();

        let e_before = state.e.coeffs().clone();

        let l_now = self.driver.lengths_sq_at(self.t);
        let l_next = self.driver.lengths_sq_at(self.t + self.dt);
        let l_half = self.driver.lengths_sq_at(self.t + 0.5 * self.dt);

        let m1_now = HostMass::new(complex, &l_now, 1);
        let m2_half = HostMass::new(complex, &l_half, 2);

        // d1^T M2 B: first M2 B, then d1^T applied to the result.
        let mut m2b = vec![0.0; m2_half.ndofs()];
        m2_half.apply_slice(state.b.coeffs().as_slice(), &mut m2b);
        let d1t_m2b = &self.d1t * DVector::from_vec(m2b);

        let mut m1e = vec![0.0; m1_now.ndofs()];
        m1_now.apply_slice(state.e.coeffs().as_slice(), &mut m1e);
        let mut rhs_full = DVector::from_vec(m1e) + self.dt * d1t_m2b;
        if let Some(j) = source {
            rhs_full -= self.dt * j.coeffs();
        }

        // Restrict to the interior (PEC edges are constrained to zero), solve
        // matrix-free, then extend by zero back onto the full mesh.
        //
        // The mass matrix is spectrally equivalent to its diagonal with a
        // mesh-independent constant, so this converges in an iteration count
        // that does not grow with the mesh. The dense factorisation it replaces
        // was cubic in the interior degree-of-freedom count and was rebuilt
        // every step, since the metric moves.
        let m1_next = HostMass::restricted(complex, &l_next, &self.interior);
        let rhs_int = self.interior.restrict(rhs_full.as_slice());

        // Warm start from E^n. Consecutive steps differ by O(dt), so the
        // initial residual is already small.
        let mut sol = self.interior.restrict(state.e.coeffs().as_slice());
        let report = pcg(&m1_next, &rhs_int, &mut sol, self.cg_tol, self.cg_max_iter);
        assert!(
            report.converged,
            "Ampere solve stalled at relative residual {:e} after {} iterations",
            report.residual, report.iterations
        );
        state.e = Cochain::new(1, DVector::from_vec(self.interior.extend_by_zero(&sol)));

        self.t += self.dt;

        let half_masses = want_half_mass.then(|| HalfMasses {
            m1: HostMass::new(complex, &l_half, 1),
            m2: m2_half,
        });
        AmpereOutcome {
            e_before,
            half_masses,
        }
    }

    /// One full leapfrog step. `source` is the (optional) electric current
    /// 1-cochain j at the half time. Advances E by dt and B by dt (staggered).
    pub fn step(&mut self, state: &mut MaxwellState, source: Option<&Cochain>) {
        // 1) Faraday half-step at the current E (metric-free, exact).
        self.faraday_step(state);
        // 2) Ampere update on the time-dependent masses.
        self.ampere_update(state, source, false);
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
        self.faraday_step(state);
        let outcome = self.ampere_update(state, source, true);
        let half = outcome
            .half_masses
            .expect("half masses were requested");
        // After the update state.e is E^{n+1}, and averaging it against the
        // captured E^n places the electric field at the same stagger point as B.
        let e_half = 0.5 * (state.e.coeffs() + &outcome.e_before);
        let ue = half.m1.quadratic_form(e_half.as_slice());
        let ub = half.m2.quadratic_form(state.b.coeffs().as_slice());
        0.5 * (ue + ub)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::FlrwDriver;
    use crate::state::MaxwellState;
    use simplicial::r#gen::cartesian::CartesianGrid;

    #[test]
    fn faraday_preserves_magnetic_gauss_law_exactly_3d() {
        let (complex, coords) = CartesianGrid::new_unit(3, 2).triangulate();
        let base = coords.to_edge_lengths_sq(&complex);
        let driver = FlrwDriver::static_metric(complex.clone(), base);
        let evolver = MaxwellEvolver::new(&driver, 0.01);

        // Start with B closed (d2 B = 0): take B = d1 of an arbitrary 1-cochain.
        let d1 = coboundary_matrix(&complex, 1);
        let seed = DVector::from_fn(complex.nsimplices(1), |i, _| (i as f64).sin());
        let b0 = &d1 * &seed;
        let e0 = DVector::from_fn(complex.nsimplices(1), |i, _| (i as f64).cos());
        let mut state = MaxwellState::new(Cochain::new(1, e0), Cochain::new(2, b0));

        assert!(
            evolver.magnetic_gauss_residual(&state) < 1e-10,
            "seed B not closed"
        );
        for _ in 0..200 {
            evolver.faraday_step(&mut state);
            assert!(
                evolver.magnetic_gauss_residual(&state) < 1e-10,
                "magnetic monopole appeared"
            );
        }
    }

    fn run_cavity(spatial_dim: usize, nsub: usize, nsteps: usize) -> f64 {
        use std::collections::HashSet;
        let (complex, coords) = CartesianGrid::new_unit(spatial_dim, nsub).triangulate();
        let base = coords.to_edge_lengths_sq(&complex);
        let driver = FlrwDriver::static_metric(complex.clone(), base);
        let dt = cfl_dt(&driver.lengths_sq_at(0.0));
        let mut evolver = MaxwellEvolver::new(&driver, dt);

        // Closed initial B, initial E satisfying PEC (zero on boundary edges).
        let d1 = coboundary_matrix(&complex, 1);
        let seed = DVector::from_fn(complex.nsimplices(1), |i, _| ((i + 1) as f64).recip());
        let b0 = &d1 * &seed;
        let boundary: HashSet<usize> = complex
            .boundary_simplices(1)
            .into_iter()
            .map(|idx| idx.kidx)
            .collect();
        let e0 = DVector::from_fn(complex.nsimplices(1), |i, _| {
            if boundary.contains(&i) {
                0.0
            } else {
                0.1 * (i as f64).cos()
            }
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
