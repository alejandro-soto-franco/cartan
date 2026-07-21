//! Staggered-leapfrog Maxwell evolver on an evolving Regge background.

use derham::cochain::Cochain;
use exterior::ExteriorGrade;
use formoniq::whitney_complex::{RelativeWhitneyComplex, WhitneyComplex};
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::CsrMatrix;
use simplicial::geometry::metric::mesh::MeshLengthsSq;
use simplicial::topology::complex::Complex;

use crate::driver::MetricDriver;
use crate::state::MaxwellState;

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

/// Assemble the grade-`k` Hodge mass on the relative (interior) complex,
/// densified for the direct solve. Interior blocks stay small on the meshes
/// this evolver targets.
fn relative_mass_dense(rel: &RelativeWhitneyComplex<'_>, grade: ExteriorGrade) -> DMatrix<f64> {
    DMatrix::from(&rel.mass(grade))
}

/// The grade-1 and grade-2 Hodge masses at the half step, which the
/// synchronized energy is measured against.
struct HalfMasses {
    m1: CsrMatrix<f64>,
    m2: CsrMatrix<f64>,
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
    dt: f64,
    t: f64,
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
        Self {
            driver,
            d1,
            d1t,
            d2,
            dt,
            t: 0.0,
        }
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

        let wc_now = WhitneyComplex::new(complex, &l_now);
        let wc_half = WhitneyComplex::new(complex, &l_half);
        let wc_next = WhitneyComplex::new(complex, &l_next);

        let m1_now = CsrMatrix::from(&wc_now.mass(1));
        let m2_half = CsrMatrix::from(&wc_half.mass(2));

        // d1^T M2 B: first M2 B, then d1^T applied to the result.
        let m2b = &m2_half * state.b.coeffs();
        let d1t_m2b = &self.d1t * &m2b;
        let mut rhs_full = &m1_now * state.e.coeffs() + self.dt * d1t_m2b;
        if let Some(j) = source {
            rhs_full -= self.dt * j.coeffs();
        }

        // Restrict to the interior (PEC edges are constrained to zero), solve,
        // then extend by zero back onto the full mesh.
        let rel_next = wc_next.relative();
        let rhs_int = rel_next.restrict(&Cochain::new(1, rhs_full));
        let m1_int = relative_mass_dense(&rel_next, 1);
        let sol = m1_int
            .cholesky()
            .expect("interior M1 must be SPD")
            .solve(rhs_int.coeffs());
        state.e = rel_next.extend_by_zero(&Cochain::new(1, sol));

        self.t += self.dt;

        let half_masses = want_half_mass.then(|| HalfMasses {
            m1: CsrMatrix::from(&wc_half.mass(1)),
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
        // After the update state.e is E^{n+1}; synchronized_energy averages it
        // against the value passed in, which is E^n.
        state.synchronized_energy(&outcome.e_before, &half.m1, &half.m2)
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
