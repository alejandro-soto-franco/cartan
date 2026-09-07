//! The k-atic Landau functional and its gradient.
//!
//! ## Why the energy is written in the invariant tensor
//!
//! Both the bulk and the elastic term are functions of `T(R, a)`, the
//! `H^`-invariant order parameter. That makes them automatically invariant
//! under the whole stabiliser, continuous part included, so the gradient
//! along a stabiliser direction vanishes identically and no projection step
//! is needed. Writing the elastic term as a rotor distance instead would need
//! a minimisation over `H^` on every edge and a hand-written projection for
//! the two continuous families.
//!
//! ## Bulk
//!
//! `f(a) = a^T A a / 2 + C(a,a,a) / 3 + (a^T B a)^2 / 4`, a polynomial in the
//! amplitudes with `B` symmetric positive definite. For one amplitude this is
//! the Landau-de Gennes bulk term for term. `A` may be indefinite: that is the
//! ordering transition. `B` positive definite is what makes the quartic
//! dominate, and it is checked at construction, so no downstream code can hold
//! a functional that is unbounded below.

use nalgebra::{DMatrix, DVector};

use cartan_core::rotor::Rotor3;

use crate::error::KaticError;
use crate::group::SymmetryGroup;
use crate::invariant::InvariantBasis;
use crate::spin::Incidence;

/// A k-atic state: one rotor and one amplitude vector per vertex.
#[derive(Clone, Debug)]
pub struct State {
    /// Frame per vertex.
    pub rotors: Vec<Rotor3>,
    /// Amplitudes per vertex, `n_amplitudes` each, laid out vertex-major.
    pub amplitudes: Vec<f64>,
    n_amplitudes: usize,
}

impl State {
    /// A uniform state: every vertex at `rotor` with the same amplitudes.
    #[must_use]
    pub fn uniform(n_vertices: usize, rotor: Rotor3, amplitudes: &[f64]) -> Self {
        Self {
            rotors: vec![rotor; n_vertices],
            amplitudes: amplitudes.repeat(n_vertices),
            n_amplitudes: amplitudes.len(),
        }
    }

    /// Number of vertices.
    #[must_use]
    pub fn n_vertices(&self) -> usize {
        self.rotors.len()
    }

    /// Amplitudes at a vertex.
    #[must_use]
    pub fn amps(&self, v: usize) -> &[f64] {
        &self.amplitudes[v * self.n_amplitudes..(v + 1) * self.n_amplitudes]
    }

    /// Largest deviation of any rotor from the unit sphere.
    #[must_use]
    pub fn worst_norm_defect(&self) -> f64 {
        self.rotors
            .iter()
            .map(|r| ((r.w * r.w + r.x * r.x + r.y * r.y + r.z * r.z).sqrt() - 1.0).abs())
            .fold(0.0, f64::max)
    }
}

/// The Landau functional for a symmetry, with its invariant basis and the
/// precomputed pieces the gradient needs.
#[derive(Clone, Debug)]
pub struct Energy {
    basis: InvariantBasis,
    generators: [DMatrix<f64>; 3],
    a: DMatrix<f64>,
    c: Vec<f64>,
    b: DMatrix<f64>,
    elastic_weight: f64,
}

impl Energy {
    /// Build the functional, rejecting a quartic form that is not positive
    /// definite.
    ///
    /// `a` and `b` are `n x n` symmetric, `c` is the flattened symmetric cubic
    /// form of side `n`, with `n` the amplitude count.
    pub fn new<H: SymmetryGroup>(
        a: DMatrix<f64>,
        c: Vec<f64>,
        b: DMatrix<f64>,
        elastic_weight: f64,
    ) -> Result<Self, KaticError> {
        let basis = InvariantBasis::for_group::<H>()?;
        let n = basis.n_amplitudes();
        assert_eq!(a.nrows(), n, "A must be n x n");
        assert_eq!(b.nrows(), n, "B must be n x n");
        assert_eq!(c.len(), n * n * n, "C must be n^3");

        // Coercivity: the quartic dominates only when B is positive definite.
        let sym = (&b + b.transpose()) * 0.5;
        let min_eig = sym
            .clone()
            .symmetric_eigen()
            .eigenvalues
            .iter()
            .fold(f64::INFINITY, |m, &v| m.min(v));
        if min_eig <= 0.0 {
            return Err(KaticError::NonCoerciveEnergy { degree: 4 });
        }

        Ok(Self {
            generators: crate::invariant::so3_generators(basis.degree()),
            basis,
            a,
            c,
            b: sym,
            elastic_weight,
        })
    }

    /// The invariant basis this functional is written in.
    #[must_use]
    pub fn basis(&self) -> &InvariantBasis {
        &self.basis
    }

    /// The `so(3)` generator for axis `a`, on the order parameter's degree.
    #[must_use]
    pub fn generator(&self, a: usize) -> &DMatrix<f64> {
        &self.generators[a]
    }

    /// Bulk energy density at one vertex.
    #[must_use]
    pub fn bulk(&self, amps: &[f64]) -> f64 {
        let n = amps.len();
        let v = DVector::from_column_slice(amps);
        let quad = (v.transpose() * &self.a * &v)[(0, 0)] * 0.5;
        let mut cubic = 0.0;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    cubic += self.c[(i * n + j) * n + k] * amps[i] * amps[j] * amps[k];
                }
            }
        }
        let s = (v.transpose() * &self.b * &v)[(0, 0)];
        quad + cubic / 3.0 + s * s / 4.0
    }

    /// The order parameter at a vertex, in monomial coordinates.
    ///
    /// The harmonic route through `rep_harmonic` was measured on 2026-09-07
    /// and is slower at both ends of the range: 946 against 538 ns per vertex
    /// at degree 2, and 5035 against 4124 at degree 6. The matrix exponential
    /// costs more than the polynomial substitution it replaces at these sizes,
    /// so the energy stays on the monomial path.
    #[must_use]
    pub fn tensor(&self, r: &Rotor3, amps: &[f64]) -> DVector<f64> {
        self.basis.order_parameter(r, amps)
    }

    /// Total energy: bulk over vertices plus the squared tensor difference
    /// over edges.
    #[must_use]
    pub fn total(&self, inc: &Incidence, state: &State) -> f64 {
        let tensors: Vec<DVector<f64>> = (0..state.n_vertices())
            .map(|v| self.tensor(&state.rotors[v], state.amps(v)))
            .collect();
        let mut e = 0.0;
        for v in 0..state.n_vertices() {
            e += self.bulk(state.amps(v));
        }
        for [u, v] in inc.edges() {
            let d = &tensors[*u] - &tensors[*v];
            e += self.elastic_weight * d.dot(&d);
        }
        e
    }

    /// Gradient in the `so(3)` directions and the amplitudes, one block per
    /// vertex: three rotation components then the amplitudes.
    #[must_use]
    pub fn gradient(&self, inc: &Incidence, state: &State) -> Vec<f64> {
        let n = self.basis.n_amplitudes();
        let nv = state.n_vertices();
        let block = 3 + n;
        // One representation matrix per vertex, reused for the tensor and for
        // every amplitude column. Rebuilding it per call was the hot path.
        let reps: Vec<DMatrix<f64>> = (0..nv).map(|v| self.basis.rep(&state.rotors[v])).collect();
        let tensors: Vec<DVector<f64>> = (0..nv)
            .map(|v| &reps[v] * self.basis.reference(state.amps(v)))
            .collect();

        // dE/dT at each vertex, from the elastic term only; the bulk depends
        // on the amplitudes directly.
        let mut dedt: Vec<DVector<f64>> = vec![DVector::zeros(tensors[0].len()); nv];
        for [u, v] in inc.edges() {
            let d = &tensors[*u] - &tensors[*v];
            dedt[*u] += 2.0 * self.elastic_weight * &d;
            dedt[*v] -= 2.0 * self.elastic_weight * &d;
        }

        let mut g = vec![0.0; nv * block];
        for v in 0..nv {
            let rho = &reps[v];
            // Rotation directions: dT/dxi_a = L_a T.
            for a in 0..3 {
                g[v * block + a] = dedt[v].dot(&(&self.generators[a] * &tensors[v]));
            }
            // Amplitude directions: dT/da_i = rho * basis column i.
            for i in 0..n {
                let col = rho * self.basis.basis_column(i);
                g[v * block + 3 + i] = dedt[v].dot(&col);
            }
            // Bulk contribution to the amplitude block.
            let amps = state.amps(v);
            for i in 0..n {
                g[v * block + 3 + i] += self.bulk_derivative(amps, i);
            }
        }
        g
    }

    /// Derivative of the bulk with respect to amplitude `i`.
    fn bulk_derivative(&self, amps: &[f64], i: usize) -> f64 {
        let n = amps.len();
        let v = DVector::from_column_slice(amps);
        let s = (v.transpose() * &self.b * &v)[(0, 0)];
        let bv = &self.b * &v;
        let mut cubic = 0.0;
        for j in 0..n {
            for k in 0..n {
                cubic += self.c[(i * n + j) * n + k] * amps[j] * amps[k];
            }
        }
        (&self.a * &v)[i] + cubic + s * bv[i]
    }
}
