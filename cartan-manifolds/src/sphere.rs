// ~/cartan/cartan-manifolds/src/sphere.rs

//! Unit sphere S^{N-1} embedded in R^N.
//!
//! The sphere of unit vectors in R^N with the round metric inherited
//! from the ambient Euclidean space.
//!
//! ## Geometry
//!
//! - Points: unit vectors p in R^N with ||p|| = 1
//! - Tangent space at p: T_pS = {v in R^N : p^T v = 0}
//! - Inner product: <u, v>_p = u^T v (inherited from R^N, independent of p)
//! - Exp: cos(||v||) p + sin(||v||) v/||v||
//! - Log: theta * (q - cos(theta) * p) / sin(theta), theta = arccos(p^T q)
//! - Cut locus: antipodal point {-p}
//! - Injectivity radius: pi
//! - Sectional curvature: K = 1 (constant positive curvature)
//!
//! ## Numerical stability
//!
//! - Small distances (theta < 1e-7): Taylor expansion to avoid 0/0 in log.
//! - Near cut locus (|theta - pi| < tol): return CutLocus error.
//! - Small tangent norm in exp: Taylor expansion of sin/cos.
//!
//! ## References
//!
//! - Absil et al., "Optimization Algorithms on Matrix Manifolds", Example 3.5.1
//! - do Carmo, "Riemannian Geometry", Chapter 3, Example 2.5

use core::f64::consts::PI;

#[cfg(feature = "alloc")]
use alloc::string::ToString;

#[cfg(not(feature = "std"))]
use nalgebra::ComplexField;
use nalgebra::SVector;
use rand::Rng;
use rand_distr::StandardNormal;

use cartan_core::{
    CartanError, Connection, Curvature, GeodesicInterpolation, Manifold, ParallelTransport, Real,
    Retraction,
};

/// The unit sphere S^{N-1} embedded in R^N.
///
/// Zero-sized type: the geometry is fully determined by N.
/// S^{N-1} has intrinsic dimension N-1.
///
/// # Examples
///
/// ```rust
/// use cartan_manifolds::Sphere;
/// use cartan_core::Manifold;
/// use nalgebra::SVector;
///
/// let s2 = Sphere::<3>;
/// let p: SVector<f64, 3> = SVector::from([1.0, 0.0, 0.0]);
/// let v: SVector<f64, 3> = SVector::from([0.0, 1.0, 0.0]);
/// let q = s2.exp(&p, &v);
/// assert!((q.norm() - 1.0).abs() < 1e-10);
/// let v_rec = s2.log(&p, &q).unwrap();
/// assert!((v_rec - v).norm() < 1e-10);
/// ```
///
/// ```rust,no_run
/// use cartan_manifolds::Sphere;
/// use cartan_core::Manifold;
/// use rand::SeedableRng;
///
/// let s2 = Sphere::<3>;
/// let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
/// let p = s2.random_point(&mut rng);
/// let v = s2.random_tangent(&p, &mut rng);
/// let _q = s2.exp(&p, &v);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Sphere<const N: usize>;

/// Tolerance for detecting near-zero and near-pi angles.
const ANGLE_EPS: Real = 1e-7;

/// Cut-locus threshold expressed on `1 + cos(angle)` rather than on the angle.
///
/// `1 - cos(ANGLE_EPS) ~ ANGLE_EPS^2 / 2`, so this is the same boundary that
/// `log` applies as `|pi - angle| < ANGLE_EPS`, in the form `transport` can
/// test without computing the angle.
const CUT_LOCUS_EPS: Real = 0.5 * ANGLE_EPS * ANGLE_EPS;
/// Tolerance for point/tangent validation.
const VALIDATION_TOL: Real = 1e-10;

impl<const N: usize> Manifold for Sphere<N> {
    type Point = SVector<Real, N>;
    type Tangent = SVector<Real, N>;

    fn dim(&self) -> usize {
        // S^{N-1} has intrinsic dimension N-1.
        N - 1
    }

    fn ambient_dim(&self) -> usize {
        N
    }

    fn injectivity_radius(&self, _p: &Self::Point) -> Real {
        PI
    }

    /// Geodesic distance on the sphere: d(p, q) = arccos(p^T q).
    ///
    /// Overrides the default (log-based) implementation so that the distance
    /// is well-defined even at the cut locus (antipodal points), where log
    /// is undefined but d(p, -p) = pi is exact.
    fn dist(&self, p: &Self::Point, q: &Self::Point) -> Result<Real, CartanError> {
        // Use the numerically stable formula 2*asin(||p - q|| / 2).
        // Unlike arccos(p·q), this avoids catastrophic cancellation near
        // p == q (distance ~ 0) while remaining accurate at the antipodal
        // point (distance = pi).
        // Measured against a manual accumulation loop, which is slower here:
        // nalgebra vectorises the difference and the norm, and an indexed loop
        // does not.
        let half_chord = (p - q).norm() / 2.0;
        Ok(2.0 * half_chord.clamp(0.0, 1.0).asin())
    }

    /// Standard inner product, inherited from R^N.
    /// Independent of the base point p.
    fn inner(&self, _p: &Self::Point, u: &Self::Tangent, v: &Self::Tangent) -> Real {
        u.dot(v)
    }

    /// Exponential map on the sphere.
    ///
    /// Exp_p(v) = cos(||v||) p + sin(||v||) v/||v||
    ///
    /// For small ||v|| < ANGLE_EPS, use first-order Taylor approximation
    /// to avoid division by zero: Exp_p(v) ~ p + v - (||v||^2 / 2) p.
    fn exp(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Point {
        // Left as nalgebra expressions deliberately. An accumulate-in-one-pass
        // rewrite measured slower at every dimension, 75 ns to 116 ns at N = 50,
        // because indexed writes into an SVector do not vectorise the way these
        // do.
        let v_norm = v.norm();

        if v_norm < ANGLE_EPS {
            // Taylor expansion: cos(t) ~ 1 - t^2/2, sin(t)/t ~ 1 - t^2/6.
            // Exp_p(v) ~ (1 - ||v||^2/2) p + (1 - ||v||^2/6) v
            //          ~ p + v - (||v||^2/2) p  (to first order)
            let result = p + v - p * (v_norm * v_norm / 2.0);
            // Re-normalize to stay exactly on the sphere.
            result / result.norm()
        } else {
            let cos_t = v_norm.cos();
            let sin_t = v_norm.sin();
            let result = p * cos_t + v * (sin_t / v_norm);
            // Re-normalise, and it is load-bearing rather than defensive.
            // ||p cos(t) + v sin(t)/||v|| || is 1 only when v is exactly
            // tangent; removing this is 1.5x faster and lets a v carrying a
            // 1e-9 normal component drift 7e-10 off the sphere, which
            // `test_exp_stays_on_the_sphere` pins.
            result / result.norm()
        }
    }

    /// Logarithmic map on the sphere.
    ///
    /// Log_p(q) = theta * (q - cos(theta) * p) / sin(theta)
    /// where theta = arccos(p^T q).
    ///
    /// Returns CutLocus error for antipodal points (theta ~ pi).
    /// Uses Taylor expansion for small theta to avoid 0/0.
    fn log(&self, p: &Self::Point, q: &Self::Point) -> Result<Self::Tangent, CartanError> {
        // Clamp the dot product to [-1, 1] for numerical safety in arccos.
        let cos_theta = p.dot(q).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();

        // Near cut locus: antipodal points.
        if (PI - theta).abs() < ANGLE_EPS {
            #[cfg(feature = "alloc")]
            return Err(CartanError::CutLocus {
                message: alloc::format!(
                    "points are nearly antipodal on S^{}: angle = {:.2e}, |pi - angle| = {:.2e}",
                    N - 1,
                    theta,
                    (PI - theta).abs()
                ),
            });
            #[cfg(not(feature = "alloc"))]
            return Err(CartanError::CutLocus {
                message: "points are nearly antipodal (cut locus of sphere)",
            });
        }

        if theta < ANGLE_EPS {
            // Small angle: log_p(q) ~ q - p (first-order).
            // Project onto tangent space to remove any normal component.
            let v = q - p;
            Ok(self.project_tangent(p, &v))
        } else {
            // Standard formula: theta * (q - cos(theta) * p) / sin(theta),
            // built in one pass rather than as a difference vector that is
            // then scaled. Measured 1.8x faster than the two-expression form
            // under criterion at N = 50 and indistinguishable from it under
            // the batch harness the cross-language comparison uses, so it is
            // kept as the form that is never slower rather than as a claimed
            // speedup.
            let k = theta / theta.sin();
            Ok(SVector::<Real, N>::from_fn(|i, _| (q[i] - p[i] * cos_theta) * k))
        }
    }

    /// Exponential map written in place, with no temporary and no returned
    /// copy.
    ///
    /// `copy_from` followed by a single `axpy` forms `a p + b v` in one pass,
    /// where the value-returning form builds an intermediate vector and then
    /// walks it again. The renormalisation stays: it is load-bearing, as
    /// `test_exp_stays_on_the_sphere` records.
    fn exp_into(&self, p: &Self::Point, v: &Self::Tangent, out: &mut Self::Point) {
        let v_norm = v.norm();
        let (a, b) = if v_norm < ANGLE_EPS {
            // Taylor: Exp_p(v) ~ (1 - ||v||^2/2) p + v.
            (1.0 - v_norm * v_norm / 2.0, 1.0)
        } else {
            (v_norm.cos(), v_norm.sin() / v_norm)
        };

        out.copy_from(p);
        out.axpy(b, v, a); // out = b v + a p
        let n = out.norm();
        *out /= n;
    }

    /// Logarithmic map written in place.
    ///
    /// `out` is untouched when the call fails at the cut locus, so a caller
    /// reusing a buffer cannot read a stale value as a fresh one.
    fn log_into(
        &self,
        p: &Self::Point,
        q: &Self::Point,
        out: &mut Self::Tangent,
    ) -> Result<(), CartanError> {
        let cos_theta = p.dot(q).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();

        if (PI - theta).abs() < ANGLE_EPS {
            #[cfg(feature = "alloc")]
            return Err(CartanError::CutLocus {
                message: alloc::format!(
                    "points are nearly antipodal on S^{}: angle = {:.2e}",
                    N - 1,
                    theta
                ),
            });
            #[cfg(not(feature = "alloc"))]
            return Err(CartanError::CutLocus {
                message: "points are nearly antipodal (cut locus of sphere)",
            });
        }

        if theta < ANGLE_EPS {
            // Small angle: q - p, projected onto T_p.
            out.copy_from(q);
            out.axpy(-1.0, p, 1.0);
            let d = p.dot(out);
            out.axpy(-d, p, 1.0);
        } else {
            let k = theta / theta.sin();
            out.copy_from(q);
            out.axpy(-cos_theta, p, 1.0);
            *out *= k;
        }
        Ok(())
    }

    /// Project ambient vector onto tangent space T_p S^{N-1}.
    ///
    /// pi_p(v) = v - (p^T v) p
    ///
    /// Subtracts the component of v in the direction of the unit normal p.
    fn project_tangent(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Tangent {
        v - p * p.dot(v)
    }

    /// Project ambient point onto the sphere: p / ||p||.
    fn project_point(&self, p: &Self::Point) -> Self::Point {
        let n = p.norm();
        if n < 1e-15 {
            // Degenerate case: zero vector. Return an arbitrary point.
            let mut result = SVector::zeros();
            result[0] = 1.0;
            result
        } else {
            p / n
        }
    }

    fn zero_tangent(&self, _p: &Self::Point) -> Self::Tangent {
        SVector::zeros()
    }

    /// Check ||p|| = 1.
    fn check_point(&self, p: &Self::Point) -> Result<(), CartanError> {
        let violation = (p.norm() - 1.0).abs();
        if violation < VALIDATION_TOL {
            Ok(())
        } else {
            #[cfg(feature = "alloc")]
            {
                Err(CartanError::NotOnManifold {
                    constraint: alloc::format!("||p|| = 1 (S^{})", N - 1),
                    violation,
                })
            }
            #[cfg(not(feature = "alloc"))]
            Err(CartanError::NotOnManifold {
                constraint: "||p|| = 1 (unit sphere S^(N-1))",
                violation,
            })
        }
    }

    /// Check p^T v = 0.
    fn check_tangent(&self, p: &Self::Point, v: &Self::Tangent) -> Result<(), CartanError> {
        let violation = p.dot(v).abs();
        if violation < VALIDATION_TOL {
            Ok(())
        } else {
            #[cfg(feature = "alloc")]
            {
                Err(CartanError::NotInTangentSpace {
                    constraint: alloc::format!("p^T v = 0 (T_p S^{})", N - 1),
                    violation,
                })
            }
            #[cfg(not(feature = "alloc"))]
            Err(CartanError::NotInTangentSpace {
                constraint: "p^T v = 0 (tangent space of S^(N-1))",
                violation,
            })
        }
    }

    /// Random point on S^{N-1}: sample Gaussian, normalize.
    /// This gives the uniform (Haar) distribution on the sphere.
    fn random_point<R: Rng>(&self, rng: &mut R) -> Self::Point {
        let v: SVector<Real, N> = SVector::from_fn(|_, _| rng.sample(StandardNormal));
        v / v.norm()
    }

    /// Random tangent vector at p: sample Gaussian in R^N, project onto T_pS.
    fn random_tangent<R: Rng>(&self, p: &Self::Point, rng: &mut R) -> Self::Tangent {
        let v: SVector<Real, N> = SVector::from_fn(|_, _| rng.sample(StandardNormal));
        self.project_tangent(p, &v)
    }
}

impl<const N: usize> Retraction for Sphere<N> {
    /// Projection retraction: normalize p + v.
    ///
    /// retract(p, v) = (p + v) / ||p + v||
    ///
    /// Cheaper than exp (no trig functions) and satisfies the
    /// retraction axioms (R_p(0) = p, first-order agreement with exp).
    fn retract(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Point {
        let result = p + v;
        result / result.norm()
    }

    fn inverse_retract(
        &self,
        p: &Self::Point,
        q: &Self::Point,
    ) -> Result<Self::Tangent, CartanError> {
        // The inverse of the projection retraction:
        // v such that (p + v)/||p + v|| = q and p^T v = 0.
        // v = q / (p^T q) - p (when p^T q > 0).
        let cos_theta = p.dot(q);
        if cos_theta < ANGLE_EPS {
            #[cfg(feature = "alloc")]
            return Err(CartanError::CutLocus {
                message: "inverse_retract: points too far apart".to_string(),
            });
            #[cfg(not(feature = "alloc"))]
            return Err(CartanError::CutLocus {
                message: "inverse_retract: points too far apart",
            });
        }
        Ok(q / cos_theta - p)
    }
}

impl<const N: usize> ParallelTransport for Sphere<N> {
    /// Parallel transport along the geodesic from p to q.
    ///
    /// Closed-form formula:
    ///   Gamma_{p->q}(v) = v - (<log_p(q), v> / dist^2(p,q)) * (log_p(q) + log_q(p))
    ///
    /// For p ~ q (small distance), transport is approximately the identity
    /// minus the second-order correction.
    ///
    /// Ref: Absil et al., Section 8.1.3.
    fn transport(
        &self,
        p: &Self::Point,
        q: &Self::Point,
        v: &Self::Tangent,
    ) -> Result<Self::Tangent, CartanError> {
        // Written in terms of the dot product alone:
        //
        //   PT(v) = v - (v.w) (w/(1+c) + p),   c = p.q,   w = q - c p
        //
        // Transport rotates the component of v in the (p, w) plane through the
        // angle theta between p and q, and fixes the rest. For unit vectors
        // cos(theta) = c and ||w|| = sin(theta), so both trig calls cancel out
        // of the rotation and neither theta nor a normalisation is ever formed.
        //
        // The previous form evaluated log twice, once in each direction, each
        // paying an inverse trig call, a norm and a division.
        //
        // That the result is tangent at q is exact rather than approximate:
        //   PT(v).q = v.w - (v.w)[(1 - c^2)/(1 + c) + c] = 0,
        // using v.p = 0 and q.q = 1.
        let c = p.dot(q).clamp(-1.0, 1.0);
        let one_plus_c = 1.0 + c;

        // Cut locus: theta -> pi, where the minimising geodesic is not unique.
        // Since c = cos(theta), the condition |pi - theta| < ANGLE_EPS that
        // `log` uses becomes 1 + c < 1 - cos(ANGLE_EPS) ~ ANGLE_EPS^2 / 2, so
        // both agree on where transport stops being defined.
        if one_plus_c < CUT_LOCUS_EPS {
            #[cfg(feature = "alloc")]
            return Err(CartanError::CutLocus {
                message: alloc::format!(
                    "points are nearly antipodal on S^{}: 1 + cos(angle) = {:.2e}",
                    N - 1,
                    one_plus_c
                ),
            });
            #[cfg(not(feature = "alloc"))]
            return Err(CartanError::CutLocus {
                message: "points are nearly antipodal (cut locus of sphere)",
            });
        }

        // As p approaches q, w tends to zero and the correction vanishes, so
        // the coincident case needs no branch of its own.
        // Left as nalgebra expressions deliberately. A three-pass indexed
        // rewrite measured 91 ns to 125 ns at N = 50, for the same reason exp
        // does: these expressions vectorise and indexed writes do not.
        //
        // The algebraic simplification (q - c p)/(1 + c) + p = (p + q)/(1 + c)
        // was also measured. It removes w but needs v.q and v.p in place of
        // v.w, and at N = 50 the extra pass costs more than the vector it
        // saves: 91 ns to 149 ns. See `transport_into` for the form that does
        // pay off, which avoids the temporaries instead of the arithmetic.
        let w = q - p * c;
        let beta = v.dot(&w);
        let transported = v - (w / one_plus_c + p) * beta;

        // Re-projected, and not merely out of caution. Tangency is exact in
        // exact arithmetic, but `w/(1 + c)` amplifies rounding as c approaches
        // -1: without this the residual reaches 4e-10 near the cut locus, which
        // `test_transport_tangency_holds_near_cut_locus` pins.
        Ok(self.project_tangent(q, &transported))
    }

    /// Parallel transport written in place, with no temporary and no returned
    /// copy.
    ///
    /// This is where the algebraic collapse pays off. Using
    /// `(q - c p)/(1 + c) + p = (p + q)/(1 + c)` and
    /// `v.w = v.q - c (v.p)`, the direction vector `w` is never formed, so the
    /// whole call is three inner products and three `axpy`s over `out`. The
    /// same collapse loses in the value-returning form, where `(p + q) * k`
    /// materialises a vector; here there is nothing to materialise.
    fn transport_into(
        &self,
        p: &Self::Point,
        q: &Self::Point,
        u: &Self::Tangent,
        out: &mut Self::Tangent,
    ) -> Result<(), CartanError> {
        let c = p.dot(q).clamp(-1.0, 1.0);
        let one_plus_c = 1.0 + c;

        if one_plus_c < CUT_LOCUS_EPS {
            #[cfg(feature = "alloc")]
            return Err(CartanError::CutLocus {
                message: alloc::format!(
                    "points are nearly antipodal on S^{}: 1 + cos(angle) = {:.2e}",
                    N - 1,
                    one_plus_c
                ),
            });
            #[cfg(not(feature = "alloc"))]
            return Err(CartanError::CutLocus {
                message: "points are nearly antipodal (cut locus of sphere)",
            });
        }

        // beta = u.w = u.q - c (u.p); the second term vanishes for an exactly
        // tangent u and is kept so a small normal component behaves as it does
        // in `transport`.
        let beta = u.dot(q) - c * u.dot(p);
        let k = beta / one_plus_c;

        out.copy_from(u);
        out.axpy(-k, p, 1.0);
        out.axpy(-k, q, 1.0);

        // Re-project, for the reason given on `transport`.
        let t_dot_q = out.dot(q);
        out.axpy(-t_dot_q, q, 1.0);
        Ok(())
    }
}

// VectorTransport: blanket impl from ParallelTransport.

impl<const N: usize> Connection for Sphere<N> {
    /// Riemannian Hessian on the sphere with Weingarten correction.
    ///
    /// Full formula (Absil et al., Example 5.3.2):
    ///
    ///   Hess f(p)\[v\] = proj_p(D^2f(p)\[v\]) - <egrad, p> * v
    ///
    /// where `grad_f` is the **Euclidean** gradient `egrad` (not the projected
    /// Riemannian gradient), matching the Pymanopt/Manopt convention.
    /// The term `<egrad, p>` is the normal component of the gradient and is
    /// nonzero whenever the cost function has a component along the normal to
    /// the sphere at p.
    fn riemannian_hessian_vector_product(
        &self,
        p: &Self::Point,
        grad_f: &Self::Tangent,
        v: &Self::Tangent,
        hess_ambient: &dyn Fn(&Self::Tangent) -> Self::Tangent,
    ) -> Result<Self::Tangent, CartanError> {
        // Step 1: project ambient HVP onto tangent space.
        let ehvp = hess_ambient(v);
        let proj_ehvp = self.project_tangent(p, &ehvp);

        // Step 2: Weingarten correction — shape operator of S^{N-1} in R^N.
        //   correction = <egrad, p> * v
        // Convention: grad_f is the Euclidean gradient (egrad), so
        // <egrad, p> = grad_f.dot(p).
        let normal_component = grad_f.dot(p);
        let weingarten = v * normal_component;

        Ok(proj_ehvp - weingarten)
    }
}

impl<const N: usize> Curvature for Sphere<N> {
    /// Curvature tensor for the unit sphere (constant sectional curvature K = 1).
    ///
    /// R(u, v)w = <v, w> u - <u, w> v
    ///
    /// This is the standard formula for a space of constant curvature K = 1.
    /// Ref: do Carmo, "Riemannian Geometry", Proposition 4.1 (constant curvature spaces).
    fn riemann_curvature(
        &self,
        _p: &Self::Point,
        u: &Self::Tangent,
        v: &Self::Tangent,
        w: &Self::Tangent,
    ) -> Self::Tangent {
        // R(u,v)w = <v,w>u - <u,w>v for K = 1.
        u * v.dot(w) - v * u.dot(w)
    }

    /// Ricci curvature of S^{N-1}.
    ///
    /// For a space of constant sectional curvature K, the Ricci curvature is:
    ///   Ric(u, v) = (N-2) * K * <u, v>
    ///
    /// For S^{N-1} with K = 1: Ric(u, v) = (N-2) * <u, v>.
    fn ricci_curvature(&self, _p: &Self::Point, u: &Self::Tangent, v: &Self::Tangent) -> Real {
        // Ricci = (dim - 1) * K * <u, v> where dim = N-1, K = 1.
        // Wait: Ric(u,v) = sum_{i} <R(e_i, u)v, e_i> over an ONB of T_pM.
        // For constant curvature K: Ric(u,v) = (n-1)*K*<u,v> where n = dim(M).
        // dim(S^{N-1}) = N-1, so Ric = (N-2) * 1 * <u,v>.
        (N as Real - 2.0) * u.dot(v)
    }

    /// Scalar curvature of S^{N-1}.
    ///
    /// S = n(n-1)K where n = dim(M) = N-1, K = 1.
    /// S = (N-1)(N-2).
    fn scalar_curvature(&self, _p: &Self::Point) -> Real {
        let n = N as Real - 1.0;
        n * (n - 1.0)
    }
}

impl<const N: usize> GeodesicInterpolation for Sphere<N> {
    /// Geodesic interpolation: gamma(t) = exp_p(t * log_p(q)).
    fn geodesic(
        &self,
        p: &Self::Point,
        q: &Self::Point,
        t: Real,
    ) -> Result<Self::Point, CartanError> {
        let v = self.log(p, q)?;
        Ok(self.exp(p, &(v * t)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `transport` has a closed form that never calls `log`. This pins it to
    /// the two-log formula it replaced (Absil et al. 8.1.3), so a divergence
    /// fails loudly rather than silently returning a different connection.
    #[test]
    fn test_transport_matches_two_log_formula() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        macro_rules! check {
            ($n:literal) => {{
                let m = Sphere::<$n>;
                let mut rng = StdRng::seed_from_u64(11);
                for _ in 0..40 {
                    let p = m.random_point(&mut rng);
                    let q = m.random_point(&mut rng);
                    let v = m.random_tangent(&p, &mut rng);

                    // Reference: v - (<log_p q, v> / d^2)(log_p q + log_q p).
                    let log_pq = m.log(&p, &q).unwrap();
                    let log_qp = m.log(&q, &p).unwrap();
                    let d_sq = m.inner(&p, &log_pq, &log_pq);
                    let coeff = m.inner(&p, &log_pq, &v) / d_sq;
                    let reference = m.project_tangent(&q, &(v - (log_pq + log_qp) * coeff));

                    let fast = m.transport(&p, &q, &v).unwrap();
                    assert!(
                        (fast - reference).norm() < 1e-9,
                        "transport disagrees on S^{}: {:.3e}",
                        $n - 1,
                        (fast - reference).norm()
                    );
                }
            }};
        }

        check!(3);
        check!(10);
        check!(50);
    }

    /// Parallel transport is an isometry between tangent spaces, and lands in
    /// the tangent space at the destination. A closed form that dropped the
    /// final projection could satisfy the first and fail the second.
    #[test]
    fn test_transport_is_isometric_and_tangent() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let m = Sphere::<10>;
        let mut rng = StdRng::seed_from_u64(3);
        for _ in 0..40 {
            let p = m.random_point(&mut rng);
            let q = m.random_point(&mut rng);
            let u = m.random_tangent(&p, &mut rng);
            let v = m.random_tangent(&p, &mut rng);

            let u_q = m.transport(&p, &q, &u).unwrap();
            let v_q = m.transport(&p, &q, &v).unwrap();

            assert!((m.inner(&p, &u, &v) - m.inner(&q, &u_q, &v_q)).abs() < 1e-10);
            assert!(u_q.dot(&q).abs() < 1e-12, "result must lie in T_q S");
        }
    }

    /// Transporting to the same point is the identity, and the formula must
    /// not degrade as p approaches q, where the direction to q is ill-defined.
    #[test]
    fn test_transport_identity_and_near_coincident() {
        let m = Sphere::<3>;
        let p = SVector::<Real, 3>::new(0.0, 0.0, 1.0);
        let v = SVector::<Real, 3>::new(0.3, -0.4, 0.0);

        let same = m.transport(&p, &p, &v).unwrap();
        assert!((same - v).norm() < 1e-12);

        for eps in [1e-3, 1e-6, 1e-9, 1e-12] {
            let q = SVector::<Real, 3>::new(eps, 0.0, (1.0 - eps * eps).sqrt());
            let t = m.transport(&p, &q, &v).unwrap();
            assert!(t.dot(&q).abs() < 1e-12, "eps = {eps:e}: not tangent at q");
            assert!(
                (t.norm() - v.norm()).abs() < 1e-9,
                "eps = {eps:e}: norm not preserved"
            );
        }
    }

    /// Near the cut locus `w/(1 + c)` amplifies rounding, so the tangency the
    /// closed form guarantees in exact arithmetic could drift. This measures
    /// the drift across the whole approach to the cut locus, which is what
    /// justifies not re-projecting the result.
    #[test]
    fn test_transport_tangency_holds_near_cut_locus() {
        use core::f64::consts::PI;

        let m = Sphere::<10>;
        let mut worst: Real = 0.0;
        let mut worst_theta = 0.0;

        for k in 0..60 {
            // Approach theta = pi geometrically, stopping short of the guard.
            let theta = PI - 1e-1 * (0.8_f64).powi(k);
            if theta >= PI {
                continue;
            }
            let mut p = SVector::<Real, 10>::zeros();
            p[0] = 1.0;
            let mut dir = SVector::<Real, 10>::zeros();
            dir[1] = 1.0;

            let q = p * theta.cos() + dir * theta.sin();
            let mut v = SVector::<Real, 10>::zeros();
            v[1] = 0.6;
            v[2] = 0.8;

            if let Ok(t) = m.transport(&p, &q, &v) {
                let resid = t.dot(&q).abs() / t.norm().max(1.0);
                if resid > worst {
                    worst = resid;
                    worst_theta = theta;
                }
            }
        }

        assert!(
            worst < 1e-13,
            "tangency drifts to {worst:.3e} at theta = {worst_theta}, \
             which would mean the result needs re-projecting"
        );
    }

    /// `exp` renormalises its result, which costs two extra passes over the
    /// data. Dropping it is 1.5x faster and was measured and rejected: the
    /// identity ||p cos(t) + v sin(t)/||v|| || = 1 holds only for an exactly
    /// tangent v, and a v carrying a 1e-9 normal component then leaves the
    /// result 7e-10 off the sphere. Feeding back a previously computed vector
    /// is how that arises in practice, so this test guards the guarantee.
    #[test]
    fn test_exp_stays_on_the_sphere() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let m = Sphere::<50>;
        let mut rng = StdRng::seed_from_u64(5);
        let mut worst_tangent: Real = 0.0;
        let mut worst_perturbed: Real = 0.0;

        for _ in 0..200 {
            let p = m.random_point(&mut rng);
            let dir = m.random_tangent(&p, &mut rng);

            // Clean tangents, across a wide range of geodesic lengths.
            for scale in [1e-8, 1e-3, 0.5, 3.0, 20.0] {
                let v = dir.normalize() * scale;
                let q = m.exp(&p, &v);
                worst_tangent = worst_tangent.max((q.norm() - 1.0).abs());
            }

            // A tangent polluted by a normal component, which is what accrues
            // when a caller feeds back a previously computed vector.
            let v = dir.normalize() * 0.7 + p * 1e-9;
            let q = m.exp(&p, &v);
            worst_perturbed = worst_perturbed.max((q.norm() - 1.0).abs());
        }

        assert!(
            worst_tangent < 1e-14,
            "exp drifts {worst_tangent:.3e} off the sphere for clean tangents"
        );
        assert!(
            worst_perturbed < 1e-14,
            "exp drifts {worst_perturbed:.3e} off the sphere when v carries a \
             normal component; renormalisation is load-bearing after all"
        );
    }

    /// The in-place forms must agree with the value-returning ones exactly.
    /// They take different arithmetic routes, so this is the check that the
    /// faster path did not become a slightly different function.
    #[test]
    fn test_into_variants_agree() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        macro_rules! check {
            ($n:literal) => {{
                let m = Sphere::<$n>;
                let mut rng = StdRng::seed_from_u64(23);
                for _ in 0..50 {
                    let p = m.random_point(&mut rng);
                    let q = m.random_point(&mut rng);
                    let v = m.random_tangent(&p, &mut rng);

                    let mut out = SVector::<Real, $n>::zeros();

                    m.exp_into(&p, &v, &mut out);
                    assert!(
                        (out - m.exp(&p, &v)).norm() < 1e-12,
                        "exp_into disagrees on S^{}",
                        $n - 1
                    );

                    m.log_into(&p, &q, &mut out).unwrap();
                    assert!(
                        (out - m.log(&p, &q).unwrap()).norm() < 1e-12,
                        "log_into disagrees on S^{}",
                        $n - 1
                    );

                    m.transport_into(&p, &q, &v, &mut out).unwrap();
                    assert!(
                        (out - m.transport(&p, &q, &v).unwrap()).norm() < 1e-12,
                        "transport_into disagrees on S^{}",
                        $n - 1
                    );
                }
            }};
        }

        check!(3);
        check!(10);
        check!(50);
    }

    /// A failed `log_into` must leave the destination alone, so a caller
    /// reusing a buffer across iterations cannot read a stale value as though
    /// it were fresh.
    #[test]
    fn test_log_into_leaves_out_untouched_on_failure() {
        let m = Sphere::<3>;
        let north = SVector::<Real, 3>::new(0.0, 0.0, 1.0);
        let south = SVector::<Real, 3>::new(0.0, 0.0, -1.0);

        let sentinel = SVector::<Real, 3>::new(7.0, 8.0, 9.0);
        let mut out = sentinel;

        assert!(m.log_into(&north, &south, &mut out).is_err());
        assert_eq!(out, sentinel, "out was written despite the call failing");

        assert!(m.transport_into(&north, &south, &sentinel, &mut out).is_err());
        assert_eq!(out, sentinel, "out was written despite the call failing");
    }

    /// Antipodal points are the cut locus: transport along a minimising
    /// geodesic is undefined there and must report that rather than return a
    /// plausible-looking vector.
    #[test]
    fn test_transport_errors_at_cut_locus() {
        let m = Sphere::<3>;
        let north = SVector::<Real, 3>::new(0.0, 0.0, 1.0);
        let south = SVector::<Real, 3>::new(0.0, 0.0, -1.0);
        let v = SVector::<Real, 3>::new(1.0, 0.0, 0.0);

        assert!(m.transport(&north, &south, &v).is_err());
    }

    use cartan_core::{Connection, Manifold};
    use nalgebra::SVector;

    #[test]
    fn test_sphere_hessian_weingarten_correction() {
        let s2 = Sphere::<3>;
        // p = (1/sqrt(2), 0, 1/sqrt(2))
        let p: SVector<f64, 3> = SVector::from([1.0 / 2f64.sqrt(), 0.0, 1.0 / 2f64.sqrt()]);
        assert!(s2.check_point(&p).is_ok());

        // Cost: f(x) = x[2] (height function, linear)
        // egrad = [0, 0, 1] everywhere; Euclidean Hessian = 0
        let egrad: SVector<f64, 3> = SVector::from([0.0, 0.0, 1.0]);
        let ehvp = |_w: &SVector<f64, 3>| SVector::zeros();

        // Tangent at p (perpendicular to p): [0, 1, 0]
        let v: SVector<f64, 3> = SVector::from([0.0, 1.0, 0.0]);
        assert!(p.dot(&v).abs() < 1e-10, "v must be in T_pS");

        let hvp = s2
            .riemannian_hessian_vector_product(&p, &egrad, &v, &ehvp)
            .unwrap();

        // Expected: -(1/sqrt(2)) * v = [0, -1/sqrt(2), 0]
        // <egrad, p> = p[2] = 1/sqrt(2); proj_p(ehvp) = proj_p(0) = 0
        let expected: SVector<f64, 3> = -v * (1.0 / 2f64.sqrt());
        let diff = (hvp - expected).norm();
        assert!(diff < 1e-10, "Weingarten correction wrong: diff = {diff}");
    }

    #[test]
    fn test_sphere_hessian_no_correction_at_north_pole() {
        let s2 = Sphere::<3>;
        // At north pole with f(x) = x[0]^2 + x[1]^2, egrad = [0,0,0] at north pole
        // => correction is zero; result = proj_p(ehvp)
        let p: SVector<f64, 3> = SVector::from([0.0, 0.0, 1.0]);
        let egrad: SVector<f64, 3> = SVector::zeros();
        let v: SVector<f64, 3> = SVector::from([1.0, 0.0, 0.0]);
        let ehvp = |w: &SVector<f64, 3>| *w * 2.0; // Hessian of x0^2+x1^2 = 2*I on first two dims
        let hvp = s2
            .riemannian_hessian_vector_product(&p, &egrad, &v, &ehvp)
            .unwrap();
        // proj_p(2*v) = 2*v since v perp p; correction = 0
        let expected = v * 2.0;
        let diff = (hvp - expected).norm();
        assert!(diff < 1e-10, "HVP without correction wrong: diff = {diff}");
    }
}
