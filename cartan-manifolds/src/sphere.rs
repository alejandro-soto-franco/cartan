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
            // Re-normalize for numerical safety.
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
            // Standard formula: theta * (q - cos(theta) * p) / sin(theta).
            let sin_theta = theta.sin();
            let v = (q - p * cos_theta) * (theta / sin_theta);
            Ok(v)
        }
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
        let log_pq = self.log(p, q)?;
        let dist_sq = self.inner(p, &log_pq, &log_pq);

        if dist_sq < ANGLE_EPS * ANGLE_EPS {
            // Points are very close: transport is approximately identity.
            // Re-project to ensure result is in T_qS.
            return Ok(self.project_tangent(q, v));
        }

        let log_qp = self.log(q, p)?;
        let coeff = self.inner(p, &log_pq, v) / dist_sq;
        let transported = v - (log_pq + log_qp) * coeff;

        // Re-project for numerical safety.
        Ok(self.project_tangent(q, &transported))
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
