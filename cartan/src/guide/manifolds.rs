//! # The manifold catalogue
//!
//! Dimensions are const generic, so a shape mismatch is a compile error rather
//! than a runtime panic. `Sphere::<3>` is the 2-sphere in R^3: the parameter is
//! the ambient dimension, and the intrinsic dimension follows from it.
//!
//! | manifold | type | intrinsic dim | curvature | tier |
//! |---|---|---|---|---|
//! | Euclidean R^N | `Euclidean<N>` | N | flat | alloc |
//! | Sphere S^(N-1) | `Sphere<N>` | N-1 | K = 1 | alloc |
//! | Special orthogonal | `SpecialOrthogonal<N>` | N(N-1)/2 | K >= 0 | alloc |
//! | Special Euclidean | `SpecialEuclidean<N>` | N(N+1)/2 | flat x sphere | alloc |
//! | SPD, affine-invariant | `Spd<N>` | N(N+1)/2 | K <= 0 | std |
//! | SPD, Bures-Wasserstein | `SpdBuresWasserstein<N>` | N(N+1)/2 | K >= 0 | std |
//! | Grassmann Gr(N, K) | `Grassmann<N, K>` | K(N-K) | 0 <= K <= 2 | alloc |
//! | Correlation | `Corr<N>` | N(N-1)/2 | flat | std |
//! | Q-tensor Sym_0(R^3) | `QTensor3` | 5 | flat | std |
//!
//! The tier column matters if you are building for a target without std. The
//! std-only entries need an eigen decomposition, whose Jacobi iteration is not
//! available in `core`. See the crate-level docs for the feature ladder.
//!
//! ## Choosing between the two SPD metrics
//!
//! Both put a geometry on symmetric positive-definite matrices, and they
//! disagree about what a straight line is. The affine-invariant metric is
//! negatively curved and sends the boundary of the cone to infinite distance,
//! so a geodesic can never reach a singular matrix. Bures-Wasserstein is
//! positively curved and the boundary sits at finite distance.
//!
//! Pick affine-invariant when degeneracy should be unreachable, such as
//! covariance interpolation that must stay invertible. Pick Bures-Wasserstein
//! when the quantity is an optimal-transport distance between Gaussians.
//!
//! ```
//! use cartan::prelude::*;
//! use cartan::manifolds::Spd;
//! use nalgebra::SMatrix;
//!
//! let spd = Spd::<2>;
//!
//! let a = SMatrix::<f64, 2, 2>::new(2.0, 0.0,
//!                                   0.0, 1.0);
//! let b = SMatrix::<f64, 2, 2>::new(1.0, 0.0,
//!                                   0.0, 4.0);
//!
//! // The geodesic midpoint is the affine-invariant mean, not the arithmetic one.
//! let v = spd.log(&a, &b).unwrap();
//! let mid = spd.exp(&a, &(v * 0.5));
//!
//! // It is equidistant from both endpoints.
//! let d_a = spd.dist(&a, &mid).unwrap();
//! let d_b = spd.dist(&mid, &b).unwrap();
//! assert!((d_a - d_b).abs() < 1e-9);
//!
//! // And it stays positive definite, as every point of the cone must.
//! assert!(spd.check_point(&mid).is_ok());
//! ```
//!
//! ## Rotations
//!
//! `SpecialOrthogonal<N>` carries the bi-invariant metric, so `exp` is the
//! matrix exponential of a skew-symmetric matrix. For N = 2 and N = 3 it uses
//! Rodrigues' formula, which is exact rather than a truncated series.
//!
//! ```
//! use cartan::prelude::*;
//! use cartan::manifolds::SpecialOrthogonal;
//!
//! let so3 = SpecialOrthogonal::<3>;
//! let mut rng = rand::rng();
//!
//! let r = so3.random_point(&mut rng);
//! let w = so3.random_tangent(&r, &mut rng);
//! let r2 = so3.exp(&r, &w);
//!
//! // The result is still a rotation: orthogonal with unit determinant.
//! assert!(so3.check_point(&r2).is_ok());
//! assert!((r2.determinant() - 1.0).abs() < 1e-10);
//! ```
