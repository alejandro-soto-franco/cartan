# Contributing: Implementing a New Manifold

Internal developer reference. Not committed to the repo (see .gitignore).

---

## Checklist

- [ ] Create `cartan-manifolds/src/<name>.rs`
- [ ] Implement `Manifold` (required)
- [ ] Implement `Retraction` (if a cheaper retraction exists)
- [ ] Implement `ParallelTransport` (required for cartan-dec compatibility)
- [ ] Implement `Connection` (required for cartan-dec compatibility)
- [ ] Implement `Curvature` (required for cartan-dec LichnerowiczLaplacian)
- [ ] Implement `GeodesicInterpolation`
- [ ] Add `pub mod <name>; pub use <name>::<Type>;` in cartan-manifolds/src/lib.rs
- [ ] Add integration tests in `cartan/tests/test_<name>.rs`
- [ ] Run the manifold harness
- [ ] Update README.md manifolds table
- [ ] Update docs/roadmap.md
- [ ] Update cartan-docs alignment: docs/cartan-docs-alignment.md

---

## File Template

```rust
// cartan-manifolds/src/mymanifold.rs

//! MyManifold -- one-line description.
//!
//! Mathematical definition, key properties, curvature values.
//! References: [Author, "Title", Year, Chapter N].

use nalgebra::{SMatrix, SVector};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use cartan_core::{CartanError, Connection, Curvature, GeodesicInterpolation,
                  Manifold, ParallelTransport, Real, Retraction};

/// MyManifold<N>.
///
/// Heavy doc comment explaining:
/// - What the manifold is geometrically
/// - Point and Tangent types
/// - Metric definition
/// - Key curvature values
/// - Injectivity radius
/// - Cut locus description
#[derive(Clone, Copy, Debug)]
pub struct MyManifold<const N: usize>;

pub type MyPoint<const N: usize> = SVector<Real, N>;
pub type MyTangent<const N: usize> = SVector<Real, N>;

impl<const N: usize> Manifold for MyManifold<N> {
    type Point = MyPoint<N>;
    type Tangent = MyTangent<N>;

    fn dim(&self) -> usize { ... }
    fn ambient_dim(&self) -> usize { ... }
    fn injectivity_radius(&self, _p: &Self::Point) -> Real { ... }
    fn inner(&self, _p: &Self::Point, u: &Self::Tangent, v: &Self::Tangent) -> Real { ... }
    fn exp(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Point { ... }
    fn log(&self, p: &Self::Point, q: &Self::Point) -> Result<Self::Tangent, CartanError> { ... }
    fn project_tangent(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Tangent { ... }
    fn project_point(&self, p: &Self::Point) -> Self::Point { ... }
    fn zero_tangent(&self, p: &Self::Point) -> Self::Tangent { ... }
    fn check_point(&self, p: &Self::Point) -> Result<(), CartanError> { ... }
    fn check_tangent(&self, p: &Self::Point, v: &Self::Tangent) -> Result<(), CartanError> { ... }
    fn random_point<R: Rng>(&self, rng: &mut R) -> Self::Point { ... }
    fn random_tangent<R: Rng>(&self, p: &Self::Point, rng: &mut R) -> Self::Tangent { ... }
}
```

---

## Manifold Harness

The test harness in `cartan/tests/common/manifold_harness.rs` runs a standard
battery of property tests on any `Manifold` impl. To use it:

```rust
// cartan/tests/test_mymanifold.rs

mod common;
use common::manifold_harness::run_manifold_harness;
use cartan::manifolds::MyManifold;

#[test]
fn mymanifold_harness() {
    run_manifold_harness(MyManifold::<3>, 1000);
}
```

The harness checks:
- exp/log roundtrip: for random p and small v, `log(p, exp(p, v)) ~= v`
- log/exp roundtrip: for nearby p and q, `exp(p, log(p, q)) ~= q`
- project_tangent idempotency
- check_point and check_tangent pass on randomly sampled points/tangents
- dist symmetry: `dist(p, q) ~= dist(q, p)`
- zero tangent has zero norm

For matrix manifolds (SO(N), SE(N) style), also run the matrix harness:

```rust
use common::matrix_harness::run_matrix_harness;

#[test]
fn mymanifold_matrix_harness() {
    run_matrix_harness(MyManifold::<3>, 100);
}
```

The matrix harness additionally checks:
- Parallel transport preserves inner product
- Retraction satisfies first and second order conditions

---

## Documentation Standards

Every public item must have a doc comment. The doc comment must explain:
- What the item is
- Mathematical definition or formula (using ASCII math or LaTeX in backtick blocks)
- Any invariants or preconditions
- What errors can be returned and why
- A reference (author, title, chapter/equation number) for non-trivial formulas

Example (from Sphere::exp):

```rust
/// Exponential map on S^{N-1}: Exp_p(v) = cos(||v||) p + sin(||v||) (v / ||v||).
///
/// Follows the great circle gamma with gamma(0) = p, gamma'(0) = v.
/// When ||v|| = 0, returns p (identity).
/// When ||v|| = pi, returns the antipodal point -p (cut locus boundary).
///
/// Ref: do Carmo, "Riemannian Geometry", Ch. 3, Example 2.
```

---

## Numerical Guidelines

- Use Taylor expansions near singularities. For Sphere::log, the formula
  `theta / sin(theta)` diverges as theta -> 0. Use the expansion
  `1 + theta^2/6 + 7*theta^4/360 + ...` for |theta| < 1e-6.
- Use the `approx` crate for all floating-point comparisons in tests.
  Set tolerances explicitly: `assert_relative_eq!(a, b, epsilon = 1e-10)`.
- Do not use `unwrap()` in library code. All errors propagate via CartanError.
- The `Real` type alias is `f64`. Do not use `f64` directly anywhere in
  cartan-core or cartan-manifolds. Always use `Real` so that a future f32
  switch is a single-line change.

---

## Curvature for cartan-dec Compatibility

cartan-dec uses the `Curvature` trait to compute the Lichnerowicz Laplacian
correction. The correction is:

```
(LichnerowiczLaplacian T)_ij =
    (BochnerLaplacian T)_ij
    - R_ik T^k_j
    - R_jk T^k_i
    + 2 R_ikjl T^kl
```

This requires `riemann_curvature(p, u, v, w)` to be implemented correctly
and efficiently. For constant-curvature manifolds (sphere, hyperbolic), the
curvature tensor has a closed-form expression:

```
R(u,v)w = K * (<v,w> u - <u,w> v)
```

Override `riemann_curvature` with this closed form. Do not use the default
implementation (which requires a basis and O(n) evaluations).
