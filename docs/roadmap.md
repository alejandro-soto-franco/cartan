# cartan Roadmap

Internal developer reference. Not committed to the repo (see .gitignore).

---

## v0.1 Status

### Done

| Component | Notes |
|-----------|-------|
| cartan-core traits | All 7 traits: Manifold, Retraction, ParallelTransport, VectorTransport, Connection, Curvature, GeodesicInterpolation |
| CartanError | 6 variants |
| cartan-nalgebra backend | SVector/SMatrix storage |
| Euclidean<N> | All 7 traits. Trivial baseline. |
| Sphere<N> | All 7 traits. exp/log, transport, Cayley retraction, curvature K=1. Weingarten correction in Connection. |
| SpecialOrthogonal<N> | All 7 traits. matrix_exp/log, Cayley retraction (formal Retraction impl), Lie bracket curvature. Weingarten correction in Connection. |
| SpecialEuclidean<N> | All 7 traits. left_jacobian, left_jacobian_inverse, SE(N) exp/log. Weingarten correction (SO block) in Connection. |
| SymmetricPositiveDefinite<N> | All 7 traits. Affine-invariant metric, K <= 0 (Cartan-Hadamard). Full Riemannian HVP via Christoffel symbols (no Weingarten needed). |
| Grassmann<N,K> | All 7 traits. Gr(N,K), principal angles, 0 <= K <= N/2. Weingarten correction (Boumal 2022 horizontal rep.) in Connection. |
| Corr<N> | All 7 traits. Correlation matrices, flat Frobenius metric, K = 0. |
| cartan-optim | RGD, RCG, RTR, Frechet mean on any Manifold. |
| cartan-geo | Geodesic curves, Jacobi fields, curvature queries. |
| cartan-dec core operators | Mesh<M,K,B>, FlatMesh alias, ExteriorDerivative (d0, d1), HodgeStar (star0, star1, star2), Operators (Laplace-Beltrami, Bochner, Lichnerowicz), advection, divergence. |
| Integration test harness | manifold_harness.rs, matrix_harness.rs in cartan/tests/common/ |
| CI | .github/workflows/ci.yml, rust.yml |

### Planned for v0.1 (remaining)

| Component | Notes |
|-----------|-------|
| Stiefel<N,K> | St(N,K). |
| Hyperbolic<N> | H^N. K = -1. Needed for Fisher-Rao agent in malliavin. |
| Simplex<N> | Probability simplex. |
| Product<M1, M2> | Product manifold. Useful for Fisher-Rao of multivariate Gaussians. |
| cargo publish v0.1 | See publishing.md for checklist. |

### cartan-dec Remaining Work

The core discrete operators are implemented. Remaining items for a production-grade DEC layer:

Phase 1 (mesh generation):
- Bowyer-Watson constrained Delaunay triangulation (currently users supply the mesh)
- Hilbert curve simplex reordering for cache locality

Phase 2 (persistent homology, independent of DEC):
- Vietoris-Rips filtration from point cloud
- Boundary matrices for k = 0, 1, 2 (over Z/2Z)
- Betti numbers and persistence diagram

Phase 3 (parallelism):
- Graph coloring (typically 4-7 colors in 2D Delaunay meshes)
- rayon::par_iter over color classes in time loops

Phase 4 (non-flat operator assembly):
- HodgeStar for generic Mesh<M, 3, 2> (currently flat-only)
- Operators::from_mesh for non-Euclidean manifolds

---

## Version Policy

- v0.x: No stability guarantees. Trait signatures may change.
- v1.0: Stable API. No breaking changes without a major version bump.
- All crates in the workspace are versioned together (same version in workspace.package).
- crates.io publishes are gated on: all tests passing, README accurate, no broken doc links.

---

## Known Gaps / Technical Debt

- cartan-docs site is scaffolded (Next.js app exists) but has no content pages yet.
  The design spec and implementation plan live in ~/cartan-docs/docs/superpowers/.
  Content pages need to be written as MDX files.

- The manifolds table in the README and cartan-docs landing page needs to be updated
  each time a new manifold is implemented. Do this as part of the implementation commit.

- HodgeStar and Operators::from_mesh are currently pinned to FlatMesh / Euclidean<2>.
  The generic Mesh<M,K,B> is available for topology queries and exterior derivative,
  but Hodge weights require flat metric methods. Extending to curved manifolds requires
  geodesic circumcenter and dual volume computations (see circumcenter() in mesh.rs for
  the tangent-space approximation already implemented).

- docs.rs metadata: add [package.metadata.docs.rs] to each Cargo.toml before publishing.
  See publishing.md.
