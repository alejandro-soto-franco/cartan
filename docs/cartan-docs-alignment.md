# cartan-docs Alignment Guide

How the cartan Rust source maps to the cartan.sotofranco.dev documentation site.

Internal developer reference. Not committed to the repo (see .gitignore).

---

## Repositories

| Concern | Repo | Local path |
|---------|------|------------|
| Rust source | github.com/alejandro-soto-franco/cartan | ~/cartan |
| Docs site | github.com/alejandro-soto-franco/cartan-docs | ~/cartan-docs |
| Deployed | cartan.sotofranco.dev | Vercel |
| Released rustdoc | docs.rs/cartan | automatic on cargo publish |
| Bleeding-edge rustdoc | cartan.sotofranco.dev/rustdoc/ | fetched at cartan-docs build time |

The cartan repo is pure Rust. It never contains MDX, CSS, or JS.
The cartan-docs repo is the standalone Next.js site. It fetches rustdoc at build time.

---

## Rustdoc Pipeline

```
cartan repo: push to main
  -> CI: cargo doc --no-deps --workspace
  -> Publish rustdoc HTML to gh-pages branch of cartan repo
  -> Fire Vercel deploy hook

cartan-docs: Vercel build triggered
  -> build script: fetch rustdoc tarball from cartan gh-pages branch archive URL
  -> unpack into public/rustdoc/
  -> next build
  -> deploy to cartan.sotofranco.dev
```

The rustdoc HTML lives on gh-pages in the cartan repo, NOT in cartan-docs git history.
The cartan-docs build script fetches it at build time via GitHub archive URL.
No Rust toolchain or cross-repo push tokens are needed in Vercel.

CI workflow to add (in cartan repo, .github/workflows/docs.yml):

```yaml
name: Rustdoc
on:
  push:
    branches: [main]
jobs:
  rustdoc:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo doc --no-deps --workspace
      - uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./target/doc
      - run: curl -X POST ${{ secrets.VERCEL_DEPLOY_HOOK }}
```

---

## Source-to-Page Mapping

For each cartan-docs content page, the corresponding cartan source:

### /getting-started
Source: README.md (Quick Start section), cartan/src/lib.rs (prelude module)
Content needed: installation (Cargo.toml snippet), first example, prelude imports

Key API note: the cartan-dec quick start uses `FlatMesh` and the two-argument form:
```rust
use cartan_dec::{FlatMesh, Operators};
use cartan_manifolds::euclidean::Euclidean;

let mesh = FlatMesh::unit_square_grid(32);
let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
```

### /concepts/manifold
Source: cartan-core/src/manifold.rs
Key items: Manifold trait definition, all 16 methods with doc comments,
           design decisions (extrinsic coords, exp total / log Result)
References: do Carmo Ch. 3, Absil Ch. 3

### /concepts/retraction
Source: cartan-core/src/retraction.rs
Key items: Retraction trait, why cheaper than exp, default delegation pattern
References: Absil Ch. 4

### /concepts/transport
Source: cartan-core/src/transport.rs
Key items: ParallelTransport, VectorTransport, blanket impl, holonomy demo setup

### /concepts/connection
Source: cartan-core/src/connection.rs
Key items: Connection trait, riemannian_hessian_vector_product, Levi-Civita connection,
           Weingarten correction convention (grad_f = Euclidean gradient, matching Pymanopt)

### /concepts/curvature
Source: cartan-core/src/curvature.rs
Key items: Curvature trait, all 4 methods, curvature table for all manifolds,
           sectional/Ricci/scalar definitions, K>0/K=0/K<0 geometry

### /concepts/geodesic
Source: cartan-core/src/geodesic.rs
Key items: GeodesicInterpolation trait, geodesic(p, q, t)

### /concepts/error-handling
Source: cartan-core/src/error.rs
Key items: CartanError variants, when each arises, how to handle CutLocus

### /manifolds/euclidean
Source: cartan-manifolds/src/euclidean.rs
Key items: trivial geometry, flat metric, all formulas, K=0

### /manifolds/sphere
Source: cartan-manifolds/src/sphere.rs
Key items: exp (great circle), log (inverse great circle), transport (Schild's ladder),
           K=1, injectivity radius pi, cut locus = antipodal point,
           Weingarten correction: Hess f(p)[v] = proj_p(D2f(p)[v]) - <egrad, p> * v

### /manifolds/special-orthogonal
Source: cartan-manifolds/src/so.rs, cartan-manifolds/src/util/
Key items: SO(N) exp (matrix exponential), log (matrix logarithm),
           Cayley retraction (formal Retraction trait impl), bi-invariant metric,
           Lie bracket curvature formula K(X,Y) = (1/4)||[X,Y]||^2 / (||X||^2 ||Y||^2 - <X,Y>^2),
           Weingarten correction: R * skew(R^T * ehvp) - 0.5 * R * sym(R^T * egrad) * R^T * V

### /manifolds/special-euclidean
Source: cartan-manifolds/src/se.rs
Key items: SE(N) as semidirect product SO(N) x R^N,
           left_jacobian, left_jacobian_inverse, exp/log roundtrip,
           Weingarten correction: SO block uses SO formula, translation block is flat (zero correction)

### /manifolds/spd
Source: cartan-manifolds/src/spd.rs
Key items: affine-invariant metric, K<=0 (Cartan-Hadamard), injectivity radius infinity,
           full Riemannian HVP via Christoffel symbols (P * ehvp * P + 0.5*(V*P^-1*G + G*P^-1*V)),
           guard: re-symmetrize and clip eigenvalues in log/project_point

### /manifolds/grassmann
Source: cartan-manifolds/src/grassmann.rs
Key items: Gr(N,K), principal angles, horizontal representation T_[Q] Gr(N,K) = {V: Q^T V = 0},
           Weingarten correction (Boumal 2022 Prop 9.46): Hess f(Q)[V] = proj_Q(ehvp) - V * sym(Q^T * G),
           dist via principal angles, exp/log via SVD-based geodesic formula

### /manifolds/corr
Source: cartan-manifolds/src/corr.rs
Key items: correlation matrices (symmetric, unit diagonal, PD), flat Frobenius metric,
           K=0, projection via nearest-correlation algorithm (Higham alternating projections),
           parallel transport = identity (flat metric)

### /demos/*
Sources: cartan/tests/ (for correctness reference), cartan-manifolds/src/sphere.rs
All demos are Three.js / D3 -- no Rust source maps directly to demos.
Demos are implemented in the cartan-docs repo.

---

## What to Update When Adding a Manifold

When a new manifold (e.g., Hyperbolic<N>) is implemented in cartan-manifolds:

1. cartan repo:
   - Add implementation in cartan-manifolds/src/hyperbolic.rs
   - Add pub mod + pub use in cartan-manifolds/src/lib.rs
   - Add integration tests in cartan/tests/test_hyperbolic.rs
   - Update README.md manifolds table (status: done)
   - Update docs/roadmap.md

2. cartan-docs repo:
   - Write content page at app/manifolds/hyperbolic/page.mdx
   - Add to left sidebar navigation in app/layout.tsx (remove "coming soon")
   - Update landing page ManifoldCard grid
   - Update /concepts/curvature page with hyperbolic curvature values

---

## MDX Frontmatter Schema

Every content page in cartan-docs uses this frontmatter:

```yaml
---
title: "Manifold Trait"
description: "The foundational trait for Riemannian geometry in cartan."
section: "concepts"
order: 1
rustdoc: "/rustdoc/cartan_core/trait.Manifold.html"
references:
  - "do Carmo, Riemannian Geometry, Ch. 3"
  - "Absil, Optimization Algorithms on Matrix Manifolds, Ch. 3"
---
```

The `rustdoc` field is the path under /rustdoc/ to the relevant rustdoc page.
The `references` array populates the margin notes at the bottom of each page.

---

## Keeping Docs in Sync

The biggest drift risk is the manifold formula tables. When fixing a bug in
exp/log/transport for a manifold, verify the corresponding formula in the cartan-docs
manifold page matches the implementation.

For the rustdoc at cartan.sotofranco.dev/rustdoc/, the pipeline is automatic
(CI builds on every push to cartan main). No manual sync needed for rustdoc.
