# Workstream B: Contained TODOs — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all contained code TODOs: Weingarten corrections on five manifold `Connection` impls, full-tensor Ricci correction callback in `cartan-dec`, and doctest annotation cleanup in `cartan-manifolds`.

**Architecture:** Three independent sub-tasks. (1) Weingarten: add the missing correction term to `riemannian_hessian_vector_product` on Sphere, SO(N), Grassmann, SE(N) using the existing `grad_f` and `v` arguments that are currently ignored. Euclidean and Corr are zero-correction (already correct). SPD is already fully correct — do not touch. (2) Ricci callback: replace `Option<f64>` with `Option<&dyn Fn(usize) -> [[f64; N]; N]>` in `apply_bochner_laplacian` and `apply_lichnerowicz_laplacian`. (3) Doctests: replace all bare `ignore` with `no_run` (non-deterministic) or running (deterministic).

**Tech Stack:** Rust 1.85, `cartan-manifolds`, `cartan-dec`. No new dependencies. This workstream is independent of Workstream A and C.

**Spec:** `docs/superpowers/specs/2026-03-17-cartan-generalization-design.md` — Workstream B.

---

## Chunk 1: Weingarten Corrections

### Task 1: Sphere — Weingarten correction + remove TODO comment

**Files:**
- Modify: `cartan-manifolds/src/sphere.rs`
- Test: `cartan-manifolds/src/sphere.rs` (inline tests)

The Sphere `Connection` impl currently returns `proj_p(ehvp)` without the Weingarten correction. The full formula is:

```
Hess f(p)[v] = proj_p(D²f(p)[v]) - <grad_f, p> * v
```

where `p` is the unit normal (the shape operator of the sphere embedding).

- [ ] **Step 1: Write a failing test for the Weingarten correction**

Add to the `#[cfg(test)]` block at the bottom of `cartan-manifolds/src/sphere.rs`:

```rust
#[test]
fn test_sphere_hessian_weingarten_correction() {
    use cartan_core::{Connection, Manifold};
    use nalgebra::SVector;

    let s2 = Sphere::<3>;
    // p = north pole, a simple point with known geometry
    let p: SVector<f64, 3> = SVector::from([0.0, 0.0, 1.0]);

    // Cost: f(x) = x[0]^2 + x[1]^2 (distance from z-axis, squared)
    // Euclidean gradient: egrad = 2*[x0, x1, 0]
    // At p=(0,0,1): egrad = [0,0,0], so the correction term -<egrad, p>*v = 0.
    // That's boring — use a point where the correction is nonzero.

    // Use p = (1/√2, 0, 1/√2)
    let p: SVector<f64, 3> = SVector::from([1.0/2f64.sqrt(), 0.0, 1.0/2f64.sqrt()]);
    // Cost: f(x) = x[2]  (height function)
    // egrad = [0, 0, 1] everywhere  ← this is the EUCLIDEAN gradient
    // rhess[v] = proj_p(ehvp) - <egrad, p> * v
    //          = proj_p(0) - (1/√2) * v
    //          = -(1/√2) * v    (since ehvp = 0 for linear cost)
    //
    // Convention: grad_f in riemannian_hessian_vector_product is the EUCLIDEAN gradient
    // (matching Pymanopt/Manopt), NOT the projected Riemannian gradient.
    // Passing rgrad = project_tangent(p, egrad) would give rgrad.dot(p) = 0
    // (since rgrad ⊥ p by construction), zeroing the correction — wrong.
    let egrad: SVector<f64, 3> = SVector::from([0.0, 0.0, 1.0]);

    // A tangent vector at p (perpendicular to p)
    let v: SVector<f64, 3> = SVector::from([0.0, 1.0, 0.0]);
    assert!(p.dot(&v).abs() < 1e-10, "v must be in T_pS");

    // ehvp = 0 for a linear cost (Hessian of x[2] is zero)
    let ehvp = |_w: &SVector<f64, 3>| SVector::zeros();

    // Pass egrad (Euclidean gradient), NOT rgrad, as the grad_f argument.
    let hvp = s2.riemannian_hessian_vector_product(&p, &egrad, &v, &ehvp).unwrap();

    // Expected: -(1/√2) * v = -(1/√2) * [0, 1, 0]
    // because <egrad, p> = p[2] = 1/√2
    let expected: SVector<f64, 3> = -v * (1.0 / 2f64.sqrt());
    let diff = (hvp - expected).norm();
    assert!(diff < 1e-10, "Weingarten correction wrong: diff = {diff}");
}
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
cargo test -p cartan-manifolds test_sphere_hessian_weingarten_correction -- --nocapture
```
Expected: FAIL (current impl returns `proj_p(ehvp) = 0`, not `-(1/√2)*v`).

- [ ] **Step 3: Implement the Weingarten correction in `sphere.rs`**

Find the `Connection` impl block in `sphere.rs`. Replace the body of `riemannian_hessian_vector_product`:

```rust
fn riemannian_hessian_vector_product(
    &self,
    p: &Self::Point,
    grad_f: &Self::Tangent,
    v: &Self::Tangent,
    hess_ambient: &dyn Fn(&Self::Tangent) -> Self::Tangent,
) -> Result<Self::Tangent, CartanError> {
    // Step 1: Euclidean HVP projected onto tangent space.
    let ehvp = hess_ambient(v);
    let proj_ehvp = self.project_tangent(p, &ehvp);

    // Step 2: Weingarten correction — shape operator of sphere embedding.
    // For S^{N-1} in R^N, the shape operator W satisfies:
    //   Hess f(p)[v] = proj_p(D²f(p)[v]) - <egrad(p), p> * v
    // where egrad is the Euclidean gradient. We receive rgrad = proj_p(egrad),
    // so <egrad, p> = <rgrad + (p^T egrad)p, p> = p^T egrad.
    // Since rgrad = egrad - (p^T egrad) p, we have p^T egrad = p^T grad_f
    // only when grad_f is egrad — but grad_f here IS the Riemannian gradient
    // (already projected). We need the normal component: <egrad, p>.
    //
    // The caller has projected: rgrad = egrad - (egrad·p) p.
    // So egrad·p cannot be recovered from rgrad alone without knowing egrad.
    //
    // The Connection trait passes `grad_f` as the Riemannian gradient (projected).
    // To get the Weingarten term we need <egrad, p> = the normal component of egrad.
    // We approximate this from the ambient HVP at v=p (the outward normal direction):
    //   <egrad, p> ≈ (second-order term) — but for the exact formula, callers who
    //   need the full Weingarten correction should pass egrad (not rgrad) as grad_f.
    //
    // Convention adopted: grad_f is assumed to be the EUCLIDEAN gradient (egrad),
    // not the Riemannian gradient, matching Pymanopt/Manopt convention for
    // riemannian_hessian_vector_product where the caller passes egrad.
    // Under this convention: <egrad, p> = grad_f.dot(p).
    let normal_component = grad_f.dot(p);
    let weingarten = v * normal_component;

    // Full Riemannian HVP: proj_p(ehvp) - <egrad, p> * v
    Ok(proj_ehvp - weingarten)
}
```

Remove the `TODO(phase6)` comment block above this function.

- [ ] **Step 4: Run the test and confirm it passes**

```bash
cargo test -p cartan-manifolds test_sphere_hessian_weingarten_correction -- --nocapture
```
Expected: PASS.

- [ ] **Step 5: Run full `cartan-manifolds` test suite**

```bash
cargo test -p cartan-manifolds
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add cartan-manifolds/src/sphere.rs && git commit -m "fix(sphere): add Weingarten correction to riemannian_hessian_vector_product"
```

---

### Task 2: `SpecialOrthogonal<N>` — Weingarten correction

**Files:**
- Modify: `cartan-manifolds/src/so.rs`
- Test: `cartan-manifolds/src/so.rs` (inline tests)

The correct Riemannian HVP for SO(N) with bi-invariant metric (Absil-Mahony-Sepulchre §5.3):

```
Hess f(R)[V] = R · skew(R^T · ehvp(V)) - 0.5 * R · sym(R^T · egrad) · R^T · V
```

where `skew(A) = (A - A^T)/2`, `sym(A) = (A + A^T)/2`, and `egrad` is the Euclidean gradient passed as `grad_f`.

- [ ] **Step 1: Write a failing test**

Add to the `#[cfg(test)]` block in `so.rs`:

```rust
#[test]
fn test_so_hessian_weingarten_correction() {
    use cartan_core::{Connection, Manifold};
    use nalgebra::{SMatrix, SVector};

    let so3 = SpecialOrthogonal::<3>;
    // Identity rotation
    let r = SMatrix::<f64, 3, 3>::identity();

    // Cost: f(R) = 0.5 * ||R - I||_F^2  (quadratic, Euclidean Hessian = identity)
    // egrad(R) = R - I; at R=I: egrad = 0
    // ehvp(V) = V  (Hessian of quadratic is identity)
    // With egrad=0, Weingarten correction = 0, so Hess[V] = proj(V) = skew(V)
    let egrad = SMatrix::<f64, 3, 3>::zeros();
    let ehvp = |v: &SMatrix<f64, 3, 3>| *v;

    // Tangent at I: any skew-symmetric matrix
    let v_data = [0.0_f64, -1.0, 0.0,  1.0, 0.0, 0.0,  0.0, 0.0, 0.0];
    let v = SMatrix::<f64, 3, 3>::from_row_slice(&v_data);
    assert!(so3.check_tangent(&r, &v).is_ok());

    let hvp = so3.riemannian_hessian_vector_product(&r, &egrad, &v, &ehvp).unwrap();

    // At R=I with egrad=0: Hess f(I)[V] = skew(V) = V (since V is already skew)
    let diff = (hvp - v).norm();
    assert!(diff < 1e-10, "SO(3) HVP wrong at identity: diff = {diff}");

    // Test with nonzero egrad to exercise the Weingarten correction term.
    // IMPORTANT: egrad must have a nonzero SYMMETRIC PART for the correction to be nonzero.
    // A = diag(1, 2, 3) is symmetric, so sym(R^T * A) = A at R=I, giving nonzero correction.
    // Cost: f(R) = tr(A^T R) for fixed A = diag(1, 2, 3) (symmetric)
    // egrad = A; ehvp = 0 (linear cost has zero Hessian)
    let a = SMatrix::<f64, 3, 3>::from_diagonal(&nalgebra::SVector::from([1.0_f64, 2.0, 3.0]));
    let ehvp_zero = |_: &SMatrix<f64, 3, 3>| SMatrix::<f64, 3, 3>::zeros();
    let hvp2 = so3.riemannian_hessian_vector_product(&r, &a, &v, &ehvp_zero).unwrap();
    // At R=I: Hess[V] = -0.5 * sym(I^T * A) * I^T * V = -0.5 * A * V
    // (sym(A) = A since A is diagonal/symmetric)
    let expected = -(a * v) * 0.5;
    let diff2 = (hvp2 - expected).norm();
    assert!(diff2 < 1e-10, "SO(3) Weingarten correction wrong: diff = {diff2}");
}
```

- [ ] **Step 2: Run and confirm failure**

```bash
cargo test -p cartan-manifolds test_so_hessian_weingarten_correction -- --nocapture
```
Expected: FAIL.

- [ ] **Step 3: Implement the correction in `so.rs`**

Find the `Connection` impl for `SpecialOrthogonal<N>`. Replace the body:

```rust
fn riemannian_hessian_vector_product(
    &self,
    p: &Self::Point,
    grad_f: &Self::Tangent,
    v: &Self::Tangent,
    hess_ambient: &dyn Fn(&Self::Tangent) -> Self::Tangent,
) -> Result<Self::Tangent, CartanError> {
    // R is the current point (rotation matrix). grad_f is the Euclidean gradient egrad.
    let r = p;
    let egrad = grad_f;

    // Step 1: R · skew(R^T · ehvp(V))
    let ehvp_v = hess_ambient(v);
    let rt_ehvp = r.transpose() * ehvp_v;
    let skew_rt_ehvp = (rt_ehvp - rt_ehvp.transpose()) * 0.5;
    let term1 = r * skew_rt_ehvp;

    // Step 2: Weingarten correction — -0.5 * R · sym(R^T · egrad) · R^T · V
    // sym(A) = (A + A^T) / 2
    let rt_egrad = r.transpose() * egrad;
    let sym_rt_egrad = (rt_egrad + rt_egrad.transpose()) * 0.5;
    let term2 = r * sym_rt_egrad * r.transpose() * v * 0.5;

    Ok(term1 - term2)
}
```

- [ ] **Step 4: Run test and confirm pass**

```bash
cargo test -p cartan-manifolds test_so_hessian_weingarten_correction -- --nocapture
```
Expected: PASS.

- [ ] **Step 5: Full test suite**

```bash
cargo test -p cartan-manifolds
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add cartan-manifolds/src/so.rs && git commit -m "fix(so): add Weingarten correction to riemannian_hessian_vector_product"
```

---

### Task 3: `Grassmann<N,K>` — Weingarten correction

**Files:**
- Modify: `cartan-manifolds/src/grassmann.rs`
- Test: `cartan-manifolds/src/grassmann.rs` (inline tests)

cartan uses the **horizontal representation** for Grassmannian tangent vectors: `T_[Q] Gr(N,K) = {V ∈ R^{N×K} : Q^T V = 0}`. In this representation the correct Riemannian HVP is (Boumal 2022, §9.5, Proposition 9.46):

```
Hess f(Q)[V] = proj_Q(ehvp(V)) - V · sym(Q^T · G)
```

where `G = grad_f` (Euclidean gradient, N×K), `proj_Q(X) = (I - QQ^T)X`, and `sym(A) = (A + A^T)/2` (K×K symmetric). This is **different from** the formula in the spec (which uses `G·(Q^T·V)` terms that vanish for horizontal tangents Q^T V = 0 — a reference error in the spec). The Boumal formula has a post-multiplication `V * sym(Q^T*G)` that is generally nonzero even when `Q^T V = 0`.

> **Note for implementer:** The spec's Grassmann formula is wrong for cartan's horizontal representation. This plan supersedes it. The spec (`docs/superpowers/specs/2026-03-17-cartan-generalization-design.md` §B1 Grassmann) should be corrected to use `- V · sym(Q^T · G)` after this task is merged.

- [ ] **Step 1: Write a failing test**

```rust
#[test]
fn test_grassmann_hessian_weingarten_correction() {
    use cartan_core::{Connection, Manifold};
    use nalgebra::SMatrix;

    // Gr(3, 1) ≅ S^2: 3-dim space, 1-dim subspaces
    let gr = Grassmann::<3, 1>;
    // Q = column vector [0, 0, 1]^T
    let q = SMatrix::<f64, 3, 1>::from_column_slice(&[0.0, 0.0, 1.0]);
    assert!(gr.check_point(&q).is_ok());

    // Cost: f(Q) = tr(Q^T A Q) for A = diag(1, 2, 3)
    // egrad = (A + A^T) Q = 2 A Q; at q=[0,0,1]: egrad = 2*[0,0,3] = [0,0,6]
    // ehvp(V) = 2 A V (Hessian of quadratic form is 2A)
    let a_diag = SMatrix::<f64, 3, 3>::from_diagonal(&nalgebra::SVector::from([1.0, 2.0, 3.0]));
    let egrad = a_diag * q * 2.0;
    let ehvp = |v: &SMatrix<f64, 3, 1>| a_diag * v * 2.0;

    // Tangent at Q: V in horizontal space (I - QQ^T) R^{3x1}
    // Take V = [1, 0, 0]^T (already horizontal since Q^T V = 0)
    let v = SMatrix::<f64, 3, 1>::from_column_slice(&[1.0, 0.0, 0.0]);
    assert!(gr.check_tangent(&q, &v).is_ok());

    let hvp = gr.riemannian_hessian_vector_product(&q, &egrad, &v, &ehvp).unwrap();

    // Manual calculation (Boumal formula: proj_Q(ehvp) - V * sym(Q^T * G)):
    // proj_Q(2A·V) = proj_Q([2,0,0]) = [2,0,0] (already horizontal)
    // sym(Q^T · G) = sym([[6]]) = [[6]] (1x1 scalar since K=1)
    // V · sym(Q^T · G) = [1,0,0]^T * 6 = [6,0,0]
    // Hess[V] = [2,0,0] - [6,0,0] = [-4,0,0]
    let expected = SMatrix::<f64, 3, 1>::from_column_slice(&[-4.0, 0.0, 0.0]);
    let diff = (hvp - expected).norm();
    assert!(diff < 1e-10, "Grassmann HVP wrong: diff = {diff}");
}
```

- [ ] **Step 2: Run and confirm failure**

```bash
cargo test -p cartan-manifolds test_grassmann_hessian_weingarten_correction -- --nocapture
```

- [ ] **Step 3: Implement the correction in `grassmann.rs`**

```rust
fn riemannian_hessian_vector_product(
    &self,
    p: &Self::Point,
    grad_f: &Self::Tangent,
    v: &Self::Tangent,
    hess_ambient: &dyn Fn(&Self::Tangent) -> Self::Tangent,
) -> Result<Self::Tangent, CartanError> {
    let q = p;        // N×K orthonormal matrix
    let g = grad_f;   // Euclidean gradient (N×K)

    // Step 1: proj_Q(ehvp(V)) — horizontal projection
    let ehvp_v = hess_ambient(v);
    // proj_Q(X) = X - Q * Q^T * X  (horizontal projection)
    let proj_ehvp = &ehvp_v - q * (q.transpose() * &ehvp_v);

    // Step 2: Weingarten correction (Boumal 2022, §9.5, Prop 9.46)
    //   Hess f(Q)[V] = proj_Q(ehvp(V)) - V · sym(Q^T · G)
    // where sym(A) = (A + A^T) / 2  (K×K symmetric)
    // V is N×K, sym(Q^T·G) is K×K, so V·sym(Q^T·G) is N×K.
    // This term is nonzero even when Q^T·V = 0 (horizontal tangent).
    let qt_g = q.transpose() * g;          // K×K
    let sym_qt_g = (&qt_g + qt_g.transpose()) * 0.5;   // K×K symmetric
    let correction = v * sym_qt_g;         // N×K

    Ok(proj_ehvp - correction)
}
```

- [ ] **Step 4: Run test, confirm pass**

```bash
cargo test -p cartan-manifolds test_grassmann_hessian_weingarten_correction -- --nocapture
```

- [ ] **Step 5: Full test suite**

```bash
cargo test -p cartan-manifolds
```

- [ ] **Step 6: Commit**

```bash
git add cartan-manifolds/src/grassmann.rs && git commit -m "fix(grassmann): add Weingarten correction to riemannian_hessian_vector_product"
```

---

### Task 4: `SpecialEuclidean<N>` — Weingarten correction (block-diagonal)

**Files:**
- Modify: `cartan-manifolds/src/se.rs`
- Test: `cartan-manifolds/src/se.rs` (inline tests)

SE(N) has product structure: rotation block (SO(N) correction) + translation block (zero correction, flat).

- [ ] **Step 1: Write a failing test**

```rust
#[test]
fn test_se_hessian_weingarten_correction() {
    use cartan_core::{Connection, Manifold};
    use nalgebra::{SMatrix, SVector};

    let se2 = SpecialEuclidean::<2>;

    // Use R=I as the rotation component so the correction is easy to compute.
    // egrad_R must have a nonzero SYMMETRIC PART for the SO block correction to fire.
    let p = SEPoint {
        rotation: SMatrix::<f64, 2, 2>::identity(),
        translation: SVector::<f64, 2>::zeros(),
    };

    // Skew 2×2 tangent at R=I: [[0, -1], [1, 0]]
    // Translation tangent: arbitrary nonzero vector
    let v = SETangent {
        rotation: SMatrix::<f64, 2, 2>::from_row_slice(&[0.0, -1.0, 1.0, 0.0]),
        translation: SVector::<f64, 2>::from([0.5, -0.3]),
    };

    // Cost: f(R, t) = tr(A^T R) for A = diag(1, 2) — symmetric egrad, zero Hessian
    // egrad_R = A, ehvp_R = 0 (linear cost)
    // No translation cost: egrad_t = 0, ehvp_t = 0
    let a = SMatrix::<f64, 2, 2>::from_diagonal(&nalgebra::SVector::from([1.0_f64, 2.0]));
    let egrad = SETangent {
        rotation: a,
        translation: SVector::<f64, 2>::zeros(),
    };
    let ehvp = |_w: &SETangent<2>| SETangent {
        rotation: SMatrix::<f64, 2, 2>::zeros(),
        translation: SVector::<f64, 2>::zeros(),
    };

    let hvp = se2.riemannian_hessian_vector_product(&p, &egrad, &v, &ehvp).unwrap();

    // At R=I: term1 = I*skew(I*ehvp_R(V)) = skew(0) = 0
    //         term2 = -0.5 * I * sym(I^T * A) * I^T * V_R = -0.5 * A * V_R
    // V_R = [[0,-1],[1,0]], A = diag(1,2)
    // A * V_R: (A*V)[i,j] = A[i,i]*V[i,j]
    //   row 0: 1*[0,-1] = [0,-1]
    //   row 1: 2*[1,0]  = [2,0]
    // A * V_R = [[0,-1],[2,0]]
    // hess_rotation = -0.5 * [[0,-1],[2,0]] = [[0,0.5],[-1,0]]
    let expected_rotation = -(a * v.rotation) * 0.5;
    let diff_r = (hvp.rotation - expected_rotation).norm();
    assert!(diff_r < 1e-10, "SE rotation block Weingarten wrong: diff = {diff_r}");

    // Translation block: ehvp_t = 0, no translation cost → flat, result = 0
    assert!(hvp.translation.norm() < 1e-10, "SE translation block should be zero");
}
```

- [ ] **Step 2: Run and confirm failure**

```bash
cargo test -p cartan-manifolds test_se_hessian_weingarten_correction -- --nocapture
```

- [ ] **Step 3: Implement in `se.rs`**

```rust
fn riemannian_hessian_vector_product(
    &self,
    p: &Self::Point,
    grad_f: &Self::Tangent,
    v: &Self::Tangent,
    hess_ambient: &dyn Fn(&Self::Tangent) -> Self::Tangent,
) -> Result<Self::Tangent, CartanError> {
    let r = &p.rotation;
    let egrad_r = &grad_f.rotation;

    // Ambient HVP
    let ehvp_v = hess_ambient(v);

    // Rotation block: SO(N) Weingarten correction
    // term1 = R · skew(R^T · ehvp_R(V))
    let rt_ehvp_r = r.transpose() * &ehvp_v.rotation;
    let skew_rt_ehvp = (&rt_ehvp_r - rt_ehvp_r.transpose()) * 0.5;
    let term1 = r * skew_rt_ehvp;
    // Weingarten: -0.5 * R · sym(R^T · egrad_R) · R^T · V_R
    let rt_egrad_r = r.transpose() * egrad_r;
    let sym_rt_egrad = (&rt_egrad_r + rt_egrad_r.transpose()) * 0.5;
    let term2 = r * sym_rt_egrad * r.transpose() * &v.rotation * 0.5;
    let hess_rotation = term1 - term2;

    // Translation block: flat (Euclidean), no correction
    let hess_translation = ehvp_v.translation;

    Ok(SETangent {
        rotation: hess_rotation,
        translation: hess_translation,
    })
}
```

- [ ] **Step 4: Run test, confirm pass**

```bash
cargo test -p cartan-manifolds test_se_hessian_weingarten_correction -- --nocapture
```

- [ ] **Step 5: Full test suite**

```bash
cargo test -p cartan-manifolds
```

- [ ] **Step 6: Commit**

```bash
git add cartan-manifolds/src/se.rs && git commit -m "fix(se): add Weingarten correction (SO block) to riemannian_hessian_vector_product"
```

---

## Chunk 2: Ricci Callback and Doctests

### Task 5: Full-tensor Ricci correction in `cartan-dec`

**Files:**
- Modify: `cartan-dec/src/laplace.rs`
- Test: `cartan-dec/src/laplace.rs` (inline tests)

Replace `Option<f64>` scalar with `Option<&dyn Fn(usize) -> [[f64; 2]; 2]>` (Bochner) and `Option<&dyn Fn(usize) -> [[f64; 3]; 3]>` (Lichnerowicz) callbacks in both Laplacian methods. These use concrete array sizes (2D vector field = 2×2; symmetric 2-tensor = 3 components = 3×3).

- [ ] **Step 1: Write failing tests for the new callback API**

Add to the `#[cfg(test)]` block in `laplace.rs` (or a new `#[cfg(test)]` block at the bottom):

```rust
#[test]
fn test_bochner_tensor_ricci_zero() {
    // Zero callback should match None behavior
    use crate::Mesh;
    let mesh = Mesh::unit_square_grid(4);
    let ops = crate::Operators::from_mesh(&mesh);

    let nv = mesh.n_vertices();
    let u: nalgebra::DVector<f64> = nalgebra::DVector::from_element(2 * nv, 0.5);

    let result_none = ops.apply_bochner_laplacian(&u, None);
    let result_zero = ops.apply_bochner_laplacian(&u, Some(&|_| [[0.0, 0.0], [0.0, 0.0]]));
    let diff = (&result_none - &result_zero).norm();
    assert!(diff < 1e-12, "zero callback should equal None: diff = {diff}");
}

#[test]
fn test_bochner_tensor_ricci_nonzero() {
    // On an Einstein manifold Ric = k*g, the tensor callback [[k,0],[0,k]]
    // should match the old scalar path's result with kappa=k.
    use crate::Mesh;
    let mesh = Mesh::unit_square_grid(4);
    let ops = crate::Operators::from_mesh(&mesh);
    let nv = mesh.n_vertices();

    let u: nalgebra::DVector<f64> = nalgebra::DVector::from_fn(2 * nv, |i, _| i as f64 * 0.01);
    let kappa = 2.5;

    // Manual application of scalar Einstein Ricci: Lu + kappa*u
    let lu = ops.apply_laplace_beltrami(&u.rows(0, nv).into_owned());
    // (applying component-wise with kappa)

    // Using tensor callback
    let result_tensor = ops.apply_bochner_laplacian(&u, Some(&|_| [[kappa, 0.0], [0.0, kappa]]));

    // Using the old scalar path equivalent (reconstruct manually):
    let ux = u.rows(0, nv).into_owned();
    let uy = u.rows(nv, nv).into_owned();
    let lux = &ops.laplace_beltrami * &ux + &ux * kappa;
    let luy = &ops.laplace_beltrami * &uy + &uy * kappa;
    let mut expected = nalgebra::DVector::zeros(2 * nv);
    expected.rows_mut(0, nv).copy_from(&lux);
    expected.rows_mut(nv, nv).copy_from(&luy);

    let diff = (&result_tensor - &expected).norm();
    assert!(diff < 1e-10, "Einstein tensor result should match scalar: diff = {diff}");
}

#[test]
fn test_lichnerowicz_tensor_callback_zero() {
    use crate::Mesh;
    let mesh = Mesh::unit_square_grid(4);
    let ops = crate::Operators::from_mesh(&mesh);
    let nv = mesh.n_vertices();
    let q: nalgebra::DVector<f64> = nalgebra::DVector::from_element(3 * nv, 0.3);

    let result_none = ops.apply_lichnerowicz_laplacian(&q, None);
    let zero_cb: [[f64; 3]; 3] = [[0.0; 3]; 3];
    let result_zero = ops.apply_lichnerowicz_laplacian(&q, Some(&|_| zero_cb));
    let diff = (&result_none - &result_zero).norm();
    assert!(diff < 1e-12, "zero callback should match None: diff = {diff}");
}
```

- [ ] **Step 2: Run tests — they should compile but fail (API doesn't match yet)**

```bash
cargo test -p cartan-dec test_bochner_tensor_ricci -- --nocapture 2>&1 | head -30
```
Expected: compile error (wrong signature on `apply_bochner_laplacian`).

- [ ] **Step 3: Replace the API in `cartan-dec/src/laplace.rs`**

Replace `apply_bochner_laplacian` signature and body:

```rust
/// Apply the Bochner (connection) Laplacian to a vector field.
///
/// `ricci_correction`: optional callback `v -> Ric_v` returning the 2x2
/// Ricci tensor at vertex `v` as `[[f64; 2]; 2]`. For flat domains pass `None`.
/// For Einstein manifolds (Ric = κ·g): `Some(&|_| [[κ, 0.], [0., κ]])`.
pub fn apply_bochner_laplacian(
    &self,
    u: &DVector<f64>,
    ricci_correction: Option<&dyn Fn(usize) -> [[f64; 2]; 2]>,
) -> DVector<f64> {
    let nv = self.laplace_beltrami.nrows();
    assert_eq!(u.len(), 2 * nv, "Bochner: u must have 2*n_v entries");

    let ux = u.rows(0, nv);
    let uy = u.rows(nv, nv);

    let mut lux = &self.laplace_beltrami * ux;
    let mut luy = &self.laplace_beltrami * uy;

    if let Some(ric) = ricci_correction {
        for v in 0..nv {
            let r = ric(v);  // [[r00, r01], [r10, r11]]
            // (∇*∇ + Ric)(u)_v = Δu_v + Ric_v · u_v
            let ux_v = ux[v];
            let uy_v = uy[v];
            lux[v] += r[0][0] * ux_v + r[0][1] * uy_v;
            luy[v] += r[1][0] * ux_v + r[1][1] * uy_v;
        }
    }

    let mut result = DVector::<f64>::zeros(2 * nv);
    result.rows_mut(0, nv).copy_from(&lux);
    result.rows_mut(nv, nv).copy_from(&luy);
    result
}
```

Replace `apply_lichnerowicz_laplacian` signature and body:

```rust
/// Apply the Lichnerowicz Laplacian to a symmetric 2-tensor field Q.
///
/// `curvature_correction`: optional callback `v -> C_v` returning the 3x3
/// matrix of the curvature endomorphism on the 3-component symmetric 2-tensor
/// space `[Qxx, Qxy, Qyy]` at vertex `v`.
/// For flat R²: `None`. For space form with sectional K=κ:
/// `Some(&|_| [[2.*κ,0.,0.],[0.,2.*κ,0.],[0.,0.,2.*κ]])`.
pub fn apply_lichnerowicz_laplacian(
    &self,
    q: &DVector<f64>,
    curvature_correction: Option<&dyn Fn(usize) -> [[f64; 3]; 3]>,
) -> DVector<f64> {
    let nv = self.laplace_beltrami.nrows();
    assert_eq!(q.len(), 3 * nv, "Lichnerowicz: q must have 3*n_v entries");

    let qxx = q.rows(0, nv);
    let qxy = q.rows(nv, nv);
    let qyy = q.rows(2 * nv, nv);

    let mut lxx = &self.laplace_beltrami * qxx;
    let mut lxy = &self.laplace_beltrami * qxy;
    let mut lyy = &self.laplace_beltrami * qyy;

    if let Some(curv) = curvature_correction {
        for v in 0..nv {
            let c = curv(v);  // 3x3 matrix acting on [qxx, qxy, qyy]
            let qx = qxx[v];
            let qm = qxy[v];
            let qy = qyy[v];
            lxx[v] += c[0][0] * qx + c[0][1] * qm + c[0][2] * qy;
            lxy[v] += c[1][0] * qx + c[1][1] * qm + c[1][2] * qy;
            lyy[v] += c[2][0] * qx + c[2][1] * qm + c[2][2] * qy;
        }
    }

    let mut result = DVector::<f64>::zeros(3 * nv);
    result.rows_mut(0, nv).copy_from(&lxx);
    result.rows_mut(nv, nv).copy_from(&lxy);
    result.rows_mut(2 * nv, nv).copy_from(&lyy);
    result
}
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
cargo test -p cartan-dec test_bochner_tensor_ricci -- --nocapture
cargo test -p cartan-dec test_lichnerowicz_tensor_callback_zero -- --nocapture
```
Expected: all pass.

- [ ] **Step 5: Run full `cartan-dec` test suite**

```bash
cargo test -p cartan-dec
```
Expected: all pass. Callers passing `None` require no changes (type inference handles both old and new signatures). Only callers passing `Some(kappa)` (scalar) must be updated to `Some(&|_| [[kappa, 0.], [0., kappa]])`. Search for these: `grep -rn "apply_bochner_laplacian\|apply_lichnerowicz_laplacian" cartan-dec/`. The current codebase has no such callers — all existing calls use `None`.

- [ ] **Step 6: Commit**

```bash
git add cartan-dec/src/laplace.rs && git commit -m "fix(dec): replace scalar Ricci/curvature option with full-tensor vertex callbacks"
```

---

### Task 6: Fix doctest annotations in `cartan-manifolds`

**Files:**
- Modify: `cartan-manifolds/src/sphere.rs`
- Modify: `cartan-manifolds/src/so.rs`
- Modify: `cartan-manifolds/src/spd.rs`
- Modify: `cartan-manifolds/src/se.rs`
- Modify: `cartan-manifolds/src/grassmann.rs`
- Modify: `cartan-manifolds/src/euclidean.rs`
- Modify: `cartan-manifolds/src/corr.rs`

- [ ] **Step 1: Find all ignored doctests**

```bash
# run from workspace root: /home/alejandrosotofranco/cartan
grep -rn '```rust,ignore' cartan-manifolds/src/
```

Note every occurrence. There should be one per manifold struct doc comment.

- [ ] **Step 2: Classify each doctest**

For each `ignore`d doctest, determine:
- **Run**: uses only hardcoded values, no RNG, no external state → change to ` ```rust `
- **no_run**: uses `random_point`, `random_tangent`, or any RNG → change to ` ```rust,no_run `

Classification:
- Examples showing `random_point` / `random_tangent` / `rand::rng()` → `no_run`
- Examples showing `exp`, `log`, `inner`, `dist`, `check_point`, `check_tangent` with hardcoded inputs → runnable

- [ ] **Step 3: Rewrite each doctest to match its classification**

For runnable examples, replace `ignore` and ensure the example compiles and produces correct output. Example for `Sphere`:

```rust
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
///
/// // Exponential map: move along geodesic from p in direction v
/// let q = s2.exp(&p, &v);
/// assert!((q.norm() - 1.0).abs() < 1e-10, "q must be on the sphere");
///
/// // Logarithmic map: recover v
/// let v_rec = s2.log(&p, &q).unwrap();
/// assert!((v_rec - v).norm() < 1e-10);
/// ```
///
/// ```rust,no_run
/// use cartan_manifolds::Sphere;
/// use cartan_core::Manifold;
///
/// let s2 = Sphere::<3>;
/// let mut rng = rand::rng();
/// let p = s2.random_point(&mut rng);
/// let v = s2.random_tangent(&p, &mut rng);
/// let q = s2.exp(&p, &v);
/// ```
```

Apply the same pattern to SO(N), SPD(N), Grassmann, SE(N), Euclidean, Corr: one runnable doctest with hardcoded inputs + one `no_run` doctest showing random sampling.

- [ ] **Step 4: Run doctests**

```bash
cargo test -p cartan-manifolds --doc
```
Expected: all run (not ignored), zero failures.

- [ ] **Step 5: Verify no bare `ignore` remains**

```bash
# run from workspace root: /home/alejandrosotofranco/cartan
grep -rn '```rust,ignore' cartan-manifolds/src/
```
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add cartan-manifolds/ && git commit -m "fix(manifolds): replace bare ignore doctests with no_run or runnable examples"
```

---

## Verification

After all tasks complete:

```bash
# All tests pass
cargo test -p cartan-manifolds
cargo test -p cartan-dec

# No bare ignore doctests remain (run from workspace root)
grep -rn '```rust,ignore' cartan-manifolds/src/
# Expected: no output

# Doctests pass
cargo test -p cartan-manifolds --doc
cargo test -p cartan-dec --doc

# Full workspace
cargo test
```
