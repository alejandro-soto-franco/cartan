# Manifold Formula Reference

Mathematical formulas for all implemented and planned cartan manifolds.
Source of truth for cartan-docs content pages and cartan-manifolds implementations.

Internal developer reference. Not committed to the repo (see .gitignore).

---

## Euclidean R^N

Points: vectors in R^N. Tangent space at any point: R^N (flat, trivial).

| Quantity | Formula |
|----------|---------|
| dim | N |
| inner(p, u, v) | u^T v |
| exp(p, v) | p + v |
| log(p, q) | q - p |
| transport(p, q, v) | v |
| retract(p, v) | p + v |
| K | 0 everywhere |
| Ricci(u,v) | 0 |
| scalar | 0 |
| injectivity radius | infinity |
| cut locus | none |

---

## Sphere S^{N-1}

Points: unit vectors in R^N, ||p|| = 1.
Tangent space at p: vectors orthogonal to p, T_pS^{N-1} = {v : p^T v = 0}.

| Quantity | Formula |
|----------|---------|
| dim | N-1 |
| inner(p, u, v) | u^T v |
| exp(p, v) | cos(t) p + sin(t) (v/t), where t = ||v|| |
| log(p, q) | theta / sin(theta) * (q - cos(theta) p), where theta = arccos(p^T q) |
| transport(p, q, v) | v - (p^T v / (1 + p^T q)) * (p + q) (Schild's ladder) |
| retract(p, v) | (p + v) / ||p + v|| (normalization retraction) |
| K(u,v) | 1 (constant sectional curvature) |
| Ricci(u,v) | (N-2) u^T v |
| scalar | (N-1)(N-2) |
| injectivity radius | pi |
| cut locus at p | antipodal point -p |
| R(u,v)w | <v,w> u - <u,w> v (constant curvature formula) |

Taylor expansion for log near theta = 0:
  theta / sin(theta) = 1 + theta^2/6 + 7*theta^4/360 + O(theta^6)
Use for |theta| < 1e-6 to avoid division by near-zero sin(theta).

---

## Special Orthogonal SO(N)

Points: N x N orthogonal matrices with det = +1, R^T R = I, det(R) = 1.
Tangent space at R: skew-symmetric matrices right-translated to R,
  T_R SO(N) = { R * Omega : Omega + Omega^T = 0 }
Lie algebra so(N): N x N skew-symmetric matrices, dim = N(N-1)/2.

Bi-invariant metric: <X, Y>_R = (1/2) tr(X^T Y), where X, Y in T_R SO(N).

| Quantity | Formula |
|----------|---------|
| dim | N(N-1)/2 |
| ambient_dim | N*N |
| inner(R, X, Y) | (1/2) tr(X^T Y) |
| exp(R, V) | R * matrix_exp(R^T V) |
| log(R, Q) | R * matrix_log(R^T Q) |
| transport(R, Q, V) | Q * R^T * V (left translation) |
| retract(R, V) | Cayley: R * (I + Omega/2)^{-1} * (I - Omega/2), Omega = R^T V |
| K(X,Y) | (1/4) ||[Omega_X, Omega_Y]||^2 / (||X||^2 ||Y||^2 - <X,Y>^2) |
| Ricci(X,Y) | (N-2)/4 * <X,Y> (for N >= 3) |
| scalar | N(N-1)(N-2)/8 |
| injectivity radius | pi |
| cut locus at R | rotations with an eigenvalue angle = pi |

matrix_exp: computed via Pade approximation or eigendecomposition.
matrix_log: principal logarithm; fails when R has eigenvalue -1 (cut locus).

Left Jacobian of SO(N) at Omega (for SE(N)):
  J(Omega) = sum_{k=0}^{inf} Omega^k / (k+1)!
  Left Jacobian inverse: J^{-1}(Omega) = I - Omega/2 + B_k terms (Bernoulli series)

---

## Special Euclidean SE(N)

Points: (N+1) x (N+1) homogeneous transformation matrices:
  T = | R  t |
      | 0  1 |
where R in SO(N), t in R^N. Represents rigid body motions.

Lie algebra se(N): (N+1) x (N+1) matrices of the form:
  Xi = | Omega  v |
       | 0      0 |
where Omega in so(N), v in R^N. dim = N(N-1)/2 + N = N(N+1)/2.

Left-invariant metric: <Xi, Eta>_T = tr(Omega_Xi^T Omega_Eta) + v_Xi^T v_Eta.

| Quantity | Formula |
|----------|---------|
| dim | N(N+1)/2 |
| exp(T, V) | T * matrix_exp(T^{-1} V) |
| log(T, S) | T * matrix_log(T^{-1} S) |
| transport | Left translation (same pattern as SO(N)) |

The SE(N) exponential in closed form (for the (R,t) representation):
  exp(Omega, v) = (exp_SO(N)(Omega), J(Omega) v)
  where J(Omega) is the left Jacobian of SO(N).

---

## Symmetric Positive Definite SPD(N) (planned)

Points: N x N symmetric positive definite matrices.
Metric: affine-invariant, <U,V>_P = tr(P^{-1} U P^{-1} V).

| Quantity | Formula |
|----------|---------|
| dim | N(N+1)/2 |
| exp(P, V) | P^{1/2} matrix_exp(P^{-1/2} V P^{-1/2}) P^{1/2} |
| log(P, Q) | P^{1/2} matrix_log(P^{-1/2} Q P^{-1/2}) P^{1/2} |
| transport(P, Q, V) | A V A^T, where A = (Q P^{-1})^{1/2} |
| K | <= 0 (non-positive, Cartan-Hadamard manifold) |
| injectivity radius | infinity |
| cut locus | none |

Reference: Bhatia, "Positive Definite Matrices", Princeton 2007, Ch. 6.

---

## Grassmann Gr(N, K) (planned)

Points: K-dimensional subspaces of R^N, represented as N x K orthonormal matrices
(equivalently: equivalence classes under right multiplication by O(K)).

| Quantity | Formula |
|----------|---------|
| dim | K(N-K) |
| K | 0 <= K <= 2 (sectional curvature bounded) |
| injectivity radius | pi/2 |
| cut locus | subspaces with a principal angle = pi/2 |

Tangent space at Y in Gr(N,K): {Z in R^{N x K} : Y^T Z = 0}.
Metric: <Z1, Z2>_Y = tr(Z1^T Z2).

Reference: Absil, Mahony, Sepulchre, "Optimization Algorithms on Matrix Manifolds",
Princeton 2008, Ch. 2.

---

## Hyperbolic H^N (planned)

Hyperboloid model: H^N = {x in R^{N+1} : <x,x>_M = -1, x_0 > 0}
where <x,y>_M = -x_0 y_0 + x_1 y_1 + ... + x_N y_N (Minkowski inner product).

| Quantity | Formula |
|----------|---------|
| dim | N |
| K | -1 (constant negative sectional curvature) |
| injectivity radius | infinity |
| cut locus | none (Cartan-Hadamard) |

exp(p, v): p cosh(||v||_M) + v/||v||_M sinh(||v||_M)
log(p, q): acosh(-<p,q>_M) / sinh(acosh(-<p,q>_M)) * (q - <p,q>_M p)

Reference: do Carmo, "Riemannian Geometry", Ch. 8 (space forms).

---

## Curvature Table Summary

| Manifold | K (sectional) | Ricci | Scalar | Inj. radius |
|----------|--------------|-------|--------|-------------|
| Euclidean R^N | 0 | 0 | 0 | infinity |
| Sphere S^{N-1} | 1 | (N-2) g | (N-1)(N-2) | pi |
| SO(N) | >= 0, <= 1/4 | (N-2)/4 g | N(N-1)(N-2)/8 | pi |
| SE(N) | varies | varies | varies | pi |
| SPD(N) | <= 0 | <= 0 | <= 0 | infinity |
| Gr(N,K) | 0 to 2 | varies | varies | pi/2 |
| H^N | -1 | -(N-1) g | -N(N-1) | infinity |
