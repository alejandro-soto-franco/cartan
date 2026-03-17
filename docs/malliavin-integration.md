# malliavin ↔ cartan Integration Notes

Internal developer reference. Not committed to the repo.

Written by Claude Code session 2026-03-17. Updated 2026-03-17 (Sub-project 2 architecture).
Audience: cartan developer (implementing manifolds), malliavin architect (designing agents).

---

## Sub-project 2 Architecture — New Requirements (2026-03-17)

The full Sub-project 2 engine design (spec: `malliavin/docs/superpowers/specs/2026-03-17-malliavin-engine-design.md`)
finalizes the geometric architecture. This section supersedes the `ManifoldVelocityAgent` pattern below
for the runtime engine. The individual agent manifold catalog (SPD, Grassmann, Fisher-Rao, etc.) remains valid.

### Core geometric model

All regime state lives on **Sym⁺(N)** with the affine-invariant metric. Regime state is a point `p` on
this manifold. Agents emit **tangent vector votes** `v ∈ Sym(N) ≅ T_p(Sym⁺(N))`. The `ManifoldThread`
applies one exponential map step per batch of votes:

```
p ← Exp_p(η · Σᵢ wᵢvᵢ)
```

There is one `ManifoldState` per `CadenceClass` (Hft, Intraday, Swing, Macro), each with its own
step size `η_cadence`. A `MetaRegimeDetector` fuses the four per-cadence centroids into a global
centroid via a weighted Fréchet mean — all on the same Sym⁺(N).

### Minimum required cartan interface (Sub-project 2)

These five functions are the exact surface area the runtime engine needs. Priority: **P0-critical**.

| Function | Signature | Formula |
|---|---|---|
| `exp_map` | `(p: SymPos, v: Sym) → SymPos` | `p½ · exp(p⁻½vp⁻½) · p½` |
| `log_map` | `(p: SymPos, q: SymPos) → Sym` | `p½ · log(p⁻½qp⁻½) · p½` |
| `geodesic_distance` | `(p: SymPos, q: SymPos) → f64` | `‖log(p⁻½qp⁻½)‖_F` |
| `frechet_mean` | `(pts: &[SymPos], w: &[f64]) → SymPos` | Iterative Karcher flow |
| `frechet_mean_gradient_step` | `(p: SymPos, samples: &[SymPos], η: f64) → SymPos` | One Karcher step (prototype calibration) |

All five operate under the affine-invariant metric on Sym⁺(N).
Per-call cost: two Cholesky factorizations + one symmetric eigendecomposition = O(N³).
For N ≤ 20: sub-microsecond on modern hardware.

### How agents interact with the new model

Agents no longer emit `AgentVote` (the velocity-based struct below). They emit `TangentVote`:

```rust
pub struct TangentVote {
    pub v: SymmetricMatrix,  // element of Sym(N) — the tangent direction
    pub weight: f64,
    pub agent_id: AgentId,
    pub timestamp_ns: u64,
}
```

The concrete `ManifoldVelocityAgent<M>` pattern (below) is still the right way to implement
individual agents. The agent's `on_data` method should:
1. Compute the manifold velocity as before (geodesic log map on the agent's native manifold)
2. **Project** that velocity into `Sym(N)` (the tangent space of the regime manifold)
3. Return a `TangentVote` with the projected tangent vector

The projection from an agent's native manifold tangent space to `Sym(N)` is agent-specific:
- `SPDVelocityAgent`: native tangent IS already `Sym(N)` — no projection needed
- `GrassmannianAgent`: project `Gr(k,N)` tangent (N×k matrix) to `Sym(N)` via `v · vᵀ`
- `LieAlgebraAgent`: project `so(N)` skew-symmetric matrix to `Sym(N)` via `(Ω + Ωᵀ)/2 = 0` for skew...
  use `Ω · Ωᵀ` (which IS symmetric positive semidefinite) as the projection
- `FisherRaoAgent`: embed H² velocity to `Sym(2)` via the Fisher information matrix

### Voronoi dispatch — what cartan needs to support

Regime dispatch uses Voronoi cells on Sym⁺(N) around prototype points. The `ManifoldThread`
calls `geodesic_distance(p, prototype_j)` after each `Exp_p` step to check cell membership.
The hysteresis band requires two distance evaluations per active regime per bar.

`frechet_mean` is called by `MetaRegimeDetector` once per Intraday tick (4 inputs, weighted).
`frechet_mean_gradient_step` is called offline during prototype calibration only.

---

---

## Context

`malliavin` is a fully autonomous regime detection and signal-routing engine for options
backtesting and live trading. Its `RegimeAgent` system uses generic manifold agents of the form:

```rust
pub struct ManifoldVelocityAgent<M: Manifold + Connection> {
    manifold:   M,
    prev_point: M::Point,
    window:     RollingMarketState,
    threshold:  f64,
}
```

Each agent embeds market state into a manifold `M`, computes the geodesic velocity
(`log(prev, current)` in the tangent space at `prev`), then interprets the speed/direction
as a regime signal with confidence proportional to `||velocity|| / threshold`.

**malliavin depends on cartan to provide the manifolds.** This document specifies exactly what
malliavin needs so cartan can prioritize its roadmap accordingly.

---

## Minimum Required Trait API

For `ManifoldVelocityAgent<M>` to compile, `M` must implement:

```rust
// From cartan-core Manifold trait (already defined):
fn log(&self, p: &Self::Point, q: &Self::Point) -> Self::Tangent;
fn inner(&self, p: &Self::Point, u: &Self::Tangent, v: &Self::Tangent) -> f64;
fn norm(&self, p: &Self::Point, v: &Self::Tangent) -> f64;
fn project_point(&self, p: Self::Point) -> Self::Point;  // for numerical stabilization
```

For time-series agents that need to carry a tangent vector forward in time:

```rust
// From cartan-core ParallelTransport trait (already defined):
fn transport(&self, p: &Self::Point, q: &Self::Point, v: &Self::Tangent) -> Self::Tangent;
```

For agents that need intrinsic curvature awareness (optional, but used by FisherRao and IVSurface):

```rust
// From cartan-core Curvature trait (already defined):
fn sectional(&self, p: &Self::Point, u: &Self::Tangent, v: &Self::Tangent) -> f64;
```

**No new traits are needed.** cartan's existing 7-trait hierarchy covers all malliavin requirements.
The work is purely manifold *implementations*.

---

## Manifold Catalog — malliavin Requirements

Priority legend: **P0** = blocks first backtest run, **P1** = needed for v1 agent suite, **P2** = research.

### P0 — SymmetricPositiveDefinite<N> (SPD(N))

Already on cartan roadmap as `SymmetricPositiveDefinite<N>`.

**malliavin agent**: `SPDVelocityAgent`
- **Point type**: `SMatrix<f64, N, N>` (symmetric, all eigenvalues > 0)
- **Embedding**: rolling N×N realized covariance matrix of log-returns (e.g. N=5 for BTC,ETH,SPX,VIX,DXY)
- **Signal**: `log_spd(Σ_{t-1}, Σ_t)` is the covariance *velocity* in the affine-invariant metric
- **Regime**: high `||velocity||` = rapid correlation shift = regime transition

**Required metric**: affine-invariant (Bures/power):
```
log_spd(P, Q) = P^{1/2} * logm(P^{-1/2} Q P^{-1/2}) * P^{1/2}
dist(P, Q)^2  = sum_i log(lambda_i(P^{-1}Q))^2
```
where `lambda_i` are generalized eigenvalues.

**Numerical notes**:
- `P^{-1/2}` via Cholesky: `L = cholesky(P)`, then `P^{-1/2} = L^{-T}`
- `logm` via symmetric eigendecomposition: `A = V diag(lambda) V^T`, `logm(A) = V diag(log(lambda)) V^T`
- Guard: clip eigenvalues to `[1e-8, ∞)` before `log` to prevent NaN from near-singular matrices
- `project_point`: re-symmetrize and clip eigenvalues: `(A + A^T)/2`, eigenvalue floor `1e-8`

**Cartan implementation note**: N is a const generic. Start with N=3,4,5 for testing.
The `injectivity_radius` is `∞` (SPD(N) is a Cartan-Hadamard manifold, K ≤ 0).

---

### P0 — Grassmann<N, K> (G(k,n))

Already on cartan roadmap as `Grassmann<N, K>`.

**malliavin agent**: `GrassmannianAgent`
- **Point type**: equivalence class of N×K matrices with orthonormal columns (a k-dimensional subspace of R^N)
- **Embedding**: top-k eigenvectors of rolling covariance (the *signal subspace*)
- **Signal**: `log_gr(U_{t-1}, U_t)` measures how fast the dominant signal subspace is rotating
- **Regime**: rapid subspace rotation = unstable eigenstructure = transitioning market

**Required metric**: canonical metric on G(k,n) via the horizontal lift:
```
// Given U, V in St(N,K) representing the same Grassmann points:
log_gr(U, V) = U_perp * B,  where B = log(U^T V)
// (thin SVD decomposition of V - U U^T V gives the geodesic)
```

More precisely, using the SVD-based formula:
```
M = U^T V
[U_svd, Sigma, V_svd^T] = svd(I - U U^T) * V  (thin)
theta_i = arctan(sigma_i)
Log_U(V) = U_svd * diag(theta) * V_svd^T       (horizontal tangent vector)
```

**Numerical notes**:
- `project_point(A)` = QR factorization, take Q (orthonormalize columns)
- Guard conjugate-pair eigenvalues in `logm` path
- `dist(U,V) = ||[theta_1,...,theta_k]||_2` where `theta_i = arccos(sigma_i(U^T V))`

**Cartan implementation note**: const generics `N` (ambient dim) and `K` (subspace dim).
Constraint `K <= N/2` for the canonical embedding. Representative: `SMatrix<f64, N, K>`.

---

### P1 — Statistical Manifold (Fisher-Rao)

**Not on cartan roadmap yet.** This is a new manifold family.

**malliavin agent**: `FisherRaoAgent`
- **Observation scale**: 5-minute bars, rolling window of ~50 bars
- **Point type**: parameters of a probability distribution (mean, variance, shape parameters)
- **Embedding**: fit a distribution to rolling return samples → parameter vector = point on manifold
- **Signal**: geodesic velocity in Fisher-Rao metric = how fast the return *distribution* is changing
- **Regime**: distributional divergence is a leading indicator of vol regime shifts

**For malliavin v1, only the multivariate Gaussian case is needed:**

The statistical manifold of `N(μ, Σ)` with N-dimensional μ and SPD Σ is isometric to:
`G = (R^N) × SPD(N)` with the Fisher information metric.

The Fisher-Rao metric on this product decomposes as:
```
g((δμ_1, δΣ_1), (δμ_2, δΣ_2)) =
    δμ_1^T Σ^{-1} δμ_2                         (mean component)
  + (1/2) tr(Σ^{-1} δΣ_1 Σ^{-1} δΣ_2)         (covariance component)
```

For **univariate** Gaussian N(μ, σ²):
```
Point: (μ, σ) ∈ R × R+   (upper half-plane)
metric: (dμ² + 2 dσ²) / σ²
```
This is the **hyperbolic plane H²** with curvature K = -1/2.

**Cartan implementation needed**: `Hyperbolic<N>` (already on roadmap).
For multivariate case: `Product<EuclideanN, SPD<N>>` with Fisher-Rao metric override.
Alternatively, wrap SPD<N+1> (the information geometry embedding).

**Practical note for malliavin**: The univariate `H²` case is sufficient for the first
`FisherRaoAgent` impl (fit N(μ,σ) to rolling 5-min returns for each asset independently).
Implement `Hyperbolic<2>` first to unblock this agent.

---

### P1 — SO(N) Lie Algebra Velocity

Already implemented: `SpecialOrthogonal<N>` is done.

**malliavin agent**: `LieAlgebraAgent`
- **Point type**: `SMatrix<f64, N, N>` orthogonal with det=1
- **Embedding**: rolling PCA rotation matrix R_t (the factor rotation in a PCA of returns)
- **Signal**: `log_so(R_{t-1}, R_t) ∈ so(N)` = skew-symmetric velocity; Frobenius norm = rotation speed
- **Regime**: fast rotation of factor loadings = factor structure instability

**This agent can be implemented NOW** — cartan already has `SpecialOrthogonal<N>`.
malliavin only needs `log`, `norm`, and optionally `transport` from the existing impl.

**Signal extraction**:
```
Omega = log_SO(R_prev, R_curr)     // skew-symmetric N×N matrix
speed = ||Omega||_F / sqrt(2)      // normalized rotation rate (rad/step)
axis  = Omega[i,j] for (i,j) s.t. |Omega[i,j]| is max  // dominant rotation axis = which factor pair
```

---

### P1 — Persistent Homology via k-Simplex (cartan-dec)

**Requires cartan-dec**, which is post-v0.1 on the roadmap.

**malliavin agent**: `PersistentHomologyAgent`
- **Observation scale**: 1-hour bars, rolling window of ~100 price vectors
- **Construction**: Vietoris-Rips filtration on point cloud of rolling return vectors
  - Each point = return vector at time t (N-dimensional, one coordinate per asset)
  - Threshold ε swept from 0 to max pairwise distance → build k-simplices as edges form
- **Signal**: Betti numbers β₀, β₁, β₂ as ε increases
  - β₀ = number of connected components (high → assets decorrelating)
  - β₁ = number of independent 1-cycles (loops in return space = cyclic arbitrage patterns)
  - β₂ = 2-cycles (bubbles/voids in distribution)
- **Regime**: rapid change in (β₀, β₁, β₂) profile = topological phase transition

**What malliavin needs from cartan-dec**:

```rust
// Build simplicial complex from point cloud with threshold epsilon
pub fn vietoris_rips(points: &[Vector<f64, N>], epsilon: f64) -> SimplicialComplex;

// Compute boundary matrices up to dimension k
impl SimplicialComplex {
    pub fn boundary_matrix(&self, k: usize) -> SparseMatrix<i8>;
}

// Compute persistent homology (birth/death pairs per dimension)
pub fn persistent_homology(complex: &SimplicialComplex) -> PersistenceDiagram;

pub struct PersistenceDiagram {
    pub pairs: Vec<(f64, f64, usize)>,  // (birth, death, dimension)
}

// Betti numbers at a given filtration level
pub fn betti_numbers(diagram: &PersistenceDiagram, epsilon: f64) -> Vec<usize>;
```

**Cartan-dec implementation roadmap addendum**:
For `PersistentHomologyAgent` in malliavin, Phase 2 (boundary operators) is sufficient.
The full LaplaceBeltrami is NOT needed — only:
1. Simplex construction (Vietoris-Rips graph from pairwise distances)
2. Boundary matrices `∂_k` for k = 0, 1, 2
3. Rank computation (via column reduction over Z/2Z = XOR operations)
4. Betti numbers: `β_k = dim(ker ∂_k) - dim(im ∂_{k+1})`

This is pure combinatorial topology — no Riemannian metric needed.
It can be implemented as a standalone `cartan-dec` feature flag, independently of the DEC solver.

**Complexity note**: Vietoris-Rips is O(n² log n) for graph, O(n³) for full complex.
For `n = 100` points in `N = 5` dimensions: manageable at 1-hour cadence.
Use sparse matrix storage (`nalgebra_sparse` or custom CSR) for boundary operators.

---

### P2 — IV Surface as Jet Bundle J¹

**Research-grade. Not needed for v1.**

**malliavin agent**: `IVSurfaceAgent`
- **Manifold**: The implied volatility surface σ(K, T) as a section of the jet bundle J¹(R² → R)
  - J¹ encodes: value σ, first derivatives ∂σ/∂K (skew), ∂σ/∂T (term structure slope), and curvature
- **Signal**: velocity of the vol surface in the J¹ metric = how fast the skew/term structure is moving
- **Regime**: rapid skew flattening = vol regime transition (risk-off to complacency or vice versa)

**What cartan needs for this**: A finite-dimensional approximation.
Model the vol surface as a parametric curve in a low-dimensional parameter space:
```
σ(K,T) ≈ SVI(a, b, ρ, m, σ_min; K, T)    // Gatheral SVI parametrization
Point = (a, b, ρ, m, σ_min) ∈ R⁵
Tangent = velocity in SVI parameter space
```
Then the "manifold" is just R⁵ with the Fisher information metric of the SVI family
(i.e., how much the *distribution of option prices* changes per unit parameter change).

**This is the same construction as FisherRao** applied to the option pricing model.
Defer to after `FisherRaoAgent` is validated.

---

### P2 — Wasserstein/Bures

**Research-grade. Blocked on SPD(N) anyway.**

**malliavin agent**: `WassersteinAgent`
- **Manifold**: (Gaussian measures, Wasserstein-2 metric) = Bures metric on SPD(N)
- **Bures distance**: `d_Bures(P, Q)² = tr(P) + tr(Q) - 2 tr((P^{1/2} Q P^{1/2})^{1/2})`
- **Signal**: Bures velocity = how fast the *distribution of returns* is moving in Wasserstein space
- **Note**: The Bures metric is equivalent to the affine-invariant metric on SPD(N) up to rescaling.
  The SPD(N) implementation covers this — no separate manifold needed.
  Implement `WassersteinAgent` on top of `SPDVelocityAgent` using the Bures distance formula directly.

---

## Implementation Priority for cartan

Based on malliavin's schedule:

| Priority | Manifold | Blocks | Notes |
|----------|----------|--------|-------|
| **P0** | `SymmetricPositiveDefinite<N>` | `SPDVelocityAgent`, `WassersteinAgent` | Highest value; implement first |
| **P0** | `Grassmann<N, K>` | `GrassmannianAgent` | Second; both P0 agents needed for first regime detector |
| **P1** | `Hyperbolic<N>` | `FisherRaoAgent` (univariate) | H² = Fisher-Rao for N(μ,σ) |
| **P1** | `SO(N)` log/norm | `LieAlgebraAgent` | **ALREADY DONE** — malliavin can use this now |
| **P1** | cartan-dec Phase 1-2 | `PersistentHomologyAgent` | Vietoris-Rips + boundary operators |
| **P2** | `Product<M1, M2>` | `FisherRaoAgent` (multivariate) | Wait until P1 FisherRao validated |
| **P2** | J¹ / SVI manifold | `IVSurfaceAgent` | After FisherRao pattern is clear |

---

## The ManifoldVelocityAgent Pattern — Contract for cartan

This is the exact Rust trait that malliavin will use. cartan must satisfy this interface:

```rust
/// The minimal bound required for ManifoldVelocityAgent<M>.
/// All of these are already in cartan-core — this is the usage contract.
pub trait RegimeManifold: Manifold + ParallelTransport + Send + Sync + 'static
where
    Self::Point:   Clone + Send + Sync,
    Self::Tangent: Clone + Send + Sync,
{}

// Blanket impl (in malliavin, not cartan):
impl<M> RegimeManifold for M
where
    M: Manifold + ParallelTransport + Send + Sync + 'static,
    M::Point:   Clone + Send + Sync,
    M::Tangent: Clone + Send + Sync,
{}
```

The agent pattern:
```rust
impl<M: RegimeManifold> RegimeAgent for ManifoldVelocityAgent<M> {
    async fn on_data(&mut self, data: &MarketData) -> Option<AgentVote> {
        let current = self.extract_point(data)?;

        // Geodesic velocity in tangent space at prev_point
        let velocity = self.manifold.log(&self.prev_point, &current);
        let speed    = self.manifold.norm(&self.prev_point, &velocity);

        // Confidence: saturates at 1.0 when speed >= threshold
        let confidence = (speed / self.threshold).min(1.0);

        // Directional signal from dominant tangent component (manifold-specific)
        let direction = self.extract_direction(&velocity);

        // Parallel transport velocity to current point for next step
        let transported = self.manifold.transport(&self.prev_point, &current, &velocity);
        self.prev_point = current;

        Some(AgentVote {
            agent_id:             M::NAME,  // const str on the manifold type
            regime:               self.classify(speed, direction),
            directional_strength: direction,
            confidence,
            diagnostics:          self.diagnostics(speed, &transported),
            timestamp_ms:         data.timestamp_ms(),
            observation_scale:    M::NATURAL_SCALE,
        })
    }
}
```

**What this requires from cartan**:
1. `Manifold::log` — already in trait ✓
2. `Manifold::norm` — already in trait ✓
3. `ParallelTransport::transport` — already in trait ✓
4. `M::Point: Clone + Send + Sync` — ensure all Point types satisfy this
5. `M::Tangent: Clone + Send + Sync` — same
6. Optional: a `const NAME: &'static str` associated constant on each manifold type
   (or via a separate `Named` marker trait — malliavin can also just use `std::any::type_name::<M>()`)

---

## Numerical Stability Requirements

These are hard requirements for production use (not just tests):

### SPD(N) — Critical

```
// ALWAYS guard before log_spd:
fn safe_log_spd(p: &SMatrix<f64,N,N>, q: &SMatrix<f64,N,N>) -> SMatrix<f64,N,N> {
    // 1. Re-symmetrize inputs: A = (A + A^T) / 2
    // 2. Eigendecompose: P = V diag(λ) V^T
    // 3. Clip λ_i = max(λ_i, 1e-8)   <- critical for near-singular covariances
    // 4. Compute P^{-1/2} = V diag(1/sqrt(λ)) V^T
    // 5. M = P^{-1/2} Q P^{-1/2}
    // 6. Eigendecompose M, clip to [1e-8, ∞), compute logm
    // 7. Result = P^{1/2} logm(M) P^{1/2}
}
```

The near-singular case arises frequently in live trading when assets are highly correlated
(e.g., BTC/ETH correlation → 0.99 during certain regimes). A robust floor is mandatory.

### Grassmann<N,K> — SVD degeneracy

```
// Guard for when U and V represent the same subspace (velocity ≈ 0):
if (1.0 - sigma_min) < 1e-10 { return zero_tangent; }
// Guard for when U and V are orthogonal (maximum distance):
// arccos(sigma_i) → π/2 — result is well-defined but distance is at injectivity_radius
```

### SO(N) — Already implemented, but note

```
// log_SO is undefined when R_prev and R_curr are antipodal (rotation by π).
// In practice, for consecutive time steps this cannot happen in financial data.
// But add a check: if ||log_SO||_F > π - 1e-6, emit a warning and return zero tangent.
```

### General — NaN propagation

All `log` implementations must return `Ok(zero_tangent)` (not NaN/panic) when:
- Input point is at the cut locus of the base point
- Input is not on the manifold (handle via `project_point` first)
- Numerical degeneracy (near-zero singular values, near-zero eigenvalues)

malliavin agents will filter `confidence = 0` votes, but must never receive NaN.

---

## Data Flow (malliavin perspective)

```
MarketDataBus
    │
    ▼
[RollingMarketState]  ← N bars of OHLCV / L2 / vol surface
    │
    ├─ extract covariance matrix Σ_t  ─→  SPDVelocityAgent
    ├─ extract PCA subspace U_t       ─→  GrassmannianAgent
    ├─ extract return distribution    ─→  FisherRaoAgent
    ├─ extract PCA rotation R_t       ─→  LieAlgebraAgent   ← usable NOW
    └─ extract point cloud            ─→  PersistentHomologyAgent

Each agent → AgentVote → VoteCache (TTL per agent) → VoteAggregator → RegimeSignal
```

The `RollingMarketState` struct is malliavin's responsibility.
cartan only needs to provide the manifold math — it never sees raw market data.

---

## Notes for cartan Developer

1. **Start with SPD(N).** It unblocks the highest-value agent and also covers Wasserstein/Bures.

2. **`SO(N)` is already usable.** malliavin can implement `LieAlgebraAgent` right now with
   `SpecialOrthogonal<N>::log` and `norm`. No cartan changes needed.

3. **The `Product<M1, M2>` manifold** will be very useful once SPD and Hyperbolic are done
   (product metric for Fisher-Rao of multivariate Gaussians). Good candidate for after those two.

4. **cartan-dec persistent homology** is independent of the Riemannian geometry stack.
   It can be implemented in parallel with the manifold work. The key dependency is only
   `nalgebra` (for sparse matrices / linear algebra over Z/2Z). No cartan-core traits needed
   for the topological computation itself — only for the smooth geometry in later DEC phases.

5. **`M::Point` must be `Clone + Send + Sync`.** Since malliavin spawns agents on tokio tasks,
   all point and tangent types must be thread-safe. For `SMatrix<f64, N, N>` (nalgebra const generic)
   this should be automatic, but verify during implementation.

6. **No `unsafe` needed** for any of these implementations. All matrix operations go through
   nalgebra's safe API. The eigendecomposition via `nalgebra::linalg::SymmetricEigen` is safe.

7. **Testing**: the existing `manifold_harness.rs` pattern (check_point, check_tangent,
   round-trip exp/log, parallel transport identity) is sufficient. For SPD(N) add:
   - `dist(P, Q) >= 0` (non-negativity)
   - `dist(P, P) = 0` (identity)
   - `dist(P, Q) = dist(Q, P)` (symmetry, holds for affine-invariant metric)
   - `log(P, exp(P, V)) ≈ V` for random V in tangent space (round-trip)
   - `||log(P, Q)||_F = dist(P, Q)` (consistency of norm and distance)
