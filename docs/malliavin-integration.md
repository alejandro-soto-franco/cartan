# malliavin <-> cartan Integration Notes

Internal developer reference. Not committed to the repo.

Written by Claude Code session 2026-03-17. Updated 2026-03-17 (Sub-project 2 architecture).
Audience: cartan developer (implementing manifolds), malliavin architect (designing agents).

---

## Sub-project 2 Architecture -- New Requirements (2026-03-17)

The full Sub-project 2 engine design (spec: `malliavin/docs/superpowers/specs/2026-03-17-malliavin-engine-design.md`)
finalizes the geometric architecture. This section supersedes the `ManifoldVelocityAgent` pattern below
for the runtime engine. The individual agent manifold catalog (SPD, Grassmann, Fisher-Rao, etc.) remains valid.

### Core geometric model

All regime state lives on **Sym+(N)** with the affine-invariant metric. Regime state is a point `p` on
this manifold. Agents emit **tangent vector votes** `v in Sym(N) = T_p(Sym+(N))`. The `ManifoldThread`
applies one exponential map step per batch of votes:

```
p <- Exp_p(eta * sum_i w_i * v_i)
```

There is one `ManifoldState` per `CadenceClass` (Hft, Intraday, Swing, Macro), each with its own
step size `eta_cadence`. A `MetaRegimeDetector` fuses the four per-cadence centroids into a global
centroid via a weighted Frechet mean -- all on the same Sym+(N).

### Minimum required cartan interface (Sub-project 2)

These five functions are the exact surface area the runtime engine needs. Priority: **P0-critical**.

| Function | Signature | Formula |
|---|---|---|
| `exp_map` | `(p: SymPos, v: Sym) -> SymPos` | `p^(1/2) * exp(p^(-1/2) v p^(-1/2)) * p^(1/2)` |
| `log_map` | `(p: SymPos, q: SymPos) -> Sym` | `p^(1/2) * log(p^(-1/2) q p^(-1/2)) * p^(1/2)` |
| `geodesic_distance` | `(p: SymPos, q: SymPos) -> f64` | `||log(p^(-1/2) q p^(-1/2))||_F` |
| `frechet_mean` | `(pts: &[SymPos], w: &[f64]) -> SymPos` | Iterative Karcher flow |
| `frechet_mean_gradient_step` | `(p: SymPos, samples: &[SymPos], eta: f64) -> SymPos` | One Karcher step (prototype calibration) |

All five operate under the affine-invariant metric on Sym+(N).
Per-call cost: two Cholesky factorizations + one symmetric eigendecomposition = O(N^3).
For N <= 20: sub-microsecond on modern hardware.

**Status: Spd<N> is implemented.** All five functions are available via `cartan_manifolds::spd::Spd<N>`.

### How agents interact with the new model

Agents emit `TangentVote`:

```rust
pub struct TangentVote {
    pub v: SymmetricMatrix,  // element of Sym(N) -- the tangent direction
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
- `SPDVelocityAgent`: native tangent IS already `Sym(N)` -- no projection needed
- `GrassmannianAgent`: project `Gr(k,N)` tangent (N x k matrix) to `Sym(N)` via `v * v^T`
- `LieAlgebraAgent`: use `Omega * Omega^T` (symmetric positive semidefinite) as the projection
- `FisherRaoAgent`: embed H^2 velocity to `Sym(2)` via the Fisher information matrix

### Voronoi dispatch -- what cartan needs to support

Regime dispatch uses Voronoi cells on Sym+(N) around prototype points. The `ManifoldThread`
calls `geodesic_distance(p, prototype_j)` after each `Exp_p` step to check cell membership.
The hysteresis band requires two distance evaluations per active regime per bar.

`frechet_mean` is called by `MetaRegimeDetector` once per Intraday tick (4 inputs, weighted).
`frechet_mean_gradient_step` is called offline during prototype calibration only.

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

## Manifold Catalog -- malliavin Requirements

Priority legend: **P0** = blocks first backtest run, **P1** = needed for v1 agent suite, **P2** = research.

### P0 -- SymmetricPositiveDefinite<N> (SPD(N)) -- DONE

**Status: Implemented as `Spd<N>` in cartan-manifolds.**

**malliavin agent**: `SPDVelocityAgent`
- **Point type**: `SMatrix<f64, N, N>` (symmetric, all eigenvalues > 0)
- **Embedding**: rolling N x N realized covariance matrix of log-returns (e.g. N=5 for BTC,ETH,SPX,VIX,DXY)
- **Signal**: `log_spd(Sigma_{t-1}, Sigma_t)` is the covariance velocity in the affine-invariant metric
- **Regime**: high `||velocity||` = rapid correlation shift = regime transition

**Required metric**: affine-invariant (Bures/power):
```
log_spd(P, Q) = P^(1/2) * logm(P^(-1/2) Q P^(-1/2)) * P^(1/2)
dist(P, Q)^2  = sum_i log(lambda_i(P^(-1)Q))^2
```
where `lambda_i` are generalized eigenvalues.

**Numerical notes**:
- `P^(-1/2)` via Cholesky: `L = cholesky(P)`, then `P^(-1/2) = L^{-T}`
- `logm` via symmetric eigendecomposition: `A = V diag(lambda) V^T`, `logm(A) = V diag(log(lambda)) V^T`
- Guard: clip eigenvalues to `[1e-8, inf)` before `log` to prevent NaN from near-singular matrices
- `project_point`: re-symmetrize and clip eigenvalues: `(A + A^T)/2`, eigenvalue floor `1e-8`

---

### P0 -- Grassmann<N, K> (G(k,n)) -- DONE

**Status: Implemented as `Grassmann<N, K>` in cartan-manifolds.**

**malliavin agent**: `GrassmannianAgent`
- **Point type**: equivalence class of N x K matrices with orthonormal columns (a k-dimensional subspace of R^N)
- **Embedding**: top-k eigenvectors of rolling covariance (the signal subspace)
- **Signal**: `log_gr(U_{t-1}, U_t)` measures how fast the dominant signal subspace is rotating
- **Regime**: rapid subspace rotation = unstable eigenstructure = transitioning market

**Required metric**: canonical metric on G(k,n) via the horizontal lift.
The Riemannian HVP uses the Boumal 2022 formula (horizontal representation):
```
Hess f(Q)[V] = proj_Q(ehvp(V)) - V * sym(Q^T * G)
```
where `G = grad_f` (Euclidean gradient, N x K), `sym(A) = (A + A^T)/2` (K x K).

**Numerical notes**:
- `project_point(A)` = QR factorization, take Q (orthonormalize columns)
- Guard conjugate-pair eigenvalues in `logm` path
- `dist(U,V) = ||[theta_1,...,theta_k]||_2` where `theta_i = arccos(sigma_i(U^T V))`

---

### P1 -- Statistical Manifold (Fisher-Rao)

**Not yet implemented.** Requires `Hyperbolic<N>` (on roadmap).

**malliavin agent**: `FisherRaoAgent`
- **Observation scale**: 5-minute bars, rolling window of ~50 bars
- **Point type**: parameters of a probability distribution (mean, variance, shape parameters)
- **Embedding**: fit a distribution to rolling return samples -> parameter vector = point on manifold
- **Signal**: geodesic velocity in Fisher-Rao metric = how fast the return distribution is changing
- **Regime**: distributional divergence is a leading indicator of vol regime shifts

**For malliavin v1, only the multivariate Gaussian case is needed:**

The statistical manifold of `N(mu, Sigma)` with N-dimensional mu and SPD Sigma is isometric to:
`G = (R^N) x SPD(N)` with the Fisher information metric.

For **univariate** Gaussian N(mu, sigma^2):
```
Point: (mu, sigma) in R x R+   (upper half-plane)
metric: (dmu^2 + 2 dsigma^2) / sigma^2
```
This is the **hyperbolic plane H^2** with curvature K = -1/2.

**cartan implementation needed**: `Hyperbolic<N>` (on roadmap, not yet done).
For multivariate case: `Product<EuclideanN, Spd<N>>` with Fisher-Rao metric override.

---

### P1 -- SO(N) Lie Algebra Velocity -- DONE

**Status: SpecialOrthogonal<N> is fully implemented and usable now.**

**malliavin agent**: `LieAlgebraAgent`
- **Point type**: `SMatrix<f64, N, N>` orthogonal with det=1
- **Embedding**: rolling PCA rotation matrix R_t (the factor rotation in a PCA of returns)
- **Signal**: `log_so(R_{t-1}, R_t) in so(N)` = skew-symmetric velocity; Frobenius norm = rotation speed
- **Regime**: fast rotation of factor loadings = factor structure instability

**Signal extraction**:
```
Omega = log_SO(R_prev, R_curr)     // skew-symmetric N x N matrix
speed = ||Omega||_F / sqrt(2)      // normalized rotation rate (rad/step)
axis  = Omega[i,j] for (i,j) s.t. |Omega[i,j]| is max  // dominant rotation axis
```

---

### P1 -- Persistent Homology via k-Simplex (cartan-dec)

**Requires cartan-dec persistent homology phase** (not yet implemented -- see roadmap.md).

**malliavin agent**: `PersistentHomologyAgent`
- **Observation scale**: 1-hour bars, rolling window of ~100 price vectors
- **Construction**: Vietoris-Rips filtration on point cloud of rolling return vectors
- **Signal**: Betti numbers beta_0, beta_1, beta_2 as epsilon increases
  - beta_0 = number of connected components (high -> assets decorrelating)
  - beta_1 = number of independent 1-cycles (cyclic arbitrage patterns)
  - beta_2 = 2-cycles (bubbles/voids in distribution)
- **Regime**: rapid change in (beta_0, beta_1, beta_2) profile = topological phase transition

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

This is pure combinatorial topology -- no Riemannian metric needed.
It can be implemented as a standalone `cartan-dec` feature flag, independently of the DEC solver.

**Complexity note**: Vietoris-Rips is O(n^2 log n) for graph, O(n^3) for full complex.
For `n = 100` points in `N = 5` dimensions: manageable at 1-hour cadence.
Use sparse matrix storage (`nalgebra_sparse` or custom CSR) for boundary operators.

---

### P2 -- IV Surface as Jet Bundle J^1

**Research-grade. Not needed for v1.**

**malliavin agent**: `IVSurfaceAgent`
- **Manifold**: The implied volatility surface sigma(K, T) as a section of the jet bundle J^1(R^2 -> R)
- **Signal**: velocity of the vol surface in the J^1 metric = how fast the skew/term structure is moving
- **Regime**: rapid skew flattening = vol regime transition (risk-off to complacency or vice versa)

Model the vol surface as a parametric curve in a low-dimensional parameter space:
```
sigma(K,T) ~= SVI(a, b, rho, m, sigma_min; K, T)    // Gatheral SVI parametrisation
Point = (a, b, rho, m, sigma_min) in R^5
```
Then the "manifold" is R^5 with the Fisher information metric of the SVI family.
Same construction as FisherRao applied to the option pricing model.
Defer until after `FisherRaoAgent` is validated.

---

### P2 -- Wasserstein/Bures

**Research-grade. Blocked on SPD(N) -- but SPD is now done.**

**malliavin agent**: `WassersteinAgent`
- **Manifold**: (Gaussian measures, Wasserstein-2 metric) = Bures metric on SPD(N)
- **Bures distance**: `d_Bures(P, Q)^2 = tr(P) + tr(Q) - 2 tr((P^(1/2) Q P^(1/2))^(1/2))`
- **Note**: The Bures metric is equivalent to the affine-invariant metric on SPD(N) up to rescaling.
  Implement `WassersteinAgent` on top of `SPDVelocityAgent` using the Bures distance formula directly.

---

## Implementation Priority for cartan

Based on malliavin's schedule:

| Priority | Manifold | Blocks | Status |
|----------|----------|--------|--------|
| **P0** | `Spd<N>` | `SPDVelocityAgent`, `WassersteinAgent` | **DONE** |
| **P0** | `Grassmann<N, K>` | `GrassmannianAgent` | **DONE** |
| **P1** | `Hyperbolic<N>` | `FisherRaoAgent` (univariate) | not started |
| **P1** | `SpecialOrthogonal<N>` log/norm | `LieAlgebraAgent` | **DONE** |
| **P1** | cartan-dec Phase 1-2 | `PersistentHomologyAgent` | not started |
| **P2** | `Product<M1, M2>` | `FisherRaoAgent` (multivariate) | not started |
| **P2** | J^1 / SVI manifold | `IVSurfaceAgent` | not started |

---

## The ManifoldVelocityAgent Pattern -- Contract for cartan

This is the exact Rust trait that malliavin will use. cartan must satisfy this interface:

```rust
/// The minimal bound required for ManifoldVelocityAgent<M>.
/// All of these are already in cartan-core -- this is the usage contract.
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
            agent_id:             M::NAME,
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
1. `Manifold::log` -- already in trait
2. `Manifold::norm` -- already in trait
3. `ParallelTransport::transport` -- already in trait
4. `M::Point: Clone + Send + Sync` -- ensure all Point types satisfy this
5. `M::Tangent: Clone + Send + Sync` -- same
6. Optional: a `const NAME: &'static str` associated constant on each manifold type
   (or via `std::any::type_name::<M>()`)

---

## Numerical Stability Requirements

These are hard requirements for production use (not just tests):

### SPD(N) -- Implemented; guards are in place

```
// ALWAYS guard before log_spd:
fn safe_log_spd(p: &SMatrix<f64,N,N>, q: &SMatrix<f64,N,N>) -> SMatrix<f64,N,N> {
    // 1. Re-symmetrize inputs: A = (A + A^T) / 2
    // 2. Eigendecompose: P = V diag(lambda) V^T
    // 3. Clip lambda_i = max(lambda_i, 1e-8)   <- critical for near-singular covariances
    // 4. Compute P^(-1/2) = V diag(1/sqrt(lambda)) V^T
    // 5. M = P^(-1/2) Q P^(-1/2)
    // 6. Eigendecompose M, clip to [1e-8, inf), compute logm
    // 7. Result = P^(1/2) logm(M) P^(1/2)
}
```

### Grassmann<N,K> -- Implemented; SVD degeneracy guards are in place

```
// Guard for when U and V represent the same subspace (velocity ~= 0):
if (1.0 - sigma_min) < 1e-10 { return zero_tangent; }
```

### SO(N) -- Implemented; note

```
// log_SO is undefined when R_prev and R_curr are antipodal (rotation by pi).
// In practice this cannot happen for consecutive financial time steps.
// Add a check: if ||log_SO||_F > pi - 1e-6, emit a warning and return zero tangent.
```

### General -- NaN propagation

All `log` implementations must return `Ok(zero_tangent)` (not NaN/panic) when:
- Input point is at the cut locus of the base point
- Input is not on the manifold (handle via `project_point` first)
- Numerical degeneracy (near-zero singular values, near-zero eigenvalues)

malliavin agents will filter `confidence = 0` votes, but must never receive NaN.

---

## Data Flow (malliavin perspective)

```
MarketDataBus
    |
    v
[RollingMarketState]  <- N bars of OHLCV / L2 / vol surface
    |
    +- extract covariance matrix Sigma_t  ->  SPDVelocityAgent     (DONE)
    +- extract PCA subspace U_t           ->  GrassmannianAgent    (DONE)
    +- extract return distribution        ->  FisherRaoAgent       (blocked on Hyperbolic<N>)
    +- extract PCA rotation R_t           ->  LieAlgebraAgent      (DONE)
    +- extract point cloud                ->  PersistentHomologyAgent (blocked on cartan-dec phase 2)

Each agent -> AgentVote -> VoteCache (TTL per agent) -> VoteAggregator -> RegimeSignal
```

The `RollingMarketState` struct is malliavin's responsibility.
cartan only needs to provide the manifold math -- it never sees raw market data.

---

## Notes for cartan Developer

1. **SPD(N) and Grassmann<N,K> are done.** malliavin can implement `SPDVelocityAgent` and
   `GrassmannianAgent` now.

2. **`SO(N)` is already usable.** malliavin can implement `LieAlgebraAgent` right now with
   `SpecialOrthogonal<N>::log` and `norm`. No cartan changes needed.

3. **Next priority: `Hyperbolic<N>`.** This unblocks `FisherRaoAgent` (univariate Gaussian case)
   and is on the v0.1 roadmap.

4. **The `Product<M1, M2>` manifold** will be very useful once Hyperbolic is done
   (product metric for Fisher-Rao of multivariate Gaussians).

5. **cartan-dec persistent homology** is independent of the Riemannian geometry stack.
   It can be implemented in parallel with manifold work. The key dependency is only
   `nalgebra` (for sparse matrices / linear algebra over Z/2Z). No cartan-core traits needed
   for the topological computation itself.

6. **`M::Point` must be `Clone + Send + Sync`.** Since malliavin spawns agents on tokio tasks,
   all point and tangent types must be thread-safe. For `SMatrix<f64, N, N>` (nalgebra const generic)
   this is automatic, but verify during implementation.

7. **No `unsafe` needed** for any of these implementations. All matrix operations go through
   nalgebra's safe API.

8. **Testing**: the existing `manifold_harness.rs` pattern is sufficient. For SPD(N), additionally test:
   - `dist(P, Q) >= 0` (non-negativity)
   - `dist(P, P) = 0` (identity)
   - `dist(P, Q) = dist(Q, P)` (symmetry, holds for affine-invariant metric)
   - `log(P, exp(P, V)) ~= V` for random V in tangent space (round-trip)
   - `||log(P, Q)||_F = dist(P, Q)` (consistency of norm and distance)
