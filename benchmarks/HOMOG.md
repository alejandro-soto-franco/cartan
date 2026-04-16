# Homogenisation benchmarks: cartan-homog vs ECHOES

Head-to-head performance and accuracy comparison of `cartan-homog` (Rust) and
ECHOES (Jean-François Barthélémy's C++/Python library, Zenodo DOI
`10.5281/zenodo.14959866`) on a fixed parameter sweep.

## Parameter sweep

190 total cases, each spanning:

| Axis | Values |
|---|---|
| Tensor order | Order2 (3×3 KM, conductivity), Order4 (6×6 KM, elasticity) |
| Scheme | VOIGT, REUSS, DIL, DILD, MT, SC, ASC, MAX, PCW, DIFF |
| Shape | sphere, oblate spheroid (ω=0.1), prolate spheroid (ω=10), penny crack |
| Volume fraction / density | {0.05, 0.10, 0.20, 0.30, 0.40} for spheres, fewer for shapes |
| Contrast | k_matrix = 1, k_inclusion = 5 (Order2); (k, μ) = (72, 32) vs (5, 2) (Order4); k_inclusion = 5e-6 for cracks |

## Results (latest run)

**Timing (156 cases where both libraries produce a value):**

| Metric | cartan | ECHOES | Ratio |
|---|---|---|---|
| Median per-call time | 110 ns | 21,350 ns | **194×** |
| Speedup range (ECHOES / cartan) | — | — | 2× – 888× |

**Accuracy (156 matched cases):**

| Metric | Value |
|---|---|
| Median \|rel err\| | 0.00e+00 (exact agreement) |
| Max \|rel err\| | 5.02e-04 |
| Cases with \|rel err\| > 1e-6 | 10 (all DIFF: RK4 ODE step differs by O(Δt^4)) |

**Hashin-Shtrikman envelope check:** all 60 sphere/Order2 cases from the
interaction-corrected scheme set (`MT, SC, ASC, MAX, PCW, DIFF`) respect the
two-phase HS bounds to machine precision. Voigt / Reuss / Dilute are
explicitly broader and excluded from the envelope check.

**Bug caught by this benchmark:** the v1.2 cartan-homog `AsymmetricSc` scheme
was using `(C_r - C^ASC)` in the contribution numerator where the correct
formula (matching ECHOES and Säevik 2014) uses `(C_r - C_0)`. The HS bound
check flagged the discrepancy (cartan's ASC undershooting HS-lower by
0.01–0.05 at high fraction); the fix landed on the `feature/homog-benchmarks`
branch and brings cartan's ASC into exact agreement with ECHOES.

## Reproducing

```bash
# 1. Build + run the Rust side (cartan).
cargo build --release --bin cartan-bench-homog
./target/release/cartan-bench-homog --out benchmarks/results/homog_cartan.jsonl

# 2. Run the Python side (ECHOES). Requires conda env with echoes wheel:
conda activate echoes-homog       # see cartan-homog-valid/python/requirements.txt
cd benchmarks/python
python bench_homog.py --out ../results/homog_echoes.jsonl

# 3. Head-to-head analysis and figures.
python analyze_homog.py \
    --cartan ../results/homog_cartan.jsonl \
    --echoes ../results/homog_echoes.jsonl \
    --out-dir ../figures/out

# 4. Hashin-Shtrikman envelope check.
python validate_bounds.py \
    --cartan ../results/homog_cartan.jsonl \
    --echoes ../results/homog_echoes.jsonl
```

Outputs:

- `benchmarks/figures/out/homog_timing.png` — log-log scatter of per-call time;
  distribution of ECHOES-to-cartan speedup ratios.
- `benchmarks/figures/out/homog_accuracy.png` — scatter of k_eff[0, 0] values.
- `benchmarks/figures/out/homog_summary.json` — machine-readable summary.

## Real-data references used

- **Hashin-Shtrikman bounds** (Hashin & Shtrikman 1963, Milton 2002 Ch. 23):
  the tightest two-phase isotropic bounds, derived from variational principles.
  Any valid interaction-corrected scheme must fall within them.
- **Analytic Mori-Tanaka formula** (sphere in iso matrix):
  `k_eff = k_0 (1 + 3·f·Δk / (3k_0 + (1-f)·Δk))`. Used in unit tests for
  cartan's MT and as a cross-check against ECHOES.
- **Budiansky-O'Connell 1976** penny-crack effective moduli: informs the
  capstone pipeline's crack-induced anisotropy assertion (separate from this
  timing benchmark).

## Why cartan is faster

The per-call overhead of ECHOES is dominated by its Python ↔ C++ boundary
(argument marshalling, GIL, reference-counted tensor construction). Once in
C++, ECHOES itself is fast — the algorithmic cost is comparable to cartan's.
The 200× median advantage therefore reflects what you gain by writing the
homogenisation loop in Rust end-to-end, not any superior numerical method.
For a one-shot call this is negligible (both take <25 μs); for parameter
sweeps over thousands of RVEs (stochastic ensembles, optimisation loops),
the cartan path is the practical choice.
