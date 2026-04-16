# Berea sandstone real-data benchmark

Head-to-head `cartan-homog` vs ECHOES on the Berea sandstone microstructure, using parameters from published literature and cross-checking against published measurements and FEM reference values.

## Inputs (from published literature)

| Quantity | Value | Source |
|---|---|---|
| Porosity ϕ | 0.195 | Andrä et al. 2013 (3-team segmentation mean, range 0.184–0.209) |
| Mineral bulk modulus K₀ | 39.75 GPa | Zimmerman 1991 |
| Mineral shear modulus G₀ | 31.34 GPa | Zimmerman 1991 |
| Pore fluid | dry (vacuum) | K_pore = G_pore ≈ 0 |

## Cross-check targets

| Target | Value | Source |
|---|---|---|
| Measured drained K @ 10 MPa effective stress | **6.6 GPa** | Hart 1995 (laboratory) |
| FEM-computed K on μCT image @ ϕ=0.22 | **≈ 13.0 GPa** | Arns et al. 2002 |
| Hashin-Shtrikman upper bound (spherical pores) | 26.99 GPa | Hashin-Shtrikman 1963, analytic |
| Hashin-Shtrikman lower bound (spherical pores, K_pore≈0) | ≈ 0 | Hashin-Shtrikman 1963, analytic |

## Results

**Head-to-head (cartan-homog vs ECHOES, Order4 elasticity):**

| Scheme | cartan K (GPa) | ECHOES K (GPa) | Δ | cartan time (ns) | ECHOES time (ns) | Speedup |
|---|---|---|---|---|---|---|
| VOIGT | 31.999 | 31.999 | 0 | 70 | 3,692 | 53× |
| REUSS | 0.000 | 0.000 | 0 | 571 | 6,152 | 11× |
| DIL | 24.625 | 24.625 | 0 | 431 | 19,011 | 44× |
| DILD | 28.794 | 28.794 | 0 | 932 | 23,624 | 25× |
| MT | 26.992 | 26.992 | 0 | 671 | 25,904 | 39× |
| SC | 24.524 | 24.524 | 0 | 97,943 | 392,399 | 4× |
| MAX | 26.992 | 26.992 | 0 | 862 | 26,389 | 31× |
| DIFF | 26.001 | 25.999 | +0.008% | 154,109 | 4,321,274 | **28×** |

**Every interaction-corrected scheme respects the HS envelope.** VOIGT (arithmetic mean) overshoots as expected; REUSS (harmonic mean) collapses to zero on dry voids as expected; DILD overshoots at this porosity because the dilute-stress dual is explicitly broader than HS.

## Interpretation: why the predictions overshoot the measured 6.6 GPa

Cartan-homog and ECHOES both predict `K_eff ≈ 24–27 GPa` for dry Berea at ϕ=0.195 under the assumption of **spherical pores in isotropic matrix**. Laboratory-measured drained bulk modulus is **6.6 GPa** (Hart 1995, 10 MPa effective stress).

This ~4× gap is a well-known phenomenon in rock physics and is **not a homogenisation error**:

1. Real Berea has grain-contact compliance, cemented boundaries with variable stiffness, and microcracks aligned along grain boundaries. These features act like oblate voids (aspect ≪ 1) with a much greater stiffness-reducing effect than isolated spheres.
2. At 10 MPa effective stress, some grain contacts remain partially open. Full contact closure requires effective stress > ~50 MPa, at which point measured K_drained rises toward the MT prediction.
3. Arns et al. 2002's FEM on actual μCT image voxels gives **≈ 13 GPa** at ϕ=0.22 — already far below the HS upper of ~27 GPa, because FEM resolves the actual grain-contact geometry that MT cannot.

**The "right" cartan-homog benchmark for real Berea stiffness would require**:
- Either a crack-density augmentation (Budiansky-O'Connell style) with penny-crack phases at ρ ≈ 0.3–0.5 on top of the sphere-pore MT,
- Or full-field voxel input (which v1.2 `FullField` supports for centred single inclusions but not yet for μCT-scanned voxel arrays — that's a v1.3 item).

## Reproducing

```bash
cargo build --release --bin cartan-bench-berea
./target/release/cartan-bench-berea benchmarks/results/berea_cartan.jsonl

conda activate echoes-homog
cd benchmarks/python
python bench_berea_echoes.py --out ../results/berea_echoes.jsonl
```

## Real-data references

- [Andrä et al. 2013 — Digital rock physics benchmarks Part II (Computers & Geosciences)](https://www.sciencedirect.com/science/article/abs/pii/S0098300412003172) — porosity 0.184–0.209, FEM/LB/FD cross-comparisons on Berea micro-CT.
- [Arns et al. 2002 — Computation of linear elastic properties from micro-CT images](https://www.sciencedirect.com/science/article/abs/pii/S1365160901000578) — FEM K ≈ 13 GPa at φ=0.22.
- [Hart & Wang 1995 — Complete poroelastic moduli of Berea and Indiana](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/95JB01242) — measured drained K_s = 6.6 GPa at 10 MPa.
- [Zimmerman 1991 — Compressibility of sandstones](https://www.sciencedirect.com/book/9780444898005) — mineral moduli K₀ = 39.75 GPa, G₀ = 31.34 GPa.
- [Hashin-Shtrikman 1963](https://www.mat.uniroma2.it/~braides/ICTP93/ICTP93Gibiansky.pdf) — variational bounds on two-phase isotropic composites.
