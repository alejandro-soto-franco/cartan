#!/usr/bin/env python3
"""Hashin-Shtrikman envelope validation.

For 2-phase isotropic media, the Hashin-Shtrikman bounds are the tightest
possible bounds on effective conductivity / bulk / shear moduli in terms of
volume fractions and phase properties alone. ANY valid homogenisation scheme
must produce results that fall within these bounds.

This script loads the cartan and ECHOES benchmark outputs and checks that every
sphere-in-iso-matrix case satisfies the HS envelope. A failure here is a
correctness bug; success confirms both libraries respect the fundamental
bounds.

References:
- Hashin, Shtrikman. A variational approach to the theory of the elastic
  behaviour of multiphase materials. J. Mech. Phys. Solids 11 (1963).
- Milton, G. The Theory of Composites. Cambridge U. Press (2002), Ch. 23.
"""
from __future__ import annotations
import argparse, json, pathlib


def hs_conductivity_bounds(k1: float, k2: float, phi2: float) -> tuple[float, float]:
    """Hashin-Shtrikman upper/lower bounds for 2-phase isotropic conductivity.

    Phases: 1 (matrix, fraction 1-phi2) and 2 (inclusion, fraction phi2).
    If k1 < k2: upper = HS with k2-coated, lower = HS with k1-coated.
    """
    k_lo, k_hi = (k1, k2) if k1 <= k2 else (k2, k1)
    f_lo, f_hi = (1 - phi2, phi2) if k1 <= k2 else (phi2, 1 - phi2)
    # Lower bound: coated with lower-conductivity phase outside.
    # k_HS- = k_lo + f_hi / (1 / (k_hi - k_lo) + f_lo / (3 k_lo))
    k_hs_lower = k_lo + f_hi / (1.0 / (k_hi - k_lo) + f_lo / (3.0 * k_lo))
    # Upper bound: coated with higher-conductivity phase outside.
    k_hs_upper = k_hi + f_lo / (1.0 / (k_lo - k_hi) + f_hi / (3.0 * k_hi))
    return k_hs_lower, k_hs_upper


def hs_bulk_bounds(k1: float, mu1: float, k2: float, mu2: float, phi2: float) -> tuple[float, float]:
    """HS bounds for 4th-order bulk modulus of 2-phase isotropic elasticity."""
    # Lower: stiffer phase "coats" softer.  Upper: softer coats stiffer.
    # Standard formula: k_HS = k + phi_other / ((k_other - k)^{-1} + 3 phi / (3k + 4 mu))
    def hs(k_inner, mu_inner, k_outer, phi_outer):
        return k_outer + phi_outer / (1.0 / (k_inner - k_outer) + 3.0 * (1 - phi_outer) / (3.0 * k_outer + 4.0 * mu_outer))
    if k1 < k2:
        lower = hs(k2, mu2, k1, phi2)
        upper = hs(k1, mu1, k2, 1 - phi2)
    else:
        lower = hs(k1, mu1, k2, 1 - phi2)
        upper = hs(k2, mu2, k1, phi2)
    return min(lower, upper), max(lower, upper)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cartan", required=True, type=pathlib.Path)
    ap.add_argument("--echoes", required=True, type=pathlib.Path)
    ap.add_argument("--tol", type=float, default=1.0e-6,
                    help="Relative tolerance beyond the bounds (accepts tiny overshoots from float error)")
    args = ap.parse_args()

    def load(p):
        return [json.loads(l) for l in p.open() if l.strip()]

    cartan = load(args.cartan)
    echoes = load(args.echoes)

    def in_bounds(k_eff: float, lo: float, hi: float, tol: float) -> bool:
        return k_eff >= lo * (1 - tol) and k_eff <= hi * (1 + tol)

    # Only interaction-corrected schemes are bounded by HS. Voigt/Reuss/Dilute
    # are explicitly broader (Voigt = arithmetic mean, Reuss = harmonic mean,
    # Dilute lacks inclusion-inclusion interaction). The bounded set is:
    HS_BOUNDED = {"MT", "SC", "ASC", "MAX", "PCW", "DIFF"}

    violations = []
    total = 0
    for src, rows in (("cartan", cartan), ("echoes", echoes)):
        for r in rows:
            if r.get("k_eff_11") is None: continue
            if r["shape"] != "sphere": continue    # HS bounds hardest for spheres; other shapes escape simple bounds
            if r["order"] != "O2": continue        # O4 bounds more subtle
            if r["scheme"] not in HS_BOUNDED: continue
            phi = r["param"]
            if not (0.0 < phi < 1.0): continue
            lo, hi = hs_conductivity_bounds(1.0, 5.0, phi)
            total += 1
            if not in_bounds(r["k_eff_11"], lo, hi, args.tol):
                violations.append((src, r["scheme"], phi, r["k_eff_11"], lo, hi))

    print(f"Hashin-Shtrikman envelope check (iso matrix k0=1, inclusion k=5):")
    print(f"  {total} sphere / O2 cases checked, {len(violations)} out-of-bounds")
    if violations:
        print(f"\n{'src':<8} {'scheme':<6} {'phi':>6} | {'k_eff':>10} {'HS_lo':>10} {'HS_hi':>10}")
        for src, sc, phi, k, lo, hi in violations:
            print(f"{src:<8} {sc:<6} {phi:>6.3f} | {k:>10.4f} {lo:>10.4f} {hi:>10.4f}")
    else:
        print("  all cases respect the bounds within tolerance")


if __name__ == "__main__":
    main()
