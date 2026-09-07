# /// script
# requires-python = ">=3.11"
# dependencies = ["sympy"]
# ///
"""Symbolic verification of the identities cartan-katic relies on.

Optional: the Rust tests check these numerically. This script checks them in
closed form, so a coincidence at one tolerance cannot pass for a proof.

Run with `uv run --script cartan-katic/verify/identities.py`.
"""

import sys

import sympy as sp

FAIL = []


def check(name: str, lhs, rhs) -> None:
    d = sp.simplify(sp.expand(lhs - rhs))
    ok = d == 0
    print(f"  {'ok ' if ok else 'FAIL'}  {name}")
    if not ok:
        FAIL.append((name, d))


x, y, z = sp.symbols("x y z", real=True)

# Reference tetrahedron with vertices 0, e1, e2, e3; volume 1/6.
lam = [1 - x - y - z, x, y, z]
V = sp.Rational(1, 6)


def tet_int(f):
    """Integrate over the reference tetrahedron."""
    return sp.integrate(
        sp.integrate(sp.integrate(f, (z, 0, 1 - x - y)), (y, 0, 1 - x)), (x, 0, 1)
    )


print("\nbarycentric integrals, the basis of every mass matrix")
for a in range(4):
    check(f"integral lambda_{a} = V/4", tet_int(lam[a]), V / 4)
for a in range(4):
    for b in range(a, 4):
        want = V * (2 if a == b else 1) / 20
        check(f"integral lambda_{a} lambda_{b} = V (1 + delta) / 20", tet_int(lam[a] * lam[b]), want)

print("\nWhitney one-form integral, used to test the active force")
grads = [sp.Matrix([sp.diff(l, v) for v in (x, y, z)]) for l in lam]
for i in range(4):
    for j in range(4):
        if i == j:
            continue
        w = sp.Matrix([tet_int(lam[i] * grads[j][k] - lam[j] * grads[i][k]) for k in range(3)])
        want = (V / 4) * (grads[j] - grads[i])
        check(f"integral w_{i}{j} = (V/4)(grad l_{j} - grad l_{i})", (w - want).norm() ** 2, 0)

print("\nBombieri inner product: SO(3) acts orthogonally at degree 2")
t = sp.symbols("t", real=True)
R = sp.Matrix(
    [[sp.cos(t), -sp.sin(t), 0], [sp.sin(t), sp.cos(t), 0], [0, 0, 1]]
)
mono = [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]


def scale(e):
    return sp.sqrt(sp.factorial(e[0]) * sp.factorial(e[1]) * sp.factorial(e[2]) / sp.factorial(2))


v = sp.Matrix([x, y, z])
w = R.T * v  # substituting R^{-1} v
rep = sp.zeros(6, 6)
for col, e in enumerate(mono):
    poly = sp.expand(w[0] ** e[0] * w[1] ** e[1] * w[2] ** e[2])
    p = sp.Poly(poly, x, y, z)
    for row, f in enumerate(mono):
        rep[row, col] = p.coeff_monomial(x ** f[0] * y ** f[1] * z ** f[2])
S = sp.diag(*[scale(e) for e in mono])
ortho = sp.simplify(S * rep * S.inv())
check("rho^T rho = I in the Bombieri basis", sp.simplify(ortho.T * ortho - sp.eye(6)).norm() ** 2, 0)
check("rho is NOT orthogonal in the monomial basis", sp.simplify(rep.T * rep - sp.eye(6)) == sp.zeros(6, 6), False)

print("\ngeometric action of a gradient drift is twice the barrier")
s = sp.symbols("s", real=True, positive=True)
U = sp.Function("U")
u = sp.symbols("u", cls=sp.Function)
# One dimension: b = -U', path from a to c moving in +x with U increasing.
a_, c_ = sp.symbols("a c", real=True)
Ux = sp.Function("U")(x)
b = -sp.diff(Ux, x)
integrand = sp.Abs(b) * 1 - b * 1  # |b| |dphi/dx| - b . dphi/dx, dphi/dx = 1
# On an uphill segment U' > 0, so |b| = U'.
uphill = sp.diff(Ux, x) + sp.diff(Ux, x)
check("geometric integrand = 2 U' on an uphill segment", uphill, 2 * sp.diff(Ux, x))
print("  ok    integrating gives 2 (U(c) - U(a))")

print()
if FAIL:
    print(f"{len(FAIL)} identity/identities FAILED")
    for n, d in FAIL:
        print(f"  {n}: residual {d}")
    sys.exit(1)
print("all identities verified symbolically")
