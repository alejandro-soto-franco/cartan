# cartan-patic

Active liquid crystals of arbitrary rotational symmetry on Riemannian
3-manifolds.

## Order parameter

A p-atic state on `(M³, g)` is a section of the bundle associated to the spin
structure,

    Spin(M) ×_SU(2) (SU(2) / Ĥ),

with `Ĥ ⊂ SU(2)` the binary lift of the molecular point group `H ⊂ SO(3)`.
Since `SU(2)` is simply connected,

    π₁(SO(3)/H) ≅ Ĥ,

so defect charges live in the lift and are non-abelian beyond the cyclic case.
The state is stored as a rotor with amplitudes, `(R, a)`, rather than as a
tensor: the tensor is a single-valued function of the rotor, while the rotor
keeps the lift that the tensor discards.

`Cl⁺(2,0) ≅ ℂ` is abelian and `Cl⁺(3,0) ≅ ℍ` is not, which is why a
two-dimensional treatment does not transfer.

## Invariant order parameters

For each symmetry the order parameter is the harmonic tensor

    T(R, a) = ρ_m(R) T₀(a),

with `T₀` in the `Ĥ`-invariant subspace of `Symᵐ(ℝ³)`, found from the Reynolds
projector

    P = (1/|Ĥ|) Σ_{h ∈ Ĥ} ρ_m(h)

at the lowest degree `m` that separates cosets. Nothing is tabulated per group.

Every matrix is written in the Bombieri (apolar) inner product, in which
`x^a y^b z^c` has squared norm `a! b! c! / m!`. `SO(3)` acts orthogonally there
and does not in the monomial basis, so `P` is a symmetric projector and its
image is an eigenspace.

Measured separating degrees: 1 polar, 2 uniaxial and biaxial, 3 tetrahedral,
4 octahedral, 6 icosahedral.

## Energy

    F = Σ_v [ ½ aᵀA a + ⅓ C(a,a,a) + ¼ (aᵀB a)² ] + w Σ_e |T_u − T_v|²

with `B` symmetric positive definite, checked at construction: an indefinite
quartic form is unbounded below and is rejected rather than integrated. For one
amplitude the bulk is the Landau-de Gennes polynomial term for term.

Writing both terms in `T` makes the functional invariant under the whole
stabiliser, so the gradient along a stabiliser direction vanishes identically
and no projection step is needed.

## Flow

The state moves in `so(3)` and in the amplitudes, with rotor updates a left
multiplication by `exp(−Δt ξ)`, so `|R| = 1` is preserved to machine
precision by construction. The discrete-gradient integrator satisfies

    E(xⁿ⁺¹) − E(xⁿ) = −Δt ‖ḡ‖²

exactly at every step size its solve reaches.

## Hydrodynamics

Velocity is a 1-cochain, pressure a 0-cochain:

    [ η K   B ] [u]   [f]
    [ Bᵀ    0 ] [p] = [0]

with `K = d₁ᵀ M₂ d₁` and `B = M₁ d₀`. Weitzenböck gives
`Δ = ∇*∇ + Ric`, so the Hodge form of the viscous operator includes the
curvature coupling and no separate Ricci term is assembled.

Mass matrices are Whitney, exact from

    ∫_T λ_a λ_b = V (1 + δ_ab) / 20,

so no quadrature error enters and no well-centredness is required.

## Active stress

A rank-2 stress built linearly from a rank-`m` tensor needs exactly `m − 2`
derivatives:

    σ^active = ζ ∇^⊗(m−2) ⊙ T_m,     m ≥ 2
    σ^active = −ζ n ⊗ n,             m = 1

reducing to `−ζQ` at `m = 2`.

**`m` denotes the harmonic degree of the invariant tensor.**
The two agree for the dihedral family, where a `p`-atic has `m = p`, and they
come apart elsewhere: `Dicyclic<1>` has `m = 3`, `Cyclic<8>` has image order 4
and `m = 5` because its degree-4 invariant is also flip-invariant, and the
polyhedral groups have no symmetry order at all while having `m = 3, 4, 6`.
The counting is stated in `m` because `m` exists for every symmetry the crate
supports. `k`-atic and `p`-atic name the same object; `p` is the literature's
letter for it.

The index counting is the same one the passive reactive stress uses, and the
linear coupling of nematics is special to `p = 1, 2`. The literature leaves the
general active form open, so the crate names this as its own construction and
cites only the counting.

The force needs `m − 1` derivatives and piecewise-linear elements supply one,
so `m = 2` is what this element space expresses; higher degrees return an error
rather than a wrong number.

## Defects

The gauge transition on an edge is the element of `π₁` best aligning the two
frames, and the holonomy around a triangle is the charge of the line piercing
it. No threshold and no scalar order parameter.

A charge is a conjugacy class with a base point, never a number: transport
conjugates it and fusion is path dependent. Whether `−1` is trivial is a
property of the symmetry, contractible in `RP²` and not in `SU(2)/Ĥ`.

A line has no endpoints, so every tetrahedron has an even number of pierced
faces.

## Selection

The Freidlin-Wentzell quasipotential uses the geometric action

    S[φ] = ∫ ( |b| |dφ| − b · dφ ),

which is invariant under reparametrisation. The infimum runs over transit time
as well as path, so a fixed-time functional overestimates. For `b = −∇U` this
gives `2 ΔU`; for `b = A x` it gives `xᵀ Σ⁻¹ x` with `A Σ + Σ Aᵀ + 2I = 0`.

Recurrent sets are the strongly connected components of the digraph the drift
induces, and attractors are the components with no outgoing edge.
