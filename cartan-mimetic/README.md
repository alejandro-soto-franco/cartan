# cartan-mimetic

Mimetic Hodge stars for compatible discretisation, consistent and positive
definite at every form degree.

A compatible discretisation represents the exterior derivative exactly and puts
every metric quantity into the inner product on k-cochains. That inner product
is the only choice the method makes, since the codifferential is its adjoint and
the Hodge Laplacian follows from the pair. This crate builds it on one simplex.

## Problem

A diagonal star is cheap, and it is what discrete exterior calculus uses. It has
one unknown per k-face, against one consistency equation per symmetric pair of
constant k-forms:

| n | k | unknowns | equations | diagonal star |
|---|---|----------|-----------|---------------|
| 2 | 0 | 3 | 1 | a 2-parameter family |
| 2 | 1 | 3 | 3 | unique, equal to `cot/2` |
| 3 | 0 | 4 | 1 | a 3-parameter family |
| 3 | 1 | 6 | 6 | unique, sign not guaranteed |
| 3 | 2 | 4 | 6 | overdetermined, generically none |

Two things follow, both of them measured in `tests/star.rs`. On a triangle the
unique diagonal star on 1-forms turns negative exactly when the triangle is
obtuse, so consistency, diagonality and positivity cannot hold at once. On a
tetrahedron no diagonal star is consistent on 2-forms at all, and the unique
1-form star is negative on every tetrahedron of a subdivided cube, which is the
ordinary way to mesh a box.

Giving up diagonality restores both properties.

## Use

```rust
use cartan_mimetic::{diagonal_star, local_star, Simplex};

let tet = Simplex::new(&[
    vec![0.0, 0.0, 0.0],
    vec![1.0, 0.0, 0.0],
    vec![0.3, 0.9, 0.0],
    vec![0.2, 0.4, 1.1],
]);

// Consistent and positive definite on any non-degenerate simplex.
let star = local_star(&tet, 2).unwrap();

// Whether the cheaper diagonal path is available on this cell at all.
assert!(!diagonal_star(&tet, 2).unwrap().is_usable());
```

`Consistency::residual` checks any star, including one built elsewhere, against
the exactness requirement. A simplex embedded in a higher dimension is treated
in its own tangent space, so a triangle in space behaves as it would in the
plane.
