# cartan-complex

Abstract simplicial complexes up to dimension two, with their incidence
operators and the Hodge decomposition of edge flows.

A neighbour graph built on data points has no manifold structure: an edge can
border any number of triangles, and the triangles do not tile a surface.
`cartan-dec` works on manifold meshes. This crate works on the complex itself,
with no geometry assumed.

## Decomposition

An edge flow `f` splits orthogonally as

```text
f = grad + curl + harm,   grad = B1^T phi,   curl = B2 y,   B1 harm = 0,   B2^T harm = 0
```

where `B1` maps edges to vertices and `B2` maps triangles to edges. The gradient
part comes from a sparse Cholesky solve of `B1 B1^T phi = B1 f`, grounded on one
vertex per connected component. The curl part is the projection of `f - grad`
onto the image of `B2`, by conjugate gradient on `B2^T B2`. The harmonic part is
what remains. All three inner products are the identity.

Triangles are enumerated as the 3-cliques of the graph, each once, in
lexicographic order.

## Use

```rust
use cartan_complex::{Complex, HodgeOptions, decompose, triangles_of};

// A filled triangle with a circulating flow on its three edges.
let graph = Complex::new(3, vec![[0, 1], [0, 2], [1, 2]]).unwrap();
let tri = triangles_of(&graph);
let complex = graph.with_triangles(tri).unwrap();

let f = [1.0, -1.0, 1.0];
let d = decompose(&complex, &f, &HodgeOptions::default()).unwrap();

assert!(d.converged);
assert!(d.harm.iter().all(|h| h.abs() < 1e-13));
assert!((d.curl[0] - 1.0).abs() < 1e-13);
```

`Decomposition::converged` reports whether the curl solve reached its tolerance.
A solve that stops on its iteration cap or on stagnation returns its true final
residual and `converged = false`.
