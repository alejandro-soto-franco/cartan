# cartan-holonomy

Levi-Civita transport, plaquette curvature and covariant jets on a point cloud
that samples a submanifold of `R^D`.

Each point gets a tangent frame of dimension `d` from the principal components
of its neighbourhood. Along an edge of the mutual nearest-neighbour graph the
transport between frames is the Grassmann geodesic map, which in frame
coordinates is the orthogonal Procrustes rotation `polar(T_i^T T_j)`.

## Quantities

- Triangle holonomy. The product of the three transports around a triangle is
  read on the triangle's plane as `K * area`. The total angle over the total
  area estimates sectional curvature.
- Curvature operator. A symmetric operator on so(d) is fitted per point to the
  triangles at that point, with the Ricci and scalar curvature it implies.
- Covariant jet. `A = nabla v` of a vector field, fitted point by point by least
  squares or over all points at once with the connection Laplacian on `End(TM)`
  as regulariser. The global fit is a preconditioned conjugate gradient solve
  through `cartan-matfree`.

Neighbour search is exact: candidate distances come from cache-sized matrix
products, and the best candidates are re-ranked on exact distances.

## Use

```rust
use cartan_holonomy::{Connection, JetOptions, fit_jet, triangle_geometry};

// x: n points of dimension dim, row-major; v: a vector field of the same shape.
let c = Connection::new(&x, n, dim, 5, 40, 20);
let k = triangle_geometry(&c, &x).curvature();
let jet = fit_jet(&c, &x, &v, None, &JetOptions { alpha: 0.3, ..JetOptions::default() });
```

The estimator overestimates the curvature of a round sphere at moderate
sampling density (0.30 for 1/4 at 6,000 points on S^2) and reads a flat plane
as flat. Frames of a smaller dimension than the manifold tilt from point to
point and read curvature on flat data, so `d` should match the intrinsic
dimension.

The Python bindings in `cartan-py` expose `cartan.CloudConnection`.
