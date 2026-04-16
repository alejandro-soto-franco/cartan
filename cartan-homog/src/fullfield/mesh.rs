//! Structured tet mesh of the unit cube. Each voxel is split into 6 tets along
//! a shared diagonal (Kuhn triangulation), producing a conforming mesh with
//! matched face discretisations on opposite cube faces — the structural
//! precondition for periodic BCs.

use crate::error::HomogError;
use alloc::vec::Vec;
use cartan_dec::Mesh;
use cartan_manifolds::Euclidean;
use nalgebra::Vector3;

#[derive(Clone, Debug)]
pub struct PeriodicCubeMeshBuilderOpts {
    pub resolution: usize,
    pub refine_depth: usize,
}

impl Default for PeriodicCubeMeshBuilderOpts {
    fn default() -> Self { Self { resolution: 8, refine_depth: 0 } }
}

pub struct PeriodicCubeMeshBuilder {
    pub opts: PeriodicCubeMeshBuilderOpts,
}

/// The 6 tetrahedra of the Kuhn triangulation of a unit cube, each given as
/// local-vertex indices into the 8-vertex cube. The cube's vertices are
/// ordered lexicographically: v[i+2j+4k] = (i, j, k) for i, j, k in {0, 1}.
const KUHN_TETS: [[usize; 4]; 6] = [
    [0, 1, 3, 7],
    [0, 1, 5, 7],
    [0, 2, 3, 7],
    [0, 2, 6, 7],
    [0, 4, 5, 7],
    [0, 4, 6, 7],
];

impl PeriodicCubeMeshBuilder {
    pub fn new(opts: &PeriodicCubeMeshBuilderOpts) -> Self {
        Self { opts: opts.clone() }
    }

    /// Build a 3D tet mesh of the unit cube with resolution N per side.
    /// Returns `(mesh, tet_barycenters)` — the second output is the
    /// per-tet barycentre used downstream for phase tagging and voxel queries.
    pub fn build(&self) -> Result<(Mesh<Euclidean<3>, 4, 3>, Vec<Vector3<f64>>), HomogError> {
        let n = self.opts.resolution;
        if n < 1 {
            return Err(HomogError::Mesh(alloc::string::String::from(
                "resolution must be >= 1")));
        }
        let h = 1.0 / (n as f64);

        // Vertices on the (N+1)^3 grid.
        let v_count = (n + 1) * (n + 1) * (n + 1);
        let mut vertices: Vec<Vector3<f64>> = Vec::with_capacity(v_count);
        for k in 0..=n { for j in 0..=n { for i in 0..=n {
            vertices.push(Vector3::new(i as f64 * h, j as f64 * h, k as f64 * h));
        }}}

        let vid = |i: usize, j: usize, k: usize| -> usize {
            k * (n + 1) * (n + 1) + j * (n + 1) + i
        };

        // Simplices (tets): 6 per voxel via the Kuhn triangulation.
        let mut simplices: Vec<[usize; 4]> = Vec::with_capacity(6 * n * n * n);
        let mut barycenters: Vec<Vector3<f64>> = Vec::with_capacity(6 * n * n * n);
        for kk in 0..n { for jj in 0..n { for ii in 0..n {
            let corners = [
                vid(ii,     jj,     kk    ),  // 0
                vid(ii + 1, jj,     kk    ),  // 1
                vid(ii,     jj + 1, kk    ),  // 2
                vid(ii + 1, jj + 1, kk    ),  // 3
                vid(ii,     jj,     kk + 1),  // 4
                vid(ii + 1, jj,     kk + 1),  // 5
                vid(ii,     jj + 1, kk + 1),  // 6
                vid(ii + 1, jj + 1, kk + 1),  // 7
            ];
            for tet_local in KUHN_TETS {
                let tet = [
                    corners[tet_local[0]], corners[tet_local[1]],
                    corners[tet_local[2]], corners[tet_local[3]],
                ];
                let bary = (vertices[tet[0]] + vertices[tet[1]]
                          + vertices[tet[2]] + vertices[tet[3]]) / 4.0;
                simplices.push(tet);
                barycenters.push(bary);
            }
        }}}

        let manifold = Euclidean::<3>;
        let mesh = Mesh::<Euclidean<3>, 4, 3>::from_simplices_generic(
            &manifold, vertices, simplices);
        Ok((mesh, barycenters))
    }
}

/// Identify vertices on each face of the unit cube (for Dirichlet / periodic BCs).
/// Returns (boundary_vertex_indices, interior_vertex_indices).
pub fn partition_boundary(vertices: &[Vector3<f64>], tol: f64) -> (Vec<usize>, Vec<usize>) {
    let mut boundary = Vec::new();
    let mut interior = Vec::new();
    for (i, v) in vertices.iter().enumerate() {
        let on_bdy = v.x < tol || v.x > 1.0 - tol
                  || v.y < tol || v.y > 1.0 - tol
                  || v.z < tol || v.z > 1.0 - tol;
        if on_bdy { boundary.push(i); } else { interior.push(i); }
    }
    (boundary, interior)
}

/// Build the periodic vertex-pair list for a structured (N+1)^3 grid. Each pair
/// is `(slave, master)` with `slave > master`; the cell-problem solve treats the
/// slave as a duplicate of master, eliminating it from the DOF set.
///
/// Pairings:
/// - face pair x=0 <-> x=1 (for j, k interior to their faces)
/// - face pair y=0 <-> y=1
/// - face pair z=0 <-> z=1
/// - edge and corner vertices are collapsed to a single representative so each
///   periodic orbit maps to exactly one master.
pub fn periodic_pairs_structured(resolution: usize) -> Vec<(usize, usize)> {
    let n = resolution;
    let vid = |i: usize, j: usize, k: usize| -> usize {
        k * (n + 1) * (n + 1) + j * (n + 1) + i
    };
    let mut pairs = Vec::new();
    // For each vertex with any coordinate equal to N, map it to the equivalent
    // vertex with that coordinate collapsed to 0. Process in a canonical order
    // so that edge/corner vertices collapse transitively down to the master.
    for k in 0..=n { for j in 0..=n { for i in 0..=n {
        let mi = if i == n { 0 } else { i };
        let mj = if j == n { 0 } else { j };
        let mk = if k == n { 0 } else { k };
        if (mi, mj, mk) != (i, j, k) {
            pairs.push((vid(i, j, k), vid(mi, mj, mk)));
        }
    }}}
    pairs
}

#[cfg(test)]
mod periodic_tests {
    use super::*;

    #[test]
    fn periodic_pairs_n2_counts() {
        // (N+1)^3 = 27; masters = N^3 = 8; slaves = 27 - 8 = 19.
        let pairs = periodic_pairs_structured(2);
        assert_eq!(pairs.len(), 19);
        // All slaves distinct.
        let mut slaves: Vec<usize> = pairs.iter().map(|&(s, _)| s).collect();
        slaves.sort(); slaves.dedup();
        assert_eq!(slaves.len(), 19);
    }

    #[test]
    fn periodic_masters_have_all_coords_less_than_n() {
        let n = 4;
        let pairs = periodic_pairs_structured(n);
        for &(_s, m) in &pairs {
            let i = m % (n + 1);
            let j = (m / (n + 1)) % (n + 1);
            let k = m / ((n + 1) * (n + 1));
            assert!(i < n && j < n && k < n, "master {m} has coord = N");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n4_builder_produces_125_vertices_and_384_tets() {
        let b = PeriodicCubeMeshBuilder::new(&PeriodicCubeMeshBuilderOpts { resolution: 4, refine_depth: 0 });
        let (mesh, bary) = b.build().unwrap();
        assert_eq!(mesh.n_vertices(), 5 * 5 * 5);
        assert_eq!(mesh.n_simplices(), 6 * 4 * 4 * 4);
        assert_eq!(bary.len(), mesh.n_simplices());
    }

    #[test]
    fn boundary_partition_counts_are_correct_for_n2() {
        let b = PeriodicCubeMeshBuilder::new(&PeriodicCubeMeshBuilderOpts { resolution: 2, refine_depth: 0 });
        let (_mesh, _bary) = b.build().unwrap();
        // (N+1)^3 = 27 vertices total; (N-1)^3 = 1 interior; boundary = 26.
        let n = 2usize;
        let verts: Vec<Vector3<f64>> = {
            let h = 1.0 / (n as f64);
            let mut v = Vec::new();
            for k in 0..=n { for j in 0..=n { for i in 0..=n {
                v.push(Vector3::new(i as f64 * h, j as f64 * h, k as f64 * h));
            }}}
            v
        };
        let (bdy, int) = partition_boundary(&verts, 1e-12);
        assert_eq!(bdy.len(), 26);
        assert_eq!(int.len(), 1);
    }
}
