//! Cell-problem assembly: piecewise-constant-K Poisson on a tet mesh.
//!
//! Solves the single-direction corrector equation
//!     ∇·(K(x)(∇χ_i + e_i)) = 0 on the unit cube,  χ_i = 0 on ∂(unit cube)
//! via standard P1 finite elements. The Dirichlet-zero BC is a simplification
//! of the periodic cell problem; for the symmetric cases used in v1.1
//! validation (matrix-only, matrix+centred sphere) it gives the same
//! effective tensor as periodic BCs because the corrector is odd around
//! the inclusion centre. Full periodic BCs are v1.2 work.

use crate::error::HomogError;
use alloc::vec::Vec;
use cartan_dec::Mesh;
use cartan_manifolds::Euclidean;
use nalgebra::{Matrix3, Vector3};
use sprs::{CsMat, TriMat};

/// Per-tet conductivity (isotropic scalar K) plus precomputed barycentric gradients
/// and volume. Built once per mesh, reused across directions.
pub struct TetData {
    pub grads: Vec<[Vector3<f64>; 4]>,  // gradient of each P1 basis function on each tet
    pub volumes: Vec<f64>,
    pub k_per_tet: Vec<f64>,             // isotropic conductivity per tet
}

/// Precompute per-tet geometry from the mesh vertices + tet indices.
pub fn build_tet_data(
    mesh: &Mesh<Euclidean<3>, 4, 3>,
    k_per_tet: Vec<f64>,
) -> Result<TetData, HomogError> {
    let nt = mesh.n_simplices();
    if k_per_tet.len() != nt {
        return Err(HomogError::Solver(alloc::format!(
            "build_tet_data: k_per_tet len {} != n_simplices {nt}", k_per_tet.len())));
    }
    let mut grads = Vec::with_capacity(nt);
    let mut volumes = Vec::with_capacity(nt);
    for s in 0..nt {
        let tet = mesh.simplices[s];
        let v: [Vector3<f64>; 4] = [
            mesh.vertices[tet[0]], mesh.vertices[tet[1]],
            mesh.vertices[tet[2]], mesh.vertices[tet[3]],
        ];
        // Reference matrix [v1-v0, v2-v0, v3-v0] (3x3).
        let jac = Matrix3::from_columns(&[v[1] - v[0], v[2] - v[0], v[3] - v[0]]);
        let det = jac.determinant();
        let vol = det.abs() / 6.0;
        if vol < 1e-18 {
            return Err(HomogError::Mesh(alloc::format!("degenerate tet {s}: vol={vol}")));
        }
        // Barycentric gradients: grad(λ_0) = -(grad(λ_1) + grad(λ_2) + grad(λ_3)),
        // grad(λ_{1..3}) = rows of (jac^{-T}) transposed, i.e., columns of jac^{-1}.
        let jinv = jac.try_inverse().ok_or(HomogError::Mesh(alloc::format!("tet {s} jacobian singular")))?;
        let jit: Matrix3<f64> = jinv.transpose();
        // grad(λ_1), λ_2, λ_3 are rows of jit (the gradient operator).
        // Actually: grad(λ_k) in physical coordinates is the row of jit (since λ_k = ξ_k
        // where ξ = J^{-1}(x - v0) for k=1..3, giving ∂λ_k/∂x = (J^{-T}) row k-1).
        let g1 = jit.column(0).into_owned();
        let g2 = jit.column(1).into_owned();
        let g3 = jit.column(2).into_owned();
        let g0 = -(g1 + g2 + g3);
        grads.push([g0, g1, g2, g3]);
        volumes.push(vol);
    }
    Ok(TetData { grads, volumes, k_per_tet })
}

/// Assemble the global sparse stiffness matrix K-weighted Laplacian
///     A[i, j] = Σ_tets  K_tet · vol_tet · (∇φ_i · ∇φ_j)
/// Dirichlet BCs handled downstream by row/column elimination on boundary vertices.
pub fn assemble_stiffness(
    mesh: &Mesh<Euclidean<3>, 4, 3>, td: &TetData,
) -> CsMat<f64> {
    let nv = mesh.n_vertices();
    let mut tri = TriMat::<f64>::new((nv, nv));
    for s in 0..mesh.n_simplices() {
        let tet = mesh.simplices[s];
        let g = &td.grads[s];
        let w = td.k_per_tet[s] * td.volumes[s];
        for a in 0..4 {
            for b in 0..4 {
                let gab = g[a].dot(&g[b]);
                if gab.abs() > 1e-30 || a == b {
                    tri.add_triplet(tet[a], tet[b], w * gab);
                }
            }
        }
    }
    tri.to_csc()
}

/// RHS for the cell problem in direction `e_dir`: Σ_tets K_tet · vol_tet · (e_dir · ∇φ_i).
/// The continuous form is b_i = -∫ K e_dir · ∇φ_i, but with our sign convention for the
/// correction field χ the sign flips to +; the resulting χ satisfies
/// (e_dir + ∇χ_i) · gradient = local flux.
pub fn assemble_rhs(
    mesh: &Mesh<Euclidean<3>, 4, 3>, td: &TetData, e_dir: usize,
) -> nalgebra::DVector<f64> {
    let nv = mesh.n_vertices();
    let mut b = nalgebra::DVector::<f64>::zeros(nv);
    for s in 0..mesh.n_simplices() {
        let tet = mesh.simplices[s];
        let g = &td.grads[s];
        let w = td.k_per_tet[s] * td.volumes[s];
        for a in 0..4 {
            b[tet[a]] -= w * g[a][e_dir];
        }
    }
    b
}

/// Apply homogeneous Dirichlet BCs (χ = 0 on boundary vertices) by zeroing their
/// rows/cols in the sparse matrix and setting A[b, b] = 1, b[b] = 0.
pub fn apply_dirichlet_zero(
    a: &mut sprs::CsMat<f64>, b: &mut nalgebra::DVector<f64>, boundary: &[usize],
) {
    use std::collections::HashSet;
    let bset: HashSet<usize> = boundary.iter().copied().collect();
    // Rebuild as dense-to-sparse (cleaner than surgery on CSC). OK for MVP sizes.
    let n = a.rows();
    let mut tri = TriMat::<f64>::new((n, n));
    for (&val, (i, j)) in a.iter() {
        if bset.contains(&i) { continue; }   // drop boundary rows
        if bset.contains(&j) { continue; }   // drop boundary cols (implicit zero)
        tri.add_triplet(i, j, val);
    }
    for &bi in boundary {
        tri.add_triplet(bi, bi, 1.0);
        b[bi] = 0.0;
    }
    *a = tri.to_csc();
}

/// Apply periodic BCs to the cell-problem system by eliminating slave DOFs.
///
/// For each `(slave, master)` pair: fold the slave row and column into the master,
/// then pin the slave to an identity row (χ[slave] = χ[master] post-solve). After
/// the solve, the caller should copy `chi[master]` into `chi[slave]`.
///
/// Additionally anchors one vertex (the "gauge vertex") to χ = 0 to remove the
/// constant-null-space mode. By convention we pick vertex 0.
pub fn apply_periodic(
    a: &mut sprs::CsMat<f64>,
    b: &mut nalgebra::DVector<f64>,
    pairs: &[(usize, usize)],
    gauge_vertex: usize,
) {
    use alloc::collections::BTreeMap;
    // slave -> master lookup.
    let mut master_of: BTreeMap<usize, usize> = BTreeMap::new();
    for &(s, m) in pairs { master_of.insert(s, m); }

    let n = a.rows();
    let mut tri = TriMat::<f64>::new((n, n));
    // First pass: fold every (i, j) entry to its canonical (master(i), master(j)).
    for (&val, (i, j)) in a.iter() {
        let mi = *master_of.get(&i).unwrap_or(&i);
        let mj = *master_of.get(&j).unwrap_or(&j);
        tri.add_triplet(mi, mj, val);
    }
    // Fold the RHS b similarly (rows only).
    let mut b_new = nalgebra::DVector::<f64>::zeros(n);
    for i in 0..n {
        let mi = *master_of.get(&i).unwrap_or(&i);
        b_new[mi] += b[i];
    }
    // Pin each slave row to identity: χ[slave] - χ[master] = 0.
    // Implement by inserting (slave, slave) = 1, (slave, master) = -1, b[slave] = 0.
    for (&s, &m) in &master_of {
        tri.add_triplet(s, s, 1.0);
        tri.add_triplet(s, m, -1.0);
        b_new[s] = 0.0;
    }
    // Pin gauge vertex: χ[gauge] = 0.
    // Zero its master-row columns and rewrite as identity. Cheapest path: add a
    // large diagonal penalty on (gauge, gauge) so the Lagrange condition dominates.
    tri.add_triplet(gauge_vertex, gauge_vertex, 1e20);
    b_new[gauge_vertex] = 0.0;

    *a = tri.to_csc();
    *b = b_new;
}

/// Copy master solution values into their paired slaves (post-solve fix-up).
pub fn expand_periodic(chi: &mut nalgebra::DVector<f64>, pairs: &[(usize, usize)]) {
    for &(s, m) in pairs {
        chi[s] = chi[m];
    }
}

/// Volume-averaged effective tensor column:
///   K_eff[:, e_dir] = Σ_tets K_tet · (e_dir + Σ_a χ[tet[a]] · ∇φ_a) · vol_tet / total_vol.
pub fn effective_column(
    mesh: &Mesh<Euclidean<3>, 4, 3>, td: &TetData,
    chi: &nalgebra::DVector<f64>, e_dir: usize,
) -> Vector3<f64> {
    let mut total_vol = 0.0;
    let mut acc = Vector3::zeros();
    let mut e_vec = Vector3::zeros();
    e_vec[e_dir] = 1.0;
    for s in 0..mesh.n_simplices() {
        let tet = mesh.simplices[s];
        let g = &td.grads[s];
        let grad_chi = chi[tet[0]] * g[0]
                     + chi[tet[1]] * g[1]
                     + chi[tet[2]] * g[2]
                     + chi[tet[3]] * g[3];
        let local = td.k_per_tet[s] * (e_vec + grad_chi);
        acc += local * td.volumes[s];
        total_vol += td.volumes[s];
    }
    acc / total_vol
}

/// Compute the full effective tensor: column i is `effective_column(chi_i, e_i)`.
pub fn effective_tensor(
    mesh: &Mesh<Euclidean<3>, 4, 3>, td: &TetData, chi_cols: [&nalgebra::DVector<f64>; 3],
) -> Matrix3<f64> {
    let c0 = effective_column(mesh, td, chi_cols[0], 0);
    let c1 = effective_column(mesh, td, chi_cols[1], 1);
    let c2 = effective_column(mesh, td, chi_cols[2], 2);
    Matrix3::from_columns(&[c0, c1, c2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fullfield::mesh::{PeriodicCubeMeshBuilder, PeriodicCubeMeshBuilderOpts};

    #[test]
    fn per_tet_data_total_volume_equals_one() {
        let b = PeriodicCubeMeshBuilder::new(&PeriodicCubeMeshBuilderOpts { resolution: 4, refine_depth: 0 });
        let (mesh, _) = b.build().unwrap();
        let td = build_tet_data(&mesh, alloc::vec![1.0; mesh.n_simplices()]).unwrap();
        let total: f64 = td.volumes.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "unit cube volume should be 1, got {total}");
    }
}

#[cfg(test)]
use nalgebra::DVector;
