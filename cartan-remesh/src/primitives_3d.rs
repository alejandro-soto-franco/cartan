//! 3D tet-mesh primitives: barycentric refinement.
//!
//! Scope: `Mesh<Euclidean<3>, 4, 3>` only. Edge-split (and therefore conforming
//! refinement on arbitrary tet meshes) is significantly more involved in 3D
//! than in 2D because the set of tets incident on an edge is arbitrary. v1.2
//! ships the simpler **barycentric refinement**: each flagged tet is replaced
//! by 4 sub-tets sharing its barycentre. This is *non-conforming* along faces
//! shared with an un-flagged neighbour, but for the full-field cell problem
//! we recompute K per tet from the barycentre anyway, so the solution loses
//! only the conforming-FE optimality rate on hanging faces — the resulting
//! K_eff is still unbiased and the refinement tracks the user-supplied
//! indicator.

use cartan_dec::Mesh;
use cartan_manifolds::euclidean::Euclidean;
use nalgebra::Vector3;

use crate::error::RemeshError;

/// Refine tets flagged by `flags` in place. Each flagged tet is replaced by 4
/// sub-tets, each sharing the barycentre of the original tet with one of the
/// 4 face triples. Returns the number of tets refined.
pub fn barycentric_refine_tets(
    mesh: &mut Mesh<Euclidean<3>, 4, 3>,
    flags: &[bool],
) -> Result<usize, RemeshError> {
    let nt = mesh.n_simplices();
    if flags.len() != nt {
        return Err(RemeshError::InvalidInput(format!(
            "flags length {} != n_simplices {nt}", flags.len())));
    }

    let mut new_simplices: Vec<[usize; 4]> = Vec::new();
    let mut flagged_indices: Vec<usize> = Vec::new();
    let mut refined = 0;
    for (s, &flag) in flags.iter().enumerate() {
        if !flag {
            new_simplices.push(mesh.simplices[s]);
            continue;
        }
        let tet = mesh.simplices[s];
        let v: [Vector3<f64>; 4] = [
            mesh.vertices[tet[0]], mesh.vertices[tet[1]],
            mesh.vertices[tet[2]], mesh.vertices[tet[3]],
        ];
        let bary = (v[0] + v[1] + v[2] + v[3]) / 4.0;
        let v_b = mesh.vertices.len();
        mesh.vertices.push(bary);

        // 4 sub-tets: each omits one original vertex, replacing it with the barycentre.
        for omit in 0..4 {
            let mut sub = [0usize; 4];
            let mut j = 0;
            for (i, &vert) in tet.iter().enumerate() {
                if i == omit { continue; }
                sub[j] = vert;
                j += 1;
            }
            sub[3] = v_b;
            new_simplices.push(sub);
        }
        flagged_indices.push(s);
        refined += 1;
    }

    mesh.simplices = new_simplices;
    mesh.rebuild_topology();
    Ok(refined)
}

/// Refinement indicator: flag tets where `|indicator_fn(barycentre)| > threshold`.
pub fn indicator_flags<F: Fn(Vector3<f64>) -> f64>(
    mesh: &Mesh<Euclidean<3>, 4, 3>,
    indicator_fn: F,
    threshold: f64,
) -> Vec<bool> {
    mesh.simplices.iter().map(|tet| {
        let bary = (mesh.vertices[tet[0]] + mesh.vertices[tet[1]]
                  + mesh.vertices[tet[2]] + mesh.vertices[tet[3]]) / 4.0;
        indicator_fn(bary).abs() > threshold
    }).collect()
}

/// Convenience: iterate barycentric refinement up to `depth` passes, applying
/// `indicator_fn` at each level. Returns the total count of refined tets.
pub fn refine_to_depth<F: Fn(Vector3<f64>) -> f64>(
    mesh: &mut Mesh<Euclidean<3>, 4, 3>,
    depth: usize,
    indicator_fn: F,
    threshold: f64,
) -> Result<usize, RemeshError> {
    let mut total = 0;
    for _ in 0..depth {
        let flags = indicator_flags(mesh, &indicator_fn, threshold);
        let n = barycentric_refine_tets(mesh, &flags)?;
        total += n;
        if n == 0 { break; }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cartan_manifolds::euclidean::Euclidean;

    fn unit_tet() -> Mesh<Euclidean<3>, 4, 3> {
        let vertices = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
        let simplices = vec![[0usize, 1, 2, 3]];
        let manifold = Euclidean::<3>;
        Mesh::<Euclidean<3>, 4, 3>::from_simplices_generic(&manifold, vertices, simplices)
    }

    #[test]
    fn refine_one_tet_produces_four_subtets_and_one_new_vertex() {
        let mut mesh = unit_tet();
        assert_eq!(mesh.n_simplices(), 1);
        assert_eq!(mesh.n_vertices(), 4);
        let n = barycentric_refine_tets(&mut mesh, &[true]).unwrap();
        assert_eq!(n, 1);
        assert_eq!(mesh.n_simplices(), 4);
        assert_eq!(mesh.n_vertices(), 5);
    }

    #[test]
    fn zero_flags_leaves_mesh_unchanged() {
        let mut mesh = unit_tet();
        let orig_tets = mesh.n_simplices();
        let orig_verts = mesh.n_vertices();
        let n = barycentric_refine_tets(&mut mesh, &[false]).unwrap();
        assert_eq!(n, 0);
        assert_eq!(mesh.n_simplices(), orig_tets);
        assert_eq!(mesh.n_vertices(), orig_verts);
    }

    #[test]
    fn indicator_flags_match_barycentre_test() {
        let mesh = unit_tet();
        let flags = indicator_flags(&mesh, |p| p.x + p.y + p.z, 0.1);
        assert_eq!(flags.len(), 1);
        // Barycentre is (0.25, 0.25, 0.25), sum = 0.75 > 0.1.
        assert!(flags[0]);
    }

    #[test]
    fn refine_to_depth_terminates_when_no_more_flags() {
        let mut mesh = unit_tet();
        let n = refine_to_depth(&mut mesh, 5, |_| 0.0, 1.0).unwrap();
        assert_eq!(n, 0);   // nothing ever flagged
        assert_eq!(mesh.n_simplices(), 1);
    }

    #[test]
    fn refine_to_depth_two_levels() {
        let mut mesh = unit_tet();
        // Always flag everything: 1 -> 4 -> 16 tets.
        let n = refine_to_depth(&mut mesh, 2, |_| 1.0, 0.5).unwrap();
        assert_eq!(n, 1 + 4);
        assert_eq!(mesh.n_simplices(), 16);
    }
}
