// ~/cartan/cartan-remesh/src/primitives.rs

//! Primitive remesh operations: split, collapse, flip, shift.
//!
//! `split_edge` is generic over `M: Manifold`. `collapse_edge` is currently
//! specialized to `Euclidean<2>` (flat triangle meshes) for foldover detection
//! via signed area. Generalization to curved manifolds is a future extension.

use cartan_core::Manifold;
use cartan_dec::Mesh;
use cartan_manifolds::euclidean::Euclidean;

use crate::error::RemeshError;
use crate::log::{EdgeCollapse, EdgeSplit, RemeshLog};

/// Split an edge by inserting a vertex at the geodesic midpoint.
///
/// The adjacent triangles are each split into two, preserving orientation.
/// Topology is rebuilt after the split.
///
/// # Panics
///
/// Panics if `edge >= mesh.n_boundaries()`.
pub fn split_edge<M: Manifold>(
    mesh: &mut Mesh<M, 3, 2>,
    manifold: &M,
    edge: usize,
) -> RemeshLog {
    assert!(edge < mesh.n_boundaries(), "edge index out of bounds");

    let [v_a, v_b] = mesh.boundaries[edge];
    let midpoint = mesh.boundary_midpoint(manifold, edge);
    let v_m = mesh.vertices.len();
    mesh.vertices.push(midpoint);

    let adjacent_faces: Vec<usize> = mesh.boundary_simplices[edge].clone();
    let mut new_triangles: Vec<[usize; 3]> = Vec::new();
    let mut faces_to_remove: Vec<usize> = Vec::new();

    for &face_idx in &adjacent_faces {
        let tri = mesh.simplices[face_idx];
        let v_opp = tri
            .iter()
            .copied()
            .find(|&v| v != v_a && v != v_b)
            .expect("triangle must have an opposite vertex");

        let pos_a = tri.iter().position(|&v| v == v_a).unwrap();
        let pos_b = tri.iter().position(|&v| v == v_b).unwrap();

        if (pos_a + 1) % 3 == pos_b {
            new_triangles.push([v_a, v_m, v_opp]);
            new_triangles.push([v_m, v_b, v_opp]);
        } else {
            new_triangles.push([v_b, v_m, v_opp]);
            new_triangles.push([v_m, v_a, v_opp]);
        }
        faces_to_remove.push(face_idx);
    }

    faces_to_remove.sort_unstable();
    for &fi in faces_to_remove.iter().rev() {
        mesh.simplices.swap_remove(fi);
    }
    for tri in &new_triangles {
        mesh.simplices.push(*tri);
    }

    mesh.rebuild_topology();

    let new_edges: Vec<usize> = mesh.vertex_boundaries[v_m].clone();

    let mut log = RemeshLog::new();
    log.splits.push(EdgeSplit {
        old_edge: edge,
        v_a,
        v_b,
        new_vertex: v_m,
        new_edges,
    });
    log
}

/// Collapse an edge on a flat 2D mesh by merging endpoints at the midpoint.
///
/// Faces containing both endpoints are removed. Returns `Err(RemeshError::Foldover)`
/// if any surviving face would flip orientation (signed area sign change).
///
/// # Panics
///
/// Panics if `edge >= mesh.n_boundaries()`.
pub fn collapse_edge(
    mesh: &mut Mesh<Euclidean<2>, 3, 2>,
    _manifold: &Euclidean<2>,
    edge: usize,
    foldover_threshold: f64,
) -> Result<RemeshLog, RemeshError> {
    assert!(edge < mesh.n_boundaries(), "edge index out of bounds");

    let [v_a, v_b] = mesh.boundaries[edge];
    let (survivor, removed) = if v_a < v_b { (v_a, v_b) } else { (v_b, v_a) };

    // Compute midpoint.
    let pa = &mesh.vertices[v_a];
    let pb = &mesh.vertices[v_b];
    let midpoint = (pa + pb) * 0.5;

    // Classify faces.
    let faces_with_both: Vec<usize> = mesh.boundary_simplices[edge].clone();
    // All faces that will be affected: those incident to either endpoint but
    // NOT containing both (those containing both are removed entirely).
    let mut faces_to_check: Vec<usize> = Vec::new();
    for &f in mesh.vertex_simplices[removed]
        .iter()
        .chain(mesh.vertex_simplices[survivor].iter())
    {
        if !faces_with_both.contains(&f) && !faces_to_check.contains(&f) {
            faces_to_check.push(f);
        }
    }

    // Foldover guard.
    for &face_idx in &faces_to_check {
        let tri = mesh.simplices[face_idx];
        let area_before = signed_area_flat(mesh, &tri);

        let mut tri_after = tri;
        for v in tri_after.iter_mut() {
            if *v == removed {
                *v = survivor;
            }
        }
        let old_pos = mesh.vertices[survivor];
        mesh.vertices[survivor] = midpoint;
        let area_after = signed_area_flat(mesh, &tri_after);
        mesh.vertices[survivor] = old_pos;

        if area_before.abs() > 1e-30 && area_after.abs() > 1e-30 {
            let cos_angle: f64 = if area_before.signum() == area_after.signum() {
                1.0
            } else {
                -1.0
            };
            let angle = cos_angle.acos();
            if angle > foldover_threshold {
                return Err(RemeshError::Foldover {
                    face: face_idx,
                    angle_rad: angle,
                    threshold: foldover_threshold,
                });
            }
        }
    }

    // Execute collapse.
    mesh.vertices[survivor] = midpoint;

    let removed_faces = faces_with_both.clone();
    let mut to_remove_sorted = faces_with_both;
    to_remove_sorted.sort_unstable();
    for &fi in to_remove_sorted.iter().rev() {
        mesh.simplices.swap_remove(fi);
    }

    for tri in mesh.simplices.iter_mut() {
        for v in tri.iter_mut() {
            if *v == removed {
                *v = survivor;
            }
        }
    }

    let last_vertex = mesh.vertices.len() - 1;
    mesh.vertices.swap_remove(removed);
    if removed != last_vertex {
        for tri in mesh.simplices.iter_mut() {
            for v in tri.iter_mut() {
                if *v == last_vertex {
                    *v = removed;
                }
            }
        }
    }

    mesh.rebuild_topology();

    let mut log = RemeshLog::new();
    log.collapses.push(EdgeCollapse {
        old_edge: edge,
        surviving_vertex: survivor,
        removed_vertex: removed,
        removed_faces,
    });
    Ok(log)
}

/// Signed area of a flat 2D triangle (cross product of edge vectors).
fn signed_area_flat(mesh: &Mesh<Euclidean<2>, 3, 2>, tri: &[usize; 3]) -> f64 {
    let [i, j, k] = *tri;
    let a = &mesh.vertices[i];
    let b = &mesh.vertices[j];
    let c = &mesh.vertices[k];
    0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
}

/// Flip the diagonal of the quad formed by two adjacent triangles.
pub fn flip_edge<M: Manifold>(
    _mesh: &mut Mesh<M, 3, 2>,
    _manifold: &M,
    _edge: usize,
) -> Result<RemeshLog, RemeshError> {
    todo!("Task 11")
}

/// Tangential Laplacian smoothing of a single vertex.
pub fn shift_vertex<M: Manifold>(
    _mesh: &mut Mesh<M, 3, 2>,
    _manifold: &M,
    _vertex: usize,
) -> RemeshLog {
    todo!("Task 11")
}
