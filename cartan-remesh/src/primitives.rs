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
use crate::log::{EdgeCollapse, EdgeFlip, EdgeSplit, RemeshLog, VertexShift};

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
///
/// Given an interior edge shared by exactly two triangles, this operation
/// replaces the shared diagonal with the opposite diagonal of the quad.
/// The flip proceeds only if the Delaunay criterion is violated (sum of
/// opposite angles exceeds pi). After flipping, topology is rebuilt from
/// scratch via `mesh.rebuild_topology()`.
///
/// # Errors
///
/// - [`RemeshError::BoundaryEdge`] if the edge has exactly 1 adjacent face.
/// - [`RemeshError::NotInteriorEdge`] if the edge has 0 or more than 2 adjacent faces.
/// - [`RemeshError::AlreadyDelaunay`] if the sum of opposite angles is at most pi.
///
/// # Panics
///
/// Panics if `edge >= mesh.n_boundaries()`.
pub fn flip_edge<M: Manifold>(
    mesh: &mut Mesh<M, 3, 2>,
    manifold: &M,
    edge: usize,
) -> Result<RemeshLog, RemeshError> {
    assert!(edge < mesh.n_boundaries(), "edge index out of bounds");

    let adj = &mesh.boundary_simplices[edge];
    let adj_count = adj.len();
    if adj_count == 1 {
        return Err(RemeshError::BoundaryEdge { edge });
    }
    if adj_count != 2 {
        return Err(RemeshError::NotInteriorEdge {
            edge,
            count: adj_count,
        });
    }

    let [v_a, v_b] = mesh.boundaries[edge];
    let face_0 = adj[0];
    let face_1 = adj[1];

    // Find the opposite vertex in each triangle (the vertex not on the shared edge).
    let opp_0 = mesh.simplices[face_0]
        .iter()
        .copied()
        .find(|&v| v != v_a && v != v_b)
        .expect("triangle must have a vertex not on the edge");
    let opp_1 = mesh.simplices[face_1]
        .iter()
        .copied()
        .find(|&v| v != v_a && v != v_b)
        .expect("triangle must have a vertex not on the edge");

    // Compute opposite angles using the manifold's log map and inner product.
    let angle_0 = opposite_angle(manifold, &mesh.vertices, opp_0, v_a, v_b);
    let angle_1 = opposite_angle(manifold, &mesh.vertices, opp_1, v_a, v_b);
    let angle_sum = angle_0 + angle_1;

    if angle_sum <= std::f64::consts::PI {
        return Err(RemeshError::AlreadyDelaunay { edge, angle_sum });
    }

    // Replace the two old triangles with two new ones using the opposite-vertex diagonal.
    // Old: [v_a, v_b, opp_0] and [v_a, v_b, opp_1] (in some winding).
    // New: [opp_0, opp_1, v_a] and [opp_1, opp_0, v_b].
    //
    // Preserve consistent CCW orientation by reading the winding of each original
    // triangle and placing the new diagonal accordingly.
    let tri_0 = mesh.simplices[face_0];
    let pos_a_in_0 = tri_0.iter().position(|&v| v == v_a).unwrap();
    let next_in_0 = tri_0[(pos_a_in_0 + 1) % 3];

    // In triangle 0, the winding order around the quad determines which vertex
    // follows v_a. If v_b follows v_a, then opp_0 precedes v_a. The new
    // triangle on the v_a side should be [opp_0, opp_1, v_a] with CCW winding
    // matching the original.
    if next_in_0 == v_b {
        // Original winding: tri_0 goes ...v_a -> v_b -> opp_0...
        // New triangles: [opp_0, opp_1, v_a] and [opp_1, opp_0, v_b]
        mesh.simplices[face_0] = [opp_0, opp_1, v_a];
        mesh.simplices[face_1] = [opp_1, opp_0, v_b];
    } else {
        // Original winding: tri_0 goes ...v_a -> opp_0 -> v_b...
        // New triangles: [opp_0, v_a, opp_1] and [opp_0, v_b, opp_1] won't work;
        // mirror: [opp_1, opp_0, v_a] and [opp_0, opp_1, v_b]
        mesh.simplices[face_0] = [opp_1, opp_0, v_a];
        mesh.simplices[face_1] = [opp_0, opp_1, v_b];
    }

    mesh.rebuild_topology();

    let mut log = RemeshLog::new();
    log.flips.push(EdgeFlip {
        old_edge: edge,
        new_edge: [opp_0, opp_1],
        affected_faces: [face_0, face_1],
    });
    Ok(log)
}

/// Compute the angle at vertex `apex` in the triangle (apex, p, q) using
/// the manifold's logarithmic map and inner product.
fn opposite_angle<M: Manifold>(
    manifold: &M,
    vertices: &[M::Point],
    apex: usize,
    p: usize,
    q: usize,
) -> f64 {
    let v_ap = manifold
        .log(&vertices[apex], &vertices[p])
        .expect("log map failed for angle computation");
    let v_aq = manifold
        .log(&vertices[apex], &vertices[q])
        .expect("log map failed for angle computation");
    let dot = manifold.inner(&vertices[apex], &v_ap, &v_aq);
    let norm_ap = manifold.norm(&vertices[apex], &v_ap);
    let norm_aq = manifold.norm(&vertices[apex], &v_aq);
    let denom = norm_ap * norm_aq;
    if denom < 1e-30 {
        return 0.0;
    }
    let cos_val = (dot / denom).clamp(-1.0, 1.0);
    cos_val.acos()
}

/// Tangential Laplacian smoothing of a single vertex.
///
/// Computes the average of `log(vertex, neighbor)` over all 1-ring neighbors,
/// producing a tangential displacement. The vertex is then moved via `exp`.
/// For 2D flat meshes, the tangent plane coincides with the embedding plane,
/// so no normal projection is needed.
///
/// Neighbors are discovered from `mesh.vertex_boundaries`: each incident edge
/// contributes the other endpoint as a neighbor.
///
/// The `old_pos_tangent` field in the returned log is currently empty because
/// the generic `Manifold::Point` type does not expose raw coordinate access.
/// Callers that need the old coordinates should snapshot them before calling.
///
/// # Panics
///
/// Panics if `vertex >= mesh.n_vertices()`.
pub fn shift_vertex<M: Manifold>(
    mesh: &mut Mesh<M, 3, 2>,
    manifold: &M,
    vertex: usize,
) -> RemeshLog {
    assert!(
        vertex < mesh.n_vertices(),
        "vertex index out of bounds"
    );

    // Collect 1-ring neighbors from incident edges.
    let mut neighbors: Vec<usize> = Vec::new();
    for &b in &mesh.vertex_boundaries[vertex] {
        let [e0, e1] = mesh.boundaries[b];
        let other = if e0 == vertex { e1 } else { e0 };
        if !neighbors.contains(&other) {
            neighbors.push(other);
        }
    }

    if neighbors.is_empty() {
        let mut log = RemeshLog::new();
        log.shifts.push(VertexShift {
            vertex,
            old_pos_tangent: Vec::new(),
        });
        return log;
    }

    // Compute tangential Laplacian: average of log(vertex, neighbor_i).
    // Tangent vectors support Add and Mul<Real>, so accumulate via those ops.
    let n = neighbors.len() as f64;
    let base = mesh.vertices[vertex].clone();
    let first_log = manifold
        .log(&base, &mesh.vertices[neighbors[0]])
        .expect("log map failed in shift_vertex");

    let mut displacement = first_log;
    for &nb in &neighbors[1..] {
        let v_log = manifold
            .log(&base, &mesh.vertices[nb])
            .expect("log map failed in shift_vertex");
        displacement = displacement + v_log;
    }

    // Average: displacement = sum / n.
    displacement = displacement * (1.0 / n);

    // Apply displacement via exponential map.
    let new_pos = manifold.exp(&base, &displacement);
    mesh.vertices[vertex] = new_pos;

    let mut log = RemeshLog::new();
    log.shifts.push(VertexShift {
        vertex,
        old_pos_tangent: Vec::new(),
    });
    log
}
