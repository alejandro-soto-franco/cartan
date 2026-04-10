// ~/cartan/cartan-geo/src/disclination/lines.rs

//! Disclination line reconstruction from individual segments.
//!
//! ## Algorithm
//!
//! Given a set of `DisclinationSegment`s (edges pierced by a disclination),
//! build a graph where each vertex index is a node and each segment is an edge.
//! BFS/DFS connected-component search yields the set of disclination lines.
//!
//! For each connected component, the vertices are ordered into a path (or loop)
//! by traversing the adjacency list. The geometric positions of each vertex
//! on the line are taken from the midpoints of the adjacent segments.
//!
//! Frenet-Serret geometry (tangent, curvature, binormal, torsion) is then
//! computed along the ordered path via central finite differences.
//!
//! ## References
//!
//! - Smalyukh, I. I. (2010). Phys. Rev. Lett. 104, 097801.
//! - Dennis, M. R. (2010). J. Phys. A 43, 494013.

use std::collections::{HashMap, HashSet, VecDeque};

use super::segments::{DisclinationCharge, DisclinationSegment};

/// A reconstructed disclination line: an ordered sequence of vertices with
/// associated Frenet-Serret geometry.
#[derive(Debug, Clone)]
pub struct DisclinationLine {
    /// Ordered vertex positions along the line (in grid units, same as midpoints).
    pub vertices: Vec<[f64; 3]>,
    /// Unit tangent vectors at each vertex (central differences, end-padded).
    pub tangents: Vec<[f64; 3]>,
    /// Curvature |dT/ds| at each vertex.
    pub curvatures: Vec<f64>,
    /// Frenet-Serret torsion tau = -N · (dB/ds) at each vertex.
    pub torsions: Vec<f64>,
    /// Topological charge of this line (from the first segment).
    pub charge: DisclinationCharge,
    /// True if the line forms a closed loop.
    pub is_loop: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Normalize a 3-vector. Returns the zero vector if the norm is below 1e-14.
fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm < 1e-14 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / norm, v[1] / norm, v[2] / norm]
    }
}

/// Subtract two 3-vectors: a - b.
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Scale a 3-vector by a scalar.
fn scale3(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// Length (norm) of a 3-vector.
fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Cross product a x b.
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product a · b.
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Compute Frenet-Serret geometry along an ordered list of positions.
///
/// Returns (tangents, curvatures, torsions), each of length equal to `positions`.
///
/// Uses central differences for interior points and one-sided at the ends.
fn compute_frenet(positions: &[[f64; 3]]) -> (Vec<[f64; 3]>, Vec<f64>, Vec<f64>) {
    let n = positions.len();
    if n < 2 {
        let tangents = vec![[1.0, 0.0, 0.0]; n];
        let curvatures = vec![0.0; n];
        let torsions = vec![0.0; n];
        return (tangents, curvatures, torsions);
    }

    // Step 1: compute tangents using central differences on positions.
    let mut tangents = vec![[0.0f64; 3]; n];
    for i in 0..n {
        let prev = if i == 0 {
            positions[0]
        } else {
            positions[i - 1]
        };
        let next = if i == n - 1 {
            positions[n - 1]
        } else {
            positions[i + 1]
        };
        let diff = sub3(next, prev);
        tangents[i] = normalize3(diff);
    }

    if n < 3 {
        let curvatures = vec![0.0; n];
        let torsions = vec![0.0; n];
        return (tangents, curvatures, torsions);
    }

    // Step 2: compute curvature and normal N via central differences of tangent.
    let mut normals = vec![[0.0f64; 3]; n];
    let mut curvatures = vec![0.0f64; n];
    for i in 0..n {
        let t_prev = if i == 0 { tangents[0] } else { tangents[i - 1] };
        let t_next = if i == n - 1 {
            tangents[n - 1]
        } else {
            tangents[i + 1]
        };
        let p_prev = if i == 0 {
            positions[0]
        } else {
            positions[i - 1]
        };
        let p_next = if i == n - 1 {
            positions[n - 1]
        } else {
            positions[i + 1]
        };
        // ds = arc length step (approximate)
        let ds = norm3(sub3(p_next, p_prev));
        let dt = sub3(t_next, t_prev);
        if ds > 1e-14 {
            let dt_ds = scale3(dt, 1.0 / ds);
            curvatures[i] = norm3(dt_ds);
            normals[i] = normalize3(dt_ds);
        } else {
            curvatures[i] = 0.0;
            normals[i] = [0.0, 0.0, 0.0];
        }
    }

    // Step 3: binormal B = T x N.
    let mut binormals = vec![[0.0f64; 3]; n];
    for i in 0..n {
        binormals[i] = normalize3(cross3(tangents[i], normals[i]));
    }

    // Step 4: torsion tau = -N · (dB/ds) via central differences on B.
    let mut torsions = vec![0.0f64; n];
    for i in 0..n {
        let b_prev = if i == 0 {
            binormals[0]
        } else {
            binormals[i - 1]
        };
        let b_next = if i == n - 1 {
            binormals[n - 1]
        } else {
            binormals[i + 1]
        };
        let p_prev = if i == 0 {
            positions[0]
        } else {
            positions[i - 1]
        };
        let p_next = if i == n - 1 {
            positions[n - 1]
        } else {
            positions[i + 1]
        };
        let ds = norm3(sub3(p_next, p_prev));
        let db = sub3(b_next, b_prev);
        if ds > 1e-14 {
            let db_ds = scale3(db, 1.0 / ds);
            torsions[i] = -dot3(normals[i], db_ds);
        } else {
            torsions[i] = 0.0;
        }
    }

    (tangents, curvatures, torsions)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Connect individual disclination segments into continuous lines.
///
/// Builds a vertex-adjacency graph from the segment edge pairs, then finds
/// connected components via BFS. Each component is ordered into a path (or
/// loop) and its Frenet-Serret geometry is computed.
///
/// # Parameters
///
/// - `segs`: slice of `DisclinationSegment`s, one per pierced edge.
/// - `_dx`: grid spacing (currently unused; geometry is taken from midpoints).
///
/// # Returns
///
/// A `Vec<DisclinationLine>`, one per connected component. Empty if `segs` is empty.
pub fn connect_disclination_lines(segs: &[DisclinationSegment], _dx: f64) -> Vec<DisclinationLine> {
    if segs.is_empty() {
        return Vec::new();
    }

    // Build adjacency: vertex index -> list of (neighbor vertex index, segment index)
    let mut adj: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (seg_idx, seg) in segs.iter().enumerate() {
        let (u, v) = seg.edge;
        adj.entry(u).or_default().push((v, seg_idx));
        adj.entry(v).or_default().push((u, seg_idx));
    }

    // BFS to find connected components
    let mut visited_vertices: HashSet<usize> = HashSet::new();
    let mut lines = Vec::new();

    for &start_v in adj.keys() {
        if visited_vertices.contains(&start_v) {
            continue;
        }

        // BFS: collect all vertices in this component and the edges between them
        let mut component_vertices: Vec<usize> = Vec::new();
        let mut component_segs: HashSet<usize> = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start_v);
        visited_vertices.insert(start_v);

        while let Some(v) = queue.pop_front() {
            component_vertices.push(v);
            if let Some(neighbors) = adj.get(&v) {
                for &(u, seg_idx) in neighbors {
                    component_segs.insert(seg_idx);
                    if !visited_vertices.contains(&u) {
                        visited_vertices.insert(u);
                        queue.push_back(u);
                    }
                }
            }
        }

        // Build an ordered path through this component.
        // Find a degree-1 vertex (endpoint) if it exists; otherwise start from any vertex.
        let degree: HashMap<usize, usize> = component_vertices
            .iter()
            .map(|&v| {
                let d = adj.get(&v).map(|n| n.len()).unwrap_or(0);
                (v, d)
            })
            .collect();

        let path_start = component_vertices
            .iter()
            .find(|&&v| *degree.get(&v).unwrap_or(&0) == 1)
            .copied()
            .unwrap_or(component_vertices[0]);

        // Traverse the path greedily.
        // Note: this handles degree-≤2 chains correctly. At a junction vertex
        // (degree ≥ 3, e.g. T- or Y-junction) the walk terminates at the junction
        // and branches beyond the first are silently dropped. Such topologies are
        // rare in equilibrium nematics and not expected in the current simulation
        // regime, but callers should be aware that junction branches are omitted.
        let mut path: Vec<usize> = Vec::new();
        let mut path_visited: HashSet<usize> = HashSet::new();
        let mut current = path_start;
        path.push(current);
        path_visited.insert(current);

        loop {
            // Find unvisited neighbor
            let next = adj.get(&current).and_then(|neighbors| {
                neighbors
                    .iter()
                    .find(|&&(u, _)| !path_visited.contains(&u))
                    .map(|&(u, _)| u)
            });
            match next {
                Some(u) => {
                    path.push(u);
                    path_visited.insert(u);
                    current = u;
                }
                None => break,
            }
        }

        // Check if loop: first and last vertex are adjacent
        let is_loop = adj
            .get(&path[path.len() - 1])
            .map(|n| n.iter().any(|&(u, _)| u == path[0]))
            .unwrap_or(false)
            && path.len() > 2;

        // Build geometric positions from midpoints of segments along the path.
        // For each vertex in the path, collect the midpoint of the first adjacent segment.
        let mut positions: Vec<[f64; 3]> = Vec::with_capacity(path.len());
        for &v in &path {
            // Use the midpoint of the first segment incident on this vertex
            if let Some(neighbors) = adj.get(&v) {
                if let Some(&(_, seg_idx)) = neighbors.first() {
                    positions.push(segs[seg_idx].midpoint);
                } else {
                    positions.push([0.0, 0.0, 0.0]);
                }
            } else {
                positions.push([0.0, 0.0, 0.0]);
            }
        }

        // For paths built from segs with edge (i, i+1) and midpoints in order,
        // we want to capture the actual traversal order of midpoints.
        // Build positions in path traversal order using the edge midpoints.
        // For each consecutive pair in path, find the segment connecting them.
        let mut ordered_positions: Vec<[f64; 3]> = Vec::with_capacity(path.len());
        for k in 0..path.len() {
            let v = path[k];
            if k + 1 < path.len() {
                let u = path[k + 1];
                // Find the segment for edge (v, u) or (u, v)
                if let Some(neighbors) = adj.get(&v) {
                    if let Some(&(_, seg_idx)) = neighbors.iter().find(|&&(nb, _)| nb == u) {
                        ordered_positions.push(segs[seg_idx].midpoint);
                        continue;
                    }
                }
            }
            // Last vertex: use midpoint of last segment
            ordered_positions.push(positions[k]);
        }

        // Compute Frenet-Serret geometry
        let (tangents, curvatures, torsions) = compute_frenet(&ordered_positions);

        // Charge from first segment in this component
        let first_seg_idx = *component_segs.iter().next().unwrap_or(&0);
        let charge = segs[first_seg_idx].charge.clone();

        lines.push(DisclinationLine {
            vertices: ordered_positions,
            tangents,
            curvatures,
            torsions,
            charge,
            is_loop,
        });
    }

    lines
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::segments::Sign;
    use super::*;

    #[test]
    fn test_connect_single_segment() {
        let segs = vec![DisclinationSegment {
            edge: (0, 1),
            charge: DisclinationCharge::Half(Sign::Positive),
            midpoint: [0.5, 0.0, 0.0],
        }];
        let lines = connect_disclination_lines(&segs, 1.0);
        assert_eq!(lines.len(), 1);
        assert!(!lines[0].is_loop);
    }

    #[test]
    fn test_frenet_torsion_nonzero_on_helix() {
        // A helical sequence of segments must produce nonzero torsion.
        // Helix: x=cos(t), y=sin(t), z=t/5 for t in [0, 2pi] with 20 steps.
        use std::f64::consts::PI;
        let n = 20usize;
        let mut segs = Vec::new();
        let positions: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let t = 2.0 * PI * (i as f64) / (n as f64);
                [t.cos(), t.sin(), t / 5.0]
            })
            .collect();
        // Build synthetic segments along the helix
        for i in 0..n - 1 {
            segs.push(DisclinationSegment {
                edge: (i, i + 1),
                charge: DisclinationCharge::Half(Sign::Positive),
                midpoint: [
                    (positions[i][0] + positions[i + 1][0]) / 2.0,
                    (positions[i][1] + positions[i + 1][1]) / 2.0,
                    (positions[i][2] + positions[i + 1][2]) / 2.0,
                ],
            });
        }
        let lines = connect_disclination_lines(&segs, 1.0);
        assert!(!lines.is_empty());
        // At least one interior vertex must have nonzero torsion
        let max_torsion = lines[0].torsions.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            max_torsion > 1e-6,
            "Helical line must have nonzero torsion, max was {}",
            max_torsion
        );
    }
}
