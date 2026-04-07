// ~/cartan/cartan-remesh/src/driver.rs

//! Adaptive remeshing driver and predicate.
//!
//! The driver orchestrates the split/collapse pipeline based on edge length
//! bounds and a curvature-CFL criterion. Flip and smoothing passes are
//! deferred to a future extension (Task 11 must land first).

use cartan_manifolds::euclidean::Euclidean;
use cartan_dec::Mesh;

use crate::config::RemeshConfig;
use crate::log::RemeshLog;
use crate::primitives::{collapse_edge, split_edge};

/// Compute the curvature-CFL upper bound on edge length at a vertex.
///
/// Given mean curvature H and Gaussian curvature K at a vertex, the larger
/// principal curvature magnitude is:
///
///   k_max = |H| + sqrt(max(H^2 - K, 0))
///
/// The CFL bound is then `curvature_scale / sqrt(k_max)`. Returns `f64::MAX`
/// when k_max is negligible (flat region, no curvature constraint).
fn curvature_cfl_bound(h: f64, k: f64, curvature_scale: f64) -> f64 {
    let k_max = h.abs() + (h * h - k).max(0.0).sqrt();
    if k_max < 1e-30 {
        return f64::MAX;
    }
    curvature_scale / k_max.sqrt()
}

/// Check whether the mesh needs remeshing based on edge length and curvature criteria.
///
/// Returns `true` if any of the following hold:
/// - Some edge length exceeds `config.max_edge_length`.
/// - Some edge length is below `config.min_edge_length`.
/// - Some edge violates the curvature-CFL criterion: its length exceeds
///   `config.curvature_scale / sqrt(k_max)` where `k_max` is the larger
///   principal curvature magnitude at the vertex with higher curvature.
///
/// The curvature arrays must have one entry per vertex.
///
/// # Panics
///
/// Panics if the curvature arrays have different lengths than vertex count.
pub fn needs_remesh(
    mesh: &Mesh<Euclidean<2>, 3, 2>,
    manifold: &Euclidean<2>,
    mean_curvatures: &[f64],
    gaussian_curvatures: &[f64],
    config: &RemeshConfig,
) -> bool {
    let nv = mesh.n_vertices();
    assert_eq!(mean_curvatures.len(), nv, "mean_curvatures length mismatch");
    assert_eq!(
        gaussian_curvatures.len(),
        nv,
        "gaussian_curvatures length mismatch"
    );

    for e in 0..mesh.n_boundaries() {
        let len_e = mesh.edge_length(manifold, e);

        if len_e > config.max_edge_length {
            return true;
        }
        if len_e < config.min_edge_length {
            return true;
        }

        // Curvature-CFL: check both endpoints, use the tighter bound.
        let [va, vb] = mesh.boundaries[e];
        let cfl_a = curvature_cfl_bound(
            mean_curvatures[va],
            gaussian_curvatures[va],
            config.curvature_scale,
        );
        let cfl_b = curvature_cfl_bound(
            mean_curvatures[vb],
            gaussian_curvatures[vb],
            config.curvature_scale,
        );
        let cfl_bound = cfl_a.min(cfl_b);
        if len_e > cfl_bound {
            return true;
        }
    }

    false
}

/// Run the adaptive remeshing pipeline: split long edges, then collapse short edges.
///
/// The pipeline proceeds in two passes:
///
/// 1. **Split pass**: edges exceeding `max_edge_length` or the curvature-CFL
///    bound are split in order of decreasing length (longest first). After
///    each split the topology is rebuilt, so edge indices change. The pass
///    repeats until no more edges need splitting.
///
/// 2. **Collapse pass**: edges shorter than `min_edge_length` are collapsed in
///    order of increasing length (shortest first). Collapses that would cause
///    foldover are skipped. The pass repeats until no more edges need collapse.
///
/// Flip and tangential smoothing passes are not yet included (pending Task 11).
///
/// **Curvature staleness**: after each split or collapse, the input curvature
/// arrays become stale because the mesh topology has changed and new vertices
/// lack curvature estimates. The current implementation uses the original
/// curvature arrays for the split pass only. The collapse pass uses only the
/// `min_edge_length` criterion, ignoring curvature. A future extension should
/// recompute curvatures between passes.
///
/// # Panics
///
/// Panics if curvature arrays have different lengths than the initial vertex count.
pub fn adaptive_remesh(
    mesh: &mut Mesh<Euclidean<2>, 3, 2>,
    manifold: &Euclidean<2>,
    mean_curvatures: &[f64],
    gaussian_curvatures: &[f64],
    config: &RemeshConfig,
) -> RemeshLog {
    let nv = mesh.n_vertices();
    assert_eq!(mean_curvatures.len(), nv, "mean_curvatures length mismatch");
    assert_eq!(
        gaussian_curvatures.len(),
        nv,
        "gaussian_curvatures length mismatch"
    );

    let mut log = RemeshLog::new();

    // ── Split pass ───────────────────────────────────────────────────────
    // Repeat until no edges exceed the length bounds.
    loop {
        // Collect edges that need splitting, with their lengths.
        let mut to_split: Vec<(usize, f64)> = Vec::new();
        for e in 0..mesh.n_boundaries() {
            let len_e = mesh.edge_length(manifold, e);
            if len_e > config.max_edge_length {
                to_split.push((e, len_e));
                continue;
            }
            // Curvature-CFL check (only for original vertices that have curvature data).
            let [va, vb] = mesh.boundaries[e];
            if va < mean_curvatures.len() && vb < mean_curvatures.len() {
                let cfl_a = curvature_cfl_bound(
                    mean_curvatures[va],
                    gaussian_curvatures[va],
                    config.curvature_scale,
                );
                let cfl_b = curvature_cfl_bound(
                    mean_curvatures[vb],
                    gaussian_curvatures[vb],
                    config.curvature_scale,
                );
                let cfl_bound = cfl_a.min(cfl_b);
                if len_e > cfl_bound {
                    to_split.push((e, len_e));
                }
            }
        }

        if to_split.is_empty() {
            break;
        }

        // Sort by length descending: split the longest edge first.
        to_split.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Split just the longest edge, then re-scan (topology changes invalidate indices).
        let (edge, _) = to_split[0];
        let split_log = split_edge(mesh, manifold, edge);
        log.merge(split_log);
    }

    // ── Collapse pass ────────────────────────────────────────────────────
    // Repeat until no edges are below the minimum length.
    loop {
        let mut to_collapse: Vec<(usize, f64)> = Vec::new();
        for e in 0..mesh.n_boundaries() {
            let len_e = mesh.edge_length(manifold, e);
            if len_e < config.min_edge_length {
                to_collapse.push((e, len_e));
            }
        }

        if to_collapse.is_empty() {
            break;
        }

        // Sort by length ascending: collapse the shortest edge first.
        to_collapse.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let (edge, _) = to_collapse[0];
        // Collapse may fail due to foldover; skip and try next.
        match collapse_edge(mesh, manifold, edge, config.foldover_threshold) {
            Ok(collapse_log) => {
                log.merge(collapse_log);
            }
            Err(_) => {
                // Foldover detected: skip this edge. If all short edges cause
                // foldover, break to avoid infinite loop.
                let remaining: Vec<(usize, f64)> = to_collapse
                    .iter()
                    .skip(1)
                    .copied()
                    .collect();
                let mut collapsed_any = false;
                for (e, _) in remaining {
                    // Edge index may have shifted, re-validate.
                    if e >= mesh.n_boundaries() {
                        continue;
                    }
                    if let Ok(cl) = collapse_edge(mesh, manifold, e, config.foldover_threshold) {
                        log.merge(cl);
                        collapsed_any = true;
                        break;
                    }
                }
                if !collapsed_any {
                    break;
                }
            }
        }
    }

    log
}
