//! Hausdorff-distance gate on adaptive-refinement indicator vs analytic
//! transition-layer set. Used by the capstone sandstone pipeline test (A2).
//!
//! The analytic transition layers are the depth bands where the crack-density
//! derivative `|ρ'(z)|` exceeds a user-chosen threshold; the refined-simplex
//! set is the barycentre cloud of tets flagged for refinement by the same
//! indicator on the discrete mesh. Agreement between the two (small Hausdorff
//! distance) demonstrates that a curvature-CFL-style refinement driver
//! applied to the K-weighted metric would track the physical transition
//! layers rather than spurious artefacts.

use alloc::vec::Vec;
use nalgebra::Vector3;

/// Compute one-sided Hausdorff distance: sup over a ∈ A of min over b ∈ B of |a - b|.
pub fn one_sided_hausdorff(a: &[Vector3<f64>], b: &[Vector3<f64>]) -> f64 {
    if a.is_empty() || b.is_empty() { return f64::INFINITY; }
    let mut max_min = 0.0_f64;
    for ai in a {
        let mut min_d = f64::INFINITY;
        for bi in b {
            let d = (ai - bi).norm();
            if d < min_d { min_d = d; }
        }
        if min_d > max_min { max_min = min_d; }
    }
    max_min
}

/// Symmetric Hausdorff distance: max of the two one-sided distances.
pub fn hausdorff_distance(a: &[Vector3<f64>], b: &[Vector3<f64>]) -> f64 {
    one_sided_hausdorff(a, b).max(one_sided_hausdorff(b, a))
}

/// Per-tet refinement flag based on a depth-derivative threshold.
/// Returns the barycentres of tets where `|indicator_fn(z)| > threshold`.
pub fn refined_barycentres<F: Fn(f64) -> f64>(
    barycentres: &[Vector3<f64>],
    indicator_fn: F,
    threshold: f64,
) -> Vec<Vector3<f64>> {
    barycentres.iter()
        .filter(|b| indicator_fn(b.z).abs() > threshold)
        .copied()
        .collect()
}

/// Sample the analytic transition-layer set: tuples (x, y, z) on a grid where
/// `|indicator_fn(z)| > threshold`. The xy sampling is coarse because the
/// transition layers are depth-only.
pub fn analytic_transition_points<F: Fn(f64) -> f64>(
    l_x: f64, l_y: f64, h: f64,
    n_xy: usize, n_z: usize,
    indicator_fn: F, threshold: f64,
) -> Vec<Vector3<f64>> {
    let mut pts = Vec::new();
    for ix in 0..n_xy {
        for iy in 0..n_xy {
            for iz in 0..n_z {
                let z = (iz as f64 + 0.5) * h / (n_z as f64);
                if indicator_fn(z).abs() > threshold {
                    let x = (ix as f64 + 0.5) * l_x / (n_xy as f64);
                    let y = (iy as f64 + 0.5) * l_y / (n_xy as f64);
                    pts.push(Vector3::new(x, y, z));
                }
            }
        }
    }
    pts
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn hausdorff_zero_for_identical_sets() {
        let a = alloc::vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        let b = a.clone();
        assert_relative_eq!(hausdorff_distance(&a, &b), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn hausdorff_captures_uniform_translation() {
        let a = alloc::vec![Vector3::new(0.0, 0.0, 0.0)];
        let b = alloc::vec![Vector3::new(3.0, 4.0, 0.0)];
        assert_relative_eq!(hausdorff_distance(&a, &b), 5.0, epsilon = 1e-12);
    }

    #[test]
    fn analytic_transition_points_are_depth_only() {
        // Indicator = 1 for z in [0.4, 0.6], 0 elsewhere.
        let ind = |z: f64| if z > 0.4 && z < 0.6 { 1.0 } else { 0.0 };
        let pts = analytic_transition_points(1.0, 1.0, 1.0, 4, 20, ind, 0.5);
        assert!(!pts.is_empty());
        for p in &pts {
            assert!(p.z > 0.4 && p.z < 0.6, "point outside transition band: {p:?}");
        }
    }

    #[test]
    fn refined_barycentres_filter_by_threshold() {
        let bary = alloc::vec![
            Vector3::new(0.5, 0.5, 0.1),
            Vector3::new(0.5, 0.5, 0.5),
            Vector3::new(0.5, 0.5, 0.9),
        ];
        let ind = |z: f64| (z - 0.5).abs() * 10.0;
        let refined = refined_barycentres(&bary, ind, 2.0);
        assert_eq!(refined.len(), 2);   // z = 0.1 and 0.9 flagged, 0.5 below threshold
    }
}
