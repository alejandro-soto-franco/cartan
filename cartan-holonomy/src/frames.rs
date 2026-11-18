//! Tangent frames from local principal components.

use nalgebra::DMatrix;
use rayon::prelude::*;

/// The top-`d` principal directions of the centred neighbourhood of each point,
/// the point itself included.
///
/// Returns `n` frames of shape `(dim, d)`, row-major and concatenated. `nbrs[i]`
/// lists the neighbours of point `i`; the first `k` of them are used.
pub fn frames(
    x: &[f64],
    n: usize,
    dim: usize,
    nbrs: &[Vec<usize>],
    k: usize,
    d: usize,
) -> Vec<f64> {
    assert!(d <= dim, "frame dimension exceeds the ambient dimension");
    (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let nb = &nbrs[i][..k];
            let m = nb.len() + 1;
            let mut a = DMatrix::<f64>::zeros(m, dim);
            for (r, &j) in std::iter::once(&i).chain(nb.iter()).enumerate() {
                for c in 0..dim {
                    a[(r, c)] = x[j * dim + c];
                }
            }
            let mean = a.row_mean();
            for mut row in a.row_iter_mut() {
                row -= &mean;
            }
            let svd = a.svd(false, true);
            let vt = svd.v_t.expect("right singular vectors");
            let mut order: Vec<usize> = (0..svd.singular_values.len()).collect();
            order.sort_by(|&p, &q| {
                svd.singular_values[q]
                    .partial_cmp(&svd.singular_values[p])
                    .unwrap()
            });
            let mut out = vec![0.0; dim * d];
            for c in 0..d {
                for r in 0..dim {
                    out[r * d + c] = vt[(order[c], r)];
                }
            }
            out
        })
        .collect()
}
