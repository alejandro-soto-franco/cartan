//! Exact k nearest neighbours.

use faer::{Accum, MatMut, MatRef, Par};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A distance ordered by `total_cmp`.
#[derive(Clone, Copy, PartialEq)]
struct Key(f64);
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Key {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0)
    }
}

const BLOCK: usize = 64;
const MARGIN: usize = 8;
const TILE: usize = 2048;

/// Exact squared Euclidean distance, summed in coordinate order.
fn dist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum()
}

/// The `k` nearest neighbours of every point, self excluded, nearest first.
///
/// `x` holds `n` points of dimension `dim`, row-major. Candidate distances
/// `|x_j|^2 - 2 x_i . x_j` come from matrix products on tiles of 64 rows by
/// 2,048 columns, small enough to stay in cache, and a bounded heap per row
/// keeps the `k + 8` best. Those are ranked on exact distances with ties to the
/// lower index, which makes the result equal to an exhaustive search.
pub fn knn(x: &[f64], n: usize, dim: usize, k: usize) -> Vec<Vec<usize>> {
    assert_eq!(x.len(), n * dim, "point array has the wrong length");
    assert!(k < n, "k must be smaller than the number of points");
    let norms: Vec<f64> = (0..n)
        .map(|i| dist2(&x[i * dim..(i + 1) * dim], &vec![0.0; dim]))
        .collect();
    let all = MatRef::from_row_major_slice(x, n, dim);
    let cand = (k + MARGIN).min(n - 1);
    (0..n.div_ceil(BLOCK))
        .into_par_iter()
        .flat_map_iter(|b| {
            let lo = b * BLOCK;
            let hi = (lo + BLOCK).min(n);
            let rows = hi - lo;
            let blk = MatRef::from_row_major_slice(&x[lo * dim..hi * dim], rows, dim);
            // Bounded max-heaps of the best candidates per row on the approximate
            // distance; |x_i|^2 is common to the row and left out. Columns are
            // taken in tiles that stay in cache.
            let mut heaps: Vec<BinaryHeap<(Key, usize)>> = (0..rows)
                .map(|_| BinaryHeap::with_capacity(cand + 1))
                .collect();
            let mut g = vec![0.0; rows * TILE];
            for c0 in (0..n).step_by(TILE) {
                let c1 = (c0 + TILE).min(n);
                let w = c1 - c0;
                faer::linalg::matmul::matmul(
                    MatMut::from_row_major_slice_mut(&mut g[..rows * w], rows, w),
                    Accum::Replace,
                    blk,
                    all.subrows(c0, w).transpose(),
                    1.0,
                    Par::Seq,
                );
                for (r, heap) in heaps.iter_mut().enumerate() {
                    let i = lo + r;
                    for (t, j) in (c0..c1).enumerate() {
                        if j == i {
                            continue;
                        }
                        let v = norms[j] - 2.0 * g[r * w + t];
                        if heap.len() < cand {
                            heap.push((Key(v), j));
                        } else if v < heap.peek().unwrap().0.0 {
                            heap.pop();
                            heap.push((Key(v), j));
                        }
                    }
                }
            }
            heaps
                .into_iter()
                .enumerate()
                .map(|(r, heap)| {
                    let i = lo + r;
                    let xi = &x[i * dim..(i + 1) * dim];
                    let mut d: Vec<(f64, usize)> = heap
                        .into_iter()
                        .map(|(_, j)| (dist2(xi, &x[j * dim..(j + 1) * dim]), j))
                        .collect();
                    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    d.iter().take(k).map(|&(_, j)| j).collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}
