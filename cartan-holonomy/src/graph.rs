//! Mutual nearest-neighbour graph and its triangles.

use rayon::prelude::*;
use std::collections::HashSet;

/// The mutual `k`-nearest-neighbour graph: `i ~ j` when each lists the other
/// among its first `k` neighbours.
///
/// A symmetrised graph gives hub points thousands of edges and a triangle count
/// quadratic in that degree. The mutual graph caps every degree at `k`.
#[derive(Debug, Clone)]
pub struct MutualGraph {
    /// Number of vertices.
    pub n: usize,
    /// Sorted neighbour list of every vertex.
    pub adj: Vec<Vec<usize>>,
    /// Undirected edges `(i, j)` with `i < j`, in lexicographic order.
    pub edges: Vec<[usize; 2]>,
}

impl MutualGraph {
    /// Builds the graph from neighbour lists ordered nearest first.
    pub fn new(nbrs: &[Vec<usize>], k: usize) -> Self {
        let n = nbrs.len();
        let near: Vec<HashSet<usize>> = nbrs
            .iter()
            .map(|r| r[..k].iter().copied().collect())
            .collect();
        let mut sets: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        for i in 0..n {
            for &j in &nbrs[i][..k] {
                if near[j].contains(&i) {
                    sets[i].insert(j);
                    sets[j].insert(i);
                }
            }
        }
        let adj: Vec<Vec<usize>> = sets
            .into_iter()
            .map(|s| {
                let mut v: Vec<usize> = s.into_iter().collect();
                v.sort_unstable();
                v
            })
            .collect();
        let edges = (0..n)
            .flat_map(|i| adj[i].iter().filter(move |&&j| j > i).map(move |&j| [i, j]))
            .collect();
        Self { n, adj, edges }
    }

    /// Whether `i ~ j`.
    pub fn adjacent(&self, i: usize, j: usize) -> bool {
        self.adj[i].binary_search(&j).is_ok()
    }

    /// The 3-cliques `(i, j, k)`, `i < j < k`, in lexicographic order.
    pub fn triangles(&self) -> Vec<[usize; 3]> {
        (0..self.n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let up: Vec<usize> = self.adj[i].iter().copied().filter(|&j| j > i).collect();
                let mut local = Vec::new();
                for (a, &j) in up.iter().enumerate() {
                    for &k in &up[a + 1..] {
                        if self.adjacent(j, k) {
                            local.push([i, j, k]);
                        }
                    }
                }
                local
            })
            .collect()
    }
}
