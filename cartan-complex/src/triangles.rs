//! Triangle enumeration for the flag complex of a graph.
//!
//! The set is every `a < b < c` whose three pairs are edges, each emitted
//! once. A dense `n * n` adjacency matrix would take 17 GB of `bool` at 130,000
//! cells. This enumeration intersects sorted upper neighbour lists and returns
//! the set in lexicographic order.

use rayon::prelude::*;

use crate::Complex;

/// Every 3-clique `[a, b, c]` with `a < b < c`, in lexicographic order.
pub fn triangles_of(complex: &Complex) -> Vec<[u32; 3]> {
    let n = complex.n_vertices();
    let mut upper: Vec<Vec<u32>> = vec![Vec::new(); n];
    // Edges are sorted, so each upper list is built in ascending order.
    for e in complex.edges() {
        upper[e[0] as usize].push(e[1]);
    }
    (0..n)
        .into_par_iter()
        .flat_map_iter(|a| {
            let ua = &upper[a];
            let mut out = Vec::new();
            for (i, &b) in ua.iter().enumerate() {
                // Candidates c > b adjacent to a are ua[i + 1..]; keep those
                // also adjacent to b, by a merge of two sorted lists.
                let (mut x, mut y) = (&ua[i + 1..], upper[b as usize].as_slice());
                while let (Some(&p), Some(&q)) = (x.first(), y.first()) {
                    match p.cmp(&q) {
                        std::cmp::Ordering::Less => x = &x[1..],
                        std::cmp::Ordering::Greater => y = &y[1..],
                        std::cmp::Ordering::Equal => {
                            out.push([a as u32, b, p]);
                            x = &x[1..];
                            y = &y[1..];
                        }
                    }
                }
            }
            out
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Complex;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn graph(n: usize, edges: Vec<[u32; 2]>) -> Complex {
        Complex::new(n, edges).unwrap()
    }

    /// Every `a < b < c` whose three pairs are edges, by brute force. This is
    /// the definition the faster enumeration has to agree with.
    fn dense_reference(c: &Complex) -> Vec<[u32; 3]> {
        let n = c.n_vertices();
        let mut adj = vec![false; n * n];
        for e in c.edges() {
            adj[e[0] as usize * n + e[1] as usize] = true;
        }
        let has = |a: usize, b: usize| adj[a * n + b];
        let mut out = Vec::new();
        for a in 0..n {
            for b in a + 1..n {
                for cc in b + 1..n {
                    if has(a, b) && has(b, cc) && has(a, cc) {
                        out.push([a as u32, b as u32, cc as u32]);
                    }
                }
            }
        }
        out
    }

    #[test]
    fn k4_has_four_triangles() {
        let c = graph(4, vec![[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]);
        assert_eq!(
            triangles_of(&c),
            vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        );
    }

    #[test]
    fn a_square_has_none() {
        let c = graph(4, vec![[0, 1], [1, 2], [2, 3], [0, 3]]);
        assert!(triangles_of(&c).is_empty());
    }

    #[test]
    fn matches_dense_reference() {
        let mut rng = ChaCha8Rng::seed_from_u64(20261111);
        for i in 0..2000 {
            let n = rng.random_range(3..=40usize);
            let p = [0.1, 0.3, 0.6, 0.9][i % 4];
            let mut edges = Vec::new();
            for a in 0..n as u32 {
                for b in a + 1..n as u32 {
                    if rng.random::<f64>() < p {
                        edges.push([a, b]);
                    }
                }
            }
            let c = graph(n, edges);
            assert_eq!(triangles_of(&c), dense_reference(&c), "graph {i}, n = {n}");
        }
    }
}
