//! The discrete Levi-Civita connection of a point cloud.

use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::frames::frames;
use crate::graph::MutualGraph;
use crate::knn::knn;

/// Transport from the frame `tb` to the frame `ta`, in frame coordinates:
/// `polar(ta^T tb)`.
///
/// The Grassmann geodesic from `span tb` to `span ta` carries `tb` to a frame
/// `Y` of `span ta`, and the transport is `ta^T Y`. That product equals the
/// orthogonal factor of the polar decomposition of `ta^T tb`, which is what is
/// computed. The reverse transport is the transpose.
pub fn transport(ta: &DMatrix<f64>, tb: &DMatrix<f64>) -> DMatrix<f64> {
    let svd = (ta.transpose() * tb).svd(true, true);
    svd.u.expect("left singular vectors") * svd.v_t.expect("right singular vectors")
}

/// Frames, graph and edge transports of a point cloud.
#[derive(Debug, Clone)]
pub struct Connection {
    /// Number of points.
    pub n: usize,
    /// Ambient dimension.
    pub dim: usize,
    /// Frame dimension.
    pub d: usize,
    /// Tangent frames, `n` blocks of shape `(dim, d)`, row-major.
    pub frames: Vec<f64>,
    /// The mutual neighbour graph.
    pub graph: MutualGraph,
    /// CSR offsets of the directed edges at each point, in the order of
    /// `graph.adj`.
    pub start: Vec<usize>,
    /// `O_ij`, the transport from the frame at `j` to the frame at `i`, for each
    /// directed slot `(i, j)`: `d x d`, row-major.
    pub o: Vec<f64>,
}

impl Connection {
    /// Builds frames from `k_frame` neighbours, the mutual `k_graph` graph and
    /// the transport along each of its edges.
    pub fn new(x: &[f64], n: usize, dim: usize, d: usize, k_frame: usize, k_graph: usize) -> Self {
        let nbrs = knn(x, n, dim, k_frame.max(k_graph));
        let frames = frames(x, n, dim, &nbrs, k_frame, d);
        let graph = MutualGraph::new(&nbrs, k_graph);
        Self::from_parts(n, dim, d, frames, graph)
    }

    /// Builds the transports for given frames and graph.
    pub fn from_parts(
        n: usize,
        dim: usize,
        d: usize,
        frames: Vec<f64>,
        graph: MutualGraph,
    ) -> Self {
        assert_eq!(
            frames.len(),
            n * dim * d,
            "frame array has the wrong length"
        );
        let mut start = vec![0usize; n + 1];
        for i in 0..n {
            start[i + 1] = start[i] + graph.adj[i].len();
        }
        let mut c = Self {
            n,
            dim,
            d,
            frames,
            graph,
            start,
            o: Vec::new(),
        };
        let o: Vec<f64> = (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let ti = c.frame(i);
                let mut out = Vec::with_capacity(c.graph.adj[i].len() * d * d);
                for &j in &c.graph.adj[i] {
                    let q = transport(&ti, &c.frame(j));
                    for r in 0..d {
                        for s in 0..d {
                            out.push(q[(r, s)]);
                        }
                    }
                }
                out
            })
            .collect();
        c.o = o;
        c
    }

    /// The frame at `i` as a `dim x d` matrix.
    pub fn frame(&self, i: usize) -> DMatrix<f64> {
        let b = self.dim * self.d;
        DMatrix::from_row_slice(self.dim, self.d, &self.frames[i * b..(i + 1) * b])
    }

    /// Directed slot of the edge `(i, j)`.
    pub fn slot(&self, i: usize, j: usize) -> usize {
        let p = self.graph.adj[i].binary_search(&j).expect("not an edge");
        self.start[i] + p
    }

    /// `O_ij` as a matrix: from the frame at `j` to the frame at `i`.
    pub fn transport_into(&self, i: usize, j: usize) -> DMatrix<f64> {
        let dd = self.d * self.d;
        let s = self.slot(i, j);
        DMatrix::from_row_slice(self.d, self.d, &self.o[s * dd..(s + 1) * dd])
    }

    /// Coordinates of `v` (length `dim`) in the frame at `i`.
    pub fn tangent(&self, i: usize, v: &[f64]) -> Vec<f64> {
        let (dim, d) = (self.dim, self.d);
        let f = &self.frames[i * dim * d..(i + 1) * dim * d];
        (0..d)
            .map(|c| (0..dim).map(|k| f[k * d + c] * v[k]).sum())
            .collect()
    }

    /// Number of directed edges.
    pub fn directed_edges(&self) -> usize {
        self.start[self.n]
    }
}
