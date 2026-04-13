//! Discrete fiber bundle connections and covariant Laplacian.
//!
//! A discrete connection stores SO(d) frame transport matrices per mesh edge.
//! The covariant Laplacian uses these transports with cotangent weights to
//! compute the connection Laplacian on any fiber bundle section.
//!
//! ## Sign convention
//!
//! The DEC Laplacian is **positive-semidefinite** (positive at maxima):
//!
//! ```text
//! (Delta s)_v = (1/A_v) * sum_{e incident to v} w_e * (s_v - T_e(s_neighbor))
//! ```
//!
//! Elastic smoothing in physical equations enters as **-K * Delta** (minus sign).

#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

use crate::fiber::{Fiber, FiberOps, Section, VecSection};

/// Discrete connection: SO(D) frame transport per edge.
///
/// Each edge stores a D x D orthogonal matrix (row-major flat slice of
/// length D*D) representing parallel transport from vertex v0 to v1.
/// The reverse transport (v1 to v0) is the matrix transpose.
pub trait DiscreteConnection<const D: usize> {
    /// Number of edges.
    fn n_edges(&self) -> usize;

    /// Edge endpoints: `[v0, v1]` for edge `e`.
    fn edge_vertices(&self, e: usize) -> [usize; 2];

    /// SO(D) transport matrix for edge `e` (v0 -> v1), row-major flat.
    /// Returned slice has length >= D*D.
    fn frame_transport(&self, e: usize) -> &[f64];

    /// Transport a fiber element from v0 to v1 along edge `e`.
    fn transport_forward<F: Fiber>(&self, e: usize, element: &F::Element) -> F::Element {
        F::transport_by(self.frame_transport(e), D, element)
    }

    /// Transport a fiber element from v1 to v0 along edge `e` (uses R^T).
    fn transport_reverse<F: Fiber>(&self, e: usize, element: &F::Element) -> F::Element {
        let r = self.frame_transport(e);
        let mut r_t = [0.0_f64; 9]; // max D=3 -> 9 entries
        for i in 0..D {
            for j in 0..D {
                r_t[i * D + j] = r[j * D + i];
            }
        }
        F::transport_by(&r_t[..D * D], D, element)
    }
}

/// Edge-based transport storage for SO(2) (4 floats per edge).
#[derive(Clone, Debug)]
pub struct EdgeTransport2D {
    /// Edge endpoints.
    pub edges: Vec<[usize; 2]>,
    /// SO(2) transport matrices, row-major: [cos, -sin, sin, cos] per edge.
    pub transports: Vec<[f64; 4]>,
}

impl DiscreteConnection<2> for EdgeTransport2D {
    fn n_edges(&self) -> usize { self.edges.len() }
    fn edge_vertices(&self, e: usize) -> [usize; 2] { self.edges[e] }
    fn frame_transport(&self, e: usize) -> &[f64] { &self.transports[e] }
}

/// Edge-based transport storage for SO(3) (9 floats per edge).
#[derive(Clone, Debug)]
pub struct EdgeTransport3D {
    /// Edge endpoints.
    pub edges: Vec<[usize; 2]>,
    /// SO(3) transport matrices, row-major (9 floats per edge).
    pub transports: Vec<[f64; 9]>,
}

impl DiscreteConnection<3> for EdgeTransport3D {
    fn n_edges(&self) -> usize { self.edges.len() }
    fn edge_vertices(&self, e: usize) -> [usize; 2] { self.edges[e] }
    fn frame_transport(&self, e: usize) -> &[f64] { &self.transports[e] }
}

/// Generic covariant Laplacian on fiber bundle sections.
///
/// Applies the DEC connection Laplacian to a section of any fiber bundle:
///
/// ```text
/// (Delta s)_v = (1/A_v) * sum_{e incident to v} w_e * (s_v - T_e(s_neighbor))
/// ```
///
/// where T_e is the parallel transport from neighbor to v along edge e,
/// w_e is the cotangent weight, and A_v is the dual cell area.
///
/// **Positive-semidefinite** (positive at maxima). Use `-K * lap` for smoothing.
pub struct CovLaplacian {
    /// Cotangent weights per edge.
    cot_weights: Vec<f64>,
    /// Dual cell area per vertex (star_0).
    dual_areas: Vec<f64>,
    /// Edge endpoints.
    edges: Vec<[usize; 2]>,
    /// For each vertex: list of (edge_idx, is_v0) pairs.
    vertex_edges: Vec<Vec<(usize, bool)>>,
}

impl CovLaplacian {
    /// Build the Laplacian stencil from mesh topology and weights.
    ///
    /// `n_vertices` is the total vertex count.
    /// `edges` are `[v0, v1]` pairs (same order as the connection).
    /// `cot_weights[e]` is the cotangent weight for edge e.
    /// `dual_areas[v]` is the dual cell area (star_0) for vertex v.
    pub fn new(
        n_vertices: usize,
        edges: &[[usize; 2]],
        cot_weights: &[f64],
        dual_areas: &[f64],
    ) -> Self {
        let mut vertex_edges: Vec<Vec<(usize, bool)>> = vec![vec![]; n_vertices];
        for (e, &[v0, v1]) in edges.iter().enumerate() {
            vertex_edges[v0].push((e, true));
            vertex_edges[v1].push((e, false));
        }
        Self {
            cot_weights: cot_weights.to_vec(),
            dual_areas: dual_areas.to_vec(),
            edges: edges.to_vec(),
            vertex_edges,
        }
    }

    /// Apply the covariant Laplacian with a discrete connection.
    ///
    /// Generic over fiber type `F` and connection dimension `D`.
    pub fn apply<F, const D: usize, C>(
        &self,
        section: &impl Section<F>,
        conn: &C,
    ) -> VecSection<F>
    where
        F: FiberOps,
        C: DiscreteConnection<D>,
    {
        let nv = section.n_vertices();
        let mut result = VecSection::<F>::zeros(nv);

        for v in 0..nv {
            let s_v = section.at(v);

            for &(e, is_v0) in &self.vertex_edges[v] {
                let neighbor = if is_v0 { self.edges[e][1] } else { self.edges[e][0] };
                let s_neighbor = section.at(neighbor);
                let w = self.cot_weights[e];

                // Transport neighbor's value to v's frame.
                let transported = if is_v0 {
                    conn.transport_reverse::<F>(e, s_neighbor)
                } else {
                    conn.transport_forward::<F>(e, s_neighbor)
                };

                // Accumulate w * (s_v - transported).
                F::accumulate_diff(result.at_mut(v), s_v, &transported, w);
            }

            // Divide by dual area.
            if self.dual_areas[v] > 1e-30 {
                F::scale_element(result.at_mut(v), 1.0 / self.dual_areas[v]);
            }
        }

        result
    }
}
