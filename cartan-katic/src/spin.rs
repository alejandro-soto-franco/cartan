//! Spin structures on a simplicial complex.
//!
//! A rotor per edge is already a lift of an `SO(3)` transition, so the choice
//! of spin structure is implicit in any connection `cartan-core` stores. This
//! module makes it explicit and checkable.
//!
//! For each triangle the connection gives one `SU(2)` holonomy `P`, while its
//! `SO(3)` holonomy `M` has the two lifts `+/- P_ref`, with `P_ref` the
//! deterministic lift `Rotor3::from_matrix(M)`. The sign relating them is a
//! `Z_2` two-cochain `w`. A spin structure is an edge cochain `s` with
//! `delta s = w`, so one exists exactly when `w` is a coboundary, and `[w]` in
//! `H^2(M; Z_2)` is the discrete second Stiefel-Whitney class.
//!
//! Flipping a single edge changes `s` and leaves `[w]` alone: that is a gauge
//! transformation, not an obstruction. An obstruction needs a two-cycle
//! carrying an odd total `w`.
//!
//! The module takes an incidence rather than a mesh type, so it is testable
//! with no mesh generator and any complex maps onto it through one adapter.

use cartan_core::rotor::Rotor3;

use crate::error::KaticError;

/// Vertex, edge and triangle incidence of a simplicial complex.
#[derive(Clone, Debug)]
pub struct Incidence {
    n_vertices: usize,
    edges: Vec<[usize; 2]>,
    /// Triangles as vertex triples, ascending.
    triangles: Vec<[usize; 3]>,
}

impl Incidence {
    /// Build from triangles given as vertex triples. Edges are derived and
    /// deduplicated, each stored with its lower vertex first.
    #[must_use]
    pub fn from_triangles(n_vertices: usize, triangles: &[[usize; 3]]) -> Self {
        let mut edges: Vec<[usize; 2]> = Vec::new();
        let mut tris: Vec<[usize; 3]> = Vec::new();
        for t in triangles {
            let mut v = *t;
            v.sort_unstable();
            tris.push(v);
            for (a, b) in [(v[0], v[1]), (v[1], v[2]), (v[0], v[2])] {
                let e = [a, b];
                if !edges.contains(&e) {
                    edges.push(e);
                }
            }
        }
        Self {
            n_vertices,
            edges,
            triangles: tris,
        }
    }

    /// Number of vertices.
    #[must_use]
    pub fn n_vertices(&self) -> usize {
        self.n_vertices
    }

    /// Number of edges.
    #[must_use]
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Number of triangles.
    #[must_use]
    pub fn n_triangles(&self) -> usize {
        self.triangles.len()
    }

    /// Index of the edge on `a` and `b`.
    fn edge_index(&self, a: usize, b: usize) -> usize {
        let e = if a < b { [a, b] } else { [b, a] };
        self.edges
            .iter()
            .position(|x| *x == e)
            .expect("edge not in complex")
    }

    /// The three edge indices of triangle `t`.
    #[must_use]
    pub fn triangle_edges(&self, t: usize) -> [usize; 3] {
        let v = self.triangles[t];
        [
            self.edge_index(v[0], v[1]),
            self.edge_index(v[1], v[2]),
            self.edge_index(v[0], v[2]),
        ]
    }

    /// `delta^0`, vertices to edges, over GF(2): one row per edge.
    fn d0_rows(&self) -> Vec<Vec<bool>> {
        self.edges
            .iter()
            .map(|e| {
                let mut row = vec![false; self.n_vertices];
                row[e[0]] ^= true;
                row[e[1]] ^= true;
                row
            })
            .collect()
    }

    /// `delta^1`, edges to triangles, over GF(2): one row per triangle.
    fn d1_rows(&self) -> Vec<Vec<bool>> {
        (0..self.n_triangles())
            .map(|t| {
                let mut row = vec![false; self.n_edges()];
                for e in self.triangle_edges(t) {
                    row[e] ^= true;
                }
                row
            })
            .collect()
    }
}

/// Row-reduce over GF(2); returns the rank and the reduced rows.
fn row_reduce(mut rows: Vec<Vec<bool>>, width: usize) -> (usize, Vec<Vec<bool>>) {
    let mut rank = 0;
    for col in 0..width {
        let Some(pivot) = (rank..rows.len()).find(|&r| rows[r][col]) else {
            continue;
        };
        rows.swap(rank, pivot);
        let pivot_row = rows[rank].clone();
        for (r, row) in rows.iter_mut().enumerate() {
            if r != rank && row[col] {
                for (c, v) in pivot_row.iter().enumerate().take(width) {
                    row[c] ^= *v;
                }
            }
        }
        rank += 1;
    }
    (rank, rows)
}

/// Squared distance between two rotors in `R^4`.
fn dist_sq(a: &Rotor3, b: &Rotor3) -> f64 {
    let (dw, dx, dy, dz) = (a.w - b.w, a.x - b.x, a.y - b.y, a.z - b.z);
    dw * dw + dx * dx + dy * dy + dz * dz
}

/// A spin structure: the `Z_2` edge cochain trivialising the lift cocycle.
#[derive(Clone, Debug)]
pub struct SpinStructure {
    lift: Vec<bool>,
    cocycle: Vec<bool>,
    n_sectors: usize,
}

impl SpinStructure {
    /// The lift cocycle `w` of a rotor connection: one bit per triangle.
    ///
    /// `true` means the connection's `SU(2)` holonomy is the negative of the
    /// deterministic lift of its own `SO(3)` holonomy.
    #[must_use]
    pub fn cocycle_of(inc: &Incidence, edge_rotors: &[Rotor3]) -> Vec<bool> {
        (0..inc.n_triangles())
            .map(|t| {
                let [e0, e1, e2] = inc.triangle_edges(t);
                // Traverse v0 -> v1 -> v2 -> v0; the third edge is stored
                // as v0 -> v2, so it enters reversed.
                let p = edge_rotors[e0]
                    .compose(&edge_rotors[e1])
                    .compose(&edge_rotors[e2].reverse());
                let reference = Rotor3::from_matrix(&p.to_matrix());
                dist_sq(&p, &reference) > dist_sq(&p, &neg(&reference))
            })
            .collect()
    }

    /// Solve `delta s = w` for the edge cochain, or report the obstruction.
    ///
    /// Fails with the first triangle of an unsolvable system, which is the
    /// discrete appearance of a non-vanishing second Stiefel-Whitney class.
    pub fn from_connection(inc: &Incidence, edge_rotors: &[Rotor3]) -> Result<Self, KaticError> {
        assert_eq!(edge_rotors.len(), inc.n_edges(), "one rotor per edge");
        let cocycle = Self::cocycle_of(inc, edge_rotors);

        // Augmented system [delta^1 | w], reduced over GF(2).
        let width = inc.n_edges();
        let rows: Vec<Vec<bool>> = inc
            .d1_rows()
            .into_iter()
            .zip(cocycle.iter())
            .map(|(mut r, &w)| {
                r.push(w);
                r
            })
            .collect();
        let (_, reduced) = row_reduce(rows, width + 1);
        for (t, r) in reduced.iter().enumerate() {
            if r[width] && !r[..width].iter().any(|&b| b) {
                return Err(KaticError::InconsistentSpinLift { triangle: t });
            }
        }

        // Back-substitute one particular solution.
        let mut lift = vec![false; width];
        for r in &reduced {
            if let Some(p) = (0..width).find(|&c| r[c]) {
                let mut acc = r[width];
                for c in (p + 1)..width {
                    if r[c] {
                        acc ^= lift[c];
                    }
                }
                lift[p] = acc;
            }
        }

        // Inequivalent structures form a torsor over H^1(M; Z_2).
        let (rank_d1, _) = row_reduce(inc.d1_rows(), inc.n_edges());
        let (rank_d0, _) = row_reduce(inc.d0_rows(), inc.n_vertices());
        let b1 = inc.n_edges() - rank_d1 - rank_d0;
        Ok(Self {
            lift,
            cocycle,
            n_sectors: 1usize << b1,
        })
    }

    /// The trivialising edge cochain.
    #[must_use]
    pub fn lift(&self) -> &[bool] {
        &self.lift
    }

    /// The lift cocycle this structure trivialises.
    #[must_use]
    pub fn cocycle(&self) -> &[bool] {
        &self.cocycle
    }

    /// Number of inequivalent spin structures, `2^dim H^1(M; Z_2)`.
    #[must_use]
    pub fn n_sectors(&self) -> usize {
        self.n_sectors
    }
}

fn neg(r: &Rotor3) -> Rotor3 {
    Rotor3 {
        w: -r.w,
        x: -r.x,
        y: -r.y,
        z: -r.z,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A triangulated disc: one interior vertex joined to a triangulated
    /// boundary square. `b1 = 0`.
    fn disc() -> Incidence {
        Incidence::from_triangles(5, &[[0, 1, 4], [1, 2, 4], [2, 3, 4], [0, 3, 4]])
    }

    /// The boundary of a tetrahedron: a triangulated `S^2`, `b1 = 0`, and the
    /// only closed surface small enough to type out.
    fn sphere() -> Incidence {
        Incidence::from_triangles(4, &[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    }

    /// A triangulated annulus: two rings of four vertices. `b1 = 1`.
    fn annulus() -> Incidence {
        let mut t = Vec::new();
        for i in 0..4 {
            let j = (i + 1) % 4;
            t.push([i, j, 4 + i]);
            t.push([j, 4 + j, 4 + i]);
        }
        Incidence::from_triangles(8, &t)
    }

    fn identities(inc: &Incidence) -> Vec<Rotor3> {
        vec![Rotor3::IDENTITY; inc.n_edges()]
    }

    #[test]
    fn a_trivial_connection_has_a_zero_cocycle() {
        for inc in [disc(), sphere(), annulus()] {
            let w = SpinStructure::cocycle_of(&inc, &identities(&inc));
            assert!(w.iter().all(|&b| !b), "trivial connection must give w = 0");
        }
    }

    #[test]
    fn flipping_one_edge_is_a_gauge_change_not_an_obstruction() {
        let inc = sphere();
        let mut rotors = identities(&inc);
        rotors[0] = neg(&Rotor3::IDENTITY);
        let s = SpinStructure::from_connection(&inc, &rotors)
            .expect("a single flip is gauge, so it must stay consistent");
        // The flip shows up in the cocycle on the triangles containing it.
        let touched = (0..inc.n_triangles())
            .filter(|&t| inc.triangle_edges(t).contains(&0))
            .count();
        assert_eq!(s.cocycle().iter().filter(|&&b| b).count(), touched);
    }

    #[test]
    fn an_odd_cocycle_on_a_closed_surface_is_obstructed() {
        let inc = sphere();
        // The sum over a closed surface of a coboundary is even, so an odd
        // total cannot be trivialised.
        let width = inc.n_edges();
        let rows: Vec<Vec<bool>> = inc
            .d1_rows()
            .into_iter()
            .enumerate()
            .map(|(t, mut r)| {
                r.push(t == 0);
                r
            })
            .collect();
        let (_, reduced) = row_reduce(rows, width + 1);
        let unsolvable = reduced
            .iter()
            .any(|r| r[width] && !r[..width].iter().any(|&b| b));
        assert!(unsolvable, "one odd triangle on S^2 must be obstructed");
    }

    #[test]
    fn sector_count_is_two_to_the_first_betti_number() {
        assert_eq!(
            SpinStructure::from_connection(&disc(), &identities(&disc()))
                .unwrap()
                .n_sectors(),
            1
        );
        assert_eq!(
            SpinStructure::from_connection(&sphere(), &identities(&sphere()))
                .unwrap()
                .n_sectors(),
            1
        );
        assert_eq!(
            SpinStructure::from_connection(&annulus(), &identities(&annulus()))
                .unwrap()
                .n_sectors(),
            2
        );
    }

    #[test]
    fn the_lift_trivialises_the_cocycle() {
        let inc = annulus();
        let mut rotors = identities(&inc);
        rotors[2] = neg(&Rotor3::IDENTITY);
        rotors[5] = neg(&Rotor3::IDENTITY);
        let s = SpinStructure::from_connection(&inc, &rotors).expect("consistent");
        for t in 0..inc.n_triangles() {
            let d: bool = inc
                .triangle_edges(t)
                .iter()
                .fold(false, |a, &e| a ^ s.lift()[e]);
            assert_eq!(d, s.cocycle()[t], "delta s must equal w on triangle {t}");
        }
    }
}
