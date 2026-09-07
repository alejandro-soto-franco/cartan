//! Disclination lines and their non-abelian charges.
//!
//! A p-atic state is a coset `R H^`, so a rotor field is defined only up to
//! `H^` at each vertex. Fixing representatives fixes a gauge, and the
//! transition on an edge is the element of the defect group that best aligns
//! the two ends. The holonomy around a triangle is then a group element:
//! trivial when no line pierces it, and the charge of the line when one does.
//!
//! **No threshold and no scalar order parameter.** The usual method looks for
//! places where the order melts, which needs a tuned cut-off and yields no
//! charge. This one is exact and gives the charge directly.
//!
//! ## What a charge is
//!
//! `pi_1` of the order-parameter manifold is the lift `H^`, which is
//! non-abelian beyond the cyclic case, so:
//!
//! - a charge is a conjugacy class with a base point, never a number;
//! - transport conjugates it, so only the class is path independent;
//! - fusion is path dependent, which no tensor invariant can express.
//!
//! The types here offer no addition on charges, deliberately.

use cartan_core::rotor::Rotor3;

use crate::complex3::Complex3;
use crate::group::SymmetryGroup;

fn dist_sq(a: &Rotor3, b: &Rotor3) -> f64 {
    let (dw, dx, dy, dz) = (a.w - b.w, a.x - b.x, a.y - b.y, a.z - b.z);
    dw * dw + dx * dx + dy * dy + dz * dz
}

/// The element of the defect group best aligning `rv` with `ru`.
///
/// This is the gauge transition on the edge: the `h` minimising the distance
/// between `R_u` and `R_v h`.
///
/// When `-1` is the trivial charge the search runs over the group extended by
/// `-1`, because `R` and `-R` are then the same state and a search that
/// ignores the sign will report a large residual where there is none. That
/// omission breaks conservation rather than accuracy: the detector loses the
/// line exactly where the field's chart wraps, and tetrahedra there end up
/// with an odd number of pierced faces.
#[must_use]
pub fn edge_transition<H: SymmetryGroup>(ru: &Rotor3, rv: &Rotor3) -> Rotor3 {
    let mut best = Rotor3::IDENTITY;
    let mut best_d = f64::INFINITY;
    for h in H::defect_group() {
        let candidates = if H::CHARGE_MODULO_SIGN {
            vec![
                h,
                Rotor3 {
                    w: -h.w,
                    x: -h.x,
                    y: -h.y,
                    z: -h.z,
                },
            ]
        } else {
            vec![h]
        };
        for cand in candidates {
            let d = dist_sq(ru, &rv.compose(&cand));
            if d < best_d {
                best_d = d;
                best = cand;
            }
        }
    }
    best
}

/// The holonomy around a triangle, as an element of the defect group.
///
/// Traverses `a -> b -> c -> a` on the ascending vertex order, composing the
/// edge transitions. A trivial result means no line pierces the triangle.
#[must_use]
pub fn face_holonomy<H: SymmetryGroup>(rotors: &[Rotor3], tri: &[usize; 3]) -> Rotor3 {
    let [a, b, c] = *tri;
    let uab = edge_transition::<H>(&rotors[a], &rotors[b]);
    let ubc = edge_transition::<H>(&rotors[b], &rotors[c]);
    let uca = edge_transition::<H>(&rotors[c], &rotors[a]);
    uab.compose(&ubc).compose(&uca)
}

/// Whether a holonomy is the trivial charge for symmetry `H`.
///
/// Whether `-1` counts as trivial is a property of the symmetry, not a
/// convention: it is contractible in `RP^2` and is not in `SU(2)/H^` for a
/// finite lift. See [`SymmetryGroup::CHARGE_MODULO_SIGN`].
#[must_use]
pub fn is_trivial_for<H: SymmetryGroup>(h: &Rotor3, tol: f64) -> bool {
    let d = dist_sq(h, &Rotor3::IDENTITY);
    if H::CHARGE_MODULO_SIGN {
        let neg = Rotor3 {
            w: -h.w,
            x: -h.x,
            y: -h.y,
            z: -h.z,
        };
        d.min(dist_sq(&neg, &Rotor3::IDENTITY)) < tol * tol
    } else {
        d < tol * tol
    }
}

/// Whether a rotor is the identity, ignoring the symmetry.
#[must_use]
pub fn is_identity(h: &Rotor3, tol: f64) -> bool {
    dist_sq(h, &Rotor3::IDENTITY) < tol * tol
}

/// A face pierced by a disclination line, with the charge of that line.
#[derive(Clone, Copy, Debug)]
pub struct PiercedFace {
    /// Index of the triangle in the complex.
    pub triangle: usize,
    /// The holonomy, a representative of the charge class at this base point.
    pub holonomy: Rotor3,
}

/// The disclination content of one field.
#[derive(Clone, Debug)]
pub struct DefectField {
    pierced: Vec<PiercedFace>,
    per_tet: Vec<usize>,
}

impl DefectField {
    /// Detect every pierced face.
    #[must_use]
    pub fn detect<H: SymmetryGroup>(c: &Complex3, rotors: &[Rotor3], tol: f64) -> Self {
        let mut pierced = Vec::new();
        let mut is_pierced = vec![false; c.n_triangles()];
        for (i, flag) in is_pierced.iter_mut().enumerate() {
            let tri = c.triangle(i);
            let h = face_holonomy::<H>(rotors, &tri);
            if !is_trivial_for::<H>(&h, tol) {
                *flag = true;
                pierced.push(PiercedFace {
                    triangle: i,
                    holonomy: h,
                });
            }
        }
        let per_tet = (0..c.n_tets())
            .map(|t| {
                c.tet_triangles(t)
                    .iter()
                    .filter(|&&f| is_pierced[f])
                    .count()
            })
            .collect();
        Self { pierced, per_tet }
    }

    /// The pierced faces.
    #[must_use]
    pub fn pierced(&self) -> &[PiercedFace] {
        &self.pierced
    }

    /// Number of pierced faces on each tetrahedron.
    ///
    /// A disclination line has no endpoints, so a line entering a tetrahedron
    /// leaves it and this count is even everywhere. It is the sharpest cheap
    /// check on the detector.
    #[must_use]
    pub fn pierced_per_tet(&self) -> &[usize] {
        &self.per_tet
    }

    /// The largest odd count, or zero when the field is conservative.
    #[must_use]
    pub fn worst_parity_violation(&self) -> usize {
        self.per_tet.iter().filter(|c| !c.is_multiple_of(2)).count()
    }

    /// Lines as connected components of pierced faces sharing a tetrahedron.
    #[must_use]
    pub fn lines(&self, c: &Complex3) -> Vec<Vec<usize>> {
        let mut face_to_comp: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut comps: Vec<Vec<usize>> = Vec::new();
        for t in 0..c.n_tets() {
            let here: Vec<usize> = c
                .tet_triangles(t)
                .iter()
                .copied()
                .filter(|f| self.pierced.iter().any(|p| p.triangle == *f))
                .collect();
            if here.is_empty() {
                continue;
            }
            let existing: Vec<usize> = here
                .iter()
                .filter_map(|f| face_to_comp.get(f).copied())
                .collect();
            let target = existing.first().copied().unwrap_or_else(|| {
                comps.push(Vec::new());
                comps.len() - 1
            });
            for &other in &existing {
                if other != target {
                    let moved = std::mem::take(&mut comps[other]);
                    for f in &moved {
                        face_to_comp.insert(*f, target);
                    }
                    comps[target].extend(moved);
                }
            }
            for f in here {
                if let std::collections::hash_map::Entry::Vacant(e) = face_to_comp.entry(f) {
                    e.insert(target);
                    comps[target].push(f);
                }
            }
        }
        comps.into_iter().filter(|c| !c.is_empty()).collect()
    }
}
