//! Faces of a simplex, and the index set of a basis of k-forms.

/// `C(n, k)`, without overflow for the sizes a simplex reaches.
pub fn n_choose_k(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut out = 1usize;
    for i in 0..k {
        out = out * (n - i) / (i + 1);
    }
    out
}

/// The `k`-faces of a simplex on `n_vertices`, as sorted vertex index sets.
///
/// A k-face has `k + 1` vertices, so the 1-faces of a tetrahedron are its six
/// edges and the 2-faces its four triangles. Ordering is lexicographic, which
/// fixes the orientation convention: a face is oriented by its sorted vertices.
pub fn k_faces(n_vertices: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut current = Vec::with_capacity(k + 1);
    fn walk(
        start: usize,
        n: usize,
        left: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if left == 0 {
            out.push(current.clone());
            return;
        }
        for v in start..=n - left {
            current.push(v);
            walk(v + 1, n, left - 1, current, out);
            current.pop();
        }
    }
    if k < n_vertices {
        walk(0, n_vertices, k + 1, &mut current, &mut out);
    }
    out
}

/// The index sets of a basis of `k`-forms on an `n`-dimensional space, as
/// strictly increasing tuples.
pub fn form_indices(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    let mut out = Vec::new();
    let mut current = Vec::with_capacity(k);
    fn walk(
        start: usize,
        n: usize,
        left: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if left == 0 {
            out.push(current.clone());
            return;
        }
        for v in start..=n - left {
            current.push(v);
            walk(v + 1, n, left - 1, current, out);
            current.pop();
        }
    }
    if k <= n {
        walk(0, n, k, &mut current, &mut out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tetrahedron_has_six_edges_and_four_triangles() {
        assert_eq!(k_faces(4, 0).len(), 4);
        assert_eq!(k_faces(4, 1).len(), 6);
        assert_eq!(k_faces(4, 2).len(), 4);
        assert_eq!(k_faces(4, 3).len(), 1);
    }

    #[test]
    fn the_face_count_is_the_binomial_the_table_uses() {
        for n in 1..=4 {
            for k in 0..=n {
                assert_eq!(k_faces(n + 1, k).len(), n_choose_k(n + 1, k + 1));
            }
        }
    }

    #[test]
    fn the_form_basis_has_the_dimension_of_the_exterior_power() {
        assert_eq!(form_indices(3, 0).len(), 1);
        assert_eq!(form_indices(3, 1).len(), 3);
        assert_eq!(form_indices(3, 2).len(), 3);
        assert_eq!(form_indices(3, 3).len(), 1);
    }
}
