//! The mimetic star against what the counting predicts and the diagonal delivers.
//!
//! Each test states a claim the crate's documentation makes, so a change that
//! breaks one breaks the claim with it.

use cartan_mimetic::{diagonal_star, local_star, n_choose_k, Simplex};

fn simplex(points: &[&[f64]]) -> Simplex {
    Simplex::new(&points.iter().map(|p| p.to_vec()).collect::<Vec<_>>())
}

fn triangle(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Simplex {
    simplex(&[&a[..], &b[..], &c[..]])
}

fn tet(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> Simplex {
    simplex(&[&a[..], &b[..], &c[..], &d[..]])
}

/// The shapes a mesher actually produces, including the ones it should not.
fn awkward_tets() -> Vec<(&'static str, Simplex)> {
    vec![
        ("regular", tet([1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.])),
        ("reference corner", tet([0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.])),
        ("generic", tet([0., 0., 0.], [1., 0., 0.], [0.3, 0.9, 0.], [0.2, 0.4, 1.1])),
        ("flat sliver", tet([0., 0., 0.], [1., 0., 0.], [0.3, 0.9, 0.], [0.4, 0.3, 0.08])),
        ("needle", tet([0., 0., 0.], [1., 0., 0.], [0.5, 0.05, 0.], [0.5, 0.02, 0.9])),
        ("cap", tet([0., 0., 0.], [1., 0., 0.], [0.5, 0.9, 0.], [0.5, 0.3, 0.02])),
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// What the mimetic star promises
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn the_mimetic_star_is_consistent_at_every_degree() {
    for (name, s) in awkward_tets() {
        for k in 0..=3 {
            let c = s.consistency(k).expect("a non-degenerate tetrahedron");
            let m = local_star(&s, k).expect("a non-degenerate tetrahedron");
            let r = c.residual(&m);
            assert!(r < 1e-10, "{name} at k={k} left consistency residual {r:.3e}");
        }
    }
}

#[test]
fn the_mimetic_star_is_positive_definite_on_every_shape() {
    for (name, s) in awkward_tets() {
        for k in 0..=3 {
            let m = local_star(&s, k).expect("a non-degenerate tetrahedron");
            let sym = 0.5 * (&m + m.transpose());
            let min = sym
                .symmetric_eigenvalues()
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            assert!(
                min > 0.0,
                "{name} at k={k} gave smallest eigenvalue {min:.3e}"
            );
        }
    }
}

#[test]
fn a_degenerate_simplex_has_no_star() {
    // Three collinear points span no plane, so there is no inner product on
    // 1-forms to return, and a zero matrix would be worse than nothing.
    let flat = triangle([0.0, 0.0], [1.0, 0.0], [2.0, 0.0]);
    assert!(local_star(&flat, 1).is_none());
    assert!(diagonal_star(&flat, 1).is_none());
}

// ─────────────────────────────────────────────────────────────────────────────
// Where the diagonal star exists, and what it equals
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn on_a_triangle_the_diagonal_star_on_1_forms_is_cot_over_two() {
    // The consistent diagonal star is unique at n=2, k=1, and it is the one
    // discrete exterior calculus uses. Faces come back lexicographically, so
    // they are (0,1), (0,2), (1,2), and the cotangent belongs to the angle at
    // the remaining vertex.
    let p = [[0.0, 0.0], [1.0, 0.0], [0.3, 0.8]];
    let s = triangle(p[0], p[1], p[2]);
    let d = diagonal_star(&s, 1).expect("a proper triangle");
    assert!(d.is_consistent(), "residual {:.3e}", d.residual);

    let cot = |i: usize, j: usize, opp: usize| {
        let u = [p[i][0] - p[opp][0], p[i][1] - p[opp][1]];
        let v = [p[j][0] - p[opp][0], p[j][1] - p[opp][1]];
        (u[0] * v[0] + u[1] * v[1]) / (u[0] * v[1] - u[1] * v[0]).abs()
    };
    let expected = [cot(0, 1, 2) / 2.0, cot(0, 2, 1) / 2.0, cot(1, 2, 0) / 2.0];
    for (f, e) in expected.iter().enumerate() {
        assert!(
            (d.entries[f] - e).abs() < 1e-12,
            "face {f}: star {} against cot/2 {e}",
            d.entries[f]
        );
    }
}

#[test]
fn that_diagonal_star_turns_negative_exactly_when_the_triangle_is_obtuse() {
    let acute = triangle([0.0, 0.0], [1.0, 0.0], [0.5, 0.8]);
    let obtuse = triangle([0.0, 0.0], [1.0, 0.0], [-0.6, 0.35]);

    let a = diagonal_star(&acute, 1).unwrap();
    assert!(a.is_usable(), "an acute triangle should admit the diagonal star");

    let o = diagonal_star(&obtuse, 1).unwrap();
    assert!(o.is_consistent(), "the obtuse case is still consistent");
    assert!(
        !o.is_positive(),
        "an obtuse triangle should force a negative entry, got {:?}",
        o.entries.as_slice()
    );
}

#[test]
fn on_a_tetrahedron_no_diagonal_star_is_consistent_on_2_forms() {
    // Four face unknowns against six equations. The regular tetrahedron and the
    // reference corner solve it by symmetry, which is why a test on either alone
    // would miss this entirely.
    let generic = tet([0., 0., 0.], [1., 0., 0.], [0.3, 0.9, 0.], [0.2, 0.4, 1.1]);
    let d = diagonal_star(&generic, 2).unwrap();
    assert!(
        !d.is_consistent(),
        "a generic tetrahedron should admit no consistent diagonal 2-form star, residual {:.3e}",
        d.residual
    );

    let regular = tet([1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]);
    assert!(
        diagonal_star(&regular, 2).unwrap().is_consistent(),
        "the regular tetrahedron is the symmetric case that does solve"
    );
}

#[test]
fn the_tetrahedra_of_a_subdivided_cube_admit_no_usable_diagonal_star() {
    // The ordinary way to mesh a box. Every one of the six fails, at k=1 by sign
    // and at k=2 by consistency, which is why the diagonal path is not an option
    // for a three-dimensional solver.
    let c = [
        [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
        [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
    ];
    let six = [
        [0, 1, 2, 6], [0, 1, 6, 5], [0, 4, 5, 6],
        [0, 4, 6, 7], [0, 3, 7, 6], [0, 2, 3, 6],
    ];
    for (t, idx) in six.iter().enumerate() {
        let s = tet(c[idx[0]], c[idx[1]], c[idx[2]], c[idx[3]]);
        assert!(
            !diagonal_star(&s, 1).unwrap().is_usable(),
            "tetrahedron {t} unexpectedly admitted a usable diagonal 1-form star"
        );
        assert!(
            !diagonal_star(&s, 2).unwrap().is_consistent(),
            "tetrahedron {t} unexpectedly admitted a consistent diagonal 2-form star"
        );
        // And the mimetic star is fine on the same cell.
        for k in 1..=2 {
            let m = local_star(&s, k).unwrap();
            let c = s.consistency(k).unwrap();
            assert!(c.residual(&m) < 1e-10);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The counting the documentation tabulates
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn the_unknowns_and_equations_are_the_binomials_the_table_states() {
    let expected = [
        (2, 0, 3, 1),
        (2, 1, 3, 3),
        (2, 2, 1, 1),
        (3, 0, 4, 1),
        (3, 1, 6, 6),
        (3, 2, 4, 6),
        (3, 3, 1, 1),
    ];
    for (n, k, unknowns, equations) in expected {
        assert_eq!(n_choose_k(n + 1, k + 1), unknowns, "unknowns at n={n}, k={k}");
        let forms = n_choose_k(n, k);
        assert_eq!(forms * (forms + 1) / 2, equations, "equations at n={n}, k={k}");
    }
}

#[test]
fn a_triangle_in_three_dimensions_is_treated_as_two_dimensional() {
    // The surface case. A triangle embedded in space has a two-dimensional
    // tangent space, so its 1-forms span two dimensions and the star is 3x3 on
    // the edges, exactly as it would be in the plane.
    let flat = triangle([0.0, 0.0], [1.0, 0.0], [0.3, 0.8]);
    let tilted = simplex(&[
        &[0.0, 0.0, 0.0][..],
        &[1.0, 0.0, 0.0][..],
        &[0.3, 0.8, 0.0][..],
    ]);
    let a = local_star(&flat, 1).unwrap();
    let b = local_star(&tilted, 1).unwrap();
    assert_eq!(a.shape(), (3, 3));
    assert_eq!(b.shape(), (3, 3));
    for i in 0..3 {
        for j in 0..3 {
            assert!((a[(i, j)] - b[(i, j)]).abs() < 1e-12, "the embedding changed the star");
        }
    }
}
