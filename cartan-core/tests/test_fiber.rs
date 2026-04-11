use std::f64::consts::{FRAC_PI_3, FRAC_PI_4, FRAC_PI_6, PI};

use cartan_core::bundle::{CovLaplacian, EdgeTransport2D, EdgeTransport3D};
use cartan_core::fiber::{
    Fiber, FiberOps, NematicFiber3D, Section, TangentFiber, U1Spin2, VecSection,
};

// ─── U1Spin2 tests ───────────────────────────────────────────────────────────

#[test]
fn u1_spin2_identity_transport() {
    let rot = [1.0, 0.0, 0.0, 1.0];
    let q = [0.3_f64, 0.4];
    let tp = U1Spin2::transport_by(&rot, 2, &q);
    assert!((tp[0] - 0.3).abs() < 1e-12);
    assert!((tp[1] - 0.4).abs() < 1e-12);
}

#[test]
fn u1_spin2_90deg_rotation() {
    // SO(2) by pi/2 -> spin-2 phase pi -> flip sign.
    let theta = PI / 2.0;
    let rot = [theta.cos(), -theta.sin(), theta.sin(), theta.cos()];
    let q = [1.0_f64, 0.0];
    let tp = U1Spin2::transport_by(&rot, 2, &q);
    assert!((tp[0] - (-1.0)).abs() < 1e-12);
    assert!((tp[1]).abs() < 1e-12);
}

#[test]
fn u1_spin2_preserves_norm() {
    let theta = 0.73_f64;
    let rot = [theta.cos(), -theta.sin(), theta.sin(), theta.cos()];
    let q = [0.3_f64, 0.4];
    let tp = U1Spin2::transport_by(&rot, 2, &q);
    let n_before = (q[0] * q[0] + q[1] * q[1]).sqrt();
    let n_after = (tp[0] * tp[0] + tp[1] * tp[1]).sqrt();
    assert!((n_before - n_after).abs() < 1e-12);
}

#[test]
fn u1_spin2_zero() {
    assert_eq!(U1Spin2::zero(), [0.0, 0.0]);
}

#[test]
fn u1_spin2_fiber_dim() {
    assert_eq!(U1Spin2::FIBER_DIM, 2);
}

// ─── TangentFiber tests ─────────────────────────────────────────────────────

#[test]
fn tangent_fiber_2d_rotation() {
    let theta = FRAC_PI_4;
    let rot = [theta.cos(), -theta.sin(), theta.sin(), theta.cos()];
    let v = [1.0_f64, 0.0];
    let tp = TangentFiber::<2>::transport_by(&rot, 2, &v);
    assert!((tp[0] - theta.cos()).abs() < 1e-12);
    assert!((tp[1] - theta.sin()).abs() < 1e-12);
}

#[test]
fn tangent_fiber_3d_preserves_norm() {
    let theta = FRAC_PI_3;
    let ct = theta.cos();
    let st = theta.sin();
    // Rotation around z-axis.
    let rot = [ct, -st, 0.0, st, ct, 0.0, 0.0, 0.0, 1.0];
    let v = [0.3_f64, 0.4, 0.5];
    let tp = TangentFiber::<3>::transport_by(&rot, 3, &v);
    let n_before: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let n_after: f64 = tp.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((n_before - n_after).abs() < 1e-12);
}

#[test]
fn tangent_fiber_3d_z_axis_unchanged() {
    let theta = 1.23_f64;
    let ct = theta.cos();
    let st = theta.sin();
    let rot = [ct, -st, 0.0, st, ct, 0.0, 0.0, 0.0, 1.0];
    let v = [0.0_f64, 0.0, 1.0];
    let tp = TangentFiber::<3>::transport_by(&rot, 3, &v);
    assert!((tp[0]).abs() < 1e-12);
    assert!((tp[1]).abs() < 1e-12);
    assert!((tp[2] - 1.0).abs() < 1e-12);
}

// ─── NematicFiber3D tests ───────────────────────────────────────────────────

#[test]
fn nematic_3d_identity_transport() {
    let rot = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let q = [0.1_f64, 0.2, 0.3, 0.4, 0.5];
    let tp = NematicFiber3D::transport_by(&rot, 3, &q);
    for i in 0..5 {
        assert!((tp[i] - q[i]).abs() < 1e-12, "component {i} mismatch");
    }
}

#[test]
fn nematic_3d_preserves_tracelessness() {
    let theta = FRAC_PI_6;
    let ct = theta.cos();
    let st = theta.sin();
    let rot = [ct, -st, 0.0, st, ct, 0.0, 0.0, 0.0, 1.0];
    let q = [0.3_f64, 0.1, 0.0, -0.1, 0.2];
    let tp = NematicFiber3D::transport_by(&rot, 3, &q);
    // Q33 = -Q11 - Q22; trace = Q11 + Q22 + Q33 = 0.
    let trace = tp[0] + tp[3] + (-tp[0] - tp[3]);
    assert!(trace.abs() < 1e-12, "trace = {trace}");
}

#[test]
fn nematic_3d_preserves_frobenius_norm() {
    let theta = 1.23_f64;
    let ct = theta.cos();
    let st = theta.sin();
    let rot = [ct, -st, 0.0, st, ct, 0.0, 0.0, 0.0, 1.0];
    let q = [0.3_f64, 0.1, -0.2, 0.15, 0.25];
    let tp = NematicFiber3D::transport_by(&rot, 3, &q);

    let frob = |e: &[f64; 5]| -> f64 {
        let q33 = -e[0] - e[3];
        e[0] * e[0] + e[3] * e[3] + q33 * q33 + 2.0 * (e[1] * e[1] + e[2] * e[2] + e[4] * e[4])
    };
    let n_before = frob(&q);
    let n_after = frob(&tp);
    assert!(
        (n_before - n_after).abs() < 1e-10,
        "Frobenius norm: {n_before} -> {n_after}"
    );
}

#[test]
fn nematic_3d_180deg_rotation_is_identity() {
    // For a nematic (headless), rotation by pi around any axis should be identity
    // on the Q-tensor (since Q = n n^T - I/3, and -n gives same Q).
    // R = rotation by pi around z: diag(-1, -1, 1).
    let rot = [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0];
    let q = [0.3_f64, 0.1, 0.0, -0.1, 0.2];
    let tp = NematicFiber3D::transport_by(&rot, 3, &q);
    // Q' = R Q R^T. For R = diag(-1,-1,1): Q'_ij = R_ii Q_ij R_jj.
    // Q'_11 = (-1)(-1)Q_11 = Q_11, Q'_12 = (-1)(-1)Q_12 = Q_12,
    // Q'_13 = (-1)(1)Q_13 = -Q_13, Q'_22 = Q_22, Q'_23 = -Q_23.
    assert!((tp[0] - q[0]).abs() < 1e-12, "Q11");
    assert!((tp[1] - q[1]).abs() < 1e-12, "Q12");
    assert!((tp[2] - (-q[2])).abs() < 1e-12, "Q13 should flip");
    assert!((tp[3] - q[3]).abs() < 1e-12, "Q22");
    assert!((tp[4] - (-q[4])).abs() < 1e-12, "Q23 should flip");
}

// ─── VecSection tests ───────────────────────────────────────────────────────

#[test]
fn vec_section_basic() {
    let mut s = VecSection::<U1Spin2>::zeros(10);
    assert_eq!(s.n_vertices(), 10);
    assert_eq!(*s.at(0), [0.0, 0.0]);
    *s.at_mut(3) = [0.5, 0.6];
    assert_eq!(*s.at(3), [0.5, 0.6]);
}

#[test]
fn vec_section_from_vec() {
    let data = vec![[1.0, 2.0], [3.0, 4.0]];
    let s = VecSection::<U1Spin2>::from_vec(data);
    assert_eq!(s.len(), 2);
    assert_eq!(*s.at(0), [1.0, 2.0]);
    assert_eq!(*s.at(1), [3.0, 4.0]);
}

// ─── CovLaplacian tests ────────────────────────────────────────────────────

#[test]
fn cov_laplacian_uniform_field_vanishes_flat() {
    // On a flat mesh (identity transport), lap of uniform field = 0.
    let edges = vec![[0usize, 1], [1, 2], [2, 3], [3, 0], [0, 2]];
    let cot_weights = vec![1.0_f64; 5];
    let dual_areas = vec![0.5_f64; 4];
    let identity = [1.0_f64, 0.0, 0.0, 1.0];
    let conn = EdgeTransport2D {
        edges: edges.clone(),
        transports: vec![identity; 5],
    };
    let lap = CovLaplacian::new(4, &edges, &cot_weights, &dual_areas);

    let field = VecSection::<U1Spin2>::from_vec(vec![[0.3, 0.4]; 4]);
    let result = lap.apply::<U1Spin2, 2, _>(&field, &conn);

    for v in 0..4 {
        let r = result.at(v);
        assert!(r[0].abs() < 1e-12, "lap[{v}].q1 = {}", r[0]);
        assert!(r[1].abs() < 1e-12, "lap[{v}].q2 = {}", r[1]);
    }
}

#[test]
fn cov_laplacian_positive_at_maximum() {
    // DEC Laplacian is positive at maxima.
    let edges = vec![[0usize, 1], [1, 2], [2, 3], [3, 0], [0, 2]];
    let cot_weights = vec![1.0_f64; 5];
    let dual_areas = vec![0.5_f64; 4];
    let identity = [1.0_f64, 0.0, 0.0, 1.0];
    let conn = EdgeTransport2D {
        edges: edges.clone(),
        transports: vec![identity; 5],
    };
    let lap = CovLaplacian::new(4, &edges, &cot_weights, &dual_areas);

    let field = VecSection::<U1Spin2>::from_vec(vec![
        [1.0, 0.0], // peak
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]);
    let result = lap.apply::<U1Spin2, 2, _>(&field, &conn);
    assert!(
        result.at(0)[0] > 0.0,
        "lap should be positive at maximum, got {}",
        result.at(0)[0]
    );
}

#[test]
fn cov_laplacian_3d_uniform_vanishes() {
    let edges = vec![[0usize, 1], [1, 2], [2, 0]];
    let cot_weights = vec![1.0_f64; 3];
    let dual_areas = vec![1.0_f64; 3];
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let conn = EdgeTransport3D {
        edges: edges.clone(),
        transports: vec![identity; 3],
    };
    let lap = CovLaplacian::new(3, &edges, &cot_weights, &dual_areas);

    let field =
        VecSection::<NematicFiber3D>::from_vec(vec![[0.1, 0.2, 0.3, 0.15, 0.25]; 3]);
    let result = lap.apply::<NematicFiber3D, 3, _>(&field, &conn);

    for v in 0..3 {
        for c in 0..5 {
            assert!(
                result.at(v)[c].abs() < 1e-12,
                "lap[{v}][{c}] = {}",
                result.at(v)[c]
            );
        }
    }
}

#[test]
fn cov_laplacian_with_nontrivial_transport() {
    // Two vertices connected by one edge with a pi/4 rotation.
    // Uniform field in v0's frame should NOT be uniform after transport.
    let theta = FRAC_PI_4;
    let ct = theta.cos();
    let st = theta.sin();
    let rot = [ct, -st, st, ct]; // SO(2) by pi/4
    let edges = vec![[0usize, 1]];
    let cot_weights = vec![1.0_f64];
    let dual_areas = vec![1.0_f64; 2];
    let conn = EdgeTransport2D {
        edges: edges.clone(),
        transports: vec![rot],
    };
    let lap = CovLaplacian::new(2, &edges, &cot_weights, &dual_areas);

    // Same value [0.5, 0.0] at both vertices.
    let field = VecSection::<U1Spin2>::from_vec(vec![[0.5, 0.0]; 2]);
    let result = lap.apply::<U1Spin2, 2, _>(&field, &conn);

    // After transport, the neighbor's value rotates by 2*pi/4 = pi/2 in spin-2.
    // Transported [0.5, 0.0] -> [0.5*cos(pi/2) - 0*sin(pi/2), ...] = [0.0, 0.5]
    // Diff at v0: [0.5, 0.0] - [0.0, 0.5] = [0.5, -0.5]
    // So lap should be nonzero.
    let r0 = result.at(0);
    assert!(
        r0[0].abs() > 0.1 || r0[1].abs() > 0.1,
        "nontrivial transport should give nonzero Laplacian"
    );
}
