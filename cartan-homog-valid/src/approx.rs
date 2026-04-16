//! Affine-invariant SPD geodesic distance comparators.

use cartan_core::Manifold;
use cartan_manifolds::Spd;

pub fn ai_distance_order2(
    a: &nalgebra::Matrix3<f64>, b: &nalgebra::Matrix3<f64>,
) -> Option<f64> {
    let spd = Spd::<3>;
    let sa = (a + a.transpose()) * 0.5;
    let sb = (b + b.transpose()) * 0.5;
    spd.dist(&sa, &sb).ok()
}

pub fn ai_distance_order4(
    a: &nalgebra::SMatrix<f64, 6, 6>, b: &nalgebra::SMatrix<f64, 6, 6>,
) -> Option<f64> {
    let spd = Spd::<6>;
    let sa = (a + a.transpose()) * 0.5;
    let sb = (b + b.transpose()) * 0.5;
    spd.dist(&sa, &sb).ok()
}

#[macro_export]
macro_rules! assert_spd_close_o2 {
    ($a:expr, $b:expr, tol = $tol:expr, case = $case:expr) => {{
        let d = $crate::approx::ai_distance_order2(&$a, &$b)
            .expect("SPD distance failed; one operand not positive definite");
        assert!(d < $tol, "case `{}`: d_AI = {:.3e} > tol {:.3e}", $case, d, $tol);
    }};
}

#[macro_export]
macro_rules! assert_spd_close_o4 {
    ($a:expr, $b:expr, tol = $tol:expr, case = $case:expr) => {{
        let d = $crate::approx::ai_distance_order4(&$a, &$b)
            .expect("SPD distance failed; one operand not positive definite");
        assert!(d < $tol, "case `{}`: d_AI = {:.3e} > tol {:.3e}", $case, d, $tol);
    }};
}
