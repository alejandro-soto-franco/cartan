// ~/cartan/cartan-geo/src/chern_simons.rs

//! Chern-Simons 3-form on a principal `G`-bundle.
//!
//! Given a `g`-valued connection 1-form `A` on a 3-parameter chart, the
//! Chern-Simons 3-form is
//!
//! ```text
//! CS(A) = Tr( A wedge dA + (2/3) A wedge A wedge A ).
//! ```
//!
//! Its exterior derivative is the Chern-Weil polynomial `Tr(F wedge F)` with
//! `F = dA + A wedge A`. On a closed 3-manifold the integral
//!
//! ```text
//! k(P, A) = (1 / 8 pi^2) * integral_M CS(A)
//! ```
//!
//! is a secondary characteristic class of the pair `(P, A)`. In the abelian
//! `U(1)` case the cubic term vanishes and `CS = A wedge dA`.
//!
//! ## What this module provides
//!
//! - [`U1Connection`]: a real-valued `U(1)` connection on a 3-parameter chart,
//!   stored as three component functions `A_i(u, v, w)` and their analytic
//!   partial derivatives. Component bracket conventions are documented inline.
//! - [`Su2Connection`]: an `su(2)`-valued connection with the same shape,
//!   stored as `Matrix2<Complex<f64>>`-valued components and partials.
//!   `su(2)` elements are anti-Hermitian; the module does not enforce this,
//!   but the trace conventions assume it.
//! - [`cs_density_u1`] / [`cs_density_su2`]: pointwise evaluators returning
//!   the coefficient of `du wedge dv wedge dw` in `CS(A)`.
//! - [`integrate_cs_u1`] / [`integrate_cs_su2`]: tensor-product Gauss-Legendre
//!   integrators over a 3D box. Bound by the supplied node count per axis.
//!
//! ## Canonical example: `U(1)` Hopf bundle on `S^3`
//!
//! In Euler coordinates `(theta, phi, psi)` on `S^3` with `theta in [0, pi]`,
//! `phi in [0, 2 pi]`, `psi in [0, 4 pi]`, the standard Hopf connection is
//!
//! ```text
//! A = (1 / 2) ( d psi + cos theta * d phi ).
//! ```
//!
//! Direct computation gives
//!
//! ```text
//! integral_{S^3} A wedge dA = - 4 pi^2,
//! ```
//!
//! so the normalized invariant
//!
//! ```text
//! (1 / 4 pi^2) integral_{S^3} A wedge dA = - 1
//! ```
//!
//! recovers `- c_1^2 = - 1` for the Hopf bundle (`c_1 = + or - 1`). The test
//! at the bottom of this file verifies the integrator reproduces this value
//! to numerical tolerance via tensor-product Gauss-Legendre quadrature.
//!
//! ## What it does not do
//!
//! - Symbolic differentiation. The user supplies analytic partial derivatives
//!   of each connection component. Autodiff is on the roadmap but not here.
//! - Boundary terms. Closed 3-manifolds only; the integrator covers a box.
//! - Gauge transformation. The level shifts by an integer winding number
//!   under large gauge changes; that machinery lives elsewhere.
//! - Discrete exterior calculus. A simplicial discretization of `CS(A)` via
//!   the DEC infrastructure in `cartan-dec` (`ExteriorDerivative`, `HodgeStar`,
//!   `line_bundle`) is a natural follow-up but lives outside this module's
//!   parametric-chart focus.
//!
//! ## Relation to the cartan stack
//!
//! Chern-Simons lives at layer L0 alongside [`crate::curvature`] and
//! [`crate::holonomy`]: it is a secondary characteristic class built from
//! curvature, and consumes the same connection data the rest of `cartan-geo`
//! uses. There is no prior public Rust implementation of `CS(A)` at the
//! time this module was written.
//!
//! ## References
//!
//! - `chern-1974-cha-for-geo`: Chern, S.-S. and Simons, J. "Characteristic
//!   forms and geometric invariants." Annals of Mathematics 99 (1974), 48-69.
//!   DOI: 10.2307/1971013. Origin of the secondary class.
//! - `witten-1989-qua-fie-the`: Witten, E. "Quantum field theory and the Jones
//!   polynomial." Communications in Mathematical Physics 121 (1989), 351-399.
//!   DOI: 10.1007/BF01217730. The CS action as a topological gauge theory.
//! - Nakahara, M. "Geometry, Topology and Physics." Second edition, IOP, 2003.
//!   Chapter 11.5 (Chern-Simons forms). Standard reference for the
//!   conventions used here.

use alloc::boxed::Box;
use cartan_core::Real;
use nalgebra::{Complex, Matrix2};

type ScalarFn = Box<dyn Fn(f64, f64, f64) -> f64>;
type MatrixFn = Box<dyn Fn(f64, f64, f64) -> Matrix2<Complex<f64>>>;

/// Real-valued `U(1)` connection 1-form on a 3-parameter chart.
///
/// Stores three component functions `A_i(u, v, w)` of
/// `A = A_1 du + A_2 dv + A_3 dw`, plus the nine analytic partial
/// derivatives `partials[i][j] = d A_i / d u_j` (with `u_0 = u`, `u_1 = v`,
/// `u_2 = w`).
pub struct U1Connection {
    /// Component functions `A_1, A_2, A_3`.
    pub components: [ScalarFn; 3],
    /// Partial derivatives `d A_i / d u_j` indexed as `partials[i][j]`.
    pub partials: [[ScalarFn; 3]; 3],
}

/// `su(2)`-valued connection 1-form on a 3-parameter chart.
///
/// Components are `Matrix2<Complex<f64>>`-valued; for an `su(2)` connection
/// each value should be anti-Hermitian. The trace conventions in
/// [`cs_density_su2`] do not check this; they take the real part of the
/// trace under the assumption that the input is anti-Hermitian.
pub struct Su2Connection {
    /// Component functions `A_1, A_2, A_3`.
    pub components: [MatrixFn; 3],
    /// Partial derivatives `d A_i / d u_j` indexed as `partials[i][j]`.
    pub partials: [[MatrixFn; 3]; 3],
}

/// Coefficient of `du wedge dv wedge dw` in `A wedge dA` for a real-valued
/// (abelian `U(1)`) connection. The cubic term vanishes for an abelian
/// algebra and is omitted.
///
/// Expansion: for `A = A_1 du + A_2 dv + A_3 dw`,
///
/// ```text
/// A wedge dA = [ A_1 (d_v A_3 - d_w A_2)
///              - A_2 (d_u A_3 - d_w A_1)
///              + A_3 (d_u A_2 - d_v A_1) ] du wedge dv wedge dw.
/// ```
pub fn cs_density_u1(conn: &U1Connection, u: f64, v: f64, w: f64) -> Real {
    let a1 = (conn.components[0])(u, v, w);
    let a2 = (conn.components[1])(u, v, w);
    let a3 = (conn.components[2])(u, v, w);
    let d_v_a3 = (conn.partials[2][1])(u, v, w);
    let d_w_a2 = (conn.partials[1][2])(u, v, w);
    let d_u_a3 = (conn.partials[2][0])(u, v, w);
    let d_w_a1 = (conn.partials[0][2])(u, v, w);
    let d_u_a2 = (conn.partials[1][0])(u, v, w);
    let d_v_a1 = (conn.partials[0][1])(u, v, w);
    a1 * (d_v_a3 - d_w_a2) - a2 * (d_u_a3 - d_w_a1) + a3 * (d_u_a2 - d_v_a1)
}

/// Coefficient of `du wedge dv wedge dw` in `Tr(A wedge dA + (2/3) A wedge A wedge A)`
/// for an `su(2)`-valued connection.
///
/// Expansion: with `A = A_1 du + A_2 dv + A_3 dw`,
///
/// ```text
/// Tr(A wedge dA) coeff
///   = Tr[ A_1 (d_v A_3 - d_w A_2)
///       - A_2 (d_u A_3 - d_w A_1)
///       + A_3 (d_u A_2 - d_v A_1) ],
///
/// (2/3) Tr(A wedge A wedge A) coeff
///   = (2/3) Tr[ A_1 [A_2, A_3] - A_2 [A_1, A_3] + A_3 [A_1, A_2] ]
///   = 2 Tr( A_1 [A_2, A_3] )       (by cyclic trace and bracket antisymmetry).
/// ```
pub fn cs_density_su2(conn: &Su2Connection, u: f64, v: f64, w: f64) -> Real {
    let a1 = (conn.components[0])(u, v, w);
    let a2 = (conn.components[1])(u, v, w);
    let a3 = (conn.components[2])(u, v, w);
    let d_v_a3 = (conn.partials[2][1])(u, v, w);
    let d_w_a2 = (conn.partials[1][2])(u, v, w);
    let d_u_a3 = (conn.partials[2][0])(u, v, w);
    let d_w_a1 = (conn.partials[0][2])(u, v, w);
    let d_u_a2 = (conn.partials[1][0])(u, v, w);
    let d_v_a1 = (conn.partials[0][1])(u, v, w);

    // Tr( A wedge dA ) coefficient
    let lin =
        (a1 * (d_v_a3 - d_w_a2) - a2 * (d_u_a3 - d_w_a1) + a3 * (d_u_a2 - d_v_a1)).trace();

    // (2/3) Tr( A wedge A wedge A ) coefficient = 2 Tr( A_1 [A_2, A_3] )
    let comm_23 = a2 * a3 - a3 * a2;
    let cubic = Complex::new(2.0, 0.0) * (a1 * comm_23).trace();

    (lin + cubic).re
}

/// Integrate `CS(A)` over a 3D box `[u0, u1] x [v0, v1] x [w0, w1]` for the
/// abelian `U(1)` case via tensor-product Gauss-Legendre quadrature.
///
/// `n_per_axis` selects the quadrature rule. Supported values: 4, 5, 8, 16.
pub fn integrate_cs_u1(
    conn: &U1Connection,
    bounds: [(f64, f64); 3],
    n_per_axis: usize,
) -> Real {
    integrate_box(bounds, n_per_axis, |u, v, w| cs_density_u1(conn, u, v, w))
}

/// Integrate `CS(A)` over a 3D box for the non-abelian `su(2)` case.
///
/// `n_per_axis` selects the quadrature rule. Supported values: 4, 5, 8, 16.
pub fn integrate_cs_su2(
    conn: &Su2Connection,
    bounds: [(f64, f64); 3],
    n_per_axis: usize,
) -> Real {
    integrate_box(bounds, n_per_axis, |u, v, w| cs_density_su2(conn, u, v, w))
}

fn integrate_box(
    bounds: [(f64, f64); 3],
    n: usize,
    f: impl Fn(f64, f64, f64) -> f64,
) -> f64 {
    let (nodes, weights) = gauss_legendre(n);
    let half = [
        (bounds[0].1 - bounds[0].0) * 0.5,
        (bounds[1].1 - bounds[1].0) * 0.5,
        (bounds[2].1 - bounds[2].0) * 0.5,
    ];
    let mid = [
        (bounds[0].0 + bounds[0].1) * 0.5,
        (bounds[1].0 + bounds[1].1) * 0.5,
        (bounds[2].0 + bounds[2].1) * 0.5,
    ];
    let mut acc = 0.0;
    for (i, &xi) in nodes.iter().enumerate() {
        let wi = weights[i];
        let u = half[0] * xi + mid[0];
        for (j, &xj) in nodes.iter().enumerate() {
            let wj = weights[j];
            let v = half[1] * xj + mid[1];
            for (k, &xk) in nodes.iter().enumerate() {
                let wk = weights[k];
                let w = half[2] * xk + mid[2];
                acc += wi * wj * wk * f(u, v, w);
            }
        }
    }
    acc * half[0] * half[1] * half[2]
}

fn gauss_legendre(n: usize) -> (&'static [f64], &'static [f64]) {
    match n {
        4 => (&GL4_NODES, &GL4_WEIGHTS),
        5 => (&GL5_NODES, &GL5_WEIGHTS),
        8 => (&GL8_NODES, &GL8_WEIGHTS),
        16 => (&GL16_NODES, &GL16_WEIGHTS),
        _ => panic!("cartan-geo::chern_simons: unsupported Gauss-Legendre order {n}; supported: 4, 5, 8, 16"),
    }
}

// Gauss-Legendre nodes and weights on [-1, 1]. Standard tabulated values.

const GL4_NODES: [f64; 4] = [
    -0.861_136_311_594_052_5,
    -0.339_981_043_584_856_26,
    0.339_981_043_584_856_26,
    0.861_136_311_594_052_5,
];
const GL4_WEIGHTS: [f64; 4] = [
    0.347_854_845_137_453_85,
    0.652_145_154_862_546_2,
    0.652_145_154_862_546_2,
    0.347_854_845_137_453_85,
];

const GL5_NODES: [f64; 5] = [
    -0.906_179_845_938_664,
    -0.538_469_310_105_683_1,
    0.0,
    0.538_469_310_105_683_1,
    0.906_179_845_938_664,
];
const GL5_WEIGHTS: [f64; 5] = [
    0.236_926_885_056_189_08,
    0.478_628_670_499_366_47,
    0.568_888_888_888_888_9,
    0.478_628_670_499_366_47,
    0.236_926_885_056_189_08,
];

const GL8_NODES: [f64; 8] = [
    -0.960_289_856_497_536_3,
    -0.796_666_477_413_626_7,
    -0.525_532_409_916_329,
    -0.183_434_642_495_649_8,
    0.183_434_642_495_649_8,
    0.525_532_409_916_329,
    0.796_666_477_413_626_7,
    0.960_289_856_497_536_3,
];
const GL8_WEIGHTS: [f64; 8] = [
    0.101_228_536_290_376_26,
    0.222_381_034_453_374_48,
    0.313_706_645_877_887_3,
    0.362_683_783_378_362,
    0.362_683_783_378_362,
    0.313_706_645_877_887_3,
    0.222_381_034_453_374_48,
    0.101_228_536_290_376_26,
];

const GL16_NODES: [f64; 16] = [
    -0.989_400_934_991_649_9,
    -0.944_575_023_073_232_6,
    -0.865_631_202_387_831_7,
    -0.755_404_408_355_003_1,
    -0.617_876_244_402_643_7,
    -0.458_016_777_657_227_4,
    -0.281_603_550_779_258_9,
    -0.095_012_509_837_637_44,
    0.095_012_509_837_637_44,
    0.281_603_550_779_258_9,
    0.458_016_777_657_227_4,
    0.617_876_244_402_643_7,
    0.755_404_408_355_003_1,
    0.865_631_202_387_831_7,
    0.944_575_023_073_232_6,
    0.989_400_934_991_649_9,
];
const GL16_WEIGHTS: [f64; 16] = [
    0.027_152_459_411_754_096,
    0.062_253_523_938_647_89,
    0.095_158_511_682_492_78,
    0.124_628_971_255_533_87,
    0.149_595_988_816_576_73,
    0.169_156_519_395_002_54,
    0.182_603_415_044_923_6,
    0.189_450_610_455_068_5,
    0.189_450_610_455_068_5,
    0.182_603_415_044_923_6,
    0.169_156_519_395_002_54,
    0.149_595_988_816_576_73,
    0.124_628_971_255_533_87,
    0.095_158_511_682_492_78,
    0.062_253_523_938_647_89,
    0.027_152_459_411_754_096,
];

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use core::f64::consts::PI;

    /// `U(1)` Hopf bundle on `S^3` in Euler coordinates `(theta, phi, psi)`.
    ///
    /// `A = (1/2)(d psi + cos theta d phi)`, so with `u_0 = theta`, `u_1 = phi`,
    /// `u_2 = psi` the components are `A_0 = 0`, `A_1 = (1/2) cos theta`,
    /// `A_2 = 1/2`. Only one partial derivative is non-zero:
    /// `d_theta A_phi = - (1/2) sin theta`.
    ///
    /// Expected exact result over `theta in [0, pi]`, `phi in [0, 2 pi]`,
    /// `psi in [0, 4 pi]`:
    ///
    /// ```text
    /// integral A wedge dA = - 4 pi^2
    /// ```
    ///
    /// so `(1 / 4 pi^2) integral = - 1`, which is `- c_1^2` for the Hopf bundle.
    #[test]
    fn hopf_bundle_u1_chern_simons_invariant() {
        let conn = U1Connection {
            components: [
                Box::new(|_u, _v, _w| 0.0),
                Box::new(|theta, _v, _w| 0.5 * theta.cos()),
                Box::new(|_u, _v, _w| 0.5),
            ],
            partials: [
                [
                    Box::new(|_u, _v, _w| 0.0),
                    Box::new(|_u, _v, _w| 0.0),
                    Box::new(|_u, _v, _w| 0.0),
                ],
                [
                    Box::new(|theta, _v, _w| -0.5 * theta.sin()),
                    Box::new(|_u, _v, _w| 0.0),
                    Box::new(|_u, _v, _w| 0.0),
                ],
                [
                    Box::new(|_u, _v, _w| 0.0),
                    Box::new(|_u, _v, _w| 0.0),
                    Box::new(|_u, _v, _w| 0.0),
                ],
            ],
        };
        let bounds = [(0.0, PI), (0.0, 2.0 * PI), (0.0, 4.0 * PI)];
        let raw = integrate_cs_u1(&conn, bounds, 16);
        let normalized = raw / (4.0 * PI * PI);

        // Exact: raw = -4 pi^2, normalized = -1.
        assert!(
            (raw - (-4.0 * PI * PI)).abs() < 1.0e-9,
            "raw integral = {raw}, expected -4 pi^2 = {}",
            -4.0 * PI * PI
        );
        assert!(
            (normalized - (-1.0)).abs() < 1.0e-10,
            "normalized invariant = {normalized}, expected -1"
        );
    }

    /// Constant `su(2)` connection `A_i = i c sigma_i` on `T^3 = [0, 1]^3`.
    ///
    /// All partials vanish, so `Tr(A wedge dA) = 0`. The cubic term works out
    /// analytically: with `[sigma_i, sigma_j] = 2 i epsilon_{ijk} sigma_k`,
    /// `Tr(sigma_i^2) = 2`, and `A_i = i c sigma_i`, one gets
    ///
    /// ```text
    /// coefficient of du wedge dv wedge dw in A wedge A wedge A
    ///   = A_1 [A_2, A_3] - A_2 [A_1, A_3] + A_3 [A_1, A_2]
    ///   = 6 c^3 I_2,
    /// (2/3) Tr( . ) = 8 c^3.
    /// ```
    ///
    /// Integrating over the unit cube gives `8 c^3`.
    #[test]
    fn flat_su2_cubic_density_on_unit_cube() {
        let c = 0.5_f64;
        let zero = Matrix2::<Complex<f64>>::zeros();
        let i_unit = Complex::new(0.0, 1.0);
        let pauli_1 = Matrix2::new(
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
        );
        let pauli_2 = Matrix2::new(
            Complex::new(0.0, 0.0),
            Complex::new(0.0, -1.0),
            Complex::new(0.0, 1.0),
            Complex::new(0.0, 0.0),
        );
        let pauli_3 = Matrix2::new(
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-1.0, 0.0),
        );
        let prefactor = i_unit * Complex::new(c, 0.0);
        let a1 = pauli_1 * prefactor;
        let a2 = pauli_2 * prefactor;
        let a3 = pauli_3 * prefactor;

        let conn = Su2Connection {
            components: [
                Box::new(move |_u, _v, _w| a1),
                Box::new(move |_u, _v, _w| a2),
                Box::new(move |_u, _v, _w| a3),
            ],
            partials: [
                [
                    Box::new(move |_u, _v, _w| zero),
                    Box::new(move |_u, _v, _w| zero),
                    Box::new(move |_u, _v, _w| zero),
                ],
                [
                    Box::new(move |_u, _v, _w| zero),
                    Box::new(move |_u, _v, _w| zero),
                    Box::new(move |_u, _v, _w| zero),
                ],
                [
                    Box::new(move |_u, _v, _w| zero),
                    Box::new(move |_u, _v, _w| zero),
                    Box::new(move |_u, _v, _w| zero),
                ],
            ],
        };
        let bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];
        let result = integrate_cs_su2(&conn, bounds, 4);
        let expected = 8.0 * c.powi(3);
        assert!(
            (result - expected).abs() < 1.0e-10,
            "constant su(2) CS = {result}, expected {expected}"
        );
    }
}
