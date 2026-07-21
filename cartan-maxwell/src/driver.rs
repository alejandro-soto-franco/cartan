//! Prescribed time-dependent geometry: a driver supplies squared edge lengths
//! `l_e(t)^2`.
//!
//! Squared lengths are the Regge primitive: the per-cell metric tensor is
//! linear in them, so a prescribed scaling enters polynomially and an
//! indefinite signature stays representable. A driver that scales *lengths*
//! by `a(t)` therefore scales the stored quantity by `a(t)^2`.

use exterior::Dim;
use simplicial::geometry::metric::mesh::MeshLengthsSq;
use simplicial::linalg::Vector;
use simplicial::topology::complex::Complex;

/// A prescribed evolving geometry over a fixed abstract complex.
pub trait MetricDriver {
    /// The intrinsic squared edge lengths at time `t`.
    fn lengths_sq_at(&self, t: f64) -> MeshLengthsSq;
    /// The fixed abstract complex the metric lives on.
    fn complex(&self) -> &Complex;
    /// Spatial dimension of the complex.
    fn dim(&self) -> Dim {
        self.complex().dim()
    }
}

/// FLRW-style uniform scaling of lengths: `l_e(t) = a(t) l_e(0)`.
///
/// Stored squared, so the coefficient applied to the base data is `a(t)^2`.
pub struct FlrwDriver {
    complex: Complex,
    /// Base *squared* edge lengths, `l_e(0)^2`.
    base_sq: Vector,
    scale: Box<dyn Fn(f64) -> f64 + Send + Sync>,
}

impl FlrwDriver {
    pub fn new(
        complex: Complex,
        base: MeshLengthsSq,
        scale: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    ) -> Self {
        Self {
            complex,
            base_sq: base.into_vector(),
            scale,
        }
    }

    /// A static (non-evolving) background, `a(t) = 1`.
    pub fn static_metric(complex: Complex, base: MeshLengthsSq) -> Self {
        Self::new(complex, base, Box::new(|_t| 1.0))
    }

    /// The scale factor `a(t)`, applied to lengths.
    pub fn scale_factor(&self, t: f64) -> f64 {
        (self.scale)(t)
    }
}

impl MetricDriver for FlrwDriver {
    fn lengths_sq_at(&self, t: f64) -> MeshLengthsSq {
        let a = (self.scale)(t);
        // Lengths scale by a, so squared lengths scale by a^2.
        MeshLengthsSq::new_unchecked(&self.base_sq * (a * a))
    }
    fn complex(&self) -> &Complex {
        &self.complex
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use simplicial::r#gen::cartesian::CartesianGrid;

    fn unit_square_driver(scale: Box<dyn Fn(f64) -> f64 + Send + Sync>) -> FlrwDriver {
        let (complex, coords) = CartesianGrid::new_unit(2, 2).triangulate();
        let base = coords.to_edge_lengths_sq(&complex);
        FlrwDriver::new(complex, base, scale)
    }

    #[test]
    fn static_driver_returns_base_lengths_at_all_times() {
        let driver = unit_square_driver(Box::new(|_t| 1.0));
        let l0 = driver.lengths_sq_at(0.0);
        let l5 = driver.lengths_sq_at(5.0);
        assert_relative_eq!((l0.vector() - l5.vector()).norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn flrw_driver_scales_every_edge_length_by_a_of_t() {
        let driver = unit_square_driver(Box::new(|t| 1.0 + t)); // a(t) = 1 + t
        let l0 = driver.lengths_sq_at(0.0);
        let l1 = driver.lengths_sq_at(1.0); // a = 2, so l^2 scales by 4
        assert_relative_eq!(
            (l1.vector() - l0.vector() * 4.0).norm(),
            0.0,
            epsilon = 1e-12
        );
        // Restated in terms of lengths: every edge is exactly twice as long.
        for e in 0..l0.nedges() {
            assert_relative_eq!(l1.length(e), 2.0 * l0.length(e), epsilon = 1e-12);
        }
    }
}
