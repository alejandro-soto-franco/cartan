//! Prescribed time-dependent geometry: a driver supplies edge lengths l_e(t).

use cartan_exterior::Dim;
use cartan_simplicial::geometry::metric::mesh::MeshLengths;
use cartan_simplicial::linalg::Vector;
use cartan_simplicial::topology::complex::Complex;

/// A prescribed evolving geometry over a fixed abstract complex.
pub trait MetricDriver {
    /// The intrinsic edge lengths at time `t`.
    fn lengths_at(&self, t: f64) -> MeshLengths;
    /// The fixed abstract complex the metric lives on.
    fn complex(&self) -> &Complex;
    /// Spatial dimension of the complex.
    fn dim(&self) -> Dim {
        self.complex().dim()
    }
}

/// FLRW-style uniform scaling: l_e(t) = a(t) * l_e(0).
pub struct FlrwDriver {
    complex: Complex,
    base: Vector,
    scale: Box<dyn Fn(f64) -> f64 + Send + Sync>,
}

impl FlrwDriver {
    pub fn new(
        complex: Complex,
        base: MeshLengths,
        scale: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    ) -> Self {
        Self { complex, base: base.into_vector(), scale }
    }

    /// A static (non-evolving) background, a(t) = 1.
    pub fn static_metric(complex: Complex, base: MeshLengths) -> Self {
        Self::new(complex, base, Box::new(|_t| 1.0))
    }

    /// The scale factor a(t).
    pub fn scale_factor(&self, t: f64) -> f64 {
        (self.scale)(t)
    }
}

impl MetricDriver for FlrwDriver {
    fn lengths_at(&self, t: f64) -> MeshLengths {
        let a = (self.scale)(t);
        MeshLengths::new_unchecked(&self.base * a)
    }
    fn complex(&self) -> &Complex {
        &self.complex
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use cartan_simplicial::r#gen::cartesian::CartesianMeshInfo;

    fn unit_square_driver(
        scale: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    ) -> FlrwDriver {
        let (complex, coords) = CartesianMeshInfo::new_unit(2, 2).compute_coord_complex();
        let base = coords.to_edge_lengths(&complex);
        FlrwDriver::new(complex, base, scale)
    }

    #[test]
    fn static_driver_returns_base_lengths_at_all_times() {
        let driver = unit_square_driver(Box::new(|_t| 1.0));
        let l0 = driver.lengths_at(0.0);
        let l5 = driver.lengths_at(5.0);
        assert_relative_eq!((l0.vector() - l5.vector()).norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn flrw_driver_scales_every_edge_by_a_of_t() {
        let driver = unit_square_driver(Box::new(|t| 1.0 + t)); // a(t) = 1 + t
        let l0 = driver.lengths_at(0.0);
        let l1 = driver.lengths_at(1.0); // a = 2
        assert_relative_eq!(
            (l1.vector() - l0.vector() * 2.0).norm(),
            0.0,
            epsilon = 1e-12
        );
    }
}
