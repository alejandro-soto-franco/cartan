use cartan_dec::line_bundle::Section;
use crate::vtp::Field;

/// Convert a spin-2 nematic Section on a FLAT (z=0) mesh to a per-vertex 3D
/// director vector, scaled by the scalar order parameter. The director is a
/// headless axis; the renderer must draw it double-ended (use the `nematic` flag).
pub fn director_field_flat(section: &Section<2>) -> Field {
    let (q1, q2) = section.to_real_components();
    let nv = q1.len();
    let mut values = Vec::with_capacity(nv * 3);
    for i in 0..nv {
        let order = 2.0 * (q1[i] * q1[i] + q2[i] * q2[i]).sqrt();
        let theta = 0.5 * q2[i].atan2(q1[i]);
        values.push(order * theta.cos());
        values.push(order * theta.sin());
        values.push(0.0);
    }
    Field::Vector { name: "director".into(), values, nematic: true }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cartan_dec::line_bundle::Section;
    use num_complex::Complex;

    #[test]
    fn director_points_along_x_for_real_positive() {
        // z = S/2 * exp(i*2*theta). theta=0 -> z real positive.
        let mut s = Section::<2>::zeros(1);
        s.values[0] = Complex::new(0.5, 0.0); // |z|=0.5 -> scalar order 1.0, theta=0
        let field = director_field_flat(&s);
        if let Field::Vector { values, nematic, .. } = field {
            assert!(nematic);
            assert!((values[0] - 1.0).abs() < 1e-12); // x-component = order * cos(0)
            assert!(values[1].abs() < 1e-12);
            assert!(values[2].abs() < 1e-12);
        } else { panic!("expected vector field"); }
    }
}
