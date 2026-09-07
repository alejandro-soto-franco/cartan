//! The p-atic fibre: a rotor with amplitudes.
//!
//! The state at a vertex is an element of `SU(2)/H^` together with the
//! amplitudes of the `H^`-invariant order parameters. The rotor is stored
//! rather than the invariant tensor, so the lift survives and the defect class
//! stays visible; the tensor is a derived observable.

use cartan_core::fiber::{Fiber, FiberOps};
use cartan_core::rotor::{Rotor, Rotor3};

use crate::group::SymmetryGroup;

/// A rotor with amplitudes.
///
/// `Default` is the identity rotor with zero amplitudes, which is the
/// isotropic state rather than an ordered one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaticElement<A> {
    /// The frame, an element of `SU(2)` representing its coset `R H^`.
    pub rotor: Rotor3,
    /// Amplitudes of the invariant order parameters.
    pub amplitudes: A,
}

impl<A: Default> Default for PaticElement<A> {
    fn default() -> Self {
        Self {
            rotor: Rotor3::IDENTITY,
            amplitudes: A::default(),
        }
    }
}

impl<A: AsRef<[f64]>> PaticElement<A> {
    /// The element as a flat run of reals: rotor components, then amplitudes.
    pub fn to_flat(&self, out: &mut [f64]) {
        out[0] = self.rotor.w;
        out[1] = self.rotor.x;
        out[2] = self.rotor.y;
        out[3] = self.rotor.z;
        out[4..].copy_from_slice(self.amplitudes.as_ref());
    }

    /// Deviation of the rotor from the unit sphere.
    #[must_use]
    pub fn norm_defect(&self) -> f64 {
        let r = &self.rotor;
        (r.w * r.w + r.x * r.x + r.y * r.y + r.z * r.z).sqrt() - 1.0
    }
}

/// The fibre of p-atic states for symmetry `H`.
#[derive(Clone, Copy, Debug, Default)]
pub struct PaticFiber<H: SymmetryGroup>(core::marker::PhantomData<H>);

impl<H: SymmetryGroup> Fiber for PaticFiber<H> {
    type Element = PaticElement<H::Amplitudes>;

    /// Meaningful degrees of freedom: four rotor components plus the
    /// amplitudes. The amplitude array may be wider for a const-generic
    /// family, and the surplus entries are not degrees of freedom.
    const FIBER_DIM: usize = 4 + H::N_AMPLITUDES;

    fn zero() -> Self::Element {
        Self::Element::default()
    }

    fn transport_by(rotation: &[f64], d: usize, element: &Self::Element) -> Self::Element {
        debug_assert_eq!(d, 3, "PaticFiber requires d=3");
        debug_assert!(rotation.len() >= 9);
        let mut m = [0.0_f64; 9];
        m.copy_from_slice(&rotation[..9]);
        let frame = Rotor3::from_matrix(&m);
        Self::Element {
            rotor: frame.compose(&element.rotor),
            amplitudes: element.amplitudes,
        }
    }

    fn transport_by_rotor(rotor: &Rotor, element: &Self::Element) -> Self::Element {
        match rotor {
            Rotor::R3(r) => Self::Element {
                rotor: r.compose(&element.rotor),
                amplitudes: element.amplitudes,
            },
            Rotor::R2(_) => {
                debug_assert!(false, "PaticFiber requires a 3D rotor");
                *element
            }
        }
    }
}

impl<H: SymmetryGroup> FiberOps for PaticFiber<H> {
    fn accumulate_diff(
        target: &mut Self::Element,
        a: &Self::Element,
        b: &Self::Element,
        scale: f64,
    ) {
        target.rotor.w += scale * (a.rotor.w - b.rotor.w);
        target.rotor.x += scale * (a.rotor.x - b.rotor.x);
        target.rotor.y += scale * (a.rotor.y - b.rotor.y);
        target.rotor.z += scale * (a.rotor.z - b.rotor.z);
        let (ta, aa, ba) = (
            target.amplitudes.as_mut(),
            a.amplitudes.as_ref(),
            b.amplitudes.as_ref(),
        );
        for i in 0..ta.len() {
            ta[i] += scale * (aa[i] - ba[i]);
        }
    }

    fn scale_element(target: &mut Self::Element, scale: f64) {
        target.rotor.w *= scale;
        target.rotor.x *= scale;
        target.rotor.y *= scale;
        target.rotor.z *= scale;
        for v in target.amplitudes.as_mut() {
            *v *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::{AxialApolar, Dicyclic, SymmetryGroup};

    #[test]
    fn fiber_dim_is_rotor_plus_amplitudes() {
        assert_eq!(<PaticFiber<AxialApolar> as Fiber>::FIBER_DIM, 5);
        // FIBER_DIM counts meaningful degrees of freedom. The Dicyclic
        // amplitude array is width 3 so one type serves every K, and the
        // surplus entries are not degrees of freedom.
        assert_eq!(<PaticFiber<Dicyclic<2>> as Fiber>::FIBER_DIM, 6);
        assert_eq!(<PaticFiber<Dicyclic<3>> as Fiber>::FIBER_DIM, 5);
        assert_eq!(<Dicyclic<2> as SymmetryGroup>::N_AMPLITUDES, 2);
        assert_eq!(<Dicyclic<3> as SymmetryGroup>::N_AMPLITUDES, 1);
    }

    #[test]
    fn rotor_and_matrix_transport_agree() {
        let rot = Rotor3 {
            w: 0.5,
            x: 0.5,
            y: 0.5,
            z: 0.5,
        };
        let elem = PaticElement {
            rotor: Rotor3 {
                w: 0.6,
                x: 0.8,
                y: 0.0,
                z: 0.0,
            },
            amplitudes: [0.7_f64],
        };
        let via_rotor = <PaticFiber<AxialApolar>>::transport_by_rotor(&Rotor::R3(rot), &elem);
        let via_matrix = <PaticFiber<AxialApolar>>::transport_by(&rot.to_matrix(), 3, &elem);
        for (a, b) in [
            (via_rotor.rotor.w, via_matrix.rotor.w),
            (via_rotor.rotor.x, via_matrix.rotor.x),
            (via_rotor.rotor.y, via_matrix.rotor.y),
            (via_rotor.rotor.z, via_matrix.rotor.z),
        ] {
            assert!((a - b).abs() < 1e-14, "{a} vs {b}");
        }
        assert_eq!(via_rotor.amplitudes, via_matrix.amplitudes);
    }

    #[test]
    fn transport_preserves_the_unit_sphere() {
        let rot = Rotor3 {
            w: 0.5,
            x: 0.5,
            y: 0.5,
            z: 0.5,
        };
        let elem = PaticElement {
            rotor: Rotor3 {
                w: 0.6,
                x: 0.8,
                y: 0.0,
                z: 0.0,
            },
            amplitudes: [1.0_f64],
        };
        let moved = <PaticFiber<AxialApolar>>::transport_by_rotor(&Rotor::R3(rot), &elem);
        assert!(moved.norm_defect().abs() < 1e-15, "{}", moved.norm_defect());
    }

    #[test]
    fn default_is_the_isotropic_state() {
        let e = <PaticFiber<Dicyclic<2>> as Fiber>::zero();
        assert_eq!(e.rotor, Rotor3::IDENTITY);
        assert_eq!(e.amplitudes, [0.0, 0.0, 0.0]);
    }
}
