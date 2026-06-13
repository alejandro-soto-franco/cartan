//! Geometric-algebra rotors for discrete parallel transport.
//!
//! A rotor is an element of the even subalgebra Cl+(d): a unit complex number
//! for d=2 (`Rotor2`) and a unit quaternion for d=3 (`Rotor3`). Rotors carry the
//! same SO(d) rotation as a d*d matrix in fewer floats (4 vs 9 for d=3), compose
//! cheaply, renormalize trivially, and reverse by conjugation. The matrix bridge
//! (`to_matrix` / `from_matrix`) keeps them interchangeable with the existing
//! matrix transport path, which remains the equivalence oracle.

#![allow(clippy::needless_range_loop)]

/// A unit quaternion rotor in Cl+(3): `R = w + x*e23 + y*e31 + z*e12`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rotor3 {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Rotor3 {
    /// The identity rotor.
    pub const IDENTITY: Rotor3 = Rotor3 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    /// Build a rotor from a row-major SO(3) matrix (Shepperd's method).
    pub fn from_matrix(m: &[f64; 9]) -> Rotor3 {
        let (m00, m11, m22) = (m[0], m[4], m[8]);
        let trace = m00 + m11 + m22;
        let r = if trace > 0.0 {
            let s = 0.5 / (trace + 1.0).sqrt();
            Rotor3 {
                w: 0.25 / s,
                x: (m[7] - m[5]) * s,
                y: (m[2] - m[6]) * s,
                z: (m[3] - m[1]) * s,
            }
        } else if m00 > m11 && m00 > m22 {
            let s = 2.0 * (1.0 + m00 - m11 - m22).sqrt();
            Rotor3 {
                w: (m[7] - m[5]) / s,
                x: 0.25 * s,
                y: (m[1] + m[3]) / s,
                z: (m[2] + m[6]) / s,
            }
        } else if m11 > m22 {
            let s = 2.0 * (1.0 + m11 - m00 - m22).sqrt();
            Rotor3 {
                w: (m[2] - m[6]) / s,
                x: (m[1] + m[3]) / s,
                y: 0.25 * s,
                z: (m[5] + m[7]) / s,
            }
        } else {
            let s = 2.0 * (1.0 + m22 - m00 - m11).sqrt();
            Rotor3 {
                w: (m[3] - m[1]) / s,
                x: (m[2] + m[6]) / s,
                y: (m[5] + m[7]) / s,
                z: 0.25 * s,
            }
        };
        r.normalized()
    }

    /// Convert to a row-major SO(3) matrix.
    pub fn to_matrix(&self) -> [f64; 9] {
        let Rotor3 { w, x, y, z } = *self;
        [
            1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z),       2.0*(x*z + w*y),
            2.0*(x*y + w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x),
            2.0*(x*z - w*y),       2.0*(y*z + w*x),       1.0 - 2.0*(x*x + y*y),
        ]
    }

    /// Rotate a vector (`R v R~`), computed via the matrix for sign-safety.
    pub fn rotate_vec(&self, v: [f64; 3]) -> [f64; 3] {
        let m = self.to_matrix();
        [
            m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
            m[3]*v[0] + m[4]*v[1] + m[5]*v[2],
            m[6]*v[0] + m[7]*v[1] + m[8]*v[2],
        ]
    }

    /// Hamilton product: `self` then `other` (matches matrix multiply order).
    pub fn compose(&self, other: &Rotor3) -> Rotor3 {
        let a = self;
        let b = other;
        Rotor3 {
            w: a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
            x: a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
            y: a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
            z: a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        }
    }

    /// Reverse (conjugate): the inverse rotation.
    pub fn reverse(&self) -> Rotor3 {
        Rotor3 { w: self.w, x: -self.x, y: -self.y, z: -self.z }
    }

    /// Return a unit-normalized copy.
    pub fn normalized(&self) -> Rotor3 {
        let n = (self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z).sqrt();
        if n < 1e-30 {
            Rotor3::IDENTITY
        } else {
            Rotor3 { w: self.w/n, x: self.x/n, y: self.y/n, z: self.z/n }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic pseudo-random rotors via a fixed set of axis/angle pairs.
    fn sample_rotations() -> Vec<[f64; 9]> {
        let axes: &[[f64; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0], [0.3, -0.7, 0.5], [-0.2, 0.4, -0.9]];
        let angles: &[f64] = &[0.1, 0.7, 1.3, -2.0, 3.0];
        let mut out = vec![];
        for a in axes {
            let n = (a[0]*a[0] + a[1]*a[1] + a[2]*a[2]).sqrt();
            let u = [a[0]/n, a[1]/n, a[2]/n];
            for &th in angles {
                out.push(axis_angle_matrix(u, th));
            }
        }
        out
    }

    // Rodrigues rotation matrix, row-major.
    fn axis_angle_matrix(u: [f64; 3], th: f64) -> [f64; 9] {
        let (c, s) = (th.cos(), th.sin());
        let t = 1.0 - c;
        let (ux, uy, uz) = (u[0], u[1], u[2]);
        [
            c + ux*ux*t,      ux*uy*t - uz*s,   ux*uz*t + uy*s,
            uy*ux*t + uz*s,   c + uy*uy*t,      uy*uz*t - ux*s,
            uz*ux*t - uy*s,   uz*uy*t + ux*s,   c + uz*uz*t,
        ]
    }

    fn matmul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
        let mut m = [0.0; 9];
        for i in 0..3 { for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 { s += a[i*3+k] * b[k*3+j]; }
            m[i*3+j] = s;
        }}
        m
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    #[test]
    fn rotor3_matrix_roundtrip() {
        for m in sample_rotations() {
            let r = Rotor3::from_matrix(&m);
            assert!(max_abs_diff(&r.to_matrix(), &m) < 1e-12, "roundtrip {m:?}");
        }
    }

    #[test]
    fn rotor3_rotate_vec_matches_matrix() {
        let v = [0.37, -1.1, 2.4];
        for m in sample_rotations() {
            let r = Rotor3::from_matrix(&m);
            let mv = [
                m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
                m[3]*v[0] + m[4]*v[1] + m[5]*v[2],
                m[6]*v[0] + m[7]*v[1] + m[8]*v[2],
            ];
            assert!(max_abs_diff(&r.rotate_vec(v), &mv) < 1e-12);
        }
    }

    #[test]
    fn rotor3_compose_matches_matmul() {
        let rots = sample_rotations();
        let r0 = Rotor3::from_matrix(&rots[1]);
        let r1 = Rotor3::from_matrix(&rots[7]);
        let composed = r0.compose(&r1).to_matrix();
        let expected = matmul(&rots[1], &rots[7]);
        assert!(max_abs_diff(&composed, &expected) < 1e-12);
    }

    #[test]
    fn rotor3_reverse_matches_transpose() {
        for m in sample_rotations() {
            let r = Rotor3::from_matrix(&m);
            let rev = r.reverse().to_matrix();
            let mut mt = [0.0; 9];
            for i in 0..3 { for j in 0..3 { mt[i*3+j] = m[j*3+i]; } }
            assert!(max_abs_diff(&rev, &mt) < 1e-12);
        }
    }
}
