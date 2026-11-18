//! Plaquette holonomy and the curvature it reads.

use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

use crate::connection::Connection;

/// Orthonormal basis of so(d): `(e_p e_q^T - e_q e_p^T) / sqrt 2`, `p < q`.
fn so_pairs(d: usize) -> Vec<(usize, usize)> {
    (0..d)
        .flat_map(|p| (p + 1..d).map(move |q| (p, q)))
        .collect()
}

/// Coordinates of a `d x d` matrix on the so(d) basis.
fn coords(m: &DMatrix<f64>, pairs: &[(usize, usize)]) -> DVector<f64> {
    let s = 0.5f64.sqrt();
    DVector::from_iterator(
        pairs.len(),
        pairs.iter().map(|&(p, q)| s * (m[(p, q)] - m[(q, p)])),
    )
}

fn point(x: &[f64], dim: usize, i: usize) -> DVector<f64> {
    DVector::from_column_slice(&x[i * dim..(i + 1) * dim])
}

/// Per triangle, the skew part `L` of the holonomy and the oriented area
/// bivector `S`, both on the so(d) basis, read in the frame of each vertex.
/// Row `3 t + c` belongs to vertex `c` of triangle `t` and stores `L` then `S`.
fn observations(c: &Connection, x: &[f64], tris: &[[usize; 3]]) -> Vec<f64> {
    let pairs = so_pairs(c.d);
    let m = pairs.len();
    tris.par_iter()
        .flat_map_iter(|&[i, j, k]| {
            let mut buf = Vec::with_capacity(6 * m);
            for &(p, q, r) in &[(i, j, k), (j, k, i), (k, i, j)] {
                // Loop p -> q -> r -> p, read in the frame of p.
                let h = c.transport_into(p, r) * c.transport_into(r, q) * c.transport_into(q, p);
                let l = (&h - h.transpose()) * 0.5;
                let tp = c.frame(p);
                let a = tp.transpose() * (point(x, c.dim, q) - point(x, c.dim, p));
                let b = tp.transpose() * (point(x, c.dim, r) - point(x, c.dim, p));
                let s = (&a * b.transpose() - &b * a.transpose()) * 0.5;
                buf.extend(coords(&l, &pairs).iter());
                buf.extend(coords(&s, &pairs).iter());
            }
            buf
        })
        .collect()
}

/// Holonomy angle and projected area of every triangle.
#[derive(Debug, Clone)]
pub struct TriangleGeometry {
    /// The triangles, `i < j < k`.
    pub triangles: Vec<[usize; 3]>,
    /// Rotation angle of the holonomy on the triangle's plane, in the frame of
    /// its first vertex. On a surface it is `K * area` to leading order.
    pub angle: Vec<f64>,
    /// Area of the triangle projected on the tangent plane of its first vertex.
    pub area: Vec<f64>,
    obs: Vec<f64>,
}

impl TriangleGeometry {
    /// Gauss-Bonnet aggregate: total angle over total area.
    pub fn curvature(&self) -> f64 {
        self.angle.iter().sum::<f64>() / self.area.iter().sum::<f64>()
    }
}

/// Holonomy of every triangle of the connection's graph.
pub fn triangle_geometry(c: &Connection, x: &[f64]) -> TriangleGeometry {
    let triangles = c.graph.triangles();
    let obs = observations(c, x, &triangles);
    let m = so_pairs(c.d).len();
    let stride = 2 * m;
    let (angle, area): (Vec<f64>, Vec<f64>) = (0..triangles.len())
        .map(|t| {
            let o = 3 * t * stride;
            let l = DVector::from_column_slice(&obs[o..o + m]);
            let s = DVector::from_column_slice(&obs[o + m..o + 2 * m]);
            // <L, S> = 2 area <u, L w> and |S| = sqrt(2) area. The minus sign
            // makes the round sphere positive.
            let sn = s.norm();
            (-l.dot(&s) / (2f64.sqrt() * sn), sn / 2f64.sqrt())
        })
        .unzip();
    TriangleGeometry {
        triangles,
        angle,
        area,
        obs,
    }
}

/// Curvature operator per point.
#[derive(Debug, Clone, Copy)]
pub struct CurvatureOperator {
    /// Scalar curvature.
    pub scalar: f64,
    /// `Ric(v, v) / |v|^2` for the given field, `NaN` without one.
    pub ric_vv: f64,
    /// Relative residual of the fit.
    pub residual: f64,
    /// Triangle readings used.
    pub count: usize,
}

/// Symmetric curvature operator `M` on so(d) per point, `L = M S` fitted by
/// ridge least squares over the triangles at that point, with the Ricci and
/// scalar curvature it implies. Points with fewer readings than unknowns get
/// `NaN`. The system has `m (m + 1) / 2` unknowns, `m = d (d - 1) / 2`.
pub fn curvature_operator(
    c: &Connection,
    geom: &TriangleGeometry,
    v: Option<&[f64]>,
) -> Vec<CurvatureOperator> {
    let d = c.d;
    let pairs = so_pairs(d);
    let m = pairs.len();
    let stride = 2 * m;
    let tris = &geom.triangles;
    let mut start = vec![0usize; c.n + 1];
    for t in tris {
        for &p in t {
            start[p + 1] += 1;
        }
    }
    for p in 0..c.n {
        start[p + 1] += start[p];
    }
    let mut fill = start.clone();
    let mut members = vec![0usize; 3 * tris.len()];
    for (t, tri) in tris.iter().enumerate() {
        for (k, &p) in tri.iter().enumerate() {
            members[fill[p]] = 3 * t + k;
            fill[p] += 1;
        }
    }
    let sym: Vec<(usize, usize)> = (0..m).flat_map(|a| (a..m).map(move |b| (a, b))).collect();
    let obs = &geom.obs;
    let lvec = |o: usize| DVector::from_column_slice(&obs[o * stride..o * stride + m]);
    let svec = |o: usize| DVector::from_column_slice(&obs[o * stride + m..(o + 1) * stride]);
    let basis: Vec<DMatrix<f64>> = pairs
        .iter()
        .map(|&(p, q)| {
            let mut e = DMatrix::<f64>::zeros(d, d);
            e[(p, q)] = 0.5f64.sqrt();
            e[(q, p)] = -(0.5f64.sqrt());
            e
        })
        .collect();
    (0..c.n)
        .into_par_iter()
        .map(|i| {
            let mine = &members[start[i]..start[i + 1]];
            if mine.len() * m < sym.len() {
                return CurvatureOperator {
                    scalar: f64::NAN,
                    ric_vv: f64::NAN,
                    residual: f64::NAN,
                    count: mine.len(),
                };
            }
            let rows = mine.len() * m;
            let mut amat = DMatrix::<f64>::zeros(rows, sym.len());
            let mut rhs = DVector::<f64>::zeros(rows);
            for (t, &ob) in mine.iter().enumerate() {
                let (l, s) = (lvec(ob), svec(ob));
                for a in 0..m {
                    let r = t * m + a;
                    rhs[r] = l[a];
                    for (col, &(p, q)) in sym.iter().enumerate() {
                        if p == a {
                            amat[(r, col)] += s[q];
                        }
                        if q == a && p != q {
                            amat[(r, col)] += s[p];
                        }
                    }
                }
            }
            let scale = rhs.norm().max(1e-300);
            let ata = amat.transpose() * &amat
                + DMatrix::<f64>::identity(sym.len(), sym.len()) * (1e-8 * scale * scale);
            let sol = ata
                .lu()
                .solve(&(amat.transpose() * &rhs))
                .unwrap_or_else(|| DVector::zeros(sym.len()));
            let residual = (&amat * &sol - &rhs).norm() / scale;
            let mut mm = DMatrix::<f64>::zeros(m, m);
            for (col, &(p, q)) in sym.iter().enumerate() {
                mm[(p, q)] = sol[col];
                mm[(q, p)] = sol[col];
            }
            // R(x, y) = -M(x y^T - y x^T); Ric(x, y) = sum_a e_a^T R(e_a, x) y.
            let r_op = |xv: &DVector<f64>, yv: &DVector<f64>| -> DMatrix<f64> {
                let p = xv * yv.transpose() - yv * xv.transpose();
                let cc = &mm * coords(&p, &pairs);
                let mut out = DMatrix::<f64>::zeros(d, d);
                for (b, e) in basis.iter().enumerate() {
                    out += e * cc[b];
                }
                -out
            };
            let unit = |a: usize| DVector::<f64>::from_fn(d, |r, _| if r == a { 1.0 } else { 0.0 });
            let mut ric = DMatrix::<f64>::zeros(d, d);
            for a in 0..d {
                let ea = unit(a);
                for xi in 0..d {
                    let rr = r_op(&ea, &unit(xi));
                    for yi in 0..d {
                        ric[(xi, yi)] += (ea.transpose() * &rr)[(0, yi)];
                    }
                }
            }
            let ric = (&ric + ric.transpose()) * 0.5;
            let ric_vv = match v {
                Some(v) => {
                    let vi = DVector::from_vec(c.tangent(i, &v[i * c.dim..(i + 1) * c.dim]));
                    let nn = vi.norm_squared();
                    if nn > 0.0 {
                        (vi.transpose() * &ric * &vi)[(0, 0)] / nn
                    } else {
                        f64::NAN
                    }
                }
                None => f64::NAN,
            };
            CurvatureOperator {
                scalar: ric.trace(),
                ric_vv,
                residual,
                count: mine.len(),
            }
        })
        .collect()
}
