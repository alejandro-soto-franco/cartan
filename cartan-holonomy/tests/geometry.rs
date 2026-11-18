//! Checks on manifolds of known curvature and on known vector fields.

use cartan_core::Manifold;
use cartan_holonomy::{
    Connection, JetOptions, fit_jet, knn::knn, local_jet, transport, triangle_geometry,
};
use cartan_manifolds::Grassmann;
use nalgebra::{DMatrix, SMatrix};

/// Deterministic uniform numbers in [-0.5, 0.5).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    }
    fn gauss(&mut self) -> f64 {
        let (u, v) = (self.next() + 0.5, self.next() + 0.5);
        (-2.0 * u.max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// Embeds points of R^m into R^dim by a fixed rotation.
fn embed(p: &[Vec<f64>], dim: usize, rng: &mut Lcg) -> Vec<f64> {
    let q = DMatrix::<f64>::from_fn(dim, dim, |_, _| rng.next())
        .qr()
        .q();
    p.iter()
        .flat_map(|row| {
            let mut v = vec![0.0; dim];
            for (k, &c) in row.iter().enumerate() {
                for r in 0..dim {
                    v[r] += q[(r, k)] * c;
                }
            }
            v
        })
        .collect()
}

#[test]
fn transport_is_the_grassmann_geodesic_map() {
    const N: usize = 12;
    let mut rng = Lcg(7);
    let a = DMatrix::<f64>::from_fn(N, 3, |_, _| rng.next()).qr().q();
    let b = (&a + DMatrix::<f64>::from_fn(N, 3, |_, _| 0.1 * rng.next()))
        .qr()
        .q();
    let gr = Grassmann::<N, 3>;
    let ta = SMatrix::<f64, N, 3>::from_column_slice(a.as_slice());
    let tb = SMatrix::<f64, N, 3>::from_column_slice(b.as_slice());
    let y = gr.exp(&ta, &gr.log(&ta, &tb).unwrap());
    let geodesic = tb.transpose() * y;
    let o = transport(&b, &a);
    let diff = (DMatrix::from_column_slice(3, 3, geodesic.as_slice()) - &o).norm();
    assert!(diff < 1e-10, "{diff}");
    assert!((transport(&a, &b) - o.transpose()).norm() < 1e-10);
}

#[test]
fn knn_matches_exhaustive_search() {
    let (n, dim, k) = (700, 9, 12);
    let mut rng = Lcg(3);
    let x: Vec<f64> = (0..n * dim).map(|_| rng.next()).collect();
    let fast = knn(&x, n, dim, k);
    for i in 0..n {
        let mut d: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                (
                    (0..dim)
                        .map(|t| (x[i * dim + t] - x[j * dim + t]).powi(2))
                        .sum(),
                    j,
                )
            })
            .collect();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let slow: Vec<usize> = d[..k].iter().map(|&(_, j)| j).collect();
        assert_eq!(fast[i], slow, "point {i}");
    }
}

#[test]
fn sphere_is_positively_curved_and_plane_is_flat() {
    let (n, dim) = (6000, 10);
    let mut rng = Lcg(11);
    let sphere: Vec<Vec<f64>> = (0..n)
        .map(|_| {
            let g: Vec<f64> = (0..3).map(|_| rng.gauss()).collect();
            let r = g.iter().map(|v| v * v).sum::<f64>().sqrt();
            g.iter().map(|v| 2.0 * v / r).collect()
        })
        .collect();
    let plane: Vec<Vec<f64>> = (0..n)
        .map(|_| vec![4.0 * rng.next(), 4.0 * rng.next()])
        .collect();
    let xs = embed(&sphere, dim, &mut rng);
    let xp = embed(&plane, dim, &mut rng);
    let ks = triangle_geometry(&Connection::new(&xs, n, dim, 2, 20, 10), &xs).curvature();
    let kp = triangle_geometry(&Connection::new(&xp, n, dim, 2, 20, 10), &xp).curvature();
    // The estimator overestimates at this density (0.30 for 1/4); the bias falls
    // as the sampling densifies.
    assert!((0.22..0.34).contains(&ks), "sphere of radius 2: {ks}");
    assert!(kp.abs() < 1e-8, "plane: {kp}");
}

#[test]
fn rotation_field_has_its_vorticity() {
    // v = J x on a flat 3-box: |omega|^2 = |J|_F^2 = 2, no expansion or shear.
    let (n, dim) = (3000, 8);
    let mut rng = Lcg(5);
    let box3: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..3).map(|_| 2.0 * rng.next()).collect())
        .collect();
    let field: Vec<Vec<f64>> = box3.iter().map(|p| vec![-p[1], p[0], 0.0]).collect();
    let x = embed(&box3, dim, &mut Lcg(17));
    let v = embed(&field, dim, &mut Lcg(17));
    let c = Connection::new(&x, n, dim, 3, 22, 15);
    let local = local_jet(&c, &x, &v);
    let fit = fit_jet(&c, &x, &v, None, &JetOptions::default());
    assert!(fit.report.converged);
    let med = |f: &dyn Fn(usize) -> f64| {
        let mut s: Vec<f64> = (0..n).map(f).filter(|v| v.is_finite()).collect();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    };
    let lv = med(&|i| local[i].vort2);
    let fv = med(&|i| cartan_holonomy::jet::invariants(&fit.a[i * 9..(i + 1) * 9], 3).vort2);
    let fs = med(&|i| cartan_holonomy::jet::invariants(&fit.a[i * 9..(i + 1) * 9], 3).shear2);
    assert!((lv - 2.0).abs() < 1e-6, "local vorticity {lv}");
    assert!((fv - 2.0).abs() < 1e-6, "variational vorticity {fv}");
    assert!(fs < 1e-10, "variational shear {fs}");
}
