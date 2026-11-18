//! The covariant derivative of a vector field on the cloud.

// Slot loops index several parallel per-edge arrays at once.
#![allow(clippy::needless_range_loop)]

use cartan_matfree::{CgReport, MassBackend, pcg};
use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::connection::Connection;

/// Directed-edge data of a field: `r_ij = O_ij v_j - v_i` and `Delta_ij =
/// T_i^T (x_j - x_i)`, per slot.
struct EdgeData {
    r: Vec<f64>,
    delta: Vec<f64>,
}

fn edge_data(c: &Connection, x: &[f64], v: &[f64]) -> EdgeData {
    let (d, dim, dd) = (c.d, c.dim, c.d * c.d);
    let vt: Vec<Vec<f64>> = (0..c.n)
        .into_par_iter()
        .map(|i| c.tangent(i, &v[i * dim..(i + 1) * dim]))
        .collect();
    let per: Vec<(Vec<f64>, Vec<f64>)> = (0..c.n)
        .into_par_iter()
        .map(|i| {
            let mut r = Vec::new();
            let mut dl = Vec::new();
            for (k, &j) in c.graph.adj[i].iter().enumerate() {
                let q = &c.o[(c.start[i] + k) * dd..(c.start[i] + k + 1) * dd];
                for a in 0..d {
                    r.push((0..d).map(|b| q[a * d + b] * vt[j][b]).sum::<f64>() - vt[i][a]);
                }
                let dx: Vec<f64> = (0..dim).map(|t| x[j * dim + t] - x[i * dim + t]).collect();
                dl.extend(c.tangent(i, &dx));
            }
            (r, dl)
        })
        .collect();
    let (mut r, mut delta) = (Vec::new(), Vec::new());
    for (a, b) in per {
        r.extend(a);
        delta.extend(b);
    }
    EdgeData { r, delta }
}

/// Expansion, shear and vorticity of `A = nabla v` per point.
#[derive(Debug, Clone, Copy)]
pub struct LocalJet {
    /// `tr A`.
    pub theta: f64,
    /// `|S|^2` of the trace-free symmetric part.
    pub shear2: f64,
    /// `|W|^2` of the antisymmetric part.
    pub vort2: f64,
    /// Edges used.
    pub edges: usize,
}

/// Invariants of one `d x d` row-major jet.
pub fn invariants(a: &[f64], d: usize) -> LocalJet {
    let m = DMatrix::from_row_slice(d, d, a);
    let th = m.trace();
    let sym = (&m + m.transpose()) * 0.5 - DMatrix::<f64>::identity(d, d) * (th / d as f64);
    let om = (&m - m.transpose()) * 0.5;
    LocalJet {
        theta: th,
        shear2: sym.norm_squared(),
        vort2: om.norm_squared(),
        edges: 0,
    }
}

/// The covariant jet fitted at each point on its own: least squares of
/// `A_i Delta_ij = r_ij` over the edges at `i`. Points with fewer than `d + 1`
/// edges get `NaN`.
pub fn local_jet(c: &Connection, x: &[f64], v: &[f64]) -> Vec<LocalJet> {
    let e = edge_data(c, x, v);
    let d = c.d;
    (0..c.n)
        .into_par_iter()
        .map(|i| {
            let deg = c.graph.adj[i].len();
            let nan = LocalJet {
                theta: f64::NAN,
                shear2: f64::NAN,
                vort2: f64::NAN,
                edges: deg,
            };
            if deg < d + 1 {
                return nan;
            }
            let s0 = c.start[i];
            let a = DMatrix::from_row_slice(deg, d, &e.delta[s0 * d..(s0 + deg) * d]);
            let y = DMatrix::from_row_slice(deg, d, &e.r[s0 * d..(s0 + deg) * d]);
            match (a.transpose() * &a).lu().solve(&(a.transpose() * &y)) {
                Some(bt) => {
                    let b = bt.transpose();
                    let mut j = invariants(b.transpose().as_slice(), d);
                    j.edges = deg;
                    j
                }
                None => nan,
            }
        })
        .collect()
}

/// Options of the variational jet fit.
#[derive(Debug, Clone, Copy)]
pub struct JetOptions {
    /// Smoothing weight in units of the median `|Delta|^2`. Zero is the local
    /// fit with a ridge.
    pub alpha: f64,
    /// Relative ridge on each cell block, in units of the median `|Delta|^2`.
    pub ridge: f64,
    /// Relative residual at which conjugate gradients stop.
    pub tol: f64,
    /// Iteration cap.
    pub max_iter: usize,
}

impl Default for JetOptions {
    fn default() -> Self {
        Self {
            alpha: 0.3,
            ridge: 1e-8,
            tol: 1e-8,
            max_iter: 5000,
        }
    }
}

/// Result of [`fit_jet`].
#[derive(Debug, Clone)]
pub struct JetFit {
    /// `A_i` per point, `d x d` row-major, concatenated.
    pub a: Vec<f64>,
    /// The conjugate gradient report (zero iterations when `alpha = 0`).
    pub report: CgReport,
    /// Relative error `sum |r - A Delta|^2 / sum |r|^2` on held-out edges, `NaN`
    /// when none is held out.
    pub heldout: f64,
    /// The same on the training edges.
    pub train: f64,
    /// Mean `|A_i - O_ij A_j O_ij^T|^2` over directed edges over mean `|A_i|^2`.
    pub dirichlet: f64,
    /// Median `|Delta|^2`, the unit of `alpha`.
    pub scale: f64,
}

/// Normal operator `A_i C_i + ridge A_i + lam (deg_i A_i - sum_j O_ij A_j O_ij^T)`
/// with the exact cell blocks as preconditioner.
struct Normal<'a> {
    c: &'a Connection,
    cm: Vec<f64>,
    pm: Vec<f64>,
    lam: f64,
    ridge: f64,
}

fn right_mul(a: &[f64], m: &[f64], d: usize, out: &mut [f64]) {
    for r in 0..d {
        for s in 0..d {
            out[r * d + s] = (0..d).map(|k| a[r * d + k] * m[k * d + s]).sum();
        }
    }
}

/// `out_i = sum_j O_ij A_j O_ij^T`.
fn transported_sum(c: &Connection, a: &[f64], out: &mut [f64]) {
    let (d, dd) = (c.d, c.d * c.d);
    out.par_chunks_mut(dd).enumerate().for_each(|(i, oi)| {
        oi.iter_mut().for_each(|v| *v = 0.0);
        let mut tmp = vec![0.0; dd];
        for (k, &j) in c.graph.adj[i].iter().enumerate() {
            let s = c.start[i] + k;
            let q = &c.o[s * dd..(s + 1) * dd];
            let aj = &a[j * dd..(j + 1) * dd];
            right_mul(q, aj, d, &mut tmp);
            for r in 0..d {
                for t in 0..d {
                    oi[r * d + t] += (0..d).map(|u| tmp[r * d + u] * q[t * d + u]).sum::<f64>();
                }
            }
        }
    });
}

impl MassBackend for Normal<'_> {
    type Vector = Vec<f64>;
    fn ndofs(&self) -> usize {
        self.c.n * self.c.d * self.c.d
    }
    fn zeros(&self) -> Vec<f64> {
        vec![0.0; self.ndofs()]
    }
    fn from_host(&self, src: &[f64]) -> Vec<f64> {
        src.to_vec()
    }
    fn to_host(&self, src: &Vec<f64>, dst: &mut [f64]) {
        dst.copy_from_slice(src);
    }
    fn copy(&self, src: &Vec<f64>, dst: &mut Vec<f64>) {
        dst.copy_from_slice(src);
    }
    fn apply(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
        let (d, dd) = (self.c.d, self.c.d * self.c.d);
        if self.lam > 0.0 {
            transported_sum(self.c, x, y);
        }
        y.par_chunks_mut(dd).enumerate().for_each(|(i, yi)| {
            let mut ac = vec![0.0; dd];
            right_mul(
                &x[i * dd..(i + 1) * dd],
                &self.cm[i * dd..(i + 1) * dd],
                d,
                &mut ac,
            );
            let deg = self.c.graph.adj[i].len() as f64;
            for k in 0..dd {
                let xi = x[i * dd + k];
                let lap = if self.lam > 0.0 {
                    self.lam * (deg * xi - yi[k])
                } else {
                    0.0
                };
                yi[k] = ac[k] + self.ridge * xi + lap;
            }
        });
    }
    fn precondition(&self, r: &Vec<f64>, z: &mut Vec<f64>) {
        let (d, dd) = (self.c.d, self.c.d * self.c.d);
        z.par_chunks_mut(dd).enumerate().for_each(|(i, zi)| {
            right_mul(
                &r[i * dd..(i + 1) * dd],
                &self.pm[i * dd..(i + 1) * dd],
                d,
                zi,
            )
        });
    }
    fn dot(&self, a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        a.par_iter().zip(b.par_iter()).map(|(p, q)| p * q).sum()
    }
    fn axpy(&self, a: f64, x: &Vec<f64>, y: &mut Vec<f64>) {
        y.par_iter_mut()
            .zip(x.par_iter())
            .for_each(|(p, q)| *p += a * q);
    }
    fn aypx(&self, a: f64, x: &Vec<f64>, y: &mut Vec<f64>) {
        y.par_iter_mut()
            .zip(x.par_iter())
            .for_each(|(p, q)| *p = q + a * *p);
    }
}

/// The covariant jet as one section of `End(TM)`.
///
/// Minimises
///
/// ```text
/// E(A) = sum_ij w_ij |r_ij - A_i Delta_ij|^2 + lam sum_ij |A_i - O_ij A_j O_ij^T|_F^2
/// ```
///
/// over all points at once, with `lam = alpha * median |Delta|^2`. The second
/// term is the Dirichlet energy of the connection Laplacian on `End(TM)`. The
/// minimiser solves `A_i C_i + lam (L A)_i = B_i`, `C_i = sum_j w Delta Delta^T`,
/// `B_i = sum_j w r Delta^T`, by preconditioned conjugate gradients. `weights`
/// has one entry per directed slot; slots with weight zero are held out of the
/// data term and scored.
pub fn fit_jet(
    c: &Connection,
    x: &[f64],
    v: &[f64],
    weights: Option<&[f64]>,
    opts: &JetOptions,
) -> JetFit {
    let (d, dd) = (c.d, c.d * c.d);
    let ne = c.directed_edges();
    let e = edge_data(c, x, v);
    let w: Vec<f64> = match weights {
        Some(w) => {
            assert_eq!(w.len(), ne, "one weight per directed edge");
            w.to_vec()
        }
        None => vec![1.0; ne],
    };
    let mut sq: Vec<f64> = (0..ne)
        .map(|s| (0..d).map(|a| e.delta[s * d + a].powi(2)).sum())
        .collect();
    let mid = sq.len() / 2;
    let scale = *sq
        .select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap())
        .1;
    let lam = opts.alpha * scale;
    let ridge = opts.ridge * scale.max(1e-30);
    let mut cm = vec![0.0; c.n * dd];
    let mut bm = vec![0.0; c.n * dd];
    cm.par_chunks_mut(dd)
        .zip(bm.par_chunks_mut(dd))
        .enumerate()
        .for_each(|(i, (ci, bi))| {
            for s in c.start[i]..c.start[i + 1] {
                let dl = &e.delta[s * d..(s + 1) * d];
                let rr = &e.r[s * d..(s + 1) * d];
                for a in 0..d {
                    for b in 0..d {
                        ci[a * d + b] += w[s] * dl[a] * dl[b];
                        bi[a * d + b] += w[s] * rr[a] * dl[b];
                    }
                }
            }
        });
    let pm: Vec<f64> = (0..c.n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let mut m = DMatrix::from_row_slice(d, d, &cm[i * dd..(i + 1) * dd]);
            let deg = c.graph.adj[i].len() as f64;
            for a in 0..d {
                m[(a, a)] += lam * deg + ridge;
            }
            let inv = m.try_inverse().expect("singular cell block");
            (0..dd).map(move |k| inv[(k / d, k % d)])
        })
        .collect();
    let op = Normal {
        c,
        cm,
        pm,
        lam,
        ridge,
    };
    let mut a = op.zeros();
    op.precondition(&bm, &mut a);
    let report = if lam > 0.0 {
        pcg(&op, &bm, &mut a, opts.tol, opts.max_iter)
    } else {
        CgReport {
            iterations: 0,
            residual: 0.0,
            converged: true,
        }
    };
    let (mut nh, mut dh, mut nt, mut dt) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..c.n {
        let ai = &a[i * dd..(i + 1) * dd];
        for s in c.start[i]..c.start[i + 1] {
            let dl = &e.delta[s * d..(s + 1) * d];
            let rr = &e.r[s * d..(s + 1) * d];
            let (mut e2, mut r2) = (0.0, 0.0);
            for p in 0..d {
                let pred: f64 = (0..d).map(|k| ai[p * d + k] * dl[k]).sum();
                e2 += (rr[p] - pred).powi(2);
                r2 += rr[p].powi(2);
            }
            if w[s] == 0.0 {
                nh += e2;
                dh += r2;
            } else {
                nt += e2;
                dt += r2;
            }
        }
    }
    let mut ts = op.zeros();
    transported_sum(c, &a, &mut ts);
    // sum_j |A_i - O A_j O^T|^2 = deg |A_i|^2 - 2 <A_i, sum_j O A_j O^T> + sum_j |A_j|^2.
    let norm2: Vec<f64> = (0..c.n)
        .map(|i| a[i * dd..(i + 1) * dd].iter().map(|v| v * v).sum())
        .collect();
    let dir: f64 = (0..c.n)
        .map(|i| {
            let deg = c.graph.adj[i].len() as f64;
            let cross: f64 = (0..dd).map(|k| a[i * dd + k] * ts[i * dd + k]).sum();
            let nb: f64 = c.graph.adj[i].iter().map(|&j| norm2[j]).sum();
            deg * norm2[i] - 2.0 * cross + nb
        })
        .sum::<f64>()
        / ne as f64;
    let mean_norm = norm2.iter().sum::<f64>() / c.n as f64;
    let ratio = |p: f64, q: f64| if q > 0.0 { p / q } else { f64::NAN };
    JetFit {
        a,
        report,
        heldout: ratio(nh, dh),
        train: ratio(nt, dt),
        dirichlet: ratio(dir, mean_norm),
        scale,
    }
}
