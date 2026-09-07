//! Freidlin-Wentzell action, its minimiser, and Conley decomposition.
//!
//! For `dx = b(x) dt + sqrt(2 eps) dW` the action of a path is
//! `S[phi] = (1/2) integral |phi' - b(phi)|^2 dt`, and the quasipotential
//! between two states is its infimum over joining paths.
//!
//! Two cases have exact answers and both are tested here. Under a gradient
//! drift `b = -grad U` the minimising path uphill is the time reversal of the
//! downhill one, with `V = 2 (U_1 - U_0)`. Under a linear drift `b = A x` with
//! `A` stable the answer is `x^T Sigma^-1 x`, with `Sigma` from
//! `A Sigma + Sigma A^T + 2 I = 0`. The second is the one that matters: `A`
//! need not be symmetric, so a gradient-only implementation fails it.

use std::collections::HashMap;

/// A drift field on `R^n`.
pub type Drift<'a> = &'a dyn Fn(&[f64]) -> Vec<f64>;

/// The geometric action, `integral (|b| |dphi| - b . dphi)`.
///
/// The Freidlin-Wentzell infimum runs over the transit time as well as the
/// path, and for a gradient drift the optimal time is unbounded. A functional
/// with the time fixed therefore overestimates: on the quartic double well it
/// returns 0.79 where the answer is 0.5. The geometric form removes the time
/// by being invariant under reparametrisation, and it is what a geometric
/// minimum action method minimises.
///
/// For `b = -grad U` this gives `2 (U_1 - U_0)` along the steepest path, since
/// `|b| |dphi|` and `-b . dphi` each contribute `dU`.
#[must_use]
pub fn geometric_action(path: &[Vec<f64>], drift: Drift) -> f64 {
    let mut s = 0.0;
    for w in path.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        let mid: Vec<f64> = a.iter().zip(b).map(|(x, y)| 0.5 * (x + y)).collect();
        let d = drift(&mid);
        let step: Vec<f64> = a.iter().zip(b).map(|(x, y)| y - x).collect();
        let dn = d.iter().map(|v| v * v).sum::<f64>().sqrt();
        let sn = step.iter().map(|v| v * v).sum::<f64>().sqrt();
        let dot: f64 = d.iter().zip(&step).map(|(x, y)| x * y).sum();
        s += dn * sn - dot;
    }
    s
}

/// Discrete Freidlin-Wentzell action of a path at a fixed transit time.
///
/// Kept for the check that a path following the drift costs nothing. The
/// quasipotential itself uses [`geometric_action`], which does not fix the
/// time.
#[must_use]
pub fn action(path: &[Vec<f64>], drift: Drift, dt: f64) -> f64 {
    let mut s = 0.0;
    for w in path.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        let mid: Vec<f64> = a.iter().zip(b).map(|(x, y)| 0.5 * (x + y)).collect();
        let d = drift(&mid);
        let mut acc = 0.0;
        for k in 0..a.len() {
            let v = (b[k] - a[k]) / dt - d[k];
            acc += v * v;
        }
        s += 0.5 * acc * dt;
    }
    s
}

/// Relax the interior of a path to lower the geometric action, with fixed
/// endpoints.
///
/// Gradient descent with an adaptive step, the gradient taken by central
/// differences, and an arclength reparametrisation each sweep so points do not
/// bunch at the ends. That reparametrisation is what makes the geometric
/// functional usable: without it the descent stalls with most of the path
/// crowded near an endpoint.
#[must_use]
pub fn minimum_action_path(
    x0: &[f64],
    x1: &[f64],
    drift: Drift,
    n_points: usize,
    iters: usize,
    step: f64,
) -> (Vec<Vec<f64>>, f64) {
    let dim = x0.len();
    let n = n_points.max(3);
    let mut path: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            (0..dim).map(|k| x0[k] + t * (x1[k] - x0[k])).collect()
        })
        .collect();

    let h = 1e-7;
    let mut best = geometric_action(&path, drift);
    let mut lr = step;
    for it in 0..iters {
        let mut grad = vec![vec![0.0; dim]; n];
        for i in 1..n - 1 {
            for k in 0..dim {
                let orig = path[i][k];
                path[i][k] = orig + h;
                let up = geometric_action(&path, drift);
                path[i][k] = orig - h;
                let dn = geometric_action(&path, drift);
                path[i][k] = orig;
                grad[i][k] = (up - dn) / (2.0 * h);
            }
        }
        let mut trial = path.clone();
        for i in 1..n - 1 {
            for k in 0..dim {
                trial[i][k] -= lr * grad[i][k];
            }
        }
        if it % 25 == 24 {
            trial = reparametrise(&trial);
        }
        let s = geometric_action(&trial, drift);
        if s < best {
            best = s;
            path = trial;
            lr *= 1.15;
        } else {
            lr *= 0.5;
            if lr < 1e-16 {
                break;
            }
        }
    }
    (path, best)
}

/// Redistribute the interior points evenly by arclength, endpoints fixed.
fn reparametrise(path: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = path.len();
    let dim = path[0].len();
    let mut cum = vec![0.0; n];
    for i in 1..n {
        let d: f64 = (0..dim)
            .map(|k| (path[i][k] - path[i - 1][k]).powi(2))
            .sum::<f64>()
            .sqrt();
        cum[i] = cum[i - 1] + d;
    }
    let total = cum[n - 1];
    if total < 1e-300 {
        return path.to_vec();
    }
    let mut out = Vec::with_capacity(n);
    out.push(path[0].clone());
    for i in 1..n - 1 {
        let target = total * i as f64 / (n - 1) as f64;
        let j = (1..n).find(|&j| cum[j] >= target).unwrap_or(n - 1);
        let seg = (cum[j] - cum[j - 1]).max(1e-300);
        let t = (target - cum[j - 1]) / seg;
        out.push(
            (0..dim)
                .map(|k| path[j - 1][k] + t * (path[j][k] - path[j - 1][k]))
                .collect(),
        );
    }
    out.push(path[n - 1].clone());
    out
}

/// A Morse decomposition of a flow sampled on a grid.
#[derive(Clone, Debug)]
pub struct MorseDecomposition {
    /// Strongly connected components, each a list of cell indices.
    pub components: Vec<Vec<usize>>,
    /// Components with no outgoing edge to another component.
    pub attractors: Vec<usize>,
}

impl MorseDecomposition {
    /// Build from a directed graph on cells.
    ///
    /// The recurrent sets of a flow are the strongly connected components of
    /// the graph its drift induces, and the attractors are the components
    /// with no edge leaving them.
    #[must_use]
    pub fn of_graph(n: usize, edges: &[(usize, usize)]) -> Self {
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut radj: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(a, b) in edges {
            adj.entry(a).or_default().push(b);
            radj.entry(b).or_default().push(a);
        }
        // Kosaraju: order by finish time, then explore the reverse graph.
        let mut seen = vec![false; n];
        let mut order = Vec::with_capacity(n);
        for s in 0..n {
            if seen[s] {
                continue;
            }
            let mut stack = vec![(s, 0usize)];
            seen[s] = true;
            while let Some((v, i)) = stack.pop() {
                let nbrs = adj.get(&v).map(Vec::as_slice).unwrap_or(&[]);
                if i < nbrs.len() {
                    stack.push((v, i + 1));
                    let w = nbrs[i];
                    if !seen[w] {
                        seen[w] = true;
                        stack.push((w, 0));
                    }
                } else {
                    order.push(v);
                }
            }
        }
        let mut comp = vec![usize::MAX; n];
        let mut components: Vec<Vec<usize>> = Vec::new();
        for &s in order.iter().rev() {
            if comp[s] != usize::MAX {
                continue;
            }
            let id = components.len();
            let mut members = Vec::new();
            let mut stack = vec![s];
            comp[s] = id;
            while let Some(v) = stack.pop() {
                members.push(v);
                for &w in radj.get(&v).map(Vec::as_slice).unwrap_or(&[]) {
                    if comp[w] == usize::MAX {
                        comp[w] = id;
                        stack.push(w);
                    }
                }
            }
            components.push(members);
        }
        let mut leaves = vec![true; components.len()];
        for &(a, b) in edges {
            if comp[a] != comp[b] {
                leaves[comp[a]] = false;
            }
        }
        let attractors = (0..components.len()).filter(|&i| leaves[i]).collect();
        Self {
            components,
            attractors,
        }
    }

    /// Build from a one-dimensional drift sampled on a uniform grid.
    ///
    /// A cell points to its neighbour in the direction the drift moves, and to
    /// itself when the drift within it changes sign.
    #[must_use]
    pub fn of_line(xs: &[f64], drift: Drift) -> Self {
        let n = xs.len();
        let mut edges = Vec::new();
        for (i, &x) in xs.iter().enumerate() {
            let d = drift(&[x])[0];
            // A cell where the drift vanishes is a fixed point of unknown
            // stability, so it points both ways: a stable one then forms a
            // cycle with the neighbours that point back at it, and an unstable
            // one forms none. Giving it a self-loop instead would make every
            // saddle its own attractor, which is what the double-well fixture
            // caught.
            const ZERO: f64 = 1e-12;
            if d > ZERO && i + 1 < n {
                edges.push((i, i + 1));
            } else if d < -ZERO && i > 0 {
                edges.push((i, i - 1));
            } else if d.abs() <= ZERO {
                if i + 1 < n {
                    edges.push((i, i + 1));
                }
                if i > 0 {
                    edges.push((i, i - 1));
                }
            }
        }
        // A cell whose neighbours point back at it is recurrent with them.
        Self::of_graph(n, &edges)
    }
}
