// ~/cartan/cartan-remesh/src/log.rs

//! Remesh operation log.
//!
//! Every topology-changing or vertex-moving operation records its mutations
//! in a [`RemeshLog`]. Downstream solvers (volterra-dec) use this log to
//! interpolate Q-tensor, velocity, and scalar fields across remesh events.

/// A complete record of all remesh mutations applied in one pass.
#[derive(Debug, Clone, Default)]
pub struct RemeshLog {
    /// Edge splits performed (vertex insertions).
    pub splits: Vec<EdgeSplit>,
    /// Edge collapses performed (vertex removals).
    pub collapses: Vec<EdgeCollapse>,
    /// Edge flips performed (diagonal swaps).
    pub flips: Vec<EdgeFlip>,
    /// Vertex shifts performed (tangential smoothing moves).
    pub shifts: Vec<VertexShift>,
}

impl RemeshLog {
    /// Create an empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Total number of mutations recorded.
    pub fn total_mutations(&self) -> usize {
        self.splits.len() + self.collapses.len() + self.flips.len() + self.shifts.len()
    }

    /// Whether any topology-changing operation occurred (split, collapse, or flip).
    pub fn topology_changed(&self) -> bool {
        !self.splits.is_empty() || !self.collapses.is_empty() || !self.flips.is_empty()
    }

    /// Merge another log into this one (appends all entries).
    pub fn merge(&mut self, other: RemeshLog) {
        self.splits.extend(other.splits);
        self.collapses.extend(other.collapses);
        self.flips.extend(other.flips);
        self.shifts.extend(other.shifts);
    }
}

/// Record of a single edge split operation.
#[derive(Debug, Clone)]
pub struct EdgeSplit {
    pub old_edge: usize,
    pub v_a: usize,
    pub v_b: usize,
    pub new_vertex: usize,
    pub new_edges: Vec<usize>,
}

/// Record of a single edge collapse operation.
#[derive(Debug, Clone)]
pub struct EdgeCollapse {
    pub old_edge: usize,
    pub surviving_vertex: usize,
    pub removed_vertex: usize,
    pub removed_faces: Vec<usize>,
}

/// Record of a single edge flip operation.
#[derive(Debug, Clone)]
pub struct EdgeFlip {
    pub old_edge: usize,
    pub new_edge: [usize; 2],
    pub affected_faces: [usize; 2],
}

/// Record of a single vertex shift (tangential smoothing move).
#[derive(Debug, Clone)]
pub struct VertexShift {
    pub vertex: usize,
    pub old_pos_tangent: Vec<f64>,
}
