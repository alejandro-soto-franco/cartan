// ~/cartan/cartan-geo/src/disclination/events.rs

//! Disclination event detection: creation, annihilation, and reconnection.
//!
//! ## Algorithm
//!
//! Given two sets of disclination lines at consecutive frames, match lines
//! between frames using centroid proximity. Lines that appear without a match
//! are "created"; lines that disappear without a match are "annihilated".
//! Two-to-one or one-to-two mappings indicate reconnection events.
//!
//! Centroid matching is performed by minimum distance between line centroids.
//! A line in frame B is "matched" if its nearest neighbor in frame A is within
//! `proximity_threshold` grid units.
//!
//! ## References
//!
//! - Thampi, S. P. et al. (2014). Nat. Commun. 5, 3048.
//! - Doostmohammadi, A. et al. (2016). Nat. Commun. 7, 10557.

use std::collections::HashMap;

use super::lines::DisclinationLine;

/// Classification of a topological event.
#[derive(Debug, Clone, PartialEq)]
pub enum EventKind {
    /// A new disclination line appeared between frames.
    Creation,
    /// A disclination line disappeared between frames.
    Annihilation,
    /// Two lines merged into one, or one split into two.
    Reconnection,
}

/// A detected topological event between two consecutive frames.
#[derive(Debug, Clone)]
pub struct DisclinationEvent {
    /// Type of event.
    pub kind: EventKind,
    /// Frame index at which the event is recorded.
    pub frame: usize,
    /// Approximate position of the event (centroid of involved lines).
    pub position: [f64; 3],
    /// Indices into the relevant frame's line array.
    pub line_ids: Vec<usize>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the centroid (mean vertex position) of a disclination line.
fn centroid(line: &DisclinationLine) -> [f64; 3] {
    if line.vertices.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let n = line.vertices.len() as f64;
    let mut c = [0.0f64; 3];
    for v in &line.vertices {
        c[0] += v[0];
        c[1] += v[1];
        c[2] += v[2];
    }
    [c[0] / n, c[1] / n, c[2] / n]
}

/// Euclidean distance between two 3D points.
fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Detect topological events between two consecutive frames of disclination lines.
///
/// Matches each line in `lines_b` (frame B) to its nearest neighbor in `lines_a`
/// (frame A) by centroid distance. If the nearest distance exceeds
/// `proximity_threshold`, the line is unmatched.
///
/// Events emitted:
/// - `Creation`: line in B has no match in A (distance > threshold).
/// - `Annihilation`: line in A has no match in B.
/// - `Reconnection`: two lines in A match to one in B, or vice versa.
///
/// # Parameters
///
/// - `lines_a`: disclination lines in frame A (previous frame).
/// - `lines_b`: disclination lines in frame B (current frame).
/// - `frame`: frame index attached to all emitted events.
/// - `proximity_threshold`: maximum centroid distance to consider two lines "matched".
pub fn track_disclination_events(
    lines_a: &[DisclinationLine],
    lines_b: &[DisclinationLine],
    frame: usize,
    proximity_threshold: f64,
) -> Vec<DisclinationEvent> {
    let mut events = Vec::new();

    // Centroids of all lines in both frames.
    let centroids_a: Vec<[f64; 3]> = lines_a.iter().map(centroid).collect();
    let centroids_b: Vec<[f64; 3]> = lines_b.iter().map(centroid).collect();

    // For each line in B, find nearest in A.
    // b_match[b_idx] = Some(a_idx) if matched, None if unmatched.
    let mut b_match: Vec<Option<usize>> = vec![None; lines_b.len()];
    for (b_idx, cb) in centroids_b.iter().enumerate() {
        let cb = *cb;
        let best = centroids_a.iter().enumerate().min_by(|(_, ca1), (_, ca2)| {
            dist3(**ca1, cb)
                .partial_cmp(&dist3(**ca2, cb))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some((a_idx, ca)) = best {
            if dist3(*ca, cb) < proximity_threshold {
                b_match[b_idx] = Some(a_idx);
            }
        }
    }

    // For each line in A, find nearest in B.
    let mut a_match: Vec<Option<usize>> = vec![None; lines_a.len()];
    for (a_idx, ca) in centroids_a.iter().enumerate() {
        let ca = *ca;
        let best = centroids_b.iter().enumerate().min_by(|(_, cb1), (_, cb2)| {
            dist3(**cb1, ca)
                .partial_cmp(&dist3(**cb2, ca))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some((b_idx, cb)) = best {
            if dist3(*cb, ca) < proximity_threshold {
                a_match[a_idx] = Some(b_idx);
            }
        }
    }

    // Creation: lines in B with no match in A.
    for (b_idx, m) in b_match.iter().enumerate() {
        if m.is_none() {
            let pos = centroids_b[b_idx];
            events.push(DisclinationEvent {
                kind: EventKind::Creation,
                frame,
                position: pos,
                line_ids: vec![b_idx],
            });
        }
    }

    // Annihilation: lines in A with no match in B.
    for (a_idx, m) in a_match.iter().enumerate() {
        if m.is_none() {
            let pos = centroids_a[a_idx];
            events.push(DisclinationEvent {
                kind: EventKind::Annihilation,
                frame,
                position: pos,
                line_ids: vec![a_idx],
            });
        }
    }

    // Reconnection: detect many-to-one or one-to-many mappings.
    // Count how many B lines map to each A line.
    let mut a_to_b_count: HashMap<usize, Vec<usize>> = HashMap::new();
    for (b_idx, m) in b_match.iter().enumerate() {
        if let Some(a_idx) = m {
            a_to_b_count.entry(*a_idx).or_default().push(b_idx);
        }
    }

    // Count how many A lines map to each B line.
    let mut b_to_a_count: HashMap<usize, Vec<usize>> = HashMap::new();
    for (a_idx, m) in a_match.iter().enumerate() {
        if let Some(b_idx) = m {
            b_to_a_count.entry(*b_idx).or_default().push(a_idx);
        }
    }

    // One A -> many B (split/reconnection)
    for (a_idx, b_ids) in &a_to_b_count {
        if b_ids.len() > 1 {
            let pos = centroids_a[*a_idx];
            events.push(DisclinationEvent {
                kind: EventKind::Reconnection,
                frame,
                position: pos,
                line_ids: b_ids.clone(),
            });
        }
    }

    // Many A -> one B (merge/reconnection)
    for (b_idx, a_ids) in &b_to_a_count {
        if a_ids.len() > 1 {
            let pos = centroids_b[*b_idx];
            events.push(DisclinationEvent {
                kind: EventKind::Reconnection,
                frame,
                position: pos,
                line_ids: a_ids.clone(),
            });
        }
    }

    events
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::segments::{DisclinationCharge, Sign};
    use super::*;

    #[test]
    fn test_no_events_same_frame() {
        // Comparing a set of lines to itself produces no events
        let line = DisclinationLine {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            tangents: vec![[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            curvatures: vec![0.0, 0.0],
            torsions: vec![0.0, 0.0],
            charge: DisclinationCharge::Half(Sign::Positive),
            is_loop: false,
        };
        let events = track_disclination_events(&[line.clone()], &[line], 5, 2.0);
        assert!(events.is_empty());
    }
}
