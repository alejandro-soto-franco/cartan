// ~/cartan/cartan-remesh/tests/scaffold.rs

use cartan_remesh::{RemeshConfig, RemeshError, RemeshLog, EdgeSplit};

#[test]
fn remesh_log_default_is_empty() {
    let log = RemeshLog::new();
    assert_eq!(log.total_mutations(), 0);
    assert!(!log.topology_changed());
}

#[test]
fn remesh_config_default_is_sane() {
    let config = RemeshConfig::default();
    assert!(config.curvature_scale > 0.0);
    assert!(config.min_edge_length > 0.0);
    assert!(config.max_edge_length > config.min_edge_length);
    assert!(config.foldover_threshold > 0.0);
}

#[test]
fn remesh_error_display() {
    let err = RemeshError::Foldover {
        face: 42,
        angle_rad: 0.8,
        threshold: 0.5,
    };
    let msg = format!("{err}");
    assert!(msg.contains("foldover"));
    assert!(msg.contains("42"));
}

#[test]
fn remesh_log_merge() {
    let mut log_a = RemeshLog::new();
    log_a.splits.push(EdgeSplit {
        old_edge: 0,
        v_a: 0,
        v_b: 1,
        new_vertex: 5,
        new_edges: vec![6, 7, 8],
    });

    let mut log_b = RemeshLog::new();
    log_b.splits.push(EdgeSplit {
        old_edge: 3,
        v_a: 2,
        v_b: 3,
        new_vertex: 6,
        new_edges: vec![9, 10, 11],
    });

    log_a.merge(log_b);
    assert_eq!(log_a.splits.len(), 2);
    assert_eq!(log_a.total_mutations(), 2);
    assert!(log_a.topology_changed());
}
