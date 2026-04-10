//! Disclination line detection and tracking for 3D uniaxial nematics.
//! Charge classification: Z2 (half-integer, ±1/2). Enum designed for Q8 extension.

pub mod events;
pub mod lines;
pub mod segments;

pub use events::{DisclinationEvent, EventKind, track_disclination_events};
pub use lines::{DisclinationLine, connect_disclination_lines};
pub use segments::{
    DisclinationCharge, DisclinationSegment, QTensorField3D, Sign, scan_disclination_lines_3d,
};
