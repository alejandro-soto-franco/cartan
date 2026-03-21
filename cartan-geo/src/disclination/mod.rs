//! Disclination line detection and tracking for 3D uniaxial nematics.
//! Charge classification: Z2 (half-integer, ±1/2). Enum designed for Q8 extension.

pub mod segments;
pub mod lines;
pub mod events;

pub use segments::{DisclinationCharge, DisclinationSegment, Sign, QTensorField3D, scan_disclination_lines_3d};
pub use lines::{DisclinationLine, connect_disclination_lines};
pub use events::{DisclinationEvent, EventKind, track_disclination_events};
