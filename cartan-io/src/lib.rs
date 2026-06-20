//! cartan-io: VTK XML writers for cartan-dec meshes and DEC fields.
pub mod xml;
pub mod vtp;
pub use vtp::{write_vtp, Field};
