//! cartan-io: VTK XML writers for cartan-dec meshes and DEC fields.
pub mod xml;
pub mod vtp;
pub mod director;
pub mod sharp;
pub mod pvd;
pub mod vtu;
pub mod diagnostics;
pub mod blender;
pub mod feec_fields;
pub mod run;
pub use vtp::{write_vtp, Field};
pub use director::director_field_flat;
pub use sharp::sharp_1form_to_vertex_vectors;
pub use pvd::write_pvd;
