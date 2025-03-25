pub mod clustering;
pub mod constructions;
pub mod data_handling;
pub mod distance;
pub mod graph;
pub mod util;

// Python bindings are only compiled when the "python" feature is enabled
#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "python")]
pub use python::scratch;

