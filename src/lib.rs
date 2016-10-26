//! `ktensor` module
//!
//! # Examples
//!
//! ```
//! assert_eq!(ktensor::hello_world(), "Hello World!");
//! ```

pub mod math;
pub mod tensor;
pub mod context;
pub mod node;

pub use context::{Context};
pub use tensor::{Tensor};
pub use node::{Node, Runnable};

/// should return the static string `"Hello World!"`
///
/// # Examples
///
/// ```
/// assert_eq!(ktensor::hello_world(), "Hello World!");
/// ```
pub fn hello_world() -> &'static str {
    "Hello World!"
}
