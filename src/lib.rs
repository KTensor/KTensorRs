//! `ktensor` module
//!
//! # Modules
//!
//! - `math`
//! - `tensor`
//! - `context`
//! - `node`
//! - `op`
//!
//! # Structs
//!
//! - `Context`
//! - `Tensor`
//! - `Node`
//! - `State`
//! - `Variable`
pub mod math;
pub mod tensor;
pub mod context;
pub mod node;
pub mod op;
pub mod cost;

pub use math::{Vec2};
pub use tensor::{Tensor};
pub use context::{Context};
pub use node::{Node, State, Variable, init_state_f64, init_state_f32};
