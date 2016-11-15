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
pub mod run;

pub use math::{Vec2};
pub use tensor::{Tensor};
pub use context::{Context};
pub use node::{Graph, Node, State, Variable};
pub use run::{execute, train};

pub use std::sync::{Arc};
