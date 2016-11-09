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
pub use node::{Node, State, Variable};
pub use run::{execute, train};

pub mod state {
    pub use node::{init_state_f64 as init_f64};
    pub use node::{init_state_f32 as init_f32};
}

pub mod variable {
    pub use node::{init_variables_f64 as init_f64};
    pub use node::{init_variables_f32 as init_f32};
}
