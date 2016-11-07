mod dot;
mod add;
mod softmax;
mod relu;

pub use self::dot::{dot};
pub use self::add::{add};
pub use self::softmax::{softmax_f64};
pub use self::relu::{relu_f64};
