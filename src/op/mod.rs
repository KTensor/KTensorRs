mod dot;
mod add;
mod softmax;
mod relu;
mod sigmoid;

pub use self::dot::{dot};
pub use self::add::{add_f64};
pub use self::softmax::{softmax_f64};
pub use self::relu::{relu_f64};
pub use self::sigmoid::{sigmoid_f64};
