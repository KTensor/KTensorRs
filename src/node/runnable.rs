use context::{Context};
use tensor::{Tensor};

pub trait Runnable<T> {
    fn run(&self, state: Context<T>, context: Context<T>) -> Tensor<T>;
    fn train(&self, state: Context<T>, context: Context<T>);
}
