use context::{Context};
use tensor::{Tensor};

pub trait Graph<T> {
    fn get_id(&self) -> &'static str;
    fn run(&self, state: &Context<T>, variable: &Context<T>) -> Tensor<T>;
    fn train(&self, state: &Context<T>, variable: &Context<T>, delta_next: &Tensor<T>);
}
