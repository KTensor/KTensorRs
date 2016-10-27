use context::{Context};
use tensor::{Tensor};

pub trait Graph<T> {
    fn get_id(&self) -> &'static str;
    fn run(&self, state: &Context<T>, variable: &Context<T>) -> Tensor<T>;
    fn train(&self, state: &Context<T>, variable: &Context<T>, history: &mut Context<T>);
    fn backward_pass(&self, gradient: &Tensor<T>, learning_rate: &f64);
}
