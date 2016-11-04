use context::{Context};
use tensor::{Tensor};

pub trait Graph<T> where T: Copy {
    fn get_id(&self) -> &'static str;
    fn run(&self, state: &Context<T>, variable: &Context<T>) -> Tensor<T>;
    fn train(&self, state: &Context<T>, variable: &Context<T>, history: &mut Context<T>) -> Tensor<T> {
        let tensor = self.forward_pass(state, variable, history);
        let tensor_clone = tensor.clone();
        history.set(self.get_id(), tensor);
        tensor_clone
    }
    fn forward_pass(&self, state: &Context<T>, variable: &Context<T>, history: &mut Context<T>) -> Tensor<T>;
    fn backward_pass(&self, state: &mut Context<T>, variable: &Context<T>, history: &Context<T>, gradient: &Tensor<T>, learning_rate: &T);
}
