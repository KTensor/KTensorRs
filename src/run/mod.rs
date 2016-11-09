use node::{Graph};
use context::{Context};
use tensor::{Tensor};
use math::{Vec2};

pub fn execute<T>(node: &Graph<T>, state: &Context<T>, variables: &Context<T>) -> Tensor<T> where T: Copy {
    node.run(state, variables)
}

pub fn train<T>(node: &Graph<T>, state: &mut Context<T>, variables: &Context<T>, history: &mut Context<T>, rate: &T) where T: Copy {
    node.train(state, variables, history);
    node.backward_pass(state, variables, history, &Tensor::from_vec(Vec2(1, 1), vec![*rate]), rate);
}
