use std::ops::{Fn};
use context::{Context};
use tensor::{Tensor};
use node::{Graph};

pub struct Node<T> {
    id: &'static str,
    op: Box<Fn(Tensor<T>, Tensor<T>) -> Tensor<T>>,
    backprop: Box<Fn(Tensor<T>, Tensor<T>) -> Tensor<T>>,
    param: Box<(Graph<T>, Graph<T>)>,
}

impl <T> Node<T> {
    pub fn new(&self, node_id: &'static str, operation: Box<Fn(Tensor<T>, Tensor<T>) -> Tensor<T>>, back_propagation: Box<Fn(Tensor<T>, Tensor<T>) -> Tensor<T>>, parameter: Box<(Graph<T>, Graph<T>)>) -> Node<T> {
        Node {
            id: node_id,
            op: operation,
            backprop: back_propagation,
            param: parameter,
        }
    }
}

impl <T> Graph<T> for Node<T> {
    fn get_id(&self) -> &'static str {
        self.id
    }

    fn run(&self, state: &Context<T>, context: &Context<T>) -> Tensor<T> {
        (self.op)(self.param.0.run(state, context), self.param.1.run(state, context))
    }

    fn train(&self, state: &Context<T>, context: &Context<T>, history: &mut Context<T>, deltas: &mut Context<T>) {

    }
}
