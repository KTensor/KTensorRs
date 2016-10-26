use std::ops::{Fn};
use context::{Context};
use tensor::{Tensor};
use node::{Graph};

pub struct Node<T> {
    id: &'static str,
    op: Box<Fn(Tensor<T>, Tensor<T>) -> Tensor<T>>,
    param: (Tensor<T>, Tensor<T>),
}

impl <T> Node<T> {
    pub fn new(&self, node_id: &'static str, operation: Box<Fn(Tensor<T>, Tensor<T>) -> Tensor<T>>, parameter: (Tensor<T>, Tensor<T>)) -> Node<T> {
        Node {
            id: node_id,
            op: operation,
            param: parameter,
        }
    }
}

impl <T> Graph<T> for Node<T> {
    fn get_id(&self) -> &'static str {
        self.id
    }

    fn run(&self, state: &Context<T>, context: &Context<T>) -> Tensor<T> {
        
    }

    fn train(&self, state: &Context<T>, context: &Context<T>) {

    }
}
