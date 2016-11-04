use std::ops::{Add, Mul};
use node::{Node, Graph};
use tensor::{Tensor};

fn operation<T>(vec: Vec<Tensor<T>>) -> Tensor<T> where T: Mul<Output=T> + Add<Output=T> + Copy {
    &vec[0] + &vec[1]
}

fn operation_prime<T>(gradient: &Tensor<T>, _: Vec<&Tensor<T>>) -> Vec<Tensor<T>> where T: Mul<Output=T> + Add<Output=T> + Copy {
    vec![gradient.clone(), gradient.clone()]
}

pub fn add<T>(node_id: &'static str, a: Box<Graph<T>>, b: Box<Graph<T>>) -> Node<T> where T: Mul<Output=T> + Add<Output=T> + Copy {
    Node::new(node_id, operation, operation_prime, vec![a, b])
}
