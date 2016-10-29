use node::{Node, Graph};
use tensor::{Tensor};

const operation: Box<Fn(Vec<Tensor<T>>) -> Tensor<T>> = Box::new(|vec| vec[0] * vec[1]);
const operation_train: Box<Fn(Vec<&Tensor<T>>) -> Tensor<T>> = Box::new(|vec| vec[0] * vec[1]);
const operation_prime: Box<Fn(&Tensor<T>, Vec<&Tensor<T>>) -> Vec<Tensor<T>>> = Box::new(|gradient, vec| vec![gradient * vec[1].T, vec[0].T * gradient]);

pub fn dot<T>(node_id: &'static str, A: &Graph<T>, B: &Graph<T>) -> Node<T> {
    Node::new(node_id, operation, operation_train, operation_prime, vec![A, B])
}
