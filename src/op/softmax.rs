use std::ops::{Add, Mul};
use math::{Vec2};
use node::{Node, Graph};
use tensor::{Tensor};

fn operation<T>(vec: Vec<Tensor<T>>) -> Tensor<T> where T: Mul<Output=T> + Add<Output=T> + Copy {
}

fn operation_prime<T>(gradient: &Tensor<T>, vec: Vec<&Tensor<T>>) -> Vec<Tensor<T>> where T: Mul<Output=T> + Add<Output=T> + Copy {
}

fn calc_dim(dims: Vec<Vec2>) -> Vec2 {
    let Vec2(x1, y1) = dims[0];
    let Vec2(x2, y2) = dims[1];
    assert_eq!(x1, x2);
    assert_eq!(y1, y2);
    dims[0]
}

pub fn softmax<T>(node_id: &'static str, a: Box<Graph<T>>, b: Box<Graph<T>>) -> Node<T> where T: Mul<Output=T> + Add<Output=T> + Copy {
    Node::new(node_id, operation, operation_prime, vec![a, b], calc_dim)
}
