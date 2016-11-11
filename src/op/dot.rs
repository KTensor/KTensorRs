use std::string::{String};
use std::sync::{Arc};
use std::ops::{Add, Mul};
use math::{Vec2};
use node::{Node, Graph};
use tensor::{Tensor};

fn operation<T>(vec: Vec<Tensor<T>>) -> Tensor<T> where T: Mul<Output=T> + Add<Output=T> + Copy {
    &vec[0] * &vec[1]
}

fn operation_prime<T>(gradient: &Tensor<T>, vec: Vec<&Tensor<T>>) -> Vec<Tensor<T>> where T: Mul<Output=T> + Add<Output=T> + Copy {
    vec![gradient * &vec[1].transpose(), &vec[0].transpose() * gradient]
}

fn calc_dim(vec: Vec<Vec2>) -> Vec2 {
    let Vec2(x1, y1) = vec[0];
    let Vec2(x2, y2) = vec[1];
    assert_eq!(y1, x2);
    Vec2(x1, y2)
}

pub fn dot<T>(node_id: String, a: Arc<Graph<T>>, b: Arc<Graph<T>>) -> Node<T> where T: Mul<Output=T> + Add<Output=T> + Copy {
    Node::new(node_id, operation, operation_prime, vec![a, b], calc_dim)
}
