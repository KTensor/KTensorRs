use std::string::{String};
use std::sync::{Arc};
use math::{Vec2};
use node::{Node, Graph};
use tensor::{Tensor};

fn operation_f64(vec: Vec<Tensor<f64>>) -> Tensor<f64> {
    vec[0].clone() // just returns softmax
}

fn operation_prime_f64(_: &Tensor<f64>, vec: Vec<&Tensor<f64>>) -> Vec<Tensor<f64>> {
    vec![vec[0] + &(vec[1] * &-1.0)]
}

fn calc_dim(dims: Vec<Vec2>) -> Vec2 {
    let Vec2(x1, y1) = dims[0];
    let Vec2(x2, y2) = dims[1];
    assert_eq!(x1, x2);
    assert_eq!(y1, y2);
    dims[0]
}

pub fn softmax_cross_entropy_f64(node_id: String, s: Arc<Graph<f64>>, y: Arc<Graph<f64>>) -> Node<f64> {
    Node::new(node_id, operation_f64, operation_prime_f64, vec![s, y], calc_dim)
}
