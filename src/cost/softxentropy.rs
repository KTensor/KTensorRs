use std::string::{String};
use std::sync::{Arc};
use math::{Vec2};
use node::{Node, Graph};
use tensor::{Tensor};

fn operation_f64(vec: Vec<Tensor<f64>>) -> Tensor<f64> {
    // y ln a + (1 - y) ln (1 - a)
    Tensor::from_vec(Vec2(1, 1), vec![vec[0].buffer().iter().zip(vec[1].buffer().iter()).fold(0.0, |sum, (a, y)| {
        sum - (y * a.ln() + (1.0 - y) * (1.0 - a).ln())
    }) / vec[0].buffer().len() as f64])
}

fn operation_f32(vec: Vec<Tensor<f32>>) -> Tensor<f32> {
    Tensor::from_vec(Vec2(1, 1), vec![vec[0].buffer().iter().zip(vec[1].buffer().iter()).fold(0.0, |sum, (a, y)| {
        sum - (y * a.ln() + (1.0 - y) * (1.0 - a).ln())
    }) / vec[0].buffer().len() as f32])
}

fn operation_prime_f64(_: &Tensor<f64>, vec: Vec<&Tensor<f64>>) -> Vec<Tensor<f64>> {
    vec![vec[0] + &(vec[1] * &-1.0)]
}

fn operation_prime_f32(_: &Tensor<f32>, vec: Vec<&Tensor<f32>>) -> Vec<Tensor<f32>> {
    vec![vec[0] + &(vec[1] * &-1.0)]
}

fn calc_dim(dims: Vec<Vec2>) -> Vec2 {
    let Vec2(x1, y1) = dims[0];
    let Vec2(x2, y2) = dims[1];
    assert_eq!(x1, x2);
    assert_eq!(y1, y2);
    Vec2(1, 1)
}

pub fn softmax_cross_entropy_f64(node_id: String, s: Arc<Graph<f64>>, y: Arc<Graph<f64>>) -> Node<f64> {
    Node::new(node_id, operation_f64, operation_prime_f64, vec![s, y], calc_dim)
}

pub fn softmax_cross_entropy_f32(node_id: String, s: Arc<Graph<f32>>, y: Arc<Graph<f32>>) -> Node<f32> {
    Node::new(node_id, operation_f32, operation_prime_f32, vec![s, y], calc_dim)
}
