use std::string::{String};
use std::sync::{Arc};
use math::{Vec2};
use node::{Node, Graph};
use tensor::{Tensor};

fn operation_f64(vec: Vec<Tensor<f64>>) -> Tensor<f64> {
    Tensor::from_vec(vec[0].dim(), vec[0].buffer().iter().map(|&a| {
        if a > 16.0 {
            a
        } else if a < -16.0 {
            a / 16.0
        } else {
            a / 16.0 + 0.9375 * (1.0 + a.exp()).ln()
        }
    }).collect())
}

fn operation_f32(vec: Vec<Tensor<f32>>) -> Tensor<f32> {
    Tensor::from_vec(vec[0].dim(), vec[0].buffer().iter().map(|&a| {
        if a > 16.0 {
            a
        } else if a < -16.0 {
            a / 16.0
        } else {
            a / 16.0 + 0.9375 * (1.0 + a.exp()).ln()
        }
    }).collect())
}

fn operation_prime_f64(gradient: &Tensor<f64>, vec: Vec<&Tensor<f64>>) -> Vec<Tensor<f64>> {
    let threshold = vec[1].get(Vec2(0, 0));
    vec![Tensor::from_vec(gradient.dim(), vec[0].buffer().iter().zip(gradient.buffer().iter()).map(|(&a, &g)| {
        if a > threshold {
            threshold
        } else if a > 16.0 {
            g
        } else if a < -16.0 {
            g / 16.0
        } else {
            g * (0.0625 + 0.9375 * (1.0 + (-a).exp()).recip())
        }
    }).collect())]
}

fn operation_prime_f32(gradient: &Tensor<f32>, vec: Vec<&Tensor<f32>>) -> Vec<Tensor<f32>> {
    let threshold = vec[1].get(Vec2(0, 0));
    vec![Tensor::from_vec(gradient.dim(), vec[0].buffer().iter().zip(gradient.buffer().iter()).map(|(&a, &g)| {
        if a > threshold {
            threshold
        } else if a > 16.0 {
            g
        } else if a < -16.0 {
            g / 16.0
        } else {
            g * (0.0625 + 0.9375 * (1.0 + (-a).exp()).recip())
        }
    }).collect())]
}

fn calc_dim(dims: Vec<Vec2>) -> Vec2 {
    assert_eq!(dims[1].0, 1);
    assert_eq!(dims[1].1, 1);
    dims[0]
}

pub fn relu_f64(node_id: String, z: Arc<Graph<f64>>, threshold: Arc<Graph<f64>>) -> Node<f64> {
    Node::new(node_id, operation_f64, operation_prime_f64, vec![z, threshold], calc_dim)
}

pub fn relu_f32(node_id: String, z: Arc<Graph<f32>>, threshold: Arc<Graph<f32>>) -> Node<f32> {
    Node::new(node_id, operation_f32, operation_prime_f32, vec![z, threshold], calc_dim)
}
