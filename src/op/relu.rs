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
            a / 64.0
        } else {
            a / 64.0 + 0.984375 * (1.0 + a.exp()).ln()
        }
    }).collect())
}

fn operation_f32(vec: Vec<Tensor<f32>>) -> Tensor<f32> {
    Tensor::from_vec(vec[0].dim(), vec[0].buffer().iter().map(|&a| {
        if a > 16.0 {
            a
        } else if a < -16.0 {
            a / 64.0
        } else {
            a / 64.0 + 0.984375 * (1.0 + a.exp()).ln()
        }
    }).collect())
}

fn operation_prime_f64(gradient: &Tensor<f64>, vec: Vec<&Tensor<f64>>) -> Vec<Tensor<f64>> {
    vec![Tensor::from_vec(gradient.dim(), vec[0].buffer().iter().zip(gradient.buffer().iter()).map(|(&a, &g)| {
        if a > 16.0 {
            g
        } else if a < -16.0 {
            g / 64.0
        } else {
            g * (0.015625 + 0.984375 * (1.0 + (-a).exp()).recip())
        }
    }).collect())]
}

fn operation_prime_f32(gradient: &Tensor<f32>, vec: Vec<&Tensor<f32>>) -> Vec<Tensor<f32>> {
    vec![Tensor::from_vec(gradient.dim(), vec[0].buffer().iter().zip(gradient.buffer().iter()).map(|(&a, &g)| {
        if a > 16.0 {
            g
        } else if a < -16.0 {
            g / 64.0
        } else {
            g * (0.015625 + 0.984375 * (1.0 + (-a).exp()).recip())
        }
    }).collect())]
}

fn calc_dim(dims: Vec<Vec2>) -> Vec2 {
    dims[0]
}

pub fn relu_f64(node_id: String, z: Arc<Graph<f64>>) -> Node<f64> {
    Node::new(node_id, operation_f64, operation_prime_f64, vec![z], calc_dim)
}

pub fn relu_f32(node_id: String, z: Arc<Graph<f32>>) -> Node<f32> {
    Node::new(node_id, operation_f32, operation_prime_f32, vec![z], calc_dim)
}
