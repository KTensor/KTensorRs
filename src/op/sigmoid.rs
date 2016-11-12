use std::string::{String};
use std::sync::{Arc};
use math::{Vec2};
use node::{Node, Graph};
use tensor::{Tensor};

fn operation_f64(vec: Vec<Tensor<f64>>) -> Tensor<f64> {
    Tensor::from_vec(vec[0].dim(), vec[0].buffer().iter().map(|&a| {
        if a >= 0.0 {
            let z = (-a).exp();
            1.0 / (1.0 + z)
        } else {
            let z = a.exp();
            z / (1.0 + z)
        }
    }).collect())
}

fn operation_prime_f64(gradient: &Tensor<f64>, vec: Vec<&Tensor<f64>>) -> Vec<Tensor<f64>> {
    vec![Tensor::from_vec(gradient.dim(), vec[0].buffer().iter().zip(gradient.buffer().iter()).map(|(&a, &g)| {
        let mut z;
        if a >= 0.0 {
            z = (-a).exp();
            z = 1.0 / (1.0 + z);
        } else {
            z = a.exp();
            z = z / (1.0 + z);
        }
        g * z * (1.0 - z)
    }).collect())]
}

fn calc_dim(dims: Vec<Vec2>) -> Vec2 {
    dims[0]
}

pub fn sigmoid_f64(node_id: String, z: Arc<Graph<f64>>) -> Node<f64> {
    Node::new(node_id, operation_f64, operation_prime_f64, vec![z], calc_dim)
}
