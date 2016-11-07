use math::{Vec2, Matrix};
use node::{Node, Graph};
use tensor::{Tensor};

fn operation_f64(vec: Vec<Tensor<f64>>) -> Tensor<f64> {
    Tensor::new(vec[0].dim(), Matrix::new(vec[0].dim(), vec[0].buffer().iter().map(|&a| {
        if a > 16.0 {
            a
        } else if a < -16.0 {
            a / 16.0
        } else {
            a / 16.0 + 0.9375 * (1.0 + a.exp()).ln()
        }
    }).collect()))
}

fn operation_prime_f64(gradient: &Tensor<f64>, vec: Vec<&Tensor<f64>>) -> Vec<Tensor<f64>> {
    vec![Tensor::new(vec[0].dim(), Matrix::new(vec[0].dim(), vec[0].buffer().iter().zip(gradient.buffer().iter()).map(|(&a, &g)| {
        if a > 16.0 {
            g
        } else if a < -16.0 {
            g / 16.0
        } else {
            g / 16.0 + g * 0.9375 * (1.0 + (-a).exp()).recip()
        }
    }).collect()))]
}

fn calc_dim(dims: Vec<Vec2>) -> Vec2 {
    dims[0]
}

pub fn relu_f64(node_id: &'static str, z: Box<Graph<f64>>) -> Node<f64> {
    Node::new(node_id, operation_f64, operation_prime_f64, vec![z], calc_dim)
}
