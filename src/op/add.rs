use std::string::{String};
use std::sync::{Arc};
use math::{Vec2};
use node::{Node, Graph};
use tensor::{Tensor};

fn operation_f64(vec: Vec<Tensor<f64>>) -> Tensor<f64> {
    let Vec2(x1, y1) = vec[0].dim();
    let Vec2(x2, _) = vec[1].dim();
    if x1 != 1 && x2 == 1 {
        let transform = vec[1].buffer();
        Tensor::from_vec(Vec2(x1, y1), vec[0].buffer().iter().enumerate().map(|(i, &x)| x + transform[i % y1]).collect())
    } else {
        &vec[0] + &vec[1]
    }

}

fn operation_prime_f64(gradient: &Tensor<f64>, vec: Vec<&Tensor<f64>>) -> Vec<Tensor<f64>> {
    let Vec2(x1, y1) = vec[0].dim();
    let Vec2(x2, _) = vec[1].dim();
    if x1 != 1 && x2 == 1 {
        let mut vector_grad = Vec::with_capacity(y1);
        for i in 0..y1 {
            let mut k = 0.0;
            for j in 0..x1 {
                k += gradient.get(Vec2(j, i));
            }
            vector_grad.push(k / x1 as f64);
        }

        vec![gradient.clone(), Tensor::from_vec(Vec2(1, y1), vector_grad)]
    } else {
        vec![gradient.clone(), gradient.clone()]
    }
}

fn calc_dim(dims: Vec<Vec2>) -> Vec2 {
    let Vec2(x1, y1) = dims[0];
    let Vec2(x2, y2) = dims[1];
    assert!(x1 == 0 && x2 == 1 || x1 == x2);
    assert_eq!(y1, y2);
    dims[0]
}

pub fn add_f64(node_id: String, a: Arc<Graph<f64>>, b: Arc<Graph<f64>>) -> Node<f64> {
    Node::new(node_id, operation_f64, operation_prime_f64, vec![a, b], calc_dim)
}
