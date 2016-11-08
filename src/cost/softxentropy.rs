use math::{Vec2};
use node::{Node, Graph};
use tensor::{Tensor};

fn operation_f64(vec: Vec<Tensor<f64>>) -> Tensor<f64> {
    vec[0].clone()
}

fn operation_prime(_: &Tensor<f64>, vec: Vec<&Tensor<f64>>) -> Vec<Tensor<f64>> {
    vec![vec[0] + &(vec[1] * &-1.0)]
}

fn calc_dim(dims: Vec<Vec2>) -> Vec2 {
    let Vec2(x1, y1) = dims[0];
    let Vec2(x2, y2) = dims[1];
    assert_eq!(x1, x2);
    assert_eq!(y1, y2);
    Vec2(x1, 1)
}

pub fn softmax_cross_entropy_f64<'a>(node_id: &'static str, s: &'a Graph<f64>, y: &'a Graph<f64>) -> Node<'a, f64> {
    Node::new(node_id, operation_f64, operation_prime, vec![s, y], calc_dim)
}
