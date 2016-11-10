use std::string::{String};
use std::f64::{NAN};
use math::{Vec2, Matrix};
use node::{Node, Graph};
use tensor::{Tensor};

fn operation_f64(vec: Vec<Tensor<f64>>) -> Tensor<f64> {
    let z = &vec[0];
    let Vec2(row, col) = z.dim();

    let mut vec_softmax = Vec::with_capacity(row * col);

    for i in 0..row {
        let mut vec_z = Vec::with_capacity(col);
        let mut m: f64 = NAN;

        for j in 0..col {
            let k = z.get(Vec2(i, j));
            vec_z.push(k);
            m = f64::max(m, k);
        }

        let vec_n: Vec<f64> = vec_z.iter().map(|a| a - m).collect();
        let vec_f: Vec<f64> = vec_n.iter().map(|&a| a.exp()).collect();
        let g = vec_f.iter().fold(0.0, |sum, &a| sum + a);

        for i in vec_f {
            vec_softmax.push(i / g);
        }
    }

    Tensor::new(Vec2(row, col), Matrix::new(Vec2(row, col), vec_softmax))
}

fn operation_prime_f64(gradient: &Tensor<f64>, _: Vec<&Tensor<f64>>) -> Vec<Tensor<f64>> {
    vec![gradient.clone()]
}

fn calc_dim(dims: Vec<Vec2>) -> Vec2 {
    dims[0]
}

pub fn softmax_f64<'a>(node_id: String, a: &'a Graph<f64>) -> Node<'a, f64> {
    Node::new(node_id, operation_f64, operation_prime_f64, vec![a], calc_dim)
}
