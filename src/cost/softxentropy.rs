use std::ops::{Add, Mul};
use math::{Vec2};
use node::{Node, Graph};
use tensor::{Tensor};
use std::f64;

fn operation(vec: Vec<Tensor<f64>>) -> Tensor<f64> {
    let z = vec[0];
    let y = vec[1];
    let Vec2(row, col) = z.dim();

    let Cost = 0.0;

    for i in 0..row {
        let mut vec_z = Vec::with_capacity(col);
        let mut vec_y = Vec::with_capacity(col);
        let mut m: f64 = f64::NAN;

        for j in 0..col {
            let k = z.get(Vec2(i, j));
            vec_z.push(k);
            m = f64::max(m, k);
            vec_y.push(y.get(Vec2(i, j)));
        }

        let vec_n: Vec<f64> = vec_z.iter().map(|a| a - m).collect();
        let vec_f: Vec<f64> = vec_n.iter().map(|&a| a.exp()).collect();
        let g = vec_f.iter().fold(0.0, |sum, &a| sum + a);
        let log_g = g.ln();

        let cost = vec_y.iter().zip(vec_n.iter()).fold(0.0, |sum, (&a, &b)| {
            sum + (b - log_g) * -a
        });

        Cost += cost;
    }


}

fn operation_prime<T>(gradient: &Tensor<T>, vec: Vec<&Tensor<T>>) -> Vec<Tensor<T>> where T: Mul<Output=T> + Add<Output=T> + Copy {

}

fn calc_dim(dims: Vec<Vec2>) -> Vec2 {
    let Vec2(x1, y1) = dims[0];
    let Vec2(x2, y2) = dims[1];
    assert_eq!(x1, x2);
    assert_eq!(y1, y2);
    Vec2(1, 1)
}

pub fn softmax_cross_entropy_f64(node_id: &'static str, z: Box<Graph<f64>>, y: Box<Graph<f64>>) -> Node<f64> {
    Node::new(node_id, operation, operation_prime, vec![z, y], calc_dim)
}
