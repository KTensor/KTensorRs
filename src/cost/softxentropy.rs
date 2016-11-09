use math::{Vec2};
use node::{Node, Graph};
use tensor::{Tensor};

fn operation_f64(vec: Vec<Tensor<f64>>) -> Tensor<f64> {
    vec[0].clone() // just returns softmax
}

fn operation_prime(_: &Tensor<f64>, vec: Vec<&Tensor<f64>>) -> Vec<Tensor<f64>> {
    let total_grad = vec[0] + &(vec[1] * &-1.0);
    let Vec2(row, col) = total_grad.dim();

    let mut vector_grad = Vec::with_capacity(col);
    for i in 0..col {
        let mut k = 0.0;
        for j in 0..row {
            k += total_grad.get(Vec2(j, i));
        }
        vector_grad.push(k / row as f64);
    }

    vec![Tensor::from_vec(Vec2(1, col), vector_grad)]
}

fn calc_dim(dims: Vec<Vec2>) -> Vec2 {
    let Vec2(x1, y1) = dims[0];
    let Vec2(x2, y2) = dims[1];
    assert_eq!(x1, x2);
    assert_eq!(y1, y2);
    dims[0]
}

pub fn softmax_cross_entropy_f64<'a>(node_id: &'static str, s: &'a Graph<f64>, y: &'a Graph<f64>) -> Node<'a, f64> {
    Node::new(node_id, operation_f64, operation_prime, vec![s, y], calc_dim)
}
