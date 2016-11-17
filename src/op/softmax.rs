use std::string::{String};
use std::sync::{Arc};
use std::f64::{NAN as NAN_f64};
use std::f32::{NAN as NAN_f32};
use math::{Vec2, Matrix};
use node::{Node, Graph};
use tensor::{Tensor};

fn operation_f64(vec: Vec<Tensor<f64>>) -> Tensor<f64> {
    let z = &vec[0];
    let Vec2(row, col) = z.dim();

    let mut vec_softmax = Vec::with_capacity(row * col);

    for i in 0..row {
        let mut vec_z = Vec::with_capacity(col);
        let mut m: f64 = NAN_f64;

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

fn operation_f32(vec: Vec<Tensor<f32>>) -> Tensor<f32> {
    let z = &vec[0];
    let Vec2(row, col) = z.dim();

    let mut vec_softmax = Vec::with_capacity(row * col);

    for i in 0..row {
        let mut vec_z = Vec::with_capacity(col);
        let mut m: f32 = NAN_f32;

        for j in 0..col {
            let k = z.get(Vec2(i, j));
            vec_z.push(k);
            m = f32::max(m, k);
        }

        let vec_n: Vec<f32> = vec_z.iter().map(|a| a - m).collect();
        let vec_f: Vec<f32> = vec_n.iter().map(|&a| a.exp()).collect();
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

fn operation_prime_f32(gradient: &Tensor<f32>, _: Vec<&Tensor<f32>>) -> Vec<Tensor<f32>> {
    vec![gradient.clone()]
}

fn calc_dim(dims: Vec<Vec2>) -> Vec2 {
    dims[0]
}

pub fn softmax_f64(node_id: String, a: Arc<Graph<f64>>) -> Node<f64> {
    Node::new(node_id, operation_f64, operation_prime_f64, vec![a], calc_dim)
}

pub fn softmax_f32(node_id: String, a: Arc<Graph<f32>>) -> Node<f32> {
    Node::new(node_id, operation_f32, operation_prime_f32, vec![a], calc_dim)
}

pub fn softmax_round_f64(tensor: Tensor<f64>) -> Tensor<usize> {
    let Vec2(row, col) = tensor.dim();
    let mut vec_rounded = Vec::with_capacity(row);

    for i in 0..row {
        let mut m = tensor.get(Vec2(1, 1));
        let mut index: usize = 0;
        for j in 1..col {
            let k = tensor.get(Vec2(i, j));
            if k > m {
                index = j;
                m = k;
            }
        }

        vec_rounded.push(index);
    }

    Tensor::from_vec(Vec2(row, 1), vec_rounded)
}

pub fn softmax_round_f32(tensor: Tensor<f32>) -> Tensor<usize> {
    let Vec2(row, col) = tensor.dim();
    let mut vec_rounded = Vec::with_capacity(row);

    for i in 0..row {
        let mut m = tensor.get(Vec2(1, 1));
        let mut index: usize = 0;
        for j in 1..col {
            let k = tensor.get(Vec2(i, j));
            if k > m {
                index = j;
                m = k;
            }
        }

        vec_rounded.push(index);
    }

    Tensor::from_vec(Vec2(row, 1), vec_rounded)
}
