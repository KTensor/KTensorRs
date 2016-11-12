use std::string::{String};
use std::sync::{Arc};
use std::ops::{Add, Mul};
use math::{Vec2};
use node::{Node, Graph};
use tensor::{Tensor};
use std::fmt::{Display};

fn operation<T>(vec: Vec<Tensor<T>>) -> Tensor<T> where T: Mul<Output=T> + Add<Output=T> + Copy {
    let Vec2(x1, y1) = vec[0].dim();
    let Vec2(x2, y2) = vec[1].dim();
    assert_eq!(y1, y2);
    if x1 != 1 && x2 == 1 {
        let transform = vec[1].buffer();
        Tensor::from_vec(Vec2(x1, y1), vec[0].buffer().iter().enumerate().map(|(i, &x)| x + transform[i % y1]).collect())
    } else {
        &vec[0] + &vec[1]
    }

}

fn operation_prime<T>(gradient: &Tensor<T>, _: Vec<&Tensor<T>>) -> Vec<Tensor<T>> where T: Mul<Output=T> + Add<Output=T> + Copy + Display {
    let Vec2(a, b) = gradient.dim();
    println!("gradient {} {}", a, b);
    for i in gradient.buffer() {
        println!("{}", i);
    }
    vec![gradient.clone(), gradient.clone()]
}

fn calc_dim(dims: Vec<Vec2>) -> Vec2 {
    let Vec2(x1, y1) = dims[0];
    let Vec2(x2, y2) = dims[1];
    assert!(x1 == 0 && x2 == 1 || x1 == x2);
    assert_eq!(y1, y2);
    dims[0]
}

pub fn add<T>(node_id: String, a: Arc<Graph<T>>, b: Arc<Graph<T>>) -> Node<T> where T: Mul<Output=T> + Add<Output=T> + Copy + Display {
    Node::new(node_id, operation, operation_prime, vec![a, b], calc_dim)
}
