use std::string::{String};
use std::sync::{Arc};
use std::ops::{Add, Mul};
use math::{Vec2};
use node::{Node, Graph};
use tensor::{Tensor};

fn operation<T>(vec: Vec<Tensor<T>>) -> Tensor<T> where T: Mul<Output=T> + Add<Output=T> + Copy {
    let Vec2(x1, _) = vec[0].dim();
    let Vec2(x2, _) = vec[1].dim();
    if x1 != 1 && x2 == 1 {
        
    } else {
        &vec[0] + &vec[1]
    }

}

fn operation_prime<T>(gradient: &Tensor<T>, _: Vec<&Tensor<T>>) -> Vec<Tensor<T>> where T: Mul<Output=T> + Add<Output=T> + Copy {
    vec![gradient.clone(), gradient.clone()]
}

fn calc_dim(dims: Vec<Vec2>) -> Vec2 {
    let Vec2(x1, y1) = dims[0];
    let Vec2(x2, y2) = dims[1];
    assert!(x1 == 0 && x2 == 1 || x1 == x2);
    assert_eq!(y1, y2);
    dims[0]
}

pub fn add<T>(node_id: String, a: Arc<Graph<T>>, b: Arc<Graph<T>>) -> Node<T> where T: Mul<Output=T> + Add<Output=T> + Copy {
    Node::new(node_id, operation, operation_prime, vec![a, b], calc_dim)
}
