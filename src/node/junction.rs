use std::string::{String};
use std::ops::{Mul, Add};
use math::{Vec2};
use context::{Context};
use tensor::{Tensor};
use node::{Graph};

pub struct Node<'a, T: 'a> {
    id: String,
    dim: Vec2,
    op: fn(Vec<Tensor<T>>) -> Tensor<T>,
    op_prime: fn(&Tensor<T>, Vec<&Tensor<T>>) -> Vec<Tensor<T>>,
    param: Vec<&'a Graph<T>>,
}

impl <'a, T> Node<'a, T> {
    /// computation node
    ///
    /// # Arguments
    ///
    /// if this node is z = f(x, y):
    ///
    /// - `node_id`
    /// - `operation` - f
    /// - `operation_train`
    /// - `operation_prime` - f_x,y which takes in a gradient dC/dz and inputs x, y; outputs gradients dC/dx, dC/dy
    /// - `parameter` - Vec<(x, y)>
    pub fn new(node_id: String, operation: fn(Vec<Tensor<T>>) -> Tensor<T>, operation_prime: fn(&Tensor<T>, Vec<&Tensor<T>>) -> Vec<Tensor<T>>, parameter: Vec<&'a Graph<T>>, calc_dim: fn(Vec<Vec2>) -> Vec2) -> Node<'a, T> where T: Copy {
        Node {
            id: node_id,
            dim: calc_dim(parameter.iter().map(|node| node.get_dim()).collect()),
            op: operation,
            op_prime: operation_prime,
            param: parameter,
        }
    }

    pub fn with_dim(node_id: String, operation: fn(Vec<Tensor<T>>) -> Tensor<T>, operation_prime: fn(&Tensor<T>, Vec<&Tensor<T>>) -> Vec<Tensor<T>>, parameter: Vec<&'a Graph<T>>, dimension: Vec2) -> Node<'a, T> {
        Node {
            id: node_id,
            dim: dimension,
            op: operation,
            op_prime: operation_prime,
            param: parameter,
        }
    }
}

impl <'a, T> Graph<T> for Node<'a, T> where T: Copy + Mul<Output=T> + Add<Output=T> {
    fn get_id(&self) -> String {
        self.id.clone()
    }

    fn get_dim(&self) -> Vec2 {
        self.dim
    }

    fn run(&self, state: &Context<T>, variable: &Context<T>) -> Tensor<T> {
        (self.op)(self.param.iter().map(|node| node.run(state, variable)).collect())
    }

    fn forward_pass(&self, state: &Context<T>, variable: &Context<T>, history: &mut Context<T>) -> Tensor<T> {
        (self.op)(self.param.iter().map(|node| {
            node.train(state, variable, history)
        }).collect())
    }

    fn backward_pass(&self, state: &mut Context<T>, variable: &Context<T>, history: &Context<T>, gradient: &Tensor<T>, learning_rate: &T) {
        let deltas = (self.op_prime)(gradient, self.param.iter().map(|node| match history.get(node.get_id()) {
            Some(x) => x,
            None    => panic!("Node {} does not exist in history", node.get_id()),
        }).collect());
        for (delta, parameter) in deltas.iter().zip(self.param.iter()) {
            parameter.backward_pass(state, variable, history, delta, learning_rate);
        }
    }
}
