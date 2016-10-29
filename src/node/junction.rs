use std::ops::{Fn, Mul, Add};
use context::{Context};
use tensor::{Tensor};
use node::{Graph};

pub struct Node<T> {
    id: &'static str,
    op: Box<Fn(Vec<Tensor<T>>) -> Tensor<T>>,
    op_train: Box<Fn(Vec<&Tensor<T>>) -> Tensor<T>>,
    op_prime: Box<Fn(&Tensor<T>, Vec<&Tensor<T>>) -> Vec<Tensor<T>>>,
    param: Vec<Box<Graph<T>>>,
}

impl <T> Node<T> {
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
    pub fn new(node_id: &'static str, operation: Box<Fn(Vec<Tensor<T>>) -> Tensor<T>>, operation_train: Box<Fn(Vec<&Tensor<T>>) -> Tensor<T>>, operation_prime: Box<Fn(&Tensor<T>, Vec<&Tensor<T>>) -> Vec<Tensor<T>>>, parameter: Vec<Box<Graph<T>>>) -> Node<T> {
        Node {
            id: node_id,
            op: operation,
            op_train: operation_train,
            op_prime: operation_prime,
            param: parameter,
        }
    }
}

impl <T> Graph<T> for Node<T> where T: Copy + Mul<Output=T> + Add<Output=T> {
    fn get_id(&self) -> &'static str {
        self.id
    }

    fn run(&self, state: &Context<T>, variable: &Context<T>) -> Tensor<T> {
        (self.op)(self.param.iter().map(|node| node.run(state, variable)).collect())
    }

    fn train(&self, state: &Context<T>, variable: &Context<T>, history: &mut Context<T>) {
        let tensor = self.forward_pass(state, variable, history);
        history.set(self.get_id(), tensor);
    }

    fn forward_pass(&self, state: &Context<T>, variable: &Context<T>, history: &Context<T>) -> Tensor<T> {
        (self.op_train)(self.param.iter().map(|node| {
            node.forward_pass(state, variable, history);
            match history.get(node.get_id()) {
                Some(x) => x,
                None    => panic!("Node {} does not exist in history", node.get_id()),
            }
        }).collect())
    }

    fn backward_pass(&self, state: &mut Context<T>, variable: &Context<T>, history: &Context<T>, gradient: &Tensor<T>, learning_rate: &f64) {
        let deltas = (self.op_prime)(gradient, self.param.iter().map(|node| match history.get(node.get_id()) {
            Some(x) => x,
            None    => panic!("Node {} does not exist in history", node.get_id()),
        }).collect());
        for (delta, parameter) in deltas.iter().zip(self.param.iter()) {
            parameter.backward_pass(state, variable, history, delta, learning_rate);
        }
    }
}
