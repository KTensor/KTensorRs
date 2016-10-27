use std::ops::{Fn, Mul, Add};
use context::{Context};
use tensor::{Tensor};
use node::{Graph};

pub struct Node<T> {
    id: &'static str,
    op: Box<Fn(&Tensor<T>, &Tensor<T>) -> Tensor<T>>,
    a: Box<Fn(&Tensor<T>) -> Tensor<T>>,
    a_prime: Box<Fn(&Tensor<T>) -> Tensor<T>>,
    param: (Box<Graph<T>>, Box<Graph<T>>, Box<Graph<T>>),
}

impl <T> Node<T> {
    pub fn new(&self, node_id: &'static str, operation: Box<Fn(&Tensor<T>, &Tensor<T>) -> Tensor<T>>, activation_function: Box<Fn(&Tensor<T>) -> Tensor<T>>, activation_prime_function: Box<Fn(&Tensor<T>) -> Tensor<T>>, parameter: (Box<Graph<T>>, Box<Graph<T>>, Box<Graph<T>>)) -> Node<T> {
        Node {
            id: node_id,
            op: operation,
            a: activation_function,
            a_prime: activation_prime_function,
            param: parameter,
        }
    }
}

impl <T> Graph<T> for Node<T> where T: Copy + Mul<Output=T> + Add<Output=T> {
    fn get_id(&self) -> &'static str {
        self.id
    }

    fn run(&self, state: &Context<T>, variable: &Context<T>) -> Tensor<T> {
        (self.a)(&((self.op)(&self.param.0.run(state, variable), &self.param.1.run(state, variable)) + self.param.2.run(state, variable)))
    }

    fn train(&self, state: &Context<T>, variable: &Context<T>, deltas: &mut Context<T>) {
        let x = self.param.0.run(state, variable);
        let w = self.param.1.run(state, variable);
        let b = self.param.2.run(state, variable);
        let z = &((self.op)(&x, &w)) + &b;
        let a = (self.a)(&z);

    }
}
