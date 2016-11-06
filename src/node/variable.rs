use std::ops::{Mul, Add};
use math::{Vec2};
use node::{Graph};
use tensor::{Tensor};
use context::{Context};

pub struct Variable {
    id: &'static str,
    dim: Vec2,
}

impl Variable {
    pub fn new(node_id: &'static str, dimensions: Vec2) -> Variable {
        Variable {
            id: node_id,
            dim: dimensions,
        }
    }
}

impl <T> Graph<T> for Variable where T: Copy + Mul<Output=T> + Add<Output=T> {
    fn get_id(&self) -> &'static str {
        self.id
    }

    fn get_dim(&self) -> Vec2 {
        self.dim
    }

    fn run(&self, _: &Context<T>, variable: &Context<T>) -> Tensor<T> {
        match variable.get(Graph::<T>::get_id(self)) {
            Some(x) => x.clone(),
            None    => panic!("Variable {} does not exist in variable", Graph::<T>::get_id(self)),
        }
    }

    fn forward_pass(&self, _: &Context<T>, variable: &Context<T>, _: &mut Context<T>) -> Tensor<T> {
        match variable.get(Graph::<T>::get_id(self)) {
            Some(x) => x.clone(),
            None    => panic!("Variable {} does not exist in variable", Graph::<T>::get_id(self)),
        }
    }

    fn backward_pass(&self, _: &mut Context<T>, _: &Context<T>, _: &Context<T>, _: &Tensor<T>, _: &T) {}
}
