use std::ops::{Mul, Add};
use std::iter::{repeat};
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

    pub fn init_norm_f64(&self, context: &mut Context<f64>) {
        context.set(self.id, Tensor::from_vec(self.dim, repeat(0.0).take(self.dim.0 * self.dim.1).collect()));
    }

    pub fn init_norm_f32(&self, context: &mut Context<f32>) {
        context.set(self.id, Tensor::from_vec(self.dim, repeat(0.0).take(self.dim.0 * self.dim.1).collect()));
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

pub fn init_variables_f64(vec_variables: Vec<&Variable>, context: &mut Context<f64>) {
    for variable in vec_variables {
        variable.init_norm_f64(context);
    }
}

pub fn init_variables_f32(vec_variables: Vec<&Variable>, context: &mut Context<f32>) {
    for variable in vec_variables {
        variable.init_norm_f32(context);
    }
}
