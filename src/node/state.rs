use std::ops::{Mul, Add};
use math::{Vec2};
use node::{Graph};
use tensor::{Tensor};
use context::{Context};

pub struct State {
    id: &'static str,
    dim: Vec2,
}

impl State {
    pub fn new(node_id: &'static str, dimensions: Vec2) -> State {
        State {
            id: node_id,
            dim: dimensions,
        }
    }

    pub fn init_norm_f64(&self, context: &mut Context<f64>) {
        context.set(self.id, Tensor::<f64>::from_gaussian(self.dim));
    }

    pub fn init_norm_f32(&self, context: &mut Context<f32>) {
        context.set(self.id, Tensor::<f32>::from_gaussian(self.dim));
    }
}

impl <T> Graph<T> for State where T: Copy + Mul<Output=T> + Add<Output=T> {
    fn get_id(&self) -> &'static str {
        self.id
    }

    fn get_dim(&self) -> Vec2 {
        self.dim
    }

    fn run(&self, state: &Context<T>, _: &Context<T>) -> Tensor<T> {
        match state.get(Graph::<T>::get_id(self)) {
            Some(x) => x.clone(),
            None    => panic!("State {} does not exist in state", Graph::<T>::get_id(self)),
        }
    }

    fn forward_pass(&self, state: &Context<T>, _: &Context<T>, _: &mut Context<T>) -> Tensor<T> {
        match state.get(Graph::<T>::get_id(self)) {
            Some(x) => x.clone(),
            None    => panic!("State {} does not exist in state", Graph::<T>::get_id(self)),
        }
    }

    fn backward_pass(&self, state: &mut Context<T>, _: &Context<T>, history: &Context<T>, gradient: &Tensor<T>, learning_rate: &T) {
        let delta = gradient * learning_rate;
        let previous_state = match history.get(Graph::<T>::get_id(self)) {
            Some(x) => x,
            None    => panic!("State {} does not exist in state", Graph::<T>::get_id(self)),
        };
        state.set(Graph::<T>::get_id(self), previous_state + &delta);
    }
}

pub fn init_state_f64(vec_states: Vec<State>, context: &mut Context<f64>) {
    for state in vec_states {
        state.init_norm_f64(context);
    }
}

pub fn init_state_f32(vec_states: Vec<State>, context: &mut Context<f32>) {
    for state in vec_states {
        state.init_norm_f32(context);
    }
}
