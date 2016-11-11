use std::string::{String};
use std::sync::{Arc};
use std::ops::{Mul, Add};
use math::{Vec2};
use node::{Graph};
use tensor::{Tensor};
use context::{Context};

pub struct State {
    id: String,
    dim: Vec2,
}

impl State {
    pub fn new(node_id: String, dimensions: Vec2) -> State {
        State {
            id: node_id,
            dim: dimensions,
        }
    }

    pub fn get_id(&self) -> String {
        self.id.clone()
    }

    pub fn init_norm_f64(&self, context: &mut Context<f64>) {
        context.set(self.get_id(), Tensor::<f64>::from_gaussian(self.dim));
    }

    pub fn init_norm_f32(&self, context: &mut Context<f32>) {
        context.set(self.get_id(), Tensor::<f32>::from_gaussian(self.dim));
    }
}

impl <T> Graph<T> for State where T: Copy + Mul<Output=T> + Add<Output=T> {
    fn get_id(&self) -> String {
        self.id.clone()
    }

    fn get_dim(&self) -> Vec2 {
        self.dim
    }

    fn run(&self, state: &Context<T>, _: &Context<T>) -> Tensor<T> {
        match state.get(self.get_id()) {
            Some(x) => x.clone(),
            None    => panic!("State {} does not exist in state", self.get_id()),
        }
    }

    fn forward_pass(&self, state: &Context<T>, _: &Context<T>, _: &mut Context<T>) -> Tensor<T> {
        match state.get(self.get_id()) {
            Some(x) => x.clone(),
            None    => panic!("State {} does not exist in state", self.get_id()),
        }
    }

    fn backward_pass(&self, state: &mut Context<T>, _: &Context<T>, history: &Context<T>, gradient: &Tensor<T>, learning_rate: T) {
        let delta = gradient * &learning_rate;
        let previous_state = match history.get(self.get_id()) {
            Some(x) => x,
            None    => panic!("State {} does not exist in state", self.get_id()),
        };
        state.set(self.get_id(), previous_state + &delta);
    }
}

pub fn init_state_f64(vec_states: Vec<Arc<State>>, context: &mut Context<f64>) {
    for state in vec_states {
        state.init_norm_f64(context);
    }
}

pub fn init_state_f32(vec_states: Vec<Arc<State>>, context: &mut Context<f32>) {
    for state in vec_states {
        state.init_norm_f32(context);
    }
}
