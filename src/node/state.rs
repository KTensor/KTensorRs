use std::ops::{Mul, Add};
use node::{Graph};
use tensor::{Tensor};
use context::{Context};

pub struct State {
    id: &'static str,
}

impl State {
    pub fn new(node_id: &'static str) -> State {
        State {
            id: node_id
        }
    }
}

impl <T> Graph<T> for State where T: Copy + Mul<Output=T> + Add<Output=T> {
    fn get_id(&self) -> &'static str {
        self.id
    }

    fn run(&self, state: &Context<T>, variable: &Context<T>) -> Tensor<T> {
        match state.get(self.get_id()) {
            Some(x) => x.clone(),
            None    => panic!("State {} does not exist in state", self.get_id()),
        }
    }

    fn forward_pass(&self, state: &Context<T>, variable: &Context<T>, history: &mut Context<T>) -> Tensor<T> {
        match state.get(self.get_id()) {
            Some(x) => x.clone(),
            None    => panic!("State {} does not exist in state", self.get_id()),
        }
    }

    fn backward_pass(&self, state: &mut Context<T>, variable: &Context<T>, history: &Context<T>, gradient: &Tensor<T>, learning_rate: &T) {
        let delta = match history.get(self.get_id()) {
            Some(x) => x,
            None    => panic!("State {} does not exist in history", self.get_id()),
        } * learning_rate;
        let previous_state = match history.get(self.get_id()) {
            Some(x) => x,
            None    => panic!("State {} does not exist in state", self.get_id()),
        };
        state.set(self.get_id(), previous_state + &delta);
    }
}
