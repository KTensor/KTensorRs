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


pub fn init_state<T>(vec_states: Vec<State>, context: &mut Context<T>) where T: Copy + Mul<Output=T> + Add<Output=T> {

}
