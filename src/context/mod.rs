use std::collections::{HashMap};
use tensor::{Tensor};
use node::{Graph};

/// Context Map
pub struct Context<T> {
    /// map of nodeids and values
    map: HashMap<&'static str, Tensor<T>>,
}

impl <T> Context<T> where T: Copy {
    pub fn new() -> Context<T> {
        Context {
            map: HashMap::new(),
        }
    }

    pub fn with_capacity(size: usize) -> Context<T> {
        Context {
            map: HashMap::with_capacity(size),
        }
    }

    pub fn from_vec(context_vec: Vec<(&Graph<T>, Tensor<T>)>) -> Context<T> {
        let mut context_map = HashMap::with_capacity(context_vec.len());

        for (node, batch) in context_vec {
            context_map.insert(node.get_id(), batch);
        }

        Context {
            map: context_map
        }
    }

    pub fn get(&self, nodeid: &'static str) -> Option<&Tensor<T>> {
        self.map.get(nodeid)
    }

    pub fn set(&mut self, nodeid: &'static str, tensor: Tensor<T>) {
        self.map.insert(nodeid, tensor);
    }
}
