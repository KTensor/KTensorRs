use std::collections::{HashMap};
use tensor::{Tensor};
use node::{Graph};

/// Context Map
pub struct Context<T> {
    /// map of nodeids and values
    map: HashMap<&'static str, Vec<Tensor<T>>>,
}

impl <T> Context<T> {
    /// creates a new `Context`
    ///
    /// # Example
    ///
    /// ```
    ///
    /// ```
    pub fn new(context_vec: Vec<(&Graph<T>, Vec<Tensor<T>>)>) -> Context<T> {
        let mut context_map = HashMap::with_capacity(context_vec.len());

        for (node, batch) in context_vec {
            context_map.insert(node.get_id(), batch);
        }

        Context {
            map: context_map
        }
    }

    /// gets a value linked to a node
    ///
    /// # Arguments
    ///
    /// - `node` - a `Node`
    ///
    /// # Example
    ///
    /// ```
    ///
    /// ```
    pub fn get(&self, node: &Graph<T>) -> Option<&Vec<Tensor<T>>> {
        self.map.get(node.get_id())
    }

    pub fn set(&mut self, node: &Graph<T>, tensor: Vec<Tensor<T>>) {
        self.map.insert(node.get_id(), tensor);
    }
}
