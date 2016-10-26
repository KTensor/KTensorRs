use std::collections::HashMap;
use tensor::{Tensor};
use node::{Node};

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
    /// let context = ktensor::Context::new(vec![]);
    /// ```
    pub fn new(context_vec: Vec<(&Node<T>, Vec<Tensor<T>>)>) -> Context<T> {
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
    pub fn get(&self, node: &Node<T>) -> Option<&Vec<Tensor<T>>> {
        self.map.get(node.get_id())
    }

    pub fn set(&mut self, node: &Node<T>, tensor: Vec<Tensor<T>>) {
        self.map.insert(node.get_id(), tensor);
    }
}
