use std::collections::HashMap;
use tensor::{Tensor};
use node::{Node};

/// Context Map
pub struct Context<T> {
    /// map of nodeids and values
    map: HashMap<&'static str, Vec<Tensor<T>>>
}

impl <T> Context<T> {
    /// creates a new `Context`
    ///
    /// # Example
    ///
    /// ```
    /// let context = ktensor::Context::new();
    /// ```
    pub fn new(contextVec: Vec<(&Node, Vec<Tensor<T>>)>) -> Context<T> {
        let mut contextMap = HashMap::new();

        for (node, batch) in contextVec {
            contextMap.insert(node.get_id(), batch);
        }

        Context {
            map: contextMap
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
    pub fn get(&self, node: &Node) -> Option<&Vec<Tensor<T>>> {
        self.map.get(node.get_id())
    }

    pub fn set(&mut self, node: &Node, tensor: Vec<Tensor<T>>) {
        self.map.insert(node.get_id(), tensor);
    }
}
