use std::ops::{Fn};
use tensor::{Tensor};
use node::{Runnable};

/// Computational Node for a Graph
pub struct Node<T> {
    /// node key
    nodeid: &'static str,
    /// function to perform on node inputs
    operation: Fn(Tensor<T>, Tensor<T>) -> Tensor<T>,
    /// input nodes
    parameters: (Runnable<T>, Runnable<T>),
}

impl <T> Node<T> {
    pub fn new(node_id: &'static str, tensor_operation: Fn(Tensor<T>, Tensor<T>) -> Tensor<T>, node_params: Vec<Runnable<T>>) -> Node<T> {
        Node {
            nodeid: node_id,
            operation: tensor_operation,
            parameters: node_params,
        }
    }

    pub fn get_id(&self) -> &'static str {
        self.nodeid
    }
}
