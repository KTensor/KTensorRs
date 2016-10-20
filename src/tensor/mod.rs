use math::{Matrix, Vec2};

/// Encapsulates a matrix for transfer between nodes
pub struct Tensor<T> {
    /// vector of the size of the `Tensor`
    dim: Vec2,
    /// mutable value of the tensor
    matrix: Matrix<T>,
}

impl <T> Tensor<T> {
    pub fn new(dimensions: Vec2, matrix: Matrix<T>) -> Tensor<T> {
        Tensor {
            dim: dimensions,
            matrix: matrix,
        }
    }

    pub fn from_generator(){

    }
}
