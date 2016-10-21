extern crate rand;
use self::rand::{thread_rng};
use self::rand::distributions::normal::{Normal};
use self::rand::distributions::{IndependentSample};

use math::{Matrix, Vec2};

/// Encapsulates a matrix for transfer between nodes
pub struct Tensor<T> {
    /// vector of the size of the `Tensor`
    dim: Vec2,
    /// mutable value of the tensor
    matrix: Matrix<T>,
}

impl <T> Tensor<T> {
    /// returns a new `Tensor`
    ///
    /// # Arguments
    ///
    /// - `dimensions` - dimensions of the `matrix`
    /// - `matrix` - `Matrix` encapsulated by the `Tensor`
    pub fn new(dimensions: Vec2, matrix: Matrix<T>) -> Tensor<T> {
        Tensor {
            dim: dimensions,
            matrix: matrix,
        }
    }
}

impl Tensor<f64> {
    /// generates a random valued `Tensor` with a standard deviation of 1
    ///
    /// # Arguments
    ///
    /// - `dimensions` - dimensions of `Tensor`
    ///
    /// # Example
    ///
    /// ```
    /// let tensor = ktensor::Tensor::<f64>::from_gaussian(ktensor::math::Vec2(5, 5));
    ///
    /// ```
    pub fn from_gaussian(dimensions: Vec2) -> Tensor<f64> {
        let Vec2(row, col) = dimensions;
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0);
        let buf = Vec::<f64>::with_capacity(row * col).iter().map(|&_| normal.ind_sample(&mut rng)).collect();

        Tensor {
            dim: dimensions,
            matrix: Matrix::new(dimensions, buf),
        }
    }
}

impl Tensor<f32> {
    /// generates a random valued matrix with a standard deviation of 1
    ///
    ///
    /// # Arguments
    ///
    /// - `dimensions` - dimensions of `Tensor`
    pub fn from_gaussian(dimensions: Vec2) -> Tensor<f32> {
        let Vec2(row, col) = dimensions;
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0);
        let buf = Vec::<f32>::with_capacity(row * col).iter().map(|&_| normal.ind_sample(&mut rng) as f32).collect();

        Tensor {
            dim: dimensions,
            matrix: Matrix::new(dimensions, buf),
        }
    }
}
