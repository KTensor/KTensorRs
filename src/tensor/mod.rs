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
    ///
    /// # Example
    ///
    /// ```
    /// let tensor = ktensor::Tensor::new(ktensor::math::Vec2(2, 3), ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).collect()));
    /// ```
    pub fn new(dimensions: Vec2, matrix: Matrix<T>) -> Tensor<T> {
        Tensor {
            dim: dimensions,
            matrix: matrix,
        }
    }

    /// Gives ownership to the `matrix` buffer
    ///
    /// # Example
    ///
    /// ```
    /// let tensor = ktensor::Tensor::new(ktensor::math::Vec2(2, 3), ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).collect()));
    /// let tensor = tensor.to_flattened();
    /// let result = vec![0, 1, 2, 3, 4, 5];
    /// for (&i, &j) in tensor.iter().zip(result.iter()) {
    ///     assert_eq!(i, j);
    /// }
    /// ```
    pub fn to_flattened(self) -> Vec<T> {
        self.matrix.to_flattened()
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
    /// let tensor = tensor.to_flattened();
    /// for &i in tensor.iter() {
    ///     println!("{}", i);
    ///     assert!(i <= 1.0 && i >= -1.0);
    /// }
    /// ```
    pub fn from_gaussian(dimensions: Vec2) -> Tensor<f64> {
        let Vec2(row, col) = dimensions;
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 2.0);
        let buf = Vec::<f64>::with_capacity(row * col).iter().map(|&_| {
            let num = normal.ind_sample(&mut rng);
            if num > 1.0 {
                1.0
            } else if num < -1.0 {
                -1.0
            } else {
                num
            }
        }).collect();

        Tensor {
            dim: dimensions,
            matrix: Matrix::new(dimensions, buf),
        }
    }

    /// max_num_strides = (dim - width) / stride + 1
    pub fn get_convolutions(){

    }
}

impl Tensor<f32> {
    /// generates a random valued matrix with a standard deviation of 1
    ///
    ///
    /// # Arguments
    ///
    /// - `dimensions` - dimensions of `Tensor`
    ///
    /// # Example
    ///
    /// ```
    /// let tensor = ktensor::Tensor::<f32>::from_gaussian(ktensor::math::Vec2(5, 5));
    /// let tensor = tensor.to_flattened();
    /// for &i in tensor.iter() {
    ///     assert!(i <= 1.0 && i >= -1.0);
    /// }
    /// ```
    pub fn from_gaussian(dimensions: Vec2) -> Tensor<f32> {
        let Vec2(row, col) = dimensions;
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 2.0);
        let buf = Vec::<f32>::with_capacity(row * col).iter().map(|&_| {
            let num = normal.ind_sample(&mut rng) as f32;
            if num > 1.0 {
                1.0
            } else if num < -1.0 {
                -1.0
            } else {
                num
            }
        }).collect();

        Tensor {
            dim: dimensions,
            matrix: Matrix::new(dimensions, buf),
        }
    }
}
