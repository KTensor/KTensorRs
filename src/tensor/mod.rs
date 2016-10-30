extern crate rand;
use self::rand::{thread_rng};
use self::rand::distributions::normal::{Normal};
use self::rand::distributions::{IndependentSample};

use std::ops::{Add, Mul};

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
        assert_eq!(dimensions.0, matrix.dim().0);
        assert_eq!(dimensions.1, matrix.dim().1);
        Tensor {
            dim: dimensions,
            matrix: matrix,
        }
    }

    /// Returns the dimensions of the `Tensor`
    pub fn dim(&self) -> Vec2 {
        self.dim
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

    /// max_num_strides = (dim - width) / stride + 1
    pub fn get_convolutions(){

    }
}

impl <T> Tensor<T> where T: Copy {
    /// Transpose `Tensor`
    ///
    /// # Example
    ///
    /// ```
    /// let tensor = ktensor::Tensor::new(ktensor::math::Vec2(2, 3), ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).collect()));
    /// let tensor2 = ktensor::Tensor::new(ktensor::math::Vec2(2, 3), ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).collect()));
    /// let tensor2 = tensor2.transpose();
    /// let tensor3 = tensor * tensor2;
    /// let tensor3 = tensor3.to_flattened();
    /// assert_eq!(tensor3[2], 14);
    /// ```
    pub fn transpose(&self) -> Tensor<T> {
        let Vec2(x, y) = self.dim;
        Tensor {
            dim: Vec2(y, x),
            matrix: self.matrix.transpose(),
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


//////////////
// Addition //
//////////////

impl <T> Add<Tensor<T>> for Tensor<T> where T: Add<Output=T> + Copy {
    type Output = Tensor<T>;

    /// Add `Tensor`s
    ///
    /// # Arguments
    ///
    /// - `self` - this tensor
    /// - `rhs` - another tensor
    ///
    /// # Example
    ///
    /// ```
    /// let tensor1 = ktensor::Tensor::new(ktensor::math::Vec2(2, 3), ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).collect()));
    /// let tensor2 = ktensor::Tensor::new(ktensor::math::Vec2(2, 3), ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).rev().collect()));
    /// let tensor3 = tensor1 + tensor2;
    /// let tensor3 = tensor3.to_flattened();
    /// assert_eq!(tensor3[0], 5.0);
    /// ```
    fn add(self, rhs: Tensor<T>) -> Tensor<T> {
        Tensor::new(self.dim, self.matrix + rhs.matrix)
    }
}

impl <'a, 'b, T> Add<&'b Tensor<T>> for &'a Tensor<T> where T: Add<Output=T> + Copy {
    type Output = Tensor<T>;

    /// Add `Tensor`s by reference
    ///
    /// # Arguments
    ///
    /// - `self` - this tensor reference
    /// - `rhs` - another tensor reference
    ///
    /// # Example
    ///
    /// ```
    /// let tensor1 = ktensor::Tensor::new(ktensor::math::Vec2(2, 3), ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).collect()));
    /// let tensor2 = ktensor::Tensor::new(ktensor::math::Vec2(2, 3), ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).rev().collect()));
    /// let tensor3 = &tensor1 + &tensor2;
    /// let tensor1 = tensor1.to_flattened();
    /// let tensor2 = tensor2.to_flattened();
    /// let tensor3 = tensor3.to_flattened();
    /// assert_eq!(tensor1[0], 0.0);
    /// assert_eq!(tensor2[0], 5.0);
    /// assert_eq!(tensor3[0], 5.0);
    /// ```
    fn add(self, rhs: &'b Tensor<T>) -> Tensor<T> {
        Tensor::new(self.dim, &self.matrix + &rhs.matrix)
    }
}

impl <T> Add<T> for Tensor<T> where T: Add<Output=T> + Copy {
    type Output = Tensor<T>;

    /// Add `Tensor` and a constant
    ///
    /// # Arguments
    ///
    /// - `self` - this `Tensor`
    /// - `rhs` - a constant
    ///
    /// # Example
    ///
    /// ```
    /// let tensor1 = ktensor::Tensor::new(ktensor::math::Vec2(2, 3), ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).collect()));
    /// let float = 1.0;
    /// let tensor2 = tensor1 + float;
    /// let tensor2 = tensor2.to_flattened();
    /// assert_eq!(tensor2[0], 1.0);
    /// ```
    fn add(self, rhs: T) -> Tensor<T> {
        Tensor::new(self.dim, self.matrix + rhs)
    }
}

impl <'a, 'b, T> Add<&'b T> for &'a Tensor<T> where T: Add<Output=T> + Copy {
    type Output = Tensor<T>;

    /// Add `Tensor` and a constant
    ///
    /// # Arguments
    ///
    /// - `self` - this tensor reference
    /// - `rhs` - a constant reference
    ///
    /// # Example
    ///
    /// ```
    /// let tensor1 = ktensor::Tensor::new(ktensor::math::Vec2(2, 3), ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).collect()));
    /// let float = 1.0;
    /// let tensor2 = &tensor1 + &float;
    /// let tensor1 = tensor1.to_flattened();
    /// let tensor2 = tensor2.to_flattened();
    /// assert_eq!(tensor1[0], 0.0);
    /// assert_eq!(tensor2[0], 1.0);
    /// assert_eq!(float, 1.0);
    /// ```
    fn add(self, rhs: &'b T) -> Tensor<T> {
        Tensor::new(self.dim, &self.matrix + rhs)
    }
}


/////////////////////
// Multiplication  //
/////////////////////

impl <T> Mul<Tensor<T>> for Tensor<T> where T: Mul<Output=T> + Add<Output=T> + Copy {
    type Output = Tensor<T>;

    /// Multiply `Tensor`s`
    ///
    /// # Arguments
    ///
    /// - `self` - this tensor
    /// - `rhs` - another tensor
    ///
    /// # Example
    ///
    /// ```
    /// let tensor1 = ktensor::Tensor::new(ktensor::math::Vec2(2, 3), ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).collect()));
    /// let tensor2 = ktensor::Tensor::new(ktensor::math::Vec2(3, 2), ktensor::math::Matrix::new(ktensor::math::Vec2(3, 2), (0..6).map(|i| i as f64).rev().collect()));
    /// let tensor3 = tensor1 * tensor2;
    /// let tensor3 = tensor3.to_flattened();
    /// assert_eq!(tensor3.len(), 4);
    /// assert_eq!(tensor3[0], 5.0);
    /// ```
    fn mul(self, rhs: Tensor<T>) -> Tensor<T> {
        Tensor::new(Vec2(self.dim.0, rhs.dim.1), self.matrix * rhs.matrix)
    }
}

impl <'a, 'b, T> Mul<&'b Tensor<T>> for &'a Tensor<T> where T: Mul<Output=T> + Add<Output=T> + Copy {
    type Output = Tensor<T>;

    /// Multiply `Tensor`s` by reference
    ///
    /// # Arguments
    ///
    /// - `self` - this tensor reference
    /// - `rhs` - another tensor reference
    ///
    /// # Example
    ///
    /// ```
    /// let tensor1 = ktensor::Tensor::new(ktensor::math::Vec2(2, 3), ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).collect()));
    /// let tensor2 = ktensor::Tensor::new(ktensor::math::Vec2(3, 2), ktensor::math::Matrix::new(ktensor::math::Vec2(3, 2), (0..6).map(|i| i as f64).rev().collect()));
    /// let tensor3 = &tensor1 * &tensor2;
    /// let tensor1 = tensor1.to_flattened();
    /// let tensor2 = tensor2.to_flattened();
    /// let tensor3 = tensor3.to_flattened();
    /// assert_eq!(tensor3.len(), 4);
    /// assert_eq!(tensor1[0], 0.0);
    /// assert_eq!(tensor2[0], 5.0);
    /// assert_eq!(tensor3[0], 5.0);
    /// ```
    fn mul(self, rhs: &'b Tensor<T>) -> Tensor<T> {
        Tensor::new(Vec2(self.dim.0, rhs.dim.1), &self.matrix * &rhs.matrix)
    }
}

impl <T> Mul<T> for Tensor<T> where T: Mul<Output=T> + Copy {
    type Output = Tensor<T>;

    /// Multiply `Tensor` and a constant
    ///
    /// # Arguments
    ///
    /// - `self` - this tensor
    /// - `rhs` - a constant
    ///
    /// # Example
    ///
    /// ```
    /// let tensor1 = ktensor::Tensor::new(ktensor::math::Vec2(2, 3), ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).collect()));
    /// let float = 2.0;
    /// let tensor2 = tensor1 * float;
    /// let tensor2 = tensor2.to_flattened();
    /// assert_eq!(tensor2[2], 4.0);
    /// ```
    fn mul(self, rhs: T) -> Tensor<T> {
        Tensor::new(self.dim, self.matrix * rhs)
    }
}

impl <'a, 'b, T> Mul<&'b T> for &'a Tensor<T> where T: Mul<Output=T> + Copy {
    type Output = Tensor<T>;

    /// Multiply `Tensor` and a constant
    ///
    /// # Arguments
    ///
    /// - `self` - this tensor reference
    /// - `rhs` - a constant reference
    ///
    /// # Example
    ///
    /// ```
    /// let tensor1 = ktensor::Tensor::new(ktensor::math::Vec2(2, 3), ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).collect()));
    /// let float = 2.0;
    /// let tensor2 = &tensor1 * &float;
    /// let tensor1 = tensor1.to_flattened();
    /// let tensor2 = tensor2.to_flattened();
    /// assert_eq!(tensor1[2], 2.0);
    /// assert_eq!(tensor2[2], 4.0);
    /// assert_eq!(float, 2.0);
    /// ```
    fn mul(self, rhs: &'b T) -> Tensor<T> {
        Tensor::new(self.dim, &self.matrix * rhs)
    }
}


///////////////////////
// Hadamard Product  //
///////////////////////

impl <'a, T> Tensor<T> where T: Mul<Output=T> + Copy {
    /// Hadamard Product of Tensors by reference
    ///
    /// # Arguments
    ///
    /// - `self` - this tensor reference
    /// - `rhs` - another tensor reference
    ///
    /// # Example
    ///
    /// ```
    /// let tensor1 = ktensor::Tensor::new(ktensor::math::Vec2(2, 3), ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).collect()));
    /// let tensor2 = ktensor::Tensor::new(ktensor::math::Vec2(2, 3), ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).rev().collect()));
    /// let tensor3 = tensor1.product(&tensor2);
    /// let tensor1 = tensor1.to_flattened();
    /// let tensor2 = tensor2.to_flattened();
    /// let tensor3 = tensor3.to_flattened();
    /// assert_eq!(tensor1[2], 2.0);
    /// assert_eq!(tensor2[2], 3.0);
    /// assert_eq!(tensor3[2], 6.0);
    /// ```
    pub fn product(&self, rhs: &'a Tensor<T>) -> Tensor<T> {
        Tensor::new(self.dim, self.matrix.product(&rhs.matrix))
    }
}
