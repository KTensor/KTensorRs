use std::ops::{Add, Mul};

/// A pair of coordinates
#[derive(Clone, Copy)]
pub struct Vec2(pub usize, pub usize);

/// A structure of values
pub struct Matrix<T> {
    /// vector of rows and columns
    dim: Vec2,
    /// `Vector` of values in the `Matrix`
    buffer: Vec<T>,
}

impl <T> Matrix<T> {
    /// Returns a new `Matrix`
    ///
    /// # Arguments
    ///
    /// - `dimensions` - 2 dimensional array of rows and columns
    /// - `buffer` - `Vector` of values
    ///
    /// # Example
    ///
    /// ```
    /// let matrix = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), vec![0, 1, 2, 3, 4, 5]);
    /// ```
    pub fn new(dimensions: Vec2, buffer: Vec<T>) -> Matrix<T> {
        Matrix {
            dim: dimensions,
            buffer: buffer,
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

impl <T> Matrix<T> where T: Copy {
    /// Returns reference to value at `Vec2`
    ///
    /// # Arguments
    ///
    /// - `Vec2` - coordinates of the value in the `Matrix`
    ///
    /// # Example
    ///
    /// ```
    /// let matrix = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), vec![0, 1, 2, 3, 4, 5]);
    /// assert_eq!(matrix.get(ktensor::math::Vec2(1, 2)), 5);
    /// assert_eq!(matrix.get(ktensor::math::Vec2(1, 2)), 5);
    /// ```
    pub fn get(&self, Vec2(x, y): Vec2) -> T {
        self.buffer[x * self.dim.1 + y]
    }
}


//////////////
// Addition //
//////////////

impl <T> Add<Matrix<T>> for Matrix<T> where T: Add<Output=T> + Copy {
    type Output = Matrix<T>;

    /// Add Matricies
    ///
    /// # Arguments
    ///
    /// - `self` - this matrix
    /// - `rhs` - another matrix
    ///
    /// # Example
    ///
    /// ```
    /// let matrix1 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let matrix2 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
    /// let matrix3 = matrix1 + matrix2;
    /// assert_eq!(matrix3.get(ktensor::math::Vec2(0, 0)), 5.0);
    /// ```
    fn add(self, rhs: Matrix<T>) -> Matrix<T> {
        assert_eq!(self.len(), rhs.len());
        let mut buffer = Vec::with_capacity(self.len());
        for (&i, &j) in self.buffer.iter().zip(rhs.buffer.iter()) {
            buffer.push(i + j);
        }
        Matrix::new(self.dim, buffer)
    }
}

impl <'a, 'b, T> Add<&'b Matrix<T>> for &'a Matrix<T> where T: Add<Output=T> + Copy {
    type Output = Matrix<T>;

    /// Add Matricies by reference
    ///
    /// # Arguments
    ///
    /// - `self` - this matrix reference
    /// - `rhs` - another matrix reference
    ///
    /// # Example
    ///
    /// ```
    /// let matrix1 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let matrix2 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
    /// let matrix3 = &matrix1 + &matrix2;
    /// assert_eq!(matrix1.get(ktensor::math::Vec2(0, 0)), 0.0);
    /// assert_eq!(matrix2.get(ktensor::math::Vec2(0, 0)), 5.0);
    /// assert_eq!(matrix3.get(ktensor::math::Vec2(0, 0)), 5.0);
    /// ```
    fn add(self, rhs: &'b Matrix<T>) -> Matrix<T> {
        assert_eq!(self.len(), rhs.len());
        let mut buffer = Vec::with_capacity(self.len());
        for (&i, &j) in self.buffer.iter().zip(rhs.buffer.iter()) {
            buffer.push(i + j);
        }
        Matrix::new(self.dim, buffer)
    }
}

impl <T> Add<T> for Matrix<T> where T: Add<Output=T> + Copy {
    type Output = Matrix<T>;

    /// Add Matrix and a constant
    ///
    /// # Arguments
    ///
    /// - `self` - this matrix
    /// - `rhs` - a constant
    ///
    /// # Example
    ///
    /// ```
    /// let matrix1 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let float = 1.0;
    /// let matrix2 = matrix1 + float;
    /// assert_eq!(matrix2.get(ktensor::math::Vec2(0, 0)), 1.0);
    /// ```
    fn add(self, rhs: T) -> Matrix<T> {
        let mut buffer = Vec::with_capacity(self.len());
        for &i in self.buffer.iter() {
            buffer.push(i + rhs);
        }
        Matrix::new(self.dim, buffer)
    }
}

impl <'a, 'b, T> Add<&'b T> for &'a Matrix<T> where T: Add<Output=T> + Copy {
    type Output = Matrix<T>;

    /// Add Matrix and a constant
    ///
    /// # Arguments
    ///
    /// - `self` - this matrix reference
    /// - `rhs` - a constant reference
    ///
    /// # Example
    ///
    /// ```
    /// let matrix1 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let float = 1.0;
    /// let matrix2 = &matrix1 + &float;
    /// assert_eq!(matrix1.get(ktensor::math::Vec2(0, 0)), 0.0);
    /// assert_eq!(matrix2.get(ktensor::math::Vec2(0, 0)), 1.0);
    /// assert_eq!(float, 1.0);
    /// ```
    fn add(self, &rhs: &'b T) -> Matrix<T> {
        let mut buffer = Vec::with_capacity(self.len());
        for &i in self.buffer.iter() {
            buffer.push(i + rhs);
        }
        Matrix::new(self.dim, buffer)
    }
}


/////////////////////
// Multiplication  //
/////////////////////

impl <T> Mul<Matrix<T>> for Matrix<T> where T: Mul<Output=T> + Add<Output=T> + Copy {
    type Output = Matrix<T>;

    /// Multiply Matricies
    ///
    /// # Arguments
    ///
    /// - `self` - this matrix
    /// - `rhs` - another matrix
    ///
    /// # Example
    ///
    /// ```
    /// let matrix1 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let matrix2 = ktensor::math::Matrix::new(ktensor::math::Vec2(3, 2), vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
    /// let matrix3 = matrix1 * matrix2;
    /// assert_eq!(matrix3.len(), 4);
    /// assert_eq!(matrix3.get(ktensor::math::Vec2(0, 0)), 5.0);
    /// ```
    fn mul(self, rhs: Matrix<T>) -> Matrix<T> {
        let Vec2(x, y) = self.dim;
        let Vec2(x2, y2) = rhs.dim;
        assert_eq!(y, x2);
        let mut buffer = Vec::with_capacity(x * y2);
        for i in 0..x {
            for j in 0..y2 {
                let mut sum = self.get(Vec2(i, 0)) * rhs.get(Vec2(0, j));
                for k in 1..y {
                    sum = sum + self.get(Vec2(i, k)) * rhs.get(Vec2(k, j));
                }
                buffer.push(sum);
            }
        }
        Matrix::new(Vec2(x, y2), buffer)
    }
}

impl <'a, 'b, T> Mul<&'b Matrix<T>> for &'a Matrix<T> where T: Mul<Output=T> + Add<Output=T> + Copy {
    type Output = Matrix<T>;

    /// Multiply Matricies by reference
    ///
    /// # Arguments
    ///
    /// - `self` - this matrix reference
    /// - `rhs` - another matrix reference
    ///
    /// # Example
    ///
    /// ```
    /// let matrix1 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let matrix2 = ktensor::math::Matrix::new(ktensor::math::Vec2(3, 2), vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
    /// let matrix3 = &matrix1 * &matrix2;
    /// assert_eq!(matrix3.len(), 4);
    /// assert_eq!(matrix1.get(ktensor::math::Vec2(0, 0)), 0.0);
    /// assert_eq!(matrix2.get(ktensor::math::Vec2(0, 0)), 5.0);
    /// assert_eq!(matrix3.get(ktensor::math::Vec2(0, 0)), 5.0);
    /// ```
    fn mul(self, rhs: &'b Matrix<T>) -> Matrix<T> {
        let Vec2(x, y) = self.dim;
        let Vec2(x2, y2) = rhs.dim;
        assert_eq!(y, x2);
        let mut buffer = Vec::with_capacity(x * y2);
        for i in 0..x {
            for j in 0..y2 {
                let mut sum = self.get(Vec2(i, 0)) * rhs.get(Vec2(0, j));
                for k in 1..y {
                    sum = sum + self.get(Vec2(i, k)) * rhs.get(Vec2(k, j));
                }
                buffer.push(sum);
            }
        }
        Matrix::new(Vec2(x, y2), buffer)
    }
}

impl <T> Mul<T> for Matrix<T> where T: Mul<Output=T> + Copy {
    type Output = Matrix<T>;

    /// Multiply Matrix and a constant
    ///
    /// # Arguments
    ///
    /// - `self` - this matrix
    /// - `rhs` - a constant
    ///
    /// # Example
    ///
    /// ```
    /// let matrix1 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let float = 2.0;
    /// let matrix2 = matrix1 * float;
    /// assert_eq!(matrix2.get(ktensor::math::Vec2(0, 2)), 4.0);
    /// ```
    fn mul(self, rhs: T) -> Matrix<T> {
        let mut buffer = Vec::with_capacity(self.len());
        for &i in self.buffer.iter() {
            buffer.push(i * rhs);
        }
        Matrix::new(self.dim, buffer)
    }
}

impl <'a, 'b, T> Mul<&'b T> for &'a Matrix<T> where T: Mul<Output=T> + Copy {
    type Output = Matrix<T>;

    /// Multiply Matrix and a constant
    ///
    /// # Arguments
    ///
    /// - `self` - this matrix reference
    /// - `rhs` - a constant reference
    ///
    /// # Example
    ///
    /// ```
    /// let matrix1 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let float = 2.0;
    /// let matrix2 = &matrix1 * &float;
    /// assert_eq!(matrix1.get(ktensor::math::Vec2(0, 2)), 2.0);
    /// assert_eq!(matrix2.get(ktensor::math::Vec2(0, 2)), 4.0);
    /// assert_eq!(float, 2.0);
    /// ```
    fn mul(self, &rhs: &'b T) -> Matrix<T> {
        let mut buffer = Vec::with_capacity(self.len());
        for &i in self.buffer.iter() {
            buffer.push(i * rhs);
        }
        Matrix::new(self.dim, buffer)
    }
}
