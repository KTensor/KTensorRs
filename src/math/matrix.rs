use std::ops::{Add, Mul};

/// A pair of coordinates
#[derive(Clone, Copy)]
pub struct Vec2(pub usize, pub usize);

/// A structure of values
pub struct Matrix<T> {
    /// vector of rows and columns
    dim: Vec2,
    /// vector to get items from buffer
    vec_length: Vec2,
    /// `Vector` of values in the `Matrix`
    buffer: Vec<T>,
}

impl <T> Matrix<T> {
    /// Returns a new `Matrix`
    ///
    /// # Arguments
    ///
    /// - `dimensions` - dimensions of `Matrix`
    /// - `buffer` - `Vec` of values
    ///
    /// # Example
    ///
    /// ```
    /// let matrix = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).collect());
    /// ```
    pub fn new(dimensions: Vec2, buffer: Vec<T>) -> Matrix<T> {
        let Vec2(_, y) = dimensions;
        Matrix {
            dim: dimensions,
            vec_length: Vec2(y, 1),
            buffer: buffer,
        }
    }

    /// Returns the total number of items in the `Matrix`
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the dimensions of the `Matrix`
    pub fn dim(&self) -> Vec2 {
        self.dim
    }

    /// Gives ownership to the buffer
    ///
    /// # Example
    ///
    /// ```
    /// let vector = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).collect()).to_flattened();
    /// let result = vec![0, 1, 2, 3, 4, 5];
    /// for (&i, &j) in vector.iter().zip(result.iter()) {
    ///     assert_eq!(i, j);
    /// }
    /// ```
    pub fn to_flattened(self) -> Vec<T> {
        self.buffer
    }

    /// Consumes `Matrix` and returns transposed `Matrix`
    ///
    /// # Example
    ///
    /// ```
    /// let mut vector = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).collect());
    /// vector.transpose();
    /// assert_eq!(vector.get(ktensor::math::Vec2(2, 0)), 2);
    /// ```
    pub fn transpose(&mut self) {
        let Vec2(dim_x, dim_y) = self.dim;
        let Vec2(x, y) = self.vec_length;
        self.dim = Vec2(dim_y, dim_x);
        self.vec_length = Vec2(y, x);
    }

    /// Returns `Vec` of indicies from `(0, 0)` to `(x, y)` (inclusive and exclusive)
    ///
    /// # Arguments
    ///
    /// - `Vec2` - bottom corner of indicies
    ///
    /// # Example
    ///
    /// ```
    /// let matrix = ktensor::math::Matrix::new(ktensor::math::Vec2(3, 3), (0..9).collect());
    /// let indicies = matrix.get_indicies(ktensor::math::Vec2(2, 2));
    /// let matrix = matrix.to_flattened();
    /// for &i in indicies.iter() {
    ///     assert_eq!(i, matrix[i]);
    /// }
    /// ```
    pub fn get_indicies(&self, Vec2(x, y): Vec2) -> Vec<usize> {
        assert!(x-1 <= self.dim.0 && x > 0);
        assert!(y-1 <= self.dim.1 && y > 0);
        let mut buf = Vec::<usize>::with_capacity(x*y);
        for i in 0..x {
            for j in 0..y {
                buf.push(i * self.vec_length.0 + j * self.vec_length.1);
            }
        }
        buf
    }

    /// Returns `Vec` of indicies across the entire `Matrix` up to (x, y) with the specified stride length
    ///
    /// # Arguments
    ///
    /// - `Vec2` - number of strides in directions x, y
    /// - `stride` - length of stride
    ///
    /// # Example
    ///
    /// ```
    /// let matrix = ktensor::math::Matrix::new(ktensor::math::Vec2(6, 6), (0..36).collect());
    /// let indicies = matrix.get_indicies_stride(ktensor::math::Vec2(2, 2), 2);
    /// let matrix = matrix.to_flattened();
    /// assert_eq!(indicies.len(), 4);
    /// for &i in indicies.iter() {
    ///     assert_eq!(i, matrix[i]);
    /// }
    /// ```
    pub fn get_indicies_stride(&self, Vec2(x, y): Vec2, stride: usize) -> Vec<usize> {
        assert!(x <= self.dim.0/stride);
        assert!(y <= self.dim.1/stride);
        self.get_indicies(Vec2(x, y)).iter().map(|&i| i*stride).collect()
    }
}

impl <T> Matrix<T> where T: Copy {
    /// Returns value at `Vec2`
    ///
    /// # Arguments
    ///
    /// - `Vec2` - coordinates of the value in the `Matrix`
    ///
    /// # Example
    ///
    /// ```
    /// let matrix = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).collect());
    /// assert_eq!(matrix.get(ktensor::math::Vec2(1, 2)), 5);
    /// assert_eq!(matrix.get(ktensor::math::Vec2(1, 2)), 5);
    /// ```
    pub fn get(&self, Vec2(x, y): Vec2) -> T {
        self.buffer[x * self.vec_length.0 + y * self.vec_length.1]
    }

    /// Returns Matrix of values from [vec1 to vec2) (inclusive and exclusive)
    ///
    /// # Arguments
    ///
    /// - `Vec2` - coordinates of the first corner
    /// - `Vec2` - coordinates of the second corner
    ///
    /// # Example
    ///
    /// ```
    /// let matrix = ktensor::math::Matrix::new(ktensor::math::Vec2(3, 3), (0..9).collect());
    /// let slice = matrix.get_submatrix(ktensor::math::Vec2(1, 1), ktensor::math::Vec2(3, 3)).to_flattened();
    /// let result = vec![4, 5, 7, 8];
    /// for (&i, &j) in slice.iter().zip(result.iter()) {
    ///     assert_eq!(i, j);
    /// }
    /// ```
    pub fn get_submatrix(&self, Vec2(x1, y1): Vec2, Vec2(x2, y2): Vec2) -> Matrix<T> {
        assert!(x1 < x2);
        assert!(y1 < y2);
        let mut buf = Vec::<T>::with_capacity((x2-x1) * (y2-y1));
        for i in x1..x2 {
            for j in y1..y2 {
                buf.push(self.get(Vec2(i, j)));
            }
        }

        Matrix {
            dim: Vec2(x2-x1, y2-y1),
            vec_length: Vec2(y2-y1, 1),
            buffer: buf,
        }
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
    /// let matrix1 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).collect());
    /// let matrix2 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).rev().collect());
    /// let matrix3 = matrix1 + matrix2;
    /// assert_eq!(matrix3.get(ktensor::math::Vec2(0, 0)), 5.0);
    /// ```
    fn add(self, rhs: Matrix<T>) -> Matrix<T> {
        assert_eq!(self.dim.0, rhs.dim.0);
        assert_eq!(self.dim.1, rhs.dim.1);
        let mut buffer = Vec::with_capacity(self.len());
        for i in 0..self.dim.0 {
            for j in 0..self.dim.1 {
                buffer.push(self.get(Vec2(i, j)) + rhs.get(Vec2(i, j)));
            }
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
    /// let matrix1 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).collect());
    /// let matrix2 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).rev().collect());
    /// let matrix3 = &matrix1 + &matrix2;
    /// assert_eq!(matrix1.get(ktensor::math::Vec2(0, 0)), 0.0);
    /// assert_eq!(matrix2.get(ktensor::math::Vec2(0, 0)), 5.0);
    /// assert_eq!(matrix3.get(ktensor::math::Vec2(0, 0)), 5.0);
    /// ```
    fn add(self, rhs: &'b Matrix<T>) -> Matrix<T> {
        assert_eq!(self.dim.0, rhs.dim.0);
        assert_eq!(self.dim.1, rhs.dim.1);
        let mut buffer = Vec::with_capacity(self.len());
        for i in 0..self.dim.0 {
            for j in 0..self.dim.1 {
                buffer.push(self.get(Vec2(i, j)) + rhs.get(Vec2(i, j)));
            }
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
    /// let matrix1 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).collect());
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
    /// let matrix1 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).collect());
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
    /// let matrix1 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).collect());
    /// let matrix2 = ktensor::math::Matrix::new(ktensor::math::Vec2(3, 2), (0..6).map(|i| i as f64).rev().collect());
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
    /// let matrix1 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).collect());
    /// let matrix2 = ktensor::math::Matrix::new(ktensor::math::Vec2(3, 2), (0..6).map(|i| i as f64).rev().collect());
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
    /// let matrix1 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).collect());
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
    /// let matrix1 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).collect());
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


///////////////////////
// Hadamard Product  //
///////////////////////

impl <'a, T> Matrix<T> where T: Mul<Output=T> + Copy {
    /// Hadamard Product of Matricies by reference
    ///
    /// # Arguments
    ///
    /// - `self` - this matrix reference
    /// - `rhs` - another matrix reference
    ///
    /// # Example
    ///
    /// ```
    /// let matrix1 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).collect());
    /// let matrix2 = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), (0..6).map(|i| i as f64).rev().collect());
    /// let matrix3 = &matrix1.product(&matrix2);
    /// assert_eq!(matrix1.get(ktensor::math::Vec2(0, 2)), 2.0);
    /// assert_eq!(matrix2.get(ktensor::math::Vec2(0, 2)), 3.0);
    /// assert_eq!(matrix3.get(ktensor::math::Vec2(0, 2)), 6.0);
    /// ```
    pub fn product(&self, rhs: &'a Matrix<T>) -> Matrix<T> {
        assert_eq!(self.dim.0, rhs.dim.0);
        assert_eq!(self.dim.1, rhs.dim.1);
        let mut buffer = Vec::with_capacity(self.len());
        for i in 0..self.dim.0 {
            for j in 0..self.dim.1 {
                buffer.push(self.get(Vec2(i, j)) * rhs.get(Vec2(i, j)));
            }
        }
        Matrix::new(self.dim, buffer)
    }
}
