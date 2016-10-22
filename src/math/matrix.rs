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
    /// - `dimensions` - dimensions of `Matrix`
    /// - `buffer` - `Vec` of values
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

    /// Returns the total number of items in the `Matrix`
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

impl <T> Matrix<T> where T: Copy {
    /// Gives ownership to the buffer
    ///
    /// # Example
    ///
    /// ```
    /// let vector = ktensor::math::Matrix::new(ktensor::math::Vec2(2, 3), vec![0, 1, 2, 3, 4, 5]).to_flattened();
    /// let result = vec![0, 1, 2, 3, 4, 5];
    /// for (&i, &j) in vector.iter().zip(result.iter()) {
    ///     assert_eq!(i, j);
    /// }
    /// ```
    pub fn to_flattened(self) -> Vec<T> {
        self.buffer
    }

    /// Returns value at `Vec2`
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
    /// let matrix = ktensor::math::Matrix::new(ktensor::math::Vec2(3, 3), vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
    /// let slice = matrix.get_slice(ktensor::math::Vec2(1, 1), ktensor::math::Vec2(3, 3)).to_flattened();
    /// let result = vec![4, 5, 7, 8];
    /// for (&i, &j) in slice.iter().zip(result.iter()) {
    ///     assert_eq!(i, j);
    /// }
    /// ```
    pub fn get_slice(&self, Vec2(x1, y1): Vec2, Vec2(x2, y2): Vec2) -> Matrix<T> {
        assert!(x1 < x2);
        assert!(y1 < y2);
        let mut buf = Vec::<T>::with_capacity((x2-x1) * (y2-y1));
        for i in x1..x2 {
            for j in y1..y2 {
                buf.push(self.buffer[i * self.dim.1 + j]);
            }
        }

        Matrix {
            dim: Vec2(x2-x1, y2-y1),
            buffer: buf
        }
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
    /// let matrix = ktensor::math::Matrix::new(ktensor::math::Vec2(3, 3), vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
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
                buf.push(i * self.dim.1 + j);
            }
        }
        buf
    }

    /// Returns `Vec` of indicies across the entire `Matrix` with the specified stride length
    ///
    /// # Arguments
    ///
    /// - `stride` - length of stride
    ///
    /// # Example
    ///
    /// ```
    /// let matrix = ktensor::math::Matrix::new(ktensor::math::Vec2(4, 4), vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    /// let indicies = matrix.get_indicies_stride(2);
    /// let matrix = matrix.to_flattened();
    /// for &i in indicies.iter() {
    ///     assert_eq!(i, matrix[i]);
    /// }
    /// ```
    pub fn get_indicies_stride(&self, stride: usize) -> Vec<usize> {
        self.get_indicies(Vec2(self.dim.0/stride, self.dim.1/stride)).iter().map(|&i| i*stride).collect()
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
