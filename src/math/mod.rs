/// A pair of coordinates
pub struct Vec2(pub usize, pub usize);

/// A structure of values
pub struct Matrix<T> {
    /// 2 dimensional array of rows and columns
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
    /// assert_eq!(*matrix.get(ktensor::math::Vec2(1, 2)), 5);
    /// ```
    pub fn get(&self, Vec2(x, y): Vec2) -> &T {
        &self.buffer[x * self.dim.1 + y]
    }
}
