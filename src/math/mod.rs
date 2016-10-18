/// A structure of values
pub struct Matrix<T> {
    /// 2 dimensional array of rows and columns
    dim: [usize; 2],
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
    /// let matrix = ktensor::math::Matrix::new([2, 3], vec![0, 1, 2, 3, 4, 5]);
    /// ```
    pub fn new(dimensions: [usize; 2], buffer: Vec<T>) -> Matrix<T> {
        Matrix {
            dim: dimensions,
            buffer: buffer,
        }
    }
}
