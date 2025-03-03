//! definitions and implementations of datasets

use std::ops::Sub;

use crate::distance::euclidean;

pub trait Numeric: Copy + Into<f64> + Sub<Output = Self> + Default {}

impl Numeric for f32 {}
impl Numeric for f64 {}
impl Numeric for i32 {}
impl Numeric for i8 {}

pub struct VectorDataset<T: Numeric> {
    data: Box<[T]>,
    pub n: usize,
    pub dim: usize,
}

impl<T: Numeric> VectorDataset<T>
where
    T: Copy + Into<f64> + Sub<Output = T>,
{
    pub fn new(data: Box<[T]>, n: usize, dim: usize) -> VectorDataset<T> {
        assert!(
            data.len() == n * dim,
            "expected {} elements for a {}x{} dataset, got {}",
            n * dim,
            n,
            dim,
            data.len()
        );

        VectorDataset { data, n, dim }
    }

    pub fn get(&self, i: usize) -> &[T] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    pub fn compare_euclidean(&self, i: usize, j: usize) -> f64 {
        euclidean::euclidean(&self.get(i), &self.get(j), self.dim)
    }
}
