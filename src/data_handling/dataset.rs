//! definitions and implementations of datasets

use std::{iter, ops::Sub};

use crate::distance::euclidean::{self, euclidean};

use super::dataset_traits::{Numeric, Dataset};

impl<T: Numeric> Dataset<T> for VectorDataset<T> {
    fn compare_internal(&self, i: usize, j: usize) -> f64 {
        self.compare_euclidean(i, j)
    }
    fn compare(&self, q: &[T], i: usize) -> f64 {
        euclidean(q, self.get(i), self.dim)
    }
    fn size(&self) -> usize {
        self.n
    }
}

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

// impl<T:Numeric> Iterator for VectorDataset<T> {
//     type Item = &[T];
// }