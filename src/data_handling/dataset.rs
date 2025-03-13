//! definitions and implementations of datasets

use std::{ops::Sub, path::Path};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::distance::euclidean;

use super::dataset_traits::{Dataset, Numeric};

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

    /// Loads a dataset from an fbin file
    pub fn from_file(path: &Path) -> std::io::Result<VectorDataset<T>> {
        Ok(super::fbin::read_fbin(path))
    }

    pub fn get(&self, i: usize) -> &[T] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    pub fn compare_euclidean(&self, i: usize, j: usize) -> f64 {
        euclidean(&self.get(i), &self.get(j), self.dim)
    }

    /// returns id, distance pairs for a subset of the dataset relative to a query
    pub fn brute_force(&self, query: &[T], subset: &[usize]) -> Box<[(usize, f32)]> {
        let mut results: Vec<(usize, f32)> = subset
            .par_iter()
            .map(|i| (*i, self.compare(query, *i) as f32))
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.into_boxed_slice()
    }

    /// returns the index (within the arg slice) of the closest of a set of indices to a query
    pub fn closest(&self, query: &[T], candidates: &[usize]) -> usize {
        if candidates.is_empty() {
            return 0;
        }

        candidates
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, &b)| {
                let dist_a = self.compare(query, a);
                let dist_b = self.compare(query, b);
                dist_a
                    .partial_cmp(&dist_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
}

// impl<T:Numeric> Iterator for VectorDataset<T> {
//     type Item = &[T];
// }
