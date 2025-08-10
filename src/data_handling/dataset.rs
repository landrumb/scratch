//! definitions and implementations of datasets

pub use super::distance_matrix::DistanceMatrix;
pub use super::subset::Subset;

use std::path::Path;

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    distance::{euclidean, DenseVector},
    graph::IndexT,
};

use super::dataset_traits::Dataset;

impl<T: DenseVector> Dataset<T> for VectorDataset<T> {
    fn compare_internal(&self, i: usize, j: usize) -> f64 {
        self.compare_euclidean(i, j)
    }
    fn compare(&self, q: &[T], i: usize) -> f64 {
        euclidean(q, self.get(i)) as f64
    }
    fn size(&self) -> usize {
        self.n
    }
    fn get(&self, i: usize) -> &[T] {
        self.get(i)
    }
}

#[derive(Debug, Clone)]
pub struct VectorDataset<T: DenseVector> {
    data: Box<[T]>,
    pub n: usize,
    pub dim: usize,
}

impl<T: DenseVector> VectorDataset<T> {
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
        debug_assert!(i < self.n, "index out of bounds: {} >= {}", i, self.n);

        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    pub fn compare_euclidean(&self, i: usize, j: usize) -> f64 {
        debug_assert!(i < self.n, "index out of bounds: {} >= {}", i, self.n);
        debug_assert!(j < self.n, "index out of bounds: {} >= {}", j, self.n);

        euclidean(self.get(i), self.get(j)) as f64
    }

    /// returns id, distance pairs for a subset of the dataset relative to a query
    pub fn brute_force_boxed_subset(&self, query: &[T], subset: &[usize]) -> Box<[(usize, f32)]> {
        let mut results: Vec<(usize, f32)> = subset
            .par_iter()
            .map(|i| (*i, self.compare(query, *i) as f32))
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.into_boxed_slice()
    }

    /// brute force but the subset is defined by an iterator
    pub fn brute_force(
        &self,
        query: &[T],
        subset: impl Iterator<Item = usize>,
    ) -> Box<[(usize, f32)]> {
        let mut results: Vec<(usize, f32)> =
            subset.map(|i| (i, self.compare(query, i) as f32)).collect();

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

    pub fn subset_copy_iter(&self, subset: impl Iterator<Item = usize>) -> VectorDataset<T> {
        let mut data = Vec::new();
        for i in subset {
            data.extend_from_slice(self.get(i));
        }
        let new_n = data.len() / self.dim;
        VectorDataset::new(data.into_boxed_slice(), new_n, self.dim)
    }

    pub fn write_fbin(&self, path: &Path) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::{BufWriter, Write};

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write dimensions
        writer.write_all(&(self.n as u32).to_le_bytes())?;
        writer.write_all(&(self.dim as u32).to_le_bytes())?;

        // Write data
        for value in &*self.data {
            writer.write_all(unsafe {
                std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>())
            })?;
        }

        Ok(())
    }

    pub fn shuffled_copy(&self, order: &[IndexT]) -> Self {
        let mut new_data: Vec<T> = Vec::new();

        let chunks: Vec<&[T]> = self.data.chunks(self.dim).collect();

        for i in order {
            new_data.extend_from_slice(chunks[*i as usize]);
        }

        Self {
            data: new_data.into_boxed_slice(),
            n: order.len(),
            dim: self.dim,
        }
    }
}

// impl<T:Numeric> Iterator for VectorDataset<T> {
//     type Item = &[T];
// }
