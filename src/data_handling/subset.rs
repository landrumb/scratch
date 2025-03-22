//! Dataset that's a subset of another dataset

use super::dataset_traits::{Dataset, Numeric};

/// Important to note that indices here are relative to the subset, not the original dataset.
pub struct Subset<T> {
    dataset: Box<dyn Dataset<T>>,
    indices: Vec<usize>,
}

impl<T: Numeric> Subset<T> {
    pub fn new(dataset: Box<dyn Dataset<T>>, indices: Vec<usize>) -> Self {
        Subset { dataset, indices }
    }
}

impl <T: Numeric> Dataset<T> for Subset<T> {
    fn compare_internal(&self, i: usize, j: usize) -> f64 {
        self.dataset.compare_internal(self.indices[i], self.indices[j])
    }

    fn compare(&self, q: &[T], i: usize) -> f64 {
        self.dataset.compare(q, self.indices[i])
    }

    fn size(&self) -> usize {
        self.indices.len()
    }
}