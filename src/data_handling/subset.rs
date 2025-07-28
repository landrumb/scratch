//! Dataset that's a subset of another dataset

use crate::distance::SqEuclidean;

use super::dataset_traits::{Dataset, Numeric};

/// Important to note that indices here are relative to the subset, not the original dataset.
pub struct Subset<T> {
    dataset: Box<dyn Dataset<T>>,
    indices: Vec<usize>,
}

impl<T: Numeric + SqEuclidean> Subset<T> {
    pub fn new(dataset: Box<dyn Dataset<T>>, indices: Vec<usize>) -> Self {
        Subset { dataset, indices }
    }

    // Access the original dataset indices
    pub fn get_original_index(&self, subset_idx: usize) -> usize {
        self.indices[subset_idx]
    }

    // Get a vector from the underlying dataset
    pub fn get(&self, i: usize) -> &[T] {
        // Get the original dataset index and forward the call
        let orig_idx = self.indices[i];
        // ideally we would have a get method on the dataset trait
        // but for now we can just cast the dataset to vector dataset and hope for the best
        unsafe {
            let vector_dataset =
                self.dataset.as_ref() as *const _ as *const super::dataset::VectorDataset<T>;
            (*vector_dataset).get(orig_idx)
        }
    }
}

impl<T: Numeric> Dataset<T> for Subset<T> {
    fn compare_internal(&self, i: usize, j: usize) -> f64 {
        self.dataset
            .compare_internal(self.indices[i], self.indices[j])
    }

    fn compare(&self, q: &[T], i: usize) -> f64 {
        self.dataset.compare(q, self.indices[i])
    }

    fn size(&self) -> usize {
        self.indices.len()
    }
    fn get(&self, i: usize) -> &[T] {
        let orig_idx = self.indices[i];
        self.dataset.get(orig_idx)
    }
}
