//! Dataset that's a subset of another dataset

use std::sync::Arc;

use crate::distance::SqEuclidean;

use super::dataset_traits::{Dataset, Numeric};

/// Important to note that indices here are relative to the subset, not the original dataset.
pub struct Subset<T> {
    dataset: Arc<dyn Dataset<T>>,
    indices: Vec<usize>,
}

impl<T: Numeric + SqEuclidean> Subset<T> {
    pub fn new(dataset: Arc<dyn Dataset<T>>, indices: Vec<usize>) -> Self {
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

    pub fn further_subset(&self, indices: Vec<usize>) -> Self {
        let mut new_indices = Vec::new();
        for index in indices {
            new_indices.push(self.indices[index]);
        }
        Subset::new(self.dataset.clone(), new_indices)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_handling::dataset::VectorDataset;
    use crate::data_handling::dataset_traits::Dataset;
    use std::sync::Arc;

    fn sample_dataset() -> Arc<dyn Dataset<f32>> {
        let data: Box<[f32]> = vec![
            1.0, 0.0, 0.0, // Vector 0
            0.0, 1.0, 0.0, // Vector 1
            0.0, 0.0, 1.0, // Vector 2
            0.5, 0.5, 0.0, // Vector 3
        ]
        .into_boxed_slice();

        let dataset = VectorDataset::new(data, 4, 3);
        Arc::new(dataset)
    }

    #[test]
    fn subset_get_and_indices() {
        let dataset = sample_dataset();
        let indices = vec![2, 0, 3];
        let subset = Subset::new(dataset.clone(), indices.clone());

        assert_eq!(subset.size(), indices.len());
        assert_eq!(subset.get_original_index(0), 2);
        assert_eq!(subset.get_original_index(1), 0);

        assert_eq!(
            <Subset<f32> as Dataset<f32>>::get(&subset, 0),
            dataset.get(2)
        );
        assert_eq!(
            <Subset<f32> as Dataset<f32>>::get(&subset, 2),
            dataset.get(3)
        );
    }

    #[test]
    fn subset_compare_matches_dataset() {
        let dataset = sample_dataset();
        let indices = vec![2, 0, 3];
        let subset = Subset::new(dataset.clone(), indices);

        let query = dataset.get(1).to_vec();

        assert_eq!(subset.compare(&query, 0), dataset.compare(&query, 2));
        assert_eq!(
            subset.compare_internal(0, 2),
            dataset.compare_internal(2, 3)
        );
    }
}
