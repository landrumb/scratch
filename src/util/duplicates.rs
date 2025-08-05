use rand::seq::IteratorRandom;
use std::collections::HashSet;
use std::sync::Arc;

use crate::data_handling::dataset::{Subset, VectorDataset};
use crate::data_handling::dataset_traits::{Dataset, Numeric};
use crate::distance::SqEuclidean;
use crate::graph::IndexT;
use crate::util::DSU;

use rayon::{join, prelude::*};

const EXHAUSTIVE_CUTOFF: usize = 1000;

/// Finds duplicate vectors in a dataset and returns the sets of ids for each group.
/// Each set contains the indices of vectors that are identical.
pub fn duplicate_sets<T>(dataset: Arc<VectorDataset<T>>, radius: Option<f64>) -> Vec<HashSet<usize>>
where
    T: Numeric + SqEuclidean + 'static,
{
    // we use recursive parallel bisection to winnow down the search space;
    // once we have < 100 vectors or we do a partition that doesn't reduce the number of vectors,
    // we use a hash map to find duplicates.
    // doing this instead of lsh because we want to be able to find approximate duplicates
    let indices: Vec<usize> = (0..dataset.n).collect();
    let radius = radius.unwrap_or(0.000000001); // default radius for approximate equality
    let subset = Subset::new(dataset.clone(), indices);

    subset_duplicates(&subset, radius)
}

/// recursive helper function to find duplicates in a subset of the dataset
pub fn subset_duplicates<T>(subset: &Subset<T>, radius: f64) -> Vec<HashSet<usize>>
where
    T: Numeric + SqEuclidean + 'static,
{
    if subset.size() < EXHAUSTIVE_CUTOFF {
        return exhaustive_subset_duplicates(subset, radius);
    }
    let mut rng = rand::rng();

    // splitting points
    // sampling w/o replacement
    let sample: Vec<IndexT> = (0..subset.size() as IndexT)
        .choose_multiple(&mut rng, 2)
        .into_iter()
        .collect::<Vec<_>>();
    let left_point = sample[0];
    let right_point = sample[1];

    let partialities: Vec<f64> = (0..subset.size() as IndexT)
        .into_par_iter()
        .map(|i| {
            subset.compare_internal(i as usize, left_point as usize)
                - subset.compare_internal(i as usize, right_point as usize)
        })
        .collect();

    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();
    for (i, index) in (0..subset.size() as IndexT).enumerate() {
        if partialities[i] < 0.0 {
            left_indices.push(index as usize);
        } else {
            right_indices.push(index as usize);
        }
    }

    // if that accomplished nothing, we just do exhaustive search
    if left_indices.len() == subset.size() || right_indices.len() == subset.size() {
        return exhaustive_subset_duplicates(subset, radius);
    }

    let (mut left_sets, right_sets) = join(
        || subset_duplicates(&subset.further_subset(left_indices), radius),
        || subset_duplicates(&subset.further_subset(right_indices), radius),
    );

    left_sets.extend(right_sets);
    left_sets
}

/// this is potentially approximate, but assumes equality is transitive, so might be wonky
fn exhaustive_subset_duplicates<T>(subset: &Subset<T>, radius: f64) -> Vec<HashSet<usize>>
where
    T: Numeric + SqEuclidean,
{
    let mut equality_dsu = DSU::new(subset.size());
    for i in 0..subset.size() {
        for j in i + 1..subset.size() {
            if subset.compare_internal(i, j) < radius {
                equality_dsu.union(i, j);
            }
        }
    }

    equality_dsu
        .components()
        .into_iter()
        .filter(|set| set.len() > 1)
        .map(|set| {
            set.into_iter()
                .map(|i| subset.get_original_index(i))
                .collect::<HashSet<_>>()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_duplicate_sets() {
        let data: Vec<f32> = vec![1.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.5, 0.5, 2.0, 1.0];
        let dataset = VectorDataset::new(data.into_boxed_slice(), 5, 2);
        let dataset = Arc::new(dataset);
        let sets = duplicate_sets(dataset, None);
        // Expect two duplicate sets: {0,2} and {1,3}
        assert_eq!(sets.len(), 2);
        for set in sets {
            assert!(
                set == [0usize, 2].iter().cloned().collect::<HashSet<_>>()
                    || set == [1usize, 3].iter().cloned().collect::<HashSet<_>>()
            );
        }
    }

    #[test]
    fn finds_duplicate_sets_generic() {
        let data: Vec<i32> = vec![1, 0, 2, 2, 1, 0, 2, 2, 3, 1];
        let dataset = VectorDataset::new(data.into_boxed_slice(), 5, 2);
        let dataset = Arc::new(dataset);
        let sets = duplicate_sets(dataset, None);
        assert_eq!(sets.len(), 2);
        for set in sets {
            assert!(
                set == [0usize, 2].iter().cloned().collect::<HashSet<_>>()
                    || set == [1usize, 3].iter().cloned().collect::<HashSet<_>>()
            );
        }
    }
}
