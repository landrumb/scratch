use rand::seq::IndexedRandom;
use std::collections::HashSet;

use crate::data_handling::dataset::VectorDataset;
use crate::data_handling::dataset_traits::{Dataset, Numeric};
use crate::distance::SqEuclidean;
use crate::util::DSU;

use rayon::{join, prelude::*};

const EXHAUSTIVE_CUTOFF: usize = 1000;

/// Finds duplicate vectors in a dataset and returns the sets of ids for each group.
/// Each set contains the indices of vectors that are identical.
pub fn duplicate_sets<T>(dataset: &VectorDataset<T>, radius: Option<f64>) -> Vec<HashSet<usize>>
where
    T: Numeric + SqEuclidean,
{
    //! we use recursive parallel bisection to winnow down the search space;
    //! once we have < 100 vectors or we do a partition that doesn't reduce the number of vectors,
    //! we use a hash map to find duplicates.
    //! doing this instead of lsh because we want to be able to find approximate duplicates
    let indices: Vec<usize> = (0..dataset.n).collect();
    let radius = radius.unwrap_or(0.000000001); // default radius for approximate equality
    subset_duplicates(dataset, indices, radius)
}

/// recursive helper function to find duplicates in a subset of the dataset
fn subset_duplicates<T>(
    dataset: &VectorDataset<T>,
    indices: Vec<usize>,
    radius: f64,
) -> Vec<HashSet<usize>>
where
    T: Numeric + SqEuclidean,
{
    if indices.len() < EXHAUSTIVE_CUTOFF {
        return exhaustive_subset_duplicates(dataset, indices, radius);
    }
    let mut rng = rand::rng();

    // splitting points
    // sampling w/o replacement
    let sample = indices
        .choose_multiple(&mut rng, 2)
        .cloned()
        .collect::<Vec<_>>();
    let left_point = sample[0];
    let right_point = sample[1];

    let partialities: Vec<f64> = indices
        .par_iter()
        .map(|&i| {
            dataset.compare_internal(i, left_point) - dataset.compare_internal(i, right_point)
        })
        .collect();

    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();
    for (i, &index) in indices.iter().enumerate() {
        if partialities[i] < 0.0 {
            left_indices.push(index);
        } else {
            right_indices.push(index);
        }
    }

    // if that accomplished nothing, we just do exhaustive search
    if left_indices.len() == indices.len() || right_indices.len() == indices.len() {
        return exhaustive_subset_duplicates(dataset, indices, radius);
    }

    let (mut left_sets, right_sets) = join(
        || subset_duplicates(dataset, left_indices, radius),
        || subset_duplicates(dataset, right_indices, radius),
    );

    left_sets.extend(right_sets);
    left_sets
}

/// this is potentially approximate, but assumes equality is transitive, so might be wonky
fn exhaustive_subset_duplicates<T>(
    dataset: &VectorDataset<T>,
    indices: Vec<usize>,
    radius: f64,
) -> Vec<HashSet<usize>>
where
    T: Numeric + SqEuclidean,
{
    let mut equality_dsu = DSU::new(indices.len());
    for (i_index, &i) in indices.iter().enumerate() {
        for (j_index, &j) in indices.iter().enumerate().skip(i_index + 1) {
            if dataset.compare_internal(i, j) < radius {
                equality_dsu.union(i_index, j_index);
            }
        }
    }

    equality_dsu
        .components()
        .into_iter()
        .filter(|set| set.len() > 1)
        .map(|set| set.into_iter().map(|i| indices[i]).collect::<HashSet<_>>())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_duplicate_sets() {
        let data: Vec<f32> = vec![1.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.5, 0.5, 2.0, 1.0];
        let dataset = VectorDataset::new(data.into_boxed_slice(), 5, 2);
        let sets = duplicate_sets(&dataset, None);
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
        let sets = duplicate_sets(&dataset, None);
        assert_eq!(sets.len(), 2);
        for set in sets {
            assert!(
                set == [0usize, 2].iter().cloned().collect::<HashSet<_>>()
                    || set == [1usize, 3].iter().cloned().collect::<HashSet<_>>()
            );
        }
    }
}
