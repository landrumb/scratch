//! Vamana graph construction

use std::collections::HashSet;

use crate::{
    data_handling::{dataset::VectorDataset, dataset_traits::Dataset},
    graph::IndexT,
};

use rayon::prelude::*;

/// robust prune without a degree bound
pub fn robust_prune_unbounded(
    mut candidates: Vec<(IndexT, f32)>,
    alpha: f32,
    dataset: &VectorDataset<f32>,
) -> Vec<IndexT> {
    let mut new_neighbors: Vec<IndexT> = Vec::new();

    // Sort candidates by distance, descending
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    while let Some((n, _)) = candidates.pop() {
        new_neighbors.push(n);
        candidates.retain(|(i, dist)| {
            alpha * dataset.compare_internal(n as usize, *i as usize) as f32 >= *dist
        });
    }

    new_neighbors
}

/// returns the set of candidates that would be pruned by a given point with a given alpha
fn materialize_alpha_set(
    neighbor: IndexT,
    candidates: &[(IndexT, f32)],
    alpha: f32,
    dataset: &VectorDataset<f32>,
) -> Vec<IndexT> {
    let mut alpha_set: Vec<IndexT> = Vec::new();

    for (i, dist) in candidates.iter() {
        if alpha * dataset.compare_internal(neighbor as usize, *i as usize) as f32 >= *dist {
            alpha_set.push(*i);
        }
    }

    alpha_set
}

/// greedily approximates the set cover instance
/// 
/// does not update the alpha sets between iterations, but does not add a candidate if it has already been covered
pub fn naive_semi_greedy_prune(
    candidates: &[(IndexT, f32)],
    dataset: &VectorDataset<f32>,
    alpha: f32,
) -> Vec<IndexT> {
    let mut new_neighbors: Vec<IndexT> = Vec::new();
    // let mut covered: HashSet<IndexT> = HashSet::new();

    let mut alpha_sets = candidates
        .par_iter()
        .map(|(i, _)| (*i, materialize_alpha_set(*i, candidates, alpha, dataset)))
        .collect::<Vec<(IndexT, Vec<IndexT>)>>();

    // Sort alpha sets by size ascending
    alpha_sets.sort_by(|a, b| a.1.len().partial_cmp(&b.1.len()).unwrap());

    while let Some((i, set)) = alpha_sets.pop() {
        // // if the candidate is already covered, skip it
        // if covered.contains(&i) {
        //     continue;
        // }

        // add the candidate to the new neighbors
        new_neighbors.push(i);
        // add the alpha set to the covered set
        // for j in set.iter() {
        //     covered.insert(*j);
        // }

        // remove all candidates that are covered by the new neighbor
        alpha_sets.retain(|(_, set)| {
            !set.iter().any(|j| set.contains(j))
        });
        
    }

    

    new_neighbors
}
