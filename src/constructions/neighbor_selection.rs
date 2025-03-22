//! Vamana graph construction

use std::{collections::{HashMap, HashSet}, sync::Arc};

use crate::{
    data_handling::{dataset::VectorDataset, dataset_traits::Dataset},
    graph::IndexT,
};

use rayon::{prelude::*, result};

/// robust prune without a degree bound
pub fn robust_prune_unbounded(
    mut candidates: Vec<(IndexT, f32)>,
    alpha: f32,
    dataset: &dyn Dataset<f32>,
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

pub struct PairwiseDistancesHandler {
    id_distance_pairs: Vec<Box<[(IndexT, f32)]>>,
}

impl PairwiseDistancesHandler {
    pub fn new(id_distance_pairs: Box<[Box<[(IndexT, f32)]>]>) -> Self {
        // sorting the constituent lists
        let id_distance_pairs: Vec<Box<[(IndexT, f32)]>> = id_distance_pairs
            .iter()
            .map(|x| {
                let mut x = x.to_vec();
                x.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                x.into_boxed_slice()
            })
            .collect();

        PairwiseDistancesHandler {
            id_distance_pairs,
        }
    }

    /// returns a box with the ids of the candidates that are within distance d of i
    pub fn closer_than(&self, i: IndexT, d: f32) -> Box<[IndexT]> {
        let mut result = Vec::new();
        
        let last_index_covered = self.id_distance_pairs[i as usize]
            .binary_search_by(|x| {
                if x.1 < d {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            })
            .unwrap_or_else(|_| self.id_distance_pairs[i as usize].len());

        for j in 0..last_index_covered {
            result.push(self.id_distance_pairs[i as usize][j].0);
        }
        result.into_boxed_slice()
    }
}

/// returns the set of candidates that would be pruned by each point with a given alpha
/// 
/// In the current implementation, this requires materializing the sorted distance matrix.
/// might be worth making a version of this where the candidates and universe are distinct
fn materialize_alpha_sets(
    center: IndexT,
    candidates: &[IndexT],
    alpha: f32,
    dataset: &dyn Dataset<f32>,
    pairwise_distances: &PairwiseDistancesHandler,
) -> Vec<(IndexT, HashSet<IndexT>)> {
    let mut alpha_sets: HashMap<IndexT, HashSet<IndexT>> = HashMap::new();
    
    for i in candidates.iter() {
        alpha_sets.insert(*i, HashSet::new());
    }

    for p in candidates.iter() {
        let dist = dataset.compare_internal(center as usize, *p as usize) as f32;
        let memberships = pairwise_distances.closer_than(*p, dist / alpha);
        for j in memberships.iter() {
            if *j != center {
                alpha_sets.get_mut(j).unwrap().insert(*p);
            }
        }
    }

    alpha_sets.
        into_iter()
        .map(|(i, set)| (i, set))
        .collect::<Vec<(IndexT, HashSet<IndexT>)>>()
}

unsafe impl Sync for PairwiseDistancesHandler {}

/// greedily approximates the set cover instance
/// 
/// does not update the alpha sets between iterations, but does not add a candidate if it has already been covered
pub fn naive_semi_greedy_prune(
    center: IndexT,
    candidates: &[IndexT],
    dataset: &dyn Dataset<f32>,
    alpha: f32,
    pairwise_distances: &PairwiseDistancesHandler,
) -> Vec<IndexT> {
    let mut new_neighbors: Vec<IndexT> = Vec::new();

    let mut alpha_sets = materialize_alpha_sets(center, candidates, alpha, dataset, pairwise_distances);

    // Sort alpha sets by size ascending
    alpha_sets.sort_by(|a, b| a.1.len().partial_cmp(&b.1.len()).unwrap());

    while let Some((i, covered_set)) = alpha_sets.pop() {
        // add the candidate to the new neighbors
        new_neighbors.push(i);

        // remove all candidates that are covered by the new neighbor
        alpha_sets.retain(|(j, _)| {
            !covered_set.contains(j)
        });
    }

    new_neighbors
}
