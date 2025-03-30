//! Vamana graph construction

use std::collections::{HashMap, HashSet};

use crate::{
    data_handling::dataset_traits::Dataset,
    graph::IndexT,
};

/// robust prune without a degree bound
pub fn robust_prune_unbounded<T>(
    mut candidates: Vec<(IndexT, f32)>,
    alpha: f32,
    dataset: &dyn Dataset<T>,
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

/// robust prune with a degree bound
pub fn robust_prune(
    mut candidates: Vec<(IndexT, f32)>,
    alpha: f32,
    dataset: &dyn Dataset<f32>,
    degree_bound: usize,
) -> Vec<IndexT> {
    let mut new_neighbors: Vec<IndexT> = Vec::new();

    // Sort candidates by distance, descending
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    while let Some((n, _)) = candidates.pop() {
        new_neighbors.push(n);
        candidates.retain(|(i, dist)| {
            alpha * dataset.compare_internal(n as usize, *i as usize) as f32 >= *dist
        });

        if new_neighbors.len() >= degree_bound {
            break;
        }
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
        // to debug we'll just walk up the id_distance_pairs
        let mut result = Vec::new();

        for (j, distance) in self.id_distance_pairs[i as usize].iter() {
            if *distance < d {
                result.push(*j);
            } else {
                break; // Since the list is sorted, we can stop early
            }
        }

        result.into_boxed_slice()
    }

    pub fn nearest(&self, i: IndexT) -> (IndexT, f32) {
        // to debug we'll just walk up the id_distance_pairs
        self.id_distance_pairs[i as usize][1].clone()
    }
}

fn brute_force_alpha_set(
    center: IndexT,
    neighbor: IndexT,
    candidates: &[IndexT],
    alpha: f32,
    dataset: &dyn Dataset<f32>,
) -> HashSet<IndexT> {
    let mut alpha_set: HashSet<IndexT> = HashSet::new();

    for j in candidates.iter() {
        let center_dist = dataset.compare_internal(center as usize, *j as usize) as f32;
        let neighbor_dist = dataset.compare_internal(neighbor as usize, *j as usize) as f32;
        if center_dist > neighbor_dist * alpha {
            alpha_set.insert(*j);
        }
    }
    alpha_set
}

/// returns the set of candidates that would be pruned by each point with a given alpha
/// 
/// In the current implementation, this requires materializing the sorted distance matrix.
/// might be worth making a version of this where the candidates and universe are distinct
pub fn materialize_alpha_sets(
    center: IndexT,
    candidates: &[IndexT],
    alpha: f32,
    dataset: &dyn Dataset<f32>,
    pairwise_distances: &PairwiseDistancesHandler,
) -> Vec<(IndexT, HashSet<IndexT>)> {
    let mut alpha_sets: HashMap<IndexT, HashSet<IndexT>> = HashMap::new();

    if candidates.is_empty() {
        return Vec::new();
    }
    
    for i in candidates.iter() {
        alpha_sets.insert(*i, HashSet::new());
    }

    for point_to_cover in candidates.iter() {
        // DEBUG: compare internal is suspect here
        // the distance from 40 to 46 is plainly not so large
        let dist = dataset.compare_internal(center as usize, *point_to_cover as usize) as f32;


        let memberships = pairwise_distances.closer_than(*point_to_cover, dist / alpha).to_vec();
        for would_be_neighbor in memberships.iter() {
            if *would_be_neighbor != center {
                alpha_sets.get_mut(would_be_neighbor).unwrap().insert(*point_to_cover);
            }
        }
    }


    alpha_sets.
        into_iter()
        .collect::<Vec<(IndexT, HashSet<IndexT>)>>()
}

unsafe impl Sync for PairwiseDistancesHandler {}

/// greedily approximates the set cover instance
pub fn naive_semi_greedy_prune(
    center: IndexT,
    candidates: &[IndexT],
    dataset: &dyn Dataset<f32>,
    alpha: f32,
    pairwise_distances: &PairwiseDistancesHandler,
) -> Vec<IndexT> {
    let mut new_neighbors: Vec<IndexT> = Vec::new();

    if candidates.is_empty() {
        return new_neighbors;
    }

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

        // remove already pruned points from candidate lists
        for set in alpha_sets.iter_mut() {
            set.1.retain(|j| !covered_set.contains(j));
        }

        // sort the alpha sets by size ascending again
        alpha_sets.sort_by(|a, b| a.1.len().partial_cmp(&b.1.len()).unwrap());
    }

    new_neighbors
}
