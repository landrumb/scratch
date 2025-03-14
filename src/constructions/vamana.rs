//! Vamana graph construction

use crate::{
    data_handling::{dataset::VectorDataset, dataset_traits::Dataset},
    graph::IndexT
};

/// robust prune without a degree bound
pub fn robust_prune_unbounded(mut candidates: Vec<(IndexT, f32)>, alpha: f32, dataset: &VectorDataset<f32>) -> Vec<IndexT> {
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