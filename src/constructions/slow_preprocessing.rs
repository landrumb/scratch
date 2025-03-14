//! constructs the slow preprocessing vamana graph & variants

use rayon::iter::IntoParallelIterator;

use crate::constructions::vamana::robust_prune_unbounded;
use crate::data_handling::dataset::VectorDataset;
use crate::graph::{IndexT, VectorGraph};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::prelude::*;

/// constructs the slow preprocessing version of a vamana graph
pub fn build_slow_preprocesssing(dataset: &VectorDataset<f32>, alpha: f32) -> VectorGraph {
    let pb = ProgressBar::new(dataset.n as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {wide_bar:green/gray} {pos}/{len} [{elapsed_precise}]({eta})")
            .unwrap()
            .progress_chars("█▓░"),
    );
    pb.set_message("Building graph");

    let neighborhoods: Vec<Vec<IndexT>> = (0..dataset.n)
        .into_par_iter()
        .progress_with(pb)
        .map(|i| {
            let point = dataset.get(i);
            let candidates = dataset
                .brute_force_iter(point, (0..i).chain((i + 1)..dataset.n))
                .iter()
                .map(|(j, dist)| (*j as IndexT, *dist))
                .collect::<Vec<(IndexT, f32)>>();

            // Prune remainder of the dataset based on alpha

            robust_prune_unbounded(candidates, alpha, dataset)
        })
        .collect();

    VectorGraph::new(neighborhoods)
}

pub fn build_global_local_graph(
    dataset: &VectorDataset<f32>,
    edge_selection_fn: fn(
        candidates: &[(IndexT, f32)],
        dataset: &VectorDataset<f32>,
    ) -> Vec<IndexT>,
) -> VectorGraph {
    let pb = ProgressBar::new(dataset.n as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {wide_bar:.green/gray} {pos}/{len} [{elapsed_precise}]({eta})")
            .unwrap()
            .progress_chars("█▓░"),
    );
    pb.set_message("Building graph");

    let neighborhoods: Vec<Vec<IndexT>> = (0..dataset.n)
        .into_par_iter()
        .progress_with(pb)
        .map(|i| {
            let point = dataset.get(i);
            let candidates = dataset
                .brute_force_iter(point, (0..i).chain((i + 1)..dataset.n))
                .iter()
                .map(|(j, dist)| (*j as IndexT, *dist))
                .collect::<Vec<(IndexT, f32)>>();

            edge_selection_fn(&candidates, dataset)
        })
        .collect();

    VectorGraph::new(neighborhoods)
}


