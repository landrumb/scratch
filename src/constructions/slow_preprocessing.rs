//! constructs the slow preprocessing vamana graph & variants

use rayon::iter::IntoParallelIterator;

use crate::constructions::neighbor_selection::robust_prune_unbounded;
use crate::data_handling::dataset_traits::Dataset;
use crate::graph::{IndexT, VectorGraph};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::prelude::*;


/// constructs the slow preprocessing version of a vamana graph
pub fn build_slow_preprocesssing(dataset: &dyn Dataset<f32>, alpha: f32) -> VectorGraph {
    let pb = ProgressBar::new(dataset.size() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {wide_bar:.green/gray} {pos}/{len} [{elapsed_precise}]({eta})")
            .unwrap()
            .progress_chars("█▓░"),
    );
    pb.set_message("Building graph");

    let neighborhoods: Vec<Vec<IndexT>> = (0..dataset.size())
        .into_par_iter()
        .progress_with(pb)
        .map(|i| {
            let candidates = (0..i).chain((i + 1)..dataset.size())
                .map(|j| {
                    let dist = dataset.compare_internal(i, j);
                    (j as IndexT, dist as f32)
                })
                .collect::<Vec<(IndexT, f32)>>();

            // Prune remainder of the dataset based on alpha

            robust_prune_unbounded(candidates, alpha, dataset)
        })
        .collect();

    VectorGraph::new(neighborhoods)
}

pub fn build_global_local_graph<F>(
    dataset: &dyn Dataset<f32>,
    edge_selection_fn: F,
) -> VectorGraph 
where F: Fn(IndexT, &[IndexT]) -> Vec<IndexT> + Sync 
    {
    let pb = ProgressBar::new(dataset.size() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {wide_bar:.green/gray} {pos}/{len} [{elapsed_precise}]({eta})")
            .unwrap()
            .progress_chars("█▓░"),
    );
    pb.set_message("Building graph");

    let neighborhoods: Vec<Vec<IndexT>> = (0..dataset.size())
        .into_par_iter()
        .progress_with(pb)
        .map(|i| {
            let candidates = (0..i).chain((i + 1)..dataset.size())
                .map(|j| {
                    // let dist = dataset.compare_internal(i, j);
                    // (j as IndexT, dist as f32)
                    j as IndexT
                })
                .collect::<Vec<IndexT>>();

            edge_selection_fn(i as IndexT, &candidates)
        })
        .collect();

    VectorGraph::new(neighborhoods)
}




