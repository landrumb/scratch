//! Vamana graph construction

use std::iter::once;

use crate::{data_handling::dataset_traits::Dataset, graph::{beam_search_with_visited, IndexT, MutableGraph, VectorGraph}};

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::num_traits::ToPrimitive;


use super::neighbor_selection::robust_prune;

// const MAX_BATCH_SIZE: usize = 10000;
// const MAX_BATCH_FRACTION: usize = 10; // reciprocal of the fraction
// const KERNEL_SIZE: usize = 1000; // size of the initial batch of points

pub fn build_vamana_graph<T>(
    dataset: &dyn Dataset<T>,
    alpha: f32,
    degree_bound: usize,
    beam_width: usize,
    limit: Option<usize>,
    root: Option<IndexT>,
) -> VectorGraph {
    let mut neighborhoods: Vec<Vec<IndexT>> = Vec::with_capacity(dataset.size());

    for _ in 0..dataset.size() {
        neighborhoods.push(Vec::with_capacity(degree_bound));
    }

    let mut graph = VectorGraph::new(neighborhoods);

    // if root is not specified, we use the first point in the dataset
    let root = root.unwrap_or(0);

    // randomized insertion order which does not include the root. Counterintuitively,
    // the insertion order goes backwards to facilitate breaking it off
    let mut insertion_order = (0..root).chain((root + 1)..dataset.size() as IndexT).collect::<Vec<_>>();
    let mut rng = SmallRng::seed_from_u64(41901); // my birthday
    insertion_order.shuffle(&mut rng);

    // let kernel_size = min(KERNEL_SIZE, dataset.size() - 1 as usize);
    // just doing the whole thing serially for now
    let kernel_size = dataset.size() - 1_usize;

    for i in insertion_order[0..kernel_size].iter() {
        let (_, visited) = beam_search_with_visited(dataset.get(*i as usize), &graph, dataset, root, beam_width, limit);

        // println!("point 0 edges: {:?}", graph.get_neighborhood(0));

        // update neighborhood 
        let new_neighborhood = robust_prune(visited, alpha, dataset, degree_bound);

        graph.set_neighborhood(*i, &new_neighborhood);

        // undirect edges
        for j in new_neighborhood {
            let head_neighborhood = graph.get_neighborhood(j);
            if head_neighborhood.contains(i) {
                continue;
            } else if head_neighborhood.len() < degree_bound {
                graph.add_neighbor(j, *i);
            } else {
                // println!("pruning neighborhood of {}", j);
                // prune the neighborhood of j
                let candidates_with_distances = head_neighborhood
                    .iter()
                    .chain(once(i))
                    .map(|&x| (x, dataset.compare_internal(j as usize, x as usize).to_f32().unwrap()))
                    .collect::<Vec<_>>();

                
                let new_head_neighborhood = robust_prune(candidates_with_distances, alpha, dataset, degree_bound);
                graph.set_neighborhood(j, &new_head_neighborhood);
            }
            
        }
    }

    // if kernel_size == dataset.size() - 1 { // if we have already inserted all points
    //     return graph;
    // }
    // println!("kernel size: {}", kernel_size);
    // let batch_size = min(insertion_order.len() / MAX_BATCH_FRACTION, MAX_BATCH_SIZE);

    // insertion_order = insertion_order.split_off(min(KERNEL_SIZE, dataset.size() - 1 as usize));


    // // after building the kernel, we start inserting points in batches
    // while let Some(batch_indices) = insertion_order.iter().chunks(batch_size).into_iter().next() {
    //     let batch_indices = batch_indices.collect::<Vec<_>>();

    //     let _ = batch_indices
    //         .par_iter()
    //         .map(|&i| {
    //             let (_, visited) = beam_search_with_visited(dataset.get(*i as usize), &graph, dataset, root, beam_width, limit);
    //             let new_neighbors = robust_prune(visited, alpha, dataset, degree_bound);
    //             // queue undirected edges
    //             for j in new_neighbors.iter() {
    //                 graph.queue_edge(*j, *i);
    //             }
    //             graph.bulk_queue(*i, &new_neighbors);
    //         });
        
    //     // process queued edges
    //     graph.preprocess_queues(degree_bound);
    //     let queued_edges = graph.get_queued_edges();
    //     let new_neighborhoods = queued_edges
    //         .par_iter()
    //         .map(|(i, queue)| {
    //             let candidates_with_distances = queue
    //                 .iter()
    //                 .chain(graph.get_neighborhood(*i))
    //                 .map(|&j| (j, dataset.compare_internal(*i as usize, j as usize).to_f32().unwrap()))
    //                 .collect::<Vec<_>>();
    //             (i, robust_prune(candidates_with_distances, alpha, dataset, degree_bound))
    //         })
    //         .collect::<Vec<_>>();
    //     // the actual updating of the neighborhoods could be done in parallel but I don't know how to do it in rust
    //     for (i, new_neighborhood) in new_neighborhoods {
    //         graph.set_neighborhood(*i, &new_neighborhood);
    //     }
    // }

    graph
}