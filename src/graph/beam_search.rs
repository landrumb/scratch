//! Implementation of beam search over a generic graph

use rand_distr::num_traits::ToPrimitive;

use crate::data_handling::dataset_traits::Dataset;
use crate::graph::{Graph, IndexT};
use std::collections::HashSet;


pub fn beam_search_with_visited<T>(
    query: &[T],
    graph: &dyn Graph,
    dataset: &dyn Dataset<T>,
    start: IndexT,
    beam_width: usize,
    limit: Option<usize>,
) -> (Vec<(IndexT, f32)>, Vec<(IndexT, f32)>) {
    let mut frontier: Vec<(IndexT, f32)> = vec![(start, dataset.compare(query, start as usize).to_f32().unwrap())];
    frontier.reserve(beam_width);

    let mut seen: HashSet<IndexT> = HashSet::new();
    seen.insert(start);
    let mut visited: Vec<(IndexT, f32)> = vec![];

    while let Some(current) = frontier.iter().find(|x| !visited.contains(x)) {
        visited.push(*current);
        let neighbors = graph.neighbors(current.0);

        for &neighbor in neighbors {
            if seen.insert(neighbor) { // true if not already seen
                let dist = dataset.compare(query, neighbor as usize).to_f32().unwrap();
                frontier.push((neighbor, dist));
            }
        }

        // Sort the frontier by distance
        frontier.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Keep only the best beam_width elements
        if frontier.len() > beam_width {
            frontier.truncate(beam_width);
        }

        // Check for limit
        if let Some(l) = limit {
            if visited.len() >= l {
                break;
            }
        }
    }

    (frontier, visited)
}

pub fn beam_search<T>(
    query: &[T],
    graph: &dyn Graph,
    dataset: &dyn Dataset<T>,
    start: IndexT,
    beam_width: usize,
    limit: Option<usize>,
) -> Vec<IndexT> {
    let (frontier, _visited) = beam_search_with_visited(query, graph, dataset, start, beam_width, limit);
    frontier.into_iter().map(|(id, _)| id).collect()
}