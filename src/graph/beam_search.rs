//! Implementation of beam search over a generic graph

use crate::data_handling::dataset_traits::Dataset;
use crate::graph::{Graph, IndexT};

type BeamElement = (f64, IndexT, bool);

fn beam_insert<T>(
    i: IndexT,
    beam: &mut Vec<BeamElement>,
    query: &[T],
    dataset: &dyn Dataset<T>,
    beam_width: usize,
) {
    // put new element on the end of the beam
    beam.push((dataset.compare(query, i as usize), i, false));

    if beam.len() <= beam_width {
        return;
    }

    // sort the beam
    beam.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

    // drop the last element
    beam.pop();
}

fn first_unvisited_element(beam: &Vec<BeamElement>) -> Option<usize> {
    (0..beam.len()).find(|&i| !beam[i].2)
}

/// This is a bare-bones beam search.
///
/// Functionality like returning the visited list is omitted here
pub fn beam_search<T>(
    query: &[T],
    graph: &dyn Graph,
    dataset: &dyn Dataset<T>,
    start: IndexT,
    beam_width: usize,
) -> Vec<IndexT> {
    // initialize the beam and visited list
    // beam elements are (distance, index, visited)
    let mut beam = Vec::<BeamElement>::with_capacity(beam_width + 1);
    // let mut visited = Vec::<IndexT>::new();

    // insert the starting point in the beam
    beam_insert(start, &mut beam, query, dataset, beam_width);

    // while there exists an unvisited element of the beam, visit the best and update the beam
    let mut next_element = first_unvisited_element(&beam);

    while next_element.is_some() {
        let current = beam.get(next_element.unwrap()).unwrap().1;
        // visited.push(current.clone());
        beam[next_element.unwrap()].2 = true;

        // insert all the neighbors in a cartoonishly inefficient way
        for neighbor in graph.neighbors(current) {
            beam_insert(*neighbor, &mut beam, query, dataset, beam_width);
        }

        next_element = first_unvisited_element(&beam)
    }

    // return indices of beam elements
    beam.iter().map(|x| x.1).collect()
}
