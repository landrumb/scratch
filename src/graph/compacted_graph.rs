use std::{collections::HashMap, sync::Arc};

use crate::{
    data_handling::{
        dataset::VectorDataset,
        dataset_traits::{Dataset, Numeric},
    },
    distance::SqEuclidean,
    graph::{beam_search_with_visited, ClassicGraph, IndexT, VectorGraph},
};

pub struct CompactedGraphIndex<T>
where
    T: Numeric + SqEuclidean + 'static,
{
    internal_graph: ClassicGraph,
    internal_dataset: Arc<dyn Dataset<T>>,
    posting_lists: HashMap<IndexT, Box<[IndexT]>>, // contains an entry for every posting list bigger than one vector, posting list does not contain the representative index
    local_to_input_index: Box<[IndexT]>,
}

impl<T: Numeric + SqEuclidean + 'static> CompactedGraphIndex<T> {
    pub fn graph_size(&self) -> usize {
        self.internal_graph.n as usize
    }

    pub fn beam_search_post_expansion(&self, query: &[T], beam_width: usize) -> Vec<IndexT> {
        let (mut frontier, _) = beam_search_with_visited(
            query,
            &self.internal_graph,
            &*self.internal_dataset,
            0,
            beam_width,
            None,
        );

        // if anything in the beam corresponds to a posting list, try to add the rest of the posting list to the beam
        let candidates: Vec<(IndexT, f32)> = frontier
            .iter()
            .filter_map(|(id, _)| {
                self.posting_lists.get(id).map(|list| {
                    list.iter().map(|&index| {
                        (
                            index,
                            self.internal_dataset.compare(query, index as usize) as f32,
                        )
                    })
                })
            })
            .flatten()
            .collect();

        frontier.extend(candidates);
        frontier.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        frontier.truncate(beam_width);
        frontier
            .iter()
            .map(|(i, _)| self.local_to_input_index[*i as usize])
            .collect()
    }

    /// doesn't do any kind of reordering etc, the secondary points are just added to the representative's posting list and have neighborhoods of length 0. Inbound edges are directed to the representative.
    /// The representative of a posting list is the first element of the list.
    pub fn build_memory_inefficient(
        graph: VectorGraph,
        dataset: Arc<VectorDataset<T>>,
        posting_lists: Box<[Box<[IndexT]>]>,
    ) -> CompactedGraphIndex<T> {
        let mut graph = graph;
        let mut posting_lists = posting_lists;

        let n = graph.n();
        let mut secondary_points_to_representatives: HashMap<IndexT, IndexT> = HashMap::new();

        posting_lists.iter_mut().for_each(|list| list.sort());
        let mut output_posting_lists: HashMap<IndexT, Box<[IndexT]>> = HashMap::new();

        for list in posting_lists {
            let representative = list[0];
            for &index in list.iter().skip(1) {
                secondary_points_to_representatives.insert(index, representative);
                graph.clear_neighborhood(index);
            }
            output_posting_lists.insert(representative, list[1..].to_vec().into_boxed_slice());
        }

        graph.apply_to_neighborhoods(|i, neighborhood| {
            if secondary_points_to_representatives.contains_key(&i) {
                neighborhood.clear();
            } else {
                for index in neighborhood.iter_mut() {
                    if let Some(representative) = secondary_points_to_representatives.get(index) {
                        *index = *representative;
                    }
                }
                neighborhood.sort();
                neighborhood.dedup();
            }
        });

        let internal_graph = graph.into();
        let internal_dataset: Arc<dyn Dataset<T>> = dataset;
        let local_to_input_index: Box<[IndexT]> = (0..n as IndexT).collect();

        CompactedGraphIndex {
            internal_graph,
            internal_dataset,
            posting_lists: output_posting_lists,
            local_to_input_index,
        }
    }

    // /// takes a graph, a dataset, and a set of posting lists, and constructs a CompactedGraphIndex where each posting list is represented in the graph by its lowest index element.
    // /// The neighborhoods in the graph are not changed, besides replacing edges to points which are now subsumed by a posting list with edges to their representative.
    // pub fn build_naive(graph: VectorGraph, dataset: VectorDataset<T>, posting_lists: Box<[Box<[IndexT]>]>) -> CompactedGraphIndex<T> {
    //     let mut graph = graph;
    //     let mut posting_lists = posting_lists;
    //     // we sort the internal posting lists by their lowest element
    //     posting_lists.iter_mut().for_each(|list| list.sort());
    //     posting_lists.sort_by_key(|list| list[0]);

    //     let mut new_posting_lists: Vec<Vec<IndexT>> = Vec::new();

    //     let n = graph.n();
    //     let mut primary_points: Vec<IndexT> = Vec::new();
    //     let mut secondary_points: Vec<IndexT> = Vec::new();
    //     let mut input_to_local_index: Vec<IndexT> = (0..n as IndexT).collect();

    //     for list in posting_lists {
    //         if list.len() == 1 {
    //             primary_points.push(list[0]);
    //             input_to_local_index[list[0] as usize] = primary_points.len() as IndexT - 1;
    //             new_posting_lists.push(Vec::new());
    //             continue;
    //         }

    //         let representative = list[0];
    //         if let Some(last) = primary_points.last() {
    //             // every point between this and the last primary point is either already secondary or doesn't show up in any other posting list
    //             for index in last + 1..=representative {
    //                 if !secondary_points.contains(&index) {
    //                     primary_points.push(index);
    //                     // input_to_local_index[index as usize] = primary_points.len() as IndexT - 1;
    //                     new_posting_lists.push(Vec::new());
    //                 }
    //             }
    //         } else {
    //             primary_points.push(representative);
    //             // input_to_local_index[representative as usize] = primary_points.len() as IndexT - 1;
    //             new_posting_lists.push(Vec::new());
    //         }

    //         for &index in list.iter().skip(1) {
    //             // input_to_local_index[index as usize] = primary_points.len() as IndexT - 1; // point to representative
    //             secondary_points.push(index);
    //             new_posting_lists[primary_points.len() - 1].push(secondary_points.len() as IndexT - 1); // add secondary index to the representative's posting list
    //         }

    //     }

    //     let internal_graph = graph.into();
    //     let internal_dataset = dataset;
    //     let local_to_input_index = graph.n();

    //     let mut local_to_input_index: Vec<IndexT> = (0..n as IndexT).collect();
    //     CompactedGraphIndex {
    //         internal_graph, internal_dataset, posting_lists, local_to_input_index }
    // }
}
