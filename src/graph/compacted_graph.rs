use std::{collections::HashMap, sync::Arc};

use crate::{
    constructions::neighbor_selection::robust_prune_unbounded,
    data_handling::{dataset::VectorDataset, dataset_traits::Dataset},
    distance::DenseVector,
    graph::{beam_search_with_visited, ClassicGraph, IndexT, MutableGraph, VectorGraph},
};

pub struct CompactedGraphIndex<T>
where
    T: DenseVector,
{
    internal_graph: ClassicGraph,
    primary_dataset: Arc<dyn Dataset<T>>,
    secondary_dataset: Arc<dyn Dataset<T>>,
    posting_lists: HashMap<IndexT, Box<[IndexT]>>, // contains an entry for every posting list bigger than one vector, posting list does not contain the representative index
    primary_to_input_index: Box<[IndexT]>,
    secondary_to_input_index: Box<[IndexT]>,
}

impl<T: DenseVector> CompactedGraphIndex<T> {
    pub fn graph_size(&self) -> usize {
        self.internal_graph.n as usize
    }

    pub fn beam_search_post_expansion(&self, query: &[T], beam_width: usize) -> Vec<IndexT> {
        let (mut frontier, _) = beam_search_with_visited(
            query,
            &self.internal_graph,
            &*self.primary_dataset,
            0,
            beam_width,
            None,
        );

        frontier
            .iter_mut()
            .for_each(|(index, _)| *index = self.primary_to_input_index[*index as usize]);

        // if anything in the beam corresponds to a posting list, try to add the rest of the posting list to the beam
        let candidates: Vec<(IndexT, f32)> = frontier
            .iter()
            .filter_map(|(id, _)| {
                self.posting_lists.get(id).map(|list| {
                    list.iter().map(|&index| {
                        (
                            self.secondary_to_input_index[index as usize],
                            self.secondary_dataset.compare(query, index as usize) as f32,
                        )
                    })
                })
            })
            .flatten()
            .collect();

        frontier.extend(candidates);
        frontier.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        frontier.truncate(beam_width);
        frontier.iter().map(|(i, _)| *i).collect()
    }

    /// the same as beam_search_post_expansion, but instead of expanding every point in the beam, we expand the points in the visited set
    pub fn beam_search_expand_visited(&self, query: &[T], beam_width: usize) -> Vec<IndexT> {
        let (mut frontier, visited) = beam_search_with_visited(
            query,
            &self.internal_graph,
            &*self.primary_dataset,
            0,
            beam_width,
            None,
        );

        frontier
            .iter_mut()
            .for_each(|(index, _)| *index = self.primary_to_input_index[*index as usize]);

        // if anything in the visited set corresponds to a posting list, try to add the rest of the posting list to the beam
        let candidates: Vec<(IndexT, f32)> = visited
            .iter()
            .filter_map(|(id, _)| {
                self.posting_lists.get(&id).map(|list| {
                    list.iter().map(|&index| {
                        (
                            self.secondary_to_input_index[index as usize],
                            self.secondary_dataset.compare(query, index as usize) as f32,
                        )
                    })
                })
            })
            .flatten()
            .collect();

        frontier.extend(candidates);
        frontier.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        frontier.truncate(beam_width);
        frontier.iter().map(|(i, _)| *i).collect()
    }

    /// beam search but no secondary points are expanded
    pub fn beam_search_primary_points(&self, query: &[T], beam_width: usize) -> Vec<IndexT> {
        let (frontier, _) = beam_search_with_visited(
            query,
            &self.internal_graph,
            &*self.primary_dataset,
            0,
            beam_width,
            None,
        );
        frontier
            .iter()
            .map(|(i, _)| self.primary_to_input_index[*i as usize])
            .collect()
    }

    /// exhaustive search on the primary points
    pub fn exhaustive_search_primary_points(&self, query: &[T]) -> Vec<IndexT> {
        let mut frontier: Vec<(IndexT, f32)> = Vec::new();
        for i in self.primary_points().iter() {
            frontier.push((*i, self.primary_dataset.compare(query, *i as usize) as f32));
        }
        frontier.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        frontier
            .iter()
            .map(|(i, _)| self.primary_to_input_index[*i as usize])
            .collect()
    }

    /// exhaustive search on the secondary points
    pub fn exhaustive_search_secondary_points(&self, query: &[T]) -> Vec<IndexT> {
        let mut frontier: Vec<(IndexT, f32)> = Vec::new();
        for i in self.secondary_points().iter() {
            frontier.push((
                *i,
                self.secondary_dataset.compare(query, *i as usize) as f32,
            ));
        }
        frontier.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        frontier
            .iter()
            .map(|(i, _)| self.secondary_to_input_index[*i as usize])
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

        CompactedGraphIndex {
            internal_graph,
            primary_dataset: dataset.clone(),
            secondary_dataset: dataset.clone(),
            posting_lists: output_posting_lists,
            primary_to_input_index: (0..n as IndexT).collect(),
            secondary_to_input_index: (0..n as IndexT).collect(),
        }
    }

    pub fn build_memory_inefficient_robust_prune(
        graph: VectorGraph,
        dataset: Arc<VectorDataset<T>>,
        posting_lists: Box<[Box<[IndexT]>]>,
        alpha: f32,
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

                let neighbors = graph.get_neighborhood(index);
                graph.bulk_queue(representative, neighbors);

                let queue_neighbors = graph
                    .get_neighborhood_queue(index)
                    .lock()
                    .ok()
                    .unwrap()
                    .to_vec();

                for queued_neighbor in queue_neighbors {
                    // if let Some(other_representative) = secondary_points_to_representatives.get(&queued_neighbor) {
                    //     graph.queue_edge(representative, *other_representative);
                    // } else {
                    graph.queue_edge(representative, queued_neighbor);
                    // }
                }
                graph.clear_neighborhood(index);
            }
            output_posting_lists.insert(representative, list[1..].to_vec().into_boxed_slice());
        }

        // running robust prune here so it uses the secondary points when evaluating edges
        for p in output_posting_lists.keys() {
            let mut candidates = graph
                .get_neighborhood_queue(*p)
                .lock()
                .ok()
                .unwrap()
                .to_vec();

            candidates.extend_from_slice(graph.get_neighborhood(*p));

            candidates.sort();
            candidates.dedup();

            let candidates = candidates
                .iter()
                .map(|i| {
                    (
                        *i,
                        dataset.compare_internal(*i as usize, *p as usize) as f32,
                    )
                })
                .collect();

            let new_neighbors = robust_prune_unbounded(candidates, alpha, &dataset);

            graph.set_neighborhood(*p, new_neighbors.as_slice());
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

        CompactedGraphIndex {
            internal_graph,
            primary_dataset: dataset.clone(),
            secondary_dataset: dataset.clone(),
            posting_lists: output_posting_lists,
            primary_to_input_index: (0..n as IndexT).collect(),
            secondary_to_input_index: (0..n as IndexT).collect(),
        }
    }

    /// returns the indices of the points which are not in any posting list, and have edges in the graph
    pub fn primary_points(&self) -> Vec<IndexT> {
        let mut primary_points: Vec<IndexT> = Vec::new();
        for i in 0..self.graph_size() {
            if self.internal_graph.get_neighborhood(i as u32).len() > 0 {
                primary_points.push(i as IndexT);
            }
        }
        primary_points
    }

    /// returns the indices of the points which are in a posting list
    pub fn secondary_points(&self) -> Vec<IndexT> {
        let mut secondary_points: Vec<IndexT> = Vec::new();
        for (_, list) in self.posting_lists.iter() {
            for &index in list.iter() {
                secondary_points.push(index);
            }
        }
        secondary_points
    }

    pub fn get_posting_lists(&self) -> HashMap<IndexT, Box<[IndexT]>> {
        self.posting_lists.clone()
    }

    // /// takes a graph, a dataset, and a set of posting lists, and constructs a CompactedGraphIndex where each posting list is represented in the graph by its lowest index element.
    // /// The neighborhoods in the graph are not changed, besides replacing edges to points which are now subsumed by a posting list with edges to their representative.
    // pub fn build_with_reordering(
    //     graph: VectorGraph,
    //     dataset: Arc<VectorDataset<T>>,
    //     posting_lists: Box<[Box<[IndexT]>]>,
    // ) -> CompactedGraphIndex<T> {
    //     // 1. Identify primary and secondary points
    //     let n = graph.n();
    //     let mut is_secondary = vec![false; n];
    //     let mut representative_of = vec![None; n];
    //     let mut posting_lists_vec = posting_lists.into_vec();
    //     posting_lists_vec.iter_mut().for_each(|list| list.sort());
    //     posting_lists_vec.sort_by_key(|list| list[0]);

    //     for list in &posting_lists_vec {
    //         let representative = list[0] as usize;
    //         for &idx in list.iter().skip(1) {
    //             is_secondary[idx as usize] = true;
    //             representative_of[idx as usize] = Some(representative as IndexT);
    //         }
    //     }

    //     // 2. Build new order: primary points first, then secondary points
    //     let mut order = Vec::with_capacity(n);
    //     let mut primary_points = Vec::new();
    //     let mut secondary_points = Vec::new();
    //     for i in 0..n {
    //         if !is_secondary[i] {
    //             order.push(i as IndexT);
    //             primary_points.push(i as IndexT);
    //         }
    //     }
    //     for i in 0..n {
    //         if is_secondary[i] {
    //             order.push(i as IndexT);
    //             secondary_points.push(i as IndexT);
    //         }
    //     }
    //     // Map: original index -> new index
    //     let mut orig_to_new = vec![0; n];
    //     for (new_idx, &orig_idx) in order.iter().enumerate() {
    //         orig_to_new[orig_idx as usize] = new_idx as IndexT;
    //     }

    //     // 3. Reorder the dataset
    //     let reordered_dataset = Arc::new(dataset.shuffled_copy(&order));

    //     // 4. Reorder the graph
    //     let mut new_graph = ClassicGraph::new(n as IndexT, graph.max_degree());
    //     for (new_idx, &orig_idx) in order.iter().enumerate() {
    //         let orig_neighbors = graph.get_neighborhood(orig_idx);
    //         let new_neighbors: Vec<IndexT> = orig_neighbors
    //             .iter()
    //             .map(|&nbr| orig_to_new[nbr as usize])
    //             .collect();
    //         new_graph.set_neighborhood(new_idx as IndexT, &new_neighbors);
    //     }

    //     // 5. Remap posting lists to new indices
    //     let mut output_posting_lists: HashMap<IndexT, Box<[IndexT]>> = HashMap::new();
    //     for list in posting_lists_vec {
    //         let representative = orig_to_new[list[0] as usize];
    //         let remapped: Vec<IndexT> = list[1..]
    //             .iter()
    //             .map(|&idx| orig_to_new[idx as usize])
    //             .collect();
    //         if !remapped.is_empty() {
    //             output_posting_lists.insert(representative, remapped.into_boxed_slice());
    //         }
    //     }

    //     // 6. local_to_input_index: new index -> original index
    //     let local_to_input_index = order.into_boxed_slice();

    //     CompactedGraphIndex {
    //         internal_graph: new_graph,
    //         internal_dataset: reordered_dataset,
    //         posting_lists: output_posting_lists,
    //         local_to_input_index,
    //     }
    // }
}
