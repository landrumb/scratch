//! A generic graph index container

use crate::{data_handling::dataset_traits::Dataset, graph::{beam_search, Graph, IndexT}};

use super::VectorIndex;

pub struct GraphIndex<'graph_lifetime, 'dataset_lifetime, T> {
    pub graph: &'graph_lifetime dyn Graph,
    pub dataset: &'dataset_lifetime dyn Dataset<T>,
    pub root: IndexT,
}

impl<'graph_lifetime, 'dataset_lifetime, T> GraphIndex<'graph_lifetime, 'dataset_lifetime, T> {
    pub fn new(
        graph: &'graph_lifetime dyn Graph,
        dataset: &'dataset_lifetime dyn Dataset<T>,
        root: IndexT,
    ) -> Self {
        GraphIndex { graph, dataset, root }
    }
}

impl<T> VectorIndex<T> for GraphIndex<'_, '_, T> {
    /// Does k-NN search on the graph
    fn query(&self, query: &[T], parameters: super::Parameters) -> Vec<IndexT> {
        let k = parameters.get::<usize>("k");
        let beam_width = parameters.get::<usize>("beam_width").unwrap_or(&10);

        let limit = parameters.get::<usize>("limit");
        let limit = if limit.is_some() {
            Some(*limit.unwrap())
        } else {
            None
        };

        let results = beam_search(
            query,
            self.graph,
            self.dataset,
            self.root,
            *beam_width,
            limit,
        );

        if k.is_some() {
            results.into_iter().take(*k.unwrap()).collect()
        } else {
            results
        }
    }
}