use std::sync::Arc;

use crate::{data_handling::dataset_traits::Dataset, graph::VectorGraph};



/// given a graph and a dataset, returns a graph which includes only edges shorter than epsilon
pub fn filter_edges_epsilon<T: Distance>(graph: &VectorGraph, dataset: &Arc<dyn Dataset<T>>, epsilon: f32) -> VectorGraph {
    todo!()
}