use std::sync::Arc;

use crate::{
    data_handling::dataset_traits::Dataset,
    distance::DenseVector,
    graph::{IndexT, VectorGraph},
};

/// given a graph and a dataset, returns a graph which includes only edges shorter than epsilon
pub fn filter_edges_epsilon<T: DenseVector>(
    graph: &VectorGraph,
    dataset: &Arc<dyn Dataset<T>>,
    epsilon: f32,
) -> VectorGraph {
    let n = graph.n();
    let mut new_neighborhoods: Vec<Vec<IndexT>> = Vec::with_capacity(n);

    for i in 0..n {
        let neighbors = graph.get_neighborhood(i as IndexT);
        let mut filtered: Vec<IndexT> = Vec::with_capacity(neighbors.len());
        for &j in neighbors {
            let d = dataset.compare_internal(i as usize, j as usize) as f32;
            if d <= epsilon {
                filtered.push(j);
            }
        }
        new_neighborhoods.push(filtered);
    }

    VectorGraph::new(new_neighborhoods)
}

/// given a graph and a dataset, returns a graph which includes only edges with lengths below the `percentile`th percentile of all edge lengths
pub fn filter_edges_percentile<T: DenseVector>(
    graph: &VectorGraph,
    dataset: &Arc<dyn Dataset<T>>,
    percentile: f32,
) -> VectorGraph {
    let n = graph.n();
    // Collect all edge distances
    let mut all_dists: Vec<f32> = Vec::with_capacity(graph.total_edges());
    for i in 0..n {
        for &j in graph.get_neighborhood(i as IndexT) {
            let d = dataset.compare_internal(i as usize, j as usize) as f32;
            all_dists.push(d);
        }
    }

    if all_dists.is_empty() {
        return VectorGraph::empty(n);
    }

    // Clamp percentile to [0, 100]
    let p = percentile.max(0.0).min(100.0);
    all_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = all_dists.len();
    // Use inclusive index so that 100 maps to max and 0 maps to min
    let rank = (p / 100.0) * ((len - 1) as f32);
    let idx = rank.floor() as usize;
    let threshold = all_dists[idx];

    // Keep edges with distance <= threshold
    let mut new_neighborhoods: Vec<Vec<IndexT>> = Vec::with_capacity(n);
    for i in 0..n {
        let neighbors = graph.get_neighborhood(i as IndexT);
        let mut filtered: Vec<IndexT> = Vec::with_capacity(neighbors.len());
        for &j in neighbors {
            let d = dataset.compare_internal(i as usize, j as usize) as f32;
            if d <= threshold {
                filtered.push(j);
            }
        }
        new_neighborhoods.push(filtered);
    }

    VectorGraph::new(new_neighborhoods)
}

/// given a graph and a dataset, returns a graph which includes only edges which are among the `k` shortest edges for each node
pub fn filter_edges_shortest_k<T: DenseVector>(
    graph: &VectorGraph,
    dataset: &Arc<dyn Dataset<T>>,
    k: usize,
) -> VectorGraph {
    let n = graph.n();
    let mut new_neighborhoods: Vec<Vec<IndexT>> = Vec::with_capacity(n);

    for i in 0..n {
        let neighbors = graph.get_neighborhood(i as IndexT);
        if neighbors.is_empty() || k == 0 {
            new_neighborhoods.push(Vec::new());
            continue;
        }

        let mut scored: Vec<(IndexT, f32)> = neighbors
            .iter()
            .copied()
            .map(|j| (j, dataset.compare_internal(i as usize, j as usize) as f32))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let keep = scored.into_iter().take(k).map(|(j, _)| j).collect();
        new_neighborhoods.push(keep);
    }

    VectorGraph::new(new_neighborhoods)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::data_handling::dataset::VectorDataset;
    use crate::graph::VectorGraph;

    use super::{filter_edges_epsilon, filter_edges_percentile, filter_edges_shortest_k};

    fn small_dataset() -> Arc<dyn crate::data_handling::dataset_traits::Dataset<f32>> {
        // Points: 0:(0,0), 1:(1,0), 2:(0,2), 3:(3,0)
        let data: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 3.0, 0.0];
        let ds = VectorDataset::new(data.into_boxed_slice(), 4, 2);
        Arc::new(ds)
    }

    fn small_graph() -> VectorGraph {
        // 0 -> [1,2,3]
        // 1 -> [0,2,3]
        // 2 -> [0]
        // 3 -> [0,1]
        VectorGraph::new(vec![vec![1, 2, 3], vec![0, 2, 3], vec![0], vec![0, 1]])
    }

    #[test]
    fn test_filter_edges_epsilon() {
        let graph = small_graph();
        let ds = small_dataset();

        let filtered = filter_edges_epsilon(&graph, &ds, 2.0);

        assert_eq!(filtered.get_neighborhood(0), &[1, 2]);
        assert_eq!(filtered.get_neighborhood(1), &[0, 3]);
        assert_eq!(filtered.get_neighborhood(2), &[0]);
        assert_eq!(filtered.get_neighborhood(3), &[1]);
    }

    #[test]
    fn test_filter_edges_percentile_0_and_100_and_50() {
        let graph = small_graph();
        let ds = small_dataset();

        // p=0 keeps only edges with minimal distance (<=1)
        let p0 = filter_edges_percentile(&graph, &ds, 0.0);
        assert_eq!(p0.get_neighborhood(0), &[1]);
        assert_eq!(p0.get_neighborhood(1), &[0]);
        assert_eq!(p0.get_neighborhood(2), &[]);
        assert_eq!(p0.get_neighborhood(3), &[]);

        // p=50 threshold is 2.0 for this graph; same as epsilon=2.0 result
        let p50 = filter_edges_percentile(&graph, &ds, 50.0);
        assert_eq!(p50.get_neighborhood(0), &[1, 2]);
        assert_eq!(p50.get_neighborhood(1), &[0, 3]);
        assert_eq!(p50.get_neighborhood(2), &[0]);
        assert_eq!(p50.get_neighborhood(3), &[1]);

        // p=100 keeps all edges
        let p100 = filter_edges_percentile(&graph, &ds, 100.0);
        assert_eq!(p100.get_neighborhood(0), &[1, 2, 3]);
        assert_eq!(p100.get_neighborhood(1), &[0, 2, 3]);
        assert_eq!(p100.get_neighborhood(2), &[0]);
        assert_eq!(p100.get_neighborhood(3), &[0, 1]);
    }

    #[test]
    fn test_filter_edges_shortest_k() {
        let graph = small_graph();
        let ds = small_dataset();

        let k2 = filter_edges_shortest_k(&graph, &ds, 2);
        // Note: order is by distance ascending
        assert_eq!(k2.get_neighborhood(0), &[1, 2]);
        assert_eq!(k2.get_neighborhood(1), &[0, 3]);
        assert_eq!(k2.get_neighborhood(2), &[0]);
        assert_eq!(k2.get_neighborhood(3), &[1, 0]);

        let k1 = filter_edges_shortest_k(&graph, &ds, 1);
        assert_eq!(k1.get_neighborhood(0), &[1]);
        assert_eq!(k1.get_neighborhood(1), &[0]);
        assert_eq!(k1.get_neighborhood(2), &[0]);
        assert_eq!(k1.get_neighborhood(3), &[1]);
    }
}
