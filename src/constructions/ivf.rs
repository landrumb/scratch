//! an ivf implementation

use rand_distr::num_traits::ToPrimitive;

use rayon::prelude::*;

use crate::{
    clustering::kmeans::kmeans,
    data_handling::{dataset::VectorDataset, dataset_traits::Dataset},
    distance::euclidean,
    graph::IndexT,
};

use super::VectorIndex;

pub struct IVFIndex<'a> {
    pub k: usize,
    dataset: &'a VectorDataset<f32>,
    representatives: Vec<f32>,
    partitions: Vec<Vec<IndexT>>,
}

/// returns index distance pairs sorted by distance to query
///
/// we trust here that the length of the query slice is the dimension
pub fn brute_force(query: &[f32], points: &[f32]) -> Vec<(IndexT, f32)> {
    // should probably add an assertion about the length of the query dividing the length of the points
    let dim = query.len();
    let n = points.len() / dim;

    let mut results: Vec<(IndexT, f32)> = (0..n)
        .map(|i| {
            (
                i as IndexT,
                euclidean(query, &points[i * dim..(i + 1) * dim]),
            )
        })
        .collect();

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    results
}

impl<'a> IVFIndex<'a> {
    /// constructs an IVFIndex with explicitly provided partitions and representatives
    pub fn new(
        dataset: &'a VectorDataset<f32>,
        representatives: Vec<f32>,
        partitions: Vec<Vec<IndexT>>,
    ) -> IVFIndex<'a> {
        assert!(
            representatives.len() == partitions.len(),
            "Number of representatives ({}) mismatched with number of partitions ({})",
            representatives.len(),
            partitions.len()
        );

        let k = representatives.len();

        IVFIndex {
            k,
            dataset,
            representatives,
            partitions,
        }
    }

    pub fn build(
        dataset: &'a VectorDataset<f32>,
        k: usize,
        max_iter: usize,
        epsilon: f32,
    ) -> IVFIndex<'a> {
        let (representatives, assignments) =
            kmeans(dataset, k, max_iter, epsilon.to_f64().unwrap());

        // collect the assignments into partitions
        let mut partitions: Vec<Vec<IndexT>> = vec![Vec::new(); k];
        for (i, &assignment) in assignments.iter().enumerate() {
            partitions[assignment].push(i as IndexT);
        }

        IVFIndex {
            k,
            dataset,
            representatives,
            partitions,
        }
    }

    /// get a slice with a given representative
    pub fn get_representative(&self, idx: usize) -> &[f32] {
        assert!(
            idx < self.k,
            "Representative index out of bounds: {} >= {}",
            idx,
            self.k
        );
        &self.representatives[idx * self.dataset.dim..(idx + 1) * self.dataset.dim]
    }

    /// queries the vectors in a given partition, and returns results in terms of their global index
    pub fn query_partition(&self, query: &[f32], partition: usize) -> Vec<(IndexT, f32)> {
        assert!(
            partition < self.k,
            "Partition index out of bounds: {} >= {}",
            partition,
            self.k
        );

        let mut results = (0..self.partitions[partition].len())
            .into_par_iter()
            .map(|i| {
                let idx = self.partitions[partition][i];
                (
                    idx,
                    self.dataset.compare(query, idx as usize).to_f32().unwrap(),
                )
            })
            .collect::<Vec<(IndexT, f32)>>();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results
    }

    /// queries the representatives of the IVF index
    pub fn query_representatives(&self, query: &[f32]) -> Vec<(IndexT, f32)> {
        let mut results = (0..self.k)
            .into_par_iter()
            .map(|i| {
                let idx = i as IndexT;
                (idx, euclidean(query, self.get_representative(i)))
            })
            .collect::<Vec<(IndexT, f32)>>();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results
    }

    /// queries the IVF index with a given query, returning the top k results
    pub fn query(&self, query: &[f32], n_probe: usize, k: usize) -> Vec<(IndexT, f32)> {
        let representative_dists = self.query_representatives(query);
        let mut results = Vec::new();
        for (partition, _) in representative_dists.iter().take(n_probe) {
            let mut partition_results = self.query_partition(query, *partition as usize);
            results.append(&mut partition_results);
        }
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }
}

impl VectorIndex<f32> for IVFIndex<'_> {
    /// Does k-NN search on the IVF index
    fn query(&self, query: &[f32], parameters: super::Parameters) -> Vec<IndexT> {
        let k = parameters.get::<usize>("k");
        let n_probe = parameters.get::<usize>("n_probe").unwrap_or(&10);

        self.query(query, *n_probe, *k.unwrap())
            .iter()
            .map(|r| r.0)
            .collect()
    }
}
