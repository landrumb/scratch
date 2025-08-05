//! Python bindings for the scratch library
//! This module is only compiled when the "python" feature is enabled

use crate::constructions::neighbor_selection::{naive_semi_greedy_prune, PairwiseDistancesHandler};
use crate::constructions::slow_preprocessing::build_global_local_graph;
use crate::data_handling::dataset::Subset;
use crate::data_handling::dataset::VectorDataset;
use crate::data_handling::dataset_traits::Dataset;
use crate::graph::{IndexT, MutableGraph, VectorGraph};
use crate::util::clique;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use std::sync::Arc;

#[pyclass]
pub struct PyVectorDataset {
    dataset: VectorDataset<f32>,
}

#[pymethods]
impl PyVectorDataset {
    #[new]
    fn new(data: PyReadonlyArray2<f32>) -> PyResult<Self> {
        let array = data.as_array();
        let n = array.shape()[0];
        let dim = array.shape()[1];

        // Create a contiguous copy of the data
        let data_vec: Vec<f32> = array.iter().copied().collect();

        Ok(PyVectorDataset {
            dataset: VectorDataset::new(data_vec.into_boxed_slice(), n, dim),
        })
    }

    #[getter]
    fn get_n(&self) -> usize {
        self.dataset.n
    }

    #[getter]
    fn get_dim(&self) -> usize {
        self.dataset.dim
    }

    fn get_vector(&self, idx: usize, py: Python<'_>) -> PyResult<PyObject> {
        if idx >= self.dataset.n {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Index {} out of bounds for dataset with {} elements",
                idx, self.dataset.n
            )));
        }

        let vector = self.dataset.get(idx);
        Ok(PyArray1::from_slice(py, vector).into_py_any(py)?)
    }

    fn compare_internal(&self, i: usize, j: usize) -> f64 {
        Dataset::compare_internal(&self.dataset, i, j)
    }

    fn compare(&self, query: PyReadonlyArray1<f32>, i: usize) -> PyResult<f64> {
        if i >= self.dataset.n {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Index {} out of bounds for dataset with {} elements",
                i, self.dataset.n
            )));
        }

        let query_slice = query.as_slice()?;
        if query_slice.len() != self.dataset.dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Query dimension {} does not match dataset dimension {}",
                query_slice.len(),
                self.dataset.dim
            )));
        }

        Ok(Dataset::compare(&self.dataset, query_slice, i))
    }

    fn size(&self) -> usize {
        Dataset::size(&self.dataset)
    }

    fn brute_force(
        &self,
        query: PyReadonlyArray1<f32>,
        _py: Python<'_>,
    ) -> PyResult<Vec<(usize, f32)>> {
        let query_slice = query.as_slice()?;
        if query_slice.len() != self.dataset.dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Query dimension {} does not match dataset dimension {}",
                query_slice.len(),
                self.dataset.dim
            )));
        }

        let results = Dataset::brute_force(&self.dataset, query_slice);
        Ok(results.into_vec())
    }

    fn brute_force_internal(&self, q: usize, _py: Python<'_>) -> PyResult<Vec<(usize, f32)>> {
        if q >= self.dataset.n {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Index {} out of bounds for dataset with {} elements",
                q, self.dataset.n
            )));
        }

        let results = Dataset::brute_force_internal(&self.dataset, q);
        Ok(results.into_vec())
    }

    fn brute_force_subset(
        &self,
        query: PyReadonlyArray1<f32>,
        subset: Vec<usize>,
        _py: Python<'_>,
    ) -> PyResult<Vec<(usize, f32)>> {
        let query_slice = query.as_slice()?;
        if query_slice.len() != self.dataset.dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Query dimension {} does not match dataset dimension {}",
                query_slice.len(),
                self.dataset.dim
            )));
        }

        // Validate subset indices
        for &idx in &subset {
            if idx >= self.dataset.n {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Subset index {} out of bounds for dataset with {} elements",
                    idx, self.dataset.n
                )));
            }
        }

        let results = Dataset::brute_force_subset(&self.dataset, query_slice, &subset);
        Ok(results.into_vec())
    }

    fn brute_force_subset_internal(
        &self,
        q: usize,
        subset: Vec<usize>,
        _py: Python<'_>,
    ) -> PyResult<Vec<(usize, f32)>> {
        if q >= self.dataset.n {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Index {} out of bounds for dataset with {} elements",
                q, self.dataset.n
            )));
        }

        // Validate subset indices
        for &idx in &subset {
            if idx >= self.dataset.n {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Subset index {} out of bounds for dataset with {} elements",
                    idx, self.dataset.n
                )));
            }
        }

        let results = Dataset::brute_force_subset_internal(&self.dataset, q, &subset);
        Ok(results.into_vec())
    }

    fn build_global_local_graph(&self, alpha: f32) -> PyResult<PyVectorGraph> {
        // Calculate pairwise distances
        let nested_boxed_distances = (0..self.dataset.size())
            .into_par_iter()
            .map(|i| {
                self.dataset
                    .brute_force_internal(i)
                    .iter()
                    .map(|(j, dist)| (*j as IndexT, *dist))
                    .collect::<Box<[(IndexT, f32)]>>()
            })
            .collect::<Box<[Box<[(IndexT, f32)]>]>>();

        let pairwise_distances = PairwiseDistancesHandler::new(nested_boxed_distances);

        // Build the graph
        let graph = build_global_local_graph(&self.dataset, |center, candidates| {
            naive_semi_greedy_prune(
                center,
                candidates,
                &self.dataset,
                alpha,
                &pairwise_distances,
            )
        });

        Ok(PyVectorGraph { graph })
    }
}

#[pyclass]
pub struct PyVectorGraph {
    graph: VectorGraph,
}

#[pymethods]
impl PyVectorGraph {
    #[new]
    fn new(neighborhoods: Vec<Vec<u32>>) -> Self {
        PyVectorGraph {
            graph: VectorGraph::new(neighborhoods),
        }
    }

    #[staticmethod]
    fn empty(n: usize) -> Self {
        PyVectorGraph {
            graph: VectorGraph::empty(n),
        }
    }

    #[getter]
    fn n(&self) -> usize {
        self.graph.n()
    }

    fn get_neighborhood(&self, i: u32, py: Python<'_>) -> PyResult<PyObject> {
        if i >= self.graph.n() as u32 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Index {} out of bounds for graph with {} nodes",
                i,
                self.graph.n()
            )));
        }

        let neighbors = self.graph.get_neighborhood(i);
        Ok(PyArray1::from_slice(py, neighbors).into_py_any(py)?)
    }

    fn total_edges(&self) -> usize {
        self.graph.total_edges()
    }

    fn max_degree(&self) -> usize {
        self.graph.max_degree()
    }

    fn maximal_cliques(&self) -> Vec<Vec<u32>> {
        clique::maximal_cliques(&self.graph)
    }

    fn maximal_independent_cliques(&self) -> Vec<Vec<u32>> {
        clique::greedy_independent_cliques(&self.maximal_cliques())
    }

    fn maximal_bidirectional_cliques(&self) -> Vec<Vec<u32>> {
        clique::maximal_bidirectional_cliques(&self.graph)
    }

    fn maximal_independent_bidirectional_cliques(&self) -> Vec<Vec<u32>> {
        clique::greedy_independent_cliques(&self.maximal_bidirectional_cliques())
    }

    fn add_neighbor(&mut self, from: u32, to: u32) -> PyResult<()> {
        if from >= self.graph.n() as u32 || to >= self.graph.n() as u32 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Index out of bounds for graph with {} nodes",
                self.graph.n()
            )));
        }

        self.graph.add_neighbor(from, to);
        Ok(())
    }

    fn set_neighborhood(&mut self, i: u32, neighborhood: Vec<u32>) -> PyResult<()> {
        if i >= self.graph.n() as u32 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Index {} out of bounds for graph with {} nodes",
                i,
                self.graph.n()
            )));
        }

        self.graph.set_neighborhood(i, &neighborhood);
        Ok(())
    }

    fn transparent_beam_search(
        &self,
        query: PyReadonlyArray1<f32>,
        dataset: &PyVectorDataset,
        start: u32,
        beam_width: usize,
        limit: Option<usize>,
    ) -> PyResult<(Vec<(u32, f32)>, Vec<(u32, f32)>, Vec<(Vec<u32>, u32)>)> {
        if start >= self.graph.n() as u32 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Start {} out of bounds for graph with {} nodes",
                start,
                self.graph.n()
            )));
        }

        let query_slice = query.as_slice()?;
        if query_slice.len() != dataset.dataset.dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Query dimension {} does not match dataset dimension {}",
                query_slice.len(),
                dataset.dataset.dim
            )));
        }

        let (frontier, visited, steps) = crate::graph::transparent_beam_search(
            query_slice,
            &self.graph,
            &dataset.dataset,
            start,
            beam_width,
            limit,
        );

        let steps_py: Vec<(Vec<u32>, u32)> =
            steps.into_iter().map(|s| (s.beam, s.expanded)).collect();

        Ok((frontier, visited, steps_py))
    }
}

#[pyclass]
pub struct PySubset {
    subset: Subset<f32>,
}

#[pymethods]
impl PySubset {
    #[new]
    fn new(dataset: &PyVectorDataset, indices: Vec<usize>) -> Self {
        // Create a new VectorDataset to pass to Subset
        let new_dataset = Arc::new(dataset.dataset.clone());

        PySubset {
            subset: Subset::new(new_dataset, indices),
        }
    }

    #[getter]
    fn size(&self) -> usize {
        self.subset.size()
    }

    fn get_vector(&self, idx: usize, py: Python<'_>) -> PyResult<PyObject> {
        if idx >= self.subset.size() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Index {} out of bounds for subset with {} elements",
                idx,
                self.subset.size()
            )));
        }

        let vector = self.subset.get(idx);
        Ok(PyArray1::from_slice(py, vector).into_py_any(py)?)
    }

    fn build_global_local_graph(&self, alpha: f32) -> PyResult<PyVectorGraph> {
        // Calculate pairwise distances
        let nested_boxed_distances = (0..self.subset.size())
            .into_par_iter()
            .map(|i| {
                self.subset
                    .brute_force_internal(i)
                    .iter()
                    .map(|(j, dist)| (*j as IndexT, *dist))
                    .collect::<Box<[(IndexT, f32)]>>()
            })
            .collect::<Box<[Box<[(IndexT, f32)]>]>>();

        let pairwise_distances = PairwiseDistancesHandler::new(nested_boxed_distances);

        // Build the graph
        let graph = build_global_local_graph(&self.subset, |center, candidates| {
            naive_semi_greedy_prune(center, candidates, &self.subset, alpha, &pairwise_distances)
        });

        Ok(PyVectorGraph { graph })
    }
}

/// Python module for scratch library
#[pymodule]
pub fn scratch(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyVectorDataset>()?;
    m.add_class::<PyVectorGraph>()?;
    m.add_class::<PySubset>()?;
    Ok(())
}
