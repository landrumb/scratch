//! a graph implementation using a vector of vectors

use std::sync::Mutex;

use rayon::prelude::*;

use crate::graph::IndexT;

use super::{ClassicGraph, Graph, MutableGraph};

pub struct VectorGraph {
    neighborhoods: Vec<Vec<IndexT>>,
    insertion_queues: Vec<Mutex<Vec<IndexT>>>,
}

impl VectorGraph {
    /// constructs a new empty VectorGraph with the given number of nodes
    pub fn empty(n: usize) -> VectorGraph {
        let neighborhoods: Vec<Vec<IndexT>> = vec![Vec::new(); n];
        let insertion_queues: Vec<Mutex<Vec<IndexT>>> =
            (0..n).map(|_| Mutex::new(Vec::new())).collect();

        VectorGraph {
            neighborhoods,
            insertion_queues,
        }
    }

    pub fn new(neighborhoods: Vec<Vec<IndexT>>) -> VectorGraph {
        let insertion_queues: Vec<Mutex<Vec<IndexT>>> = (0..neighborhoods.len())
            .map(|_| Mutex::new(Vec::new()))
            .collect();
        VectorGraph {
            neighborhoods,
            insertion_queues,
        }
    }

    /// returns the number of nodes in the graph
    pub fn n(&self) -> usize {
        self.neighborhoods.len()
    }

    /// returns the neighborhood of a node
    pub fn get_neighborhood(&self, i: IndexT) -> &[IndexT] {
        assert!(i < self.n() as IndexT);
        &self.neighborhoods[i as usize]
    }

    /// returns all the out-neighbors queued for a node.
    ///
    /// Unlike `get_queued_edges`, does not empty the queue.
    pub fn get_neighborhood_queue(&self, i: IndexT) -> &Mutex<Vec<IndexT>> {
        assert!(i < self.n() as IndexT);
        &self.insertion_queues[i as usize]
    }

    /// sum of degrees of all nodes
    pub fn total_edges(&self) -> usize {
        self.neighborhoods.iter().map(|n| n.len()).sum()
    }

    /// maximum degree of the graph
    pub fn max_degree(&self) -> usize {
        self.neighborhoods
            .iter()
            .map(|n| n.len())
            .max()
            .unwrap_or(0)
    }

    /// pushes a new edge to the insertion queue of a node
    pub fn queue_edge(&self, from: IndexT, to: IndexT) {
        assert!(from < self.n() as IndexT && to < self.n() as IndexT);
        let mut queue = self.insertion_queues[from as usize].lock().unwrap();
        queue.push(to);
    }

    /// pushes a slice of new edges to the insertion queue of a node
    pub fn bulk_queue(&self, from: IndexT, to: &[IndexT]) {
        assert!(from < self.n() as IndexT);
        let mut queue = self.insertion_queues[from as usize].lock().unwrap();
        queue.extend_from_slice(to);
    }

    /// For each queue that is not empty and is small enough that adding the edges does not violate the
    /// degree bound, we add the edges to the graph and clear the queue. Otherwise, we do nothing.
    pub fn preprocess_queues(&mut self, degree_bound: usize) {
        self.insertion_queues
            .par_iter_mut()
            .zip(self.neighborhoods.par_iter_mut())
            .for_each(|(queue, neighborhood)| {
                let queue = queue.get_mut().unwrap();
                if queue.is_empty() {
                    return;
                }
                if queue.len() + neighborhood.len() > degree_bound {
                    return;
                }
                neighborhood.extend(queue.iter());
                queue.clear()
            });
    }

    /// returns a vector of (index, queue) pairs for all non-empty queues
    /// and clears the queues
    pub fn get_queued_edges(&mut self) -> Vec<(IndexT, Vec<IndexT>)> {
        let nonempty_queues = self
            .insertion_queues
            .par_iter_mut()
            .enumerate()
            .filter_map(|(i, queue)| {
                let queue = queue.get_mut().unwrap();
                if !queue.is_empty() {
                    let queue_copy = queue.clone();
                    queue.clear();
                    Some((i as IndexT, queue_copy))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        nonempty_queues
    }

    /// removes all edges from a node
    pub fn clear_neighborhood(&mut self, i: IndexT) {
        assert!(i < self.n() as IndexT);
        self.neighborhoods[i as usize].clear();
    }

    /// applies a function to each neighborhood
    pub fn apply_to_neighborhoods<F>(&mut self, f: F)
    where
        F: Fn(IndexT, &mut Vec<IndexT>),
    {
        self.neighborhoods
            .iter_mut()
            .enumerate()
            .for_each(|(i, neighborhood)| f(i as IndexT, neighborhood));
    }
}

impl Clone for VectorGraph {
    fn clone(&self) -> Self {
        let insertion_queues: Vec<Mutex<Vec<IndexT>>> = self.insertion_queues.iter().map(|q| Mutex::new(q.lock().unwrap().clone())).collect();
        VectorGraph {
            neighborhoods: self.neighborhoods.clone(),
            insertion_queues,
        }
    }
}

impl Graph for VectorGraph {
    fn neighbors(&self, i: IndexT) -> &[IndexT] {
        self.get_neighborhood(i)
    }
}

impl MutableGraph for VectorGraph {
    fn add_neighbor(&mut self, from: IndexT, to: IndexT) {
        assert!(from < self.n() as IndexT && to < self.n() as IndexT);
        self.neighborhoods[from as usize].push(to);
    }

    fn set_neighborhood(&mut self, i: IndexT, neighborhood: &[IndexT]) {
        assert!(i < self.n() as IndexT);
        self.neighborhoods[i as usize] = neighborhood.to_vec();
    }
}

impl From<&VectorGraph> for ClassicGraph {
    fn from(graph: &VectorGraph) -> ClassicGraph {
        let n = graph.n() as IndexT;
        let r = graph.max_degree();

        let mut output_graph = ClassicGraph::new(n, r);

        for i in 0..n {
            let neighborhood = graph.get_neighborhood(i);
            output_graph.set_neighborhood(i, neighborhood);
        }
        output_graph
    }
}

impl From<VectorGraph> for ClassicGraph {
    fn from(mut graph: VectorGraph) -> ClassicGraph {
        let n = graph.n() as IndexT;
        let r = graph.max_degree();

        let mut output_graph = ClassicGraph::new(n, r);

        // We can move out the neighborhoods without cloning to avoid extra allocations.
        for (i, neighborhood) in graph.neighborhoods.drain(..).enumerate() {
            output_graph.set_neighborhood(i as IndexT, &neighborhood);
        }
        output_graph
    }
}

unsafe impl Sync for VectorGraph {}
