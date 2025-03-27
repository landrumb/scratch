//! a graph implementation using a vector of vectors

use crate::graph::IndexT;

use super::{ClassicGraph, Graph, MutableGraph};

pub struct VectorGraph {
    neighborhoods: Vec<Vec<IndexT>>,
}

impl VectorGraph {
    /// constructs a new empty VectorGraph with the given number of nodes
    pub fn empty(n: usize) -> VectorGraph {
        VectorGraph {
            neighborhoods: vec![Vec::new(); n],
        }
    }

    pub fn new(neighborhoods: Vec<Vec<IndexT>>) -> VectorGraph {
        VectorGraph { neighborhoods }
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
