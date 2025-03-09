//! kmeans tree construction

use std::{collections::{HashMap, VecDeque}, iter::zip};

use id_tree::{self, InsertBehavior::*, Node, NodeId, Tree};

use crate::{clustering::kmeans::kmeans_subset, data_handling::dataset::VectorDataset, graph::graph::{Graph, IndexT}};

// #[cfg(feature = "verbose_kmt")]
// macro_rules! verbose_println {
//     ($($arg:tt)*) => {
//         println!($($arg)*);
//     }
// }

// #[cfg(not(feature = "verbose_kmt"))]
// macro_rules! verbose_println {
//     ($($arg:tt)*) => {};
// }

pub struct KMeansTree<'a> {
    tree: Tree<Option<usize>>,
    representatives: VectorDataset<f32>,
    leaf_to_partition_map: HashMap<NodeId, Vec<IndexT>>,
    dataset: &'a VectorDataset<f32>,
}


impl<'a> KMeansTree<'a> {

    /// constructs a graph that's a kmeans tree
    /// 
    /// given subset of points [indices], we run kmeans on the subset, 
    /// we do a zany stack based approach with a stack of (NodeID, descendants) tuples
    /// 
    /// when popping an element from the stack, we run kmeans on it, and for each kmeans cluster we:
    ///  - append the centroid to a vector storing centroids
    ///  - add a node as a child of the parent NodeID with data corresponding to the index of the centroid in question
    /// 
    /// when querying, it operates over a collection of partitions, combining results from specific partitions reached by the
    /// kmeans tree 
    pub fn build_bounded_leaf(dataset: &'a VectorDataset<f32>, k: usize, max_leaf_size: usize, max_iter: usize, epsilon:f64) -> KMeansTree<'a> {
        let mut tree: id_tree::Tree<Option<usize>> = id_tree::TreeBuilder::new().with_node_capacity(dataset.n).build();
        let mut representatives: Vec<f32> = Vec::new();
        let dim = dataset.dim;
    
        let mut leaf_to_partition_map: HashMap<NodeId, Vec<IndexT>> = HashMap::new();
    
        let mut queue: VecDeque<(NodeId, Vec<IndexT>)> = VecDeque::new();
        queue.push_back((
            tree.insert(Node::new(None), AsRoot).unwrap(), // root id
            (0..dataset.n as IndexT).collect()
        ));
    
        while queue.len() > 0 {
            let (parent_node, subset) = queue.pop_back().unwrap();
    
            // clustering the points of the subset
            let (unfolded_representatives, assignments) = kmeans_subset(dataset, k, max_iter, epsilon, &subset);
    
            // TODO: move folding representatives into 2d vec into kmeans
            let mut local_representatives: Vec<&[f32]> = Vec::new();
            for i in 0..unfolded_representatives.len() / dim {
                local_representatives.push(&unfolded_representatives[i * dim..(i+1) * dim]);
            }
    
    
            // pivoting assignments into partitions
            let mut partitions: Vec<Vec<IndexT>> = vec![Vec::new(); k];
            for (i, &assignment) in assignments.iter().enumerate() {
                partitions[assignment].push(subset[i]);
            }
    
            for (rep, part) in zip(local_representatives, partitions) {
                // add to tree
                let current_node_id = tree.insert(Node::new(Some(representatives.len() / dim)), UnderNode(&parent_node)).unwrap();
    
                // add centroid to representatives
                representatives.extend_from_slice(rep);
    
                if part.len() <= max_leaf_size { // leaf
                    leaf_to_partition_map.insert(current_node_id, part);
                } else { // internal node
                    queue.push_back((current_node_id, part));
                }
            }
        }

        let n = representatives.len() / dim;

        KMeansTree {
            tree,
            representatives : VectorDataset::new(representatives.into_boxed_slice(), n, dim),
            leaf_to_partition_map,
            dataset
        }
    }

    pub fn get_max_height(&self) -> usize {
        self.tree.height()
    }
}