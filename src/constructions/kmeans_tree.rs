//! kmeans tree construction

use std::{collections::{HashMap, VecDeque}, iter::zip};

use id_tree::{self, InsertBehavior::*, Node, NodeId, Tree};

use crate::{clustering::kmeans::kmeans_subset, data_handling::dataset::VectorDataset, distance::euclidean::euclidean, graph::graph::{Graph, IndexT}};

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
    leaf_to_partition_map: HashMap<usize, Vec<IndexT>>,
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
    
        let mut leaf_to_partition_map: HashMap<usize, Vec<IndexT>> = HashMap::new();
    
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
                    leaf_to_partition_map.insert(tree.get(&current_node_id).unwrap().data().unwrap(), part);
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

    /// The most basic query procedure possible, just walks to the corresponding leaf and queries it, returning the top k results
    pub fn query(&self, query: &[f32], k: usize) -> Box<[(IndexT, f32)]> {
        let nearest_partition = self.query_tree(query);
        if nearest_partition.is_none() {
            panic!("No nearest partition found");
        }

        // query partition indicated by the leaf node
        let partition: Box<[usize]> = self.leaf_to_partition_map.get(&nearest_partition.unwrap()).unwrap().iter().map(|x| *x as usize).collect();

        // Use the k parameter to limit results
        let results = self.dataset.brute_force(query, partition.iter().as_slice());
        let results_vec: Vec<(IndexT, f32)> = results.iter()
            .take(k)
            .map(|x| (x.0 as IndexT, x.1))
            .collect();
            
        results_vec.into_boxed_slice()
    }

    fn query_tree (&self, query: &[f32]) -> Option<usize> {
        let mut current_node = self.tree.get(self.tree.root_node_id().unwrap()).unwrap();
        
        loop {
            let child_node_ids = current_node.children();
            
            if child_node_ids.is_empty() {
                // We've reached a leaf node
                return current_node.data().clone();
            }
            
            let child_node_data = child_node_ids
                .iter()
                .map(|x| self.tree.get(x).unwrap().data().unwrap())
                .collect::<Vec<_>>();

            // Find the closest representative among the children
            let closest = self.representatives.closest(query, &child_node_data);
            
            current_node = self.tree.get(&child_node_ids[closest]).unwrap();
        }
    }

    pub fn get_max_height(&self) -> usize {
        self.tree.height()
    }
    
    pub fn get_leaf_count(&self) -> usize {
        self.leaf_to_partition_map.len()
    }
    
    pub fn get_total_leaf_points(&self) -> usize {
        self.leaf_to_partition_map.values().map(|v| v.len()).sum()
    }
    
    /// Debug function to find which partition contains a specific point index
    pub fn find_point_partition(&self, point_idx: usize) -> Option<usize> {
        for (partition_id, points) in &self.leaf_to_partition_map {
            if points.contains(&(point_idx as IndexT)) {
                return Some(*partition_id);
            }
        }
        None
    }
    
    /// Debug function to get points in a specific partition
    pub fn get_partition_points(&self, partition_id: usize) -> &Vec<IndexT> {
        self.leaf_to_partition_map.get(&partition_id).unwrap()
    }
    
    /// Debug function to return which partition a query would end up in
    pub fn debug_query_partition(&self, query: &[f32]) -> Option<usize> {
        self.query_tree(query)
    }
}