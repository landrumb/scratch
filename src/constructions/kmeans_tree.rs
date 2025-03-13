//! kmeans tree construction

use std::{
    collections::{HashMap, VecDeque},
    iter::zip,
};

use id_tree::{self, InsertBehavior::*, Node, NodeId, Tree};

/// A beam element for tree traversal, consisting of:
/// - distance: the distance to the query
/// - node_id: the ID of the node in the tree
/// - is_leaf: whether this node is a leaf
type TreeBeamElement = (f32, NodeId, bool);

use crate::{
    clustering::kmeans::kmeans_subset, 
    data_handling::dataset::VectorDataset, 
    data_handling::dataset_traits::Dataset,
    graph::IndexT,
};

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
    /// constructs a kmeans tree
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
    pub fn build_bounded_leaf(
        dataset: &'a VectorDataset<f32>,
        k: usize,
        max_leaf_size: usize,
        max_iter: usize,
        epsilon: f64,
    ) -> KMeansTree<'a> {
        let mut tree: id_tree::Tree<Option<usize>> = id_tree::TreeBuilder::new()
            .with_node_capacity(dataset.n)
            .build();
        let mut representatives: Vec<f32> = Vec::new();
        let dim = dataset.dim;

        let mut leaf_to_partition_map: HashMap<usize, Vec<IndexT>> = HashMap::new();

        let mut queue: VecDeque<(NodeId, Vec<IndexT>)> = VecDeque::new();
        queue.push_back((
            tree.insert(Node::new(None), AsRoot).unwrap(), // root id
            (0..dataset.n as IndexT).collect(),
        ));

        while queue.len() > 0 {
            let (parent_node, subset) = queue.pop_back().unwrap();

            // clustering the points of the subset
            let (unfolded_representatives, assignments) =
                kmeans_subset(dataset, k, max_iter, epsilon, &subset);

            // TODO: move folding representatives into 2d vec into kmeans
            let mut local_representatives: Vec<&[f32]> = Vec::new();
            for i in 0..unfolded_representatives.len() / dim {
                local_representatives.push(&unfolded_representatives[i * dim..(i + 1) * dim]);
            }

            // pivoting assignments into partitions
            let mut partitions: Vec<Vec<IndexT>> = vec![Vec::new(); k];
            for (i, &assignment) in assignments.iter().enumerate() {
                partitions[assignment].push(subset[i]);
            }

            for (rep, part) in zip(local_representatives, partitions) {
                // add to tree
                let current_node_id = tree
                    .insert(
                        Node::new(Some(representatives.len() / dim)),
                        UnderNode(&parent_node),
                    )
                    .unwrap();

                // add centroid to representatives
                representatives.extend_from_slice(rep);

                if part.len() <= max_leaf_size {
                    // leaf
                    leaf_to_partition_map
                        .insert(tree.get(&current_node_id).unwrap().data().unwrap(), part);
                } else {
                    // internal node
                    queue.push_back((current_node_id, part));
                }
            }
        }

        let n = representatives.len() / dim;

        KMeansTree {
            tree,
            representatives: VectorDataset::new(representatives.into_boxed_slice(), n, dim),
            leaf_to_partition_map,
            dataset,
        }
    }
    
    /// Constructs a KMeansTree with spillover, where each point can be assigned to multiple centroids
    ///
    /// Similar to build_bounded_leaf, but each point can be assigned to up to 's' closest centroids.
    /// This results in points potentially appearing in multiple partitions, which can increase recall
    /// at the cost of larger partitions and potentially more computation.
    ///
    /// The spillover parameter 's' controls how many centroids each point can be assigned to.
    pub fn build_with_spillover(
        dataset: &'a VectorDataset<f32>,
        k: usize,
        max_leaf_size: usize,
        max_iter: usize,
        epsilon: f64,
        spillover: usize,
    ) -> KMeansTree<'a> {
        if spillover == 0 || spillover == 1 {
            // If spillover is 0 or 1, it's equivalent to the standard method
            return Self::build_bounded_leaf(dataset, k, max_leaf_size, max_iter, epsilon);
        }
        
        let mut tree: id_tree::Tree<Option<usize>> = id_tree::TreeBuilder::new()
            .with_node_capacity(dataset.n)
            .build();
        let mut representatives: Vec<f32> = Vec::new();
        let dim = dataset.dim;

        let mut leaf_to_partition_map: HashMap<usize, Vec<IndexT>> = HashMap::new();

        let mut queue: VecDeque<(NodeId, Vec<IndexT>)> = VecDeque::new();
        queue.push_back((
            tree.insert(Node::new(None), AsRoot).unwrap(), // root id
            (0..dataset.n as IndexT).collect(),
        ));

        while queue.len() > 0 {
            let (parent_node, subset) = queue.pop_back().unwrap();

            // First, compute the centroids and get the multiple assignments for each point
            let (unfolded_representatives, spillover_assignments) =
                crate::clustering::kmeans::kmeans_subset_with_spillover(
                    dataset, k, max_iter, epsilon, &subset, spillover);

            // Create representative slices for each centroid
            let mut local_representatives: Vec<&[f32]> = Vec::new();
            for i in 0..unfolded_representatives.len() / dim {
                local_representatives.push(&unfolded_representatives[i * dim..(i + 1) * dim]);
            }

            // Create partitions, allowing each point to be in multiple partitions
            let mut partitions: Vec<Vec<IndexT>> = vec![Vec::new(); k];
            for (i, assignments) in spillover_assignments.iter().enumerate() {
                for &assignment in assignments {
                    partitions[assignment].push(subset[i]);
                }
            }

            for (rep, part) in zip(local_representatives, partitions) {
                // Skip empty partitions that might occur with spillover
                if part.is_empty() {
                    continue;
                }
                
                // Add to tree
                let current_node_id = tree
                    .insert(
                        Node::new(Some(representatives.len() / dim)),
                        UnderNode(&parent_node),
                    )
                    .unwrap();

                // Add centroid to representatives
                representatives.extend_from_slice(rep);

                if part.len() <= max_leaf_size {
                    // Leaf
                    leaf_to_partition_map
                        .insert(tree.get(&current_node_id).unwrap().data().unwrap(), part);
                } else {
                    // Internal node, continue partitioning
                    queue.push_back((current_node_id, part));
                }
            }
        }

        let n = representatives.len() / dim;

        KMeansTree {
            tree,
            representatives: VectorDataset::new(representatives.into_boxed_slice(), n, dim),
            leaf_to_partition_map,
            dataset,
        }
    }

    /// The most basic query procedure possible, just walks to the corresponding leaf and queries it, returning the top k results
    pub fn query(&self, query: &[f32], k: usize) -> Box<[(IndexT, f32)]> {
        let nearest_partition = self.query_tree(query);
        if nearest_partition.is_none() {
            panic!("No nearest partition found");
        }

        // query partition indicated by the leaf node
        let partition: Box<[usize]> = self
            .leaf_to_partition_map
            .get(&nearest_partition.unwrap())
            .unwrap()
            .iter()
            .map(|x| *x as usize)
            .collect();

        // Use the k parameter to limit results
        let results = self.dataset.brute_force(query, partition.iter().as_slice());
        let results_vec: Vec<(IndexT, f32)> = results
            .iter()
            .take(k)
            .map(|x| (x.0 as IndexT, x.1))
            .collect();

        results_vec.into_boxed_slice()
    }

    /// internal function to query the tree for the corresponding leaf node
    fn query_tree(&self, query: &[f32]) -> Option<usize> {
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
    
    /// Query the tree using beam search, keeping the top beam_width nodes at each level.
    /// 
    /// Unlike standard beam search, this implementation removes parent nodes from the beam
    /// when their children are added, forcing the search to eventually reach leaf nodes.
    /// When the search completes, the beam contains only leaf nodes, which are then searched
    /// to find the nearest neighbors.
    ///
    /// Parameters:
    /// - query: The query vector
    /// - beam_width: The maximum number of nodes to keep in the beam
    /// - k: The number of nearest neighbors to return
    ///
    /// Returns: The top k nearest neighbors from the partitions of the leaf nodes in the final beam
    pub fn query_beam_search(&self, query: &[f32], beam_width: usize, k: usize) -> Box<[(IndexT, f32)]> {
        if beam_width == 0 {
            return Box::new([]);
        }
        
        // Helper function to insert a node into the beam
        fn beam_insert(
            node_id: NodeId,
            query: &[f32], 
            tree: &Tree<Option<usize>>,
            representatives: &VectorDataset<f32>,
            beam: &mut Vec<TreeBeamElement>,
            beam_width: usize,
            is_leaf: bool,
        ) {
            let node = tree.get(&node_id).unwrap();
            let node_data = node.data();
            
            // Skip nodes with no representative (like the root)
            if node_data.is_none() {
                return;
            }
            
            // Calculate distance to this node's representative
            let centroid_idx = node_data.unwrap();
            let dist = representatives.compare(query, centroid_idx);
            
            // Push node to beam. We use negative distance to create a min-heap for closest nodes
            beam.push((dist as f32, node_id.clone(), is_leaf));
            
            // If the beam is too large, keep only the closest nodes
            if beam.len() > beam_width {
                // Sort by distance (ascending)
                beam.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                // Remove the farthest
                beam.truncate(beam_width);
            }
        }
        
        // Initialize the beam with the root node
        let mut beam: Vec<TreeBeamElement> = Vec::with_capacity(beam_width);
        let root_id = self.tree.root_node_id().unwrap().clone();
        let root_node = self.tree.get(&root_id).unwrap();
        
        // Process root node differently since it has no representative
        let child_node_ids = root_node.children();
        let mut has_more_internal_nodes = true;
        
        // If root has no children, return empty result
        if child_node_ids.is_empty() {
            return Box::new([]);
        }
        
        // Add all children of root to the beam
        for child_id in child_node_ids {
            let child_node = self.tree.get(child_id).unwrap();
            let is_leaf = child_node.children().is_empty();
            beam_insert(
                child_id.clone(), 
                query, 
                &self.tree, 
                &self.representatives, 
                &mut beam, 
                beam_width,
                is_leaf
            );
        }
        
        // Continue expanding nodes until all beam elements are leaves
        while has_more_internal_nodes {
            has_more_internal_nodes = false;
            
            // Track nodes to remove and add
            let mut nodes_to_remove = Vec::new();
            let mut nodes_to_add = Vec::new();
            
            // Identify internal nodes to expand
            for (i, (_, node_id, is_leaf)) in beam.iter().enumerate() {
                if !is_leaf {
                    has_more_internal_nodes = true;
                    let node = self.tree.get(node_id).unwrap();
                    
                    // Mark this node for removal
                    nodes_to_remove.push(i);
                    
                    // Gather its children to add to the beam
                    for child_id in node.children() {
                        let child_node = self.tree.get(child_id).unwrap();
                        let child_is_leaf = child_node.children().is_empty();
                        nodes_to_add.push((child_id.clone(), child_is_leaf));
                    }
                }
            }
            
            // If no more internal nodes, we're done
            if !has_more_internal_nodes {
                break;
            }
            
            // Remove processed internal nodes (in reverse order to maintain indices)
            for &i in nodes_to_remove.iter().rev() {
                beam.swap_remove(i);
            }
            
            // Add all children to the beam
            for (node_id, is_leaf) in nodes_to_add {
                beam_insert(
                    node_id, 
                    query, 
                    &self.tree, 
                    &self.representatives, 
                    &mut beam, 
                    beam_width,
                    is_leaf
                );
            }
        }
        
        // At this point, beam contains only leaf nodes
        // Gather all partitions from these leaves
        let mut all_candidates = Vec::new();
        
        for (_, node_id, _) in beam {
            let node = self.tree.get(&node_id).unwrap();
            if let Some(centroid_idx) = node.data() {
                if let Some(partition) = self.leaf_to_partition_map.get(centroid_idx) {
                    // Add partition indices to candidates
                    all_candidates.extend(partition.iter().map(|&idx| idx as usize));
                }
            }
        }
        
        // Remove duplicates
        all_candidates.sort_unstable();
        all_candidates.dedup();
        
        // Perform brute force search on the combined partitions
        let results = self.dataset.brute_force(query, &all_candidates);
        
        // Limit to k results
        let results_vec: Vec<(IndexT, f32)> = results
            .iter()
            .take(k)
            .map(|&(idx, dist)| (idx as IndexT, dist))
            .collect();
        
        results_vec.into_boxed_slice()
    }
}
