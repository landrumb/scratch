//! Implementation of beam search over a generic graph

use crate::data_handling::dataset_traits::Dataset;
use crate::graph::{Graph, IndexT};
use std::collections::HashSet;


/// This is an implementation of beam search with the logic from the ParlayANN implementation.
pub fn beam_search<T>(
    query: &[T],
    graph: &dyn Graph,
    dataset: &dyn Dataset<T>,
    start: IndexT,
    beam_width: usize,
) -> Vec<IndexT> {
    // Frontier maintains the closest points found so far
    let mut frontier: Vec<(f64, IndexT)> = Vec::with_capacity(beam_width);
    
    // The hash set provides fast duplicate checking (similar to ParlayANN's hash filter)
    let mut seen: HashSet<IndexT> = HashSet::new();
    
    // Visited nodes, ordered by insertion 
    let mut visited: Vec<(f64, IndexT)> = Vec::new();
    
    // Initialize with starting point
    let start_dist = dataset.compare(query, start as usize);
    frontier.push((start_dist, start));
    seen.insert(start);
    
    // The subset of frontier that has not been visited
    let mut unvisited_frontier: Vec<(f64, IndexT)> = Vec::with_capacity(beam_width);
    unvisited_frontier.push((start_dist, start));
    
    // Set a reasonable visit limit (similar to ParlayANN's QP.limit)
    let visit_limit = 100;
    let mut num_visited = 0;
    
    // Main loop - continue until all frontier points are visited or we reach the limit
    while !unvisited_frontier.is_empty() && num_visited < visit_limit {
        // Sort unvisited frontier by distance
        unvisited_frontier.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        
        // Get the closest unvisited point
        let current = unvisited_frontier[0];
        unvisited_frontier.remove(0);
        
        // Add to visited set
        visited.push(current);
        num_visited += 1;
        
        // Prepare candidates from neighbors
        let mut candidates: Vec<(f64, IndexT)> = Vec::new();
        
        // Process neighbors
        for &neighbor in graph.neighbors(current.1) {
            // Skip if already seen
            if seen.contains(&neighbor) {
                continue;
            }
            
            // Mark as seen
            seen.insert(neighbor);
            
            // If frontier is full, check if the neighbor is closer than the furthest point
            if frontier.len() == beam_width {
                // Get the furthest distance in the frontier
                let furthest_dist = frontier.iter().map(|x| x.0).max_by(|a, b| a.total_cmp(b)).unwrap_or(f64::MAX);
                
                // Calculate distance
                let dist = dataset.compare(query, neighbor as usize);
                
                // Skip if distance is greater than current furthest
                if dist >= furthest_dist {
                    continue;
                }
                
                candidates.push((dist, neighbor));
            } else {
                // Frontier not full, add the neighbor directly
                let dist = dataset.compare(query, neighbor as usize);
                candidates.push((dist, neighbor));
            }
        }
        
        // Sort candidates by distance
        candidates.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        
        // Merge frontier and candidates (similar to set_union in ParlayANN)
        let mut new_frontier: Vec<(f64, IndexT)> = Vec::with_capacity(beam_width);
        
        let mut i = 0;
        let mut j = 0;
        
        while i < frontier.len() && j < candidates.len() && new_frontier.len() < beam_width {
            if frontier[i].0 <= candidates[j].0 {
                new_frontier.push(frontier[i]);
                i += 1;
            } else {
                new_frontier.push(candidates[j]);
                j += 1;
            }
        }
        
        // Add remaining elements from frontier
        while i < frontier.len() && new_frontier.len() < beam_width {
            new_frontier.push(frontier[i]);
            i += 1;
        }
        
        // Add remaining elements from candidates
        while j < candidates.len() && new_frontier.len() < beam_width {
            new_frontier.push(candidates[j]);
            j += 1;
        }
        
        // Update frontier
        frontier = new_frontier;
        
        // Update unvisited frontier
        unvisited_frontier.clear();
        for &element in &frontier {
            // If the element hasn't been visited, add it to unvisited_frontier
            if !visited.iter().any(|&x| x.1 == element.1) {
                unvisited_frontier.push(element);
            }
        }
    }
    
    // Return indices of beam elements (frontier)
    frontier.iter().map(|x| x.1).collect()
}