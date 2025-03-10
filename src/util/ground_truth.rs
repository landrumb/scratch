//! loads and holds ground truth from a ground truth file

use std::path::Path;
use std::fs::File;
use std::io::{Read, Write, Result};

use crate::data_handling::dataset_traits::{Dataset, Numeric};

#[cfg(test)]
mod tests;

pub struct GroundTruth {
    neighbors: Box<[u32]>,
    distances: Box<[f32]>,
    pub n: usize,
    pub k: usize,
}

impl GroundTruth {
    /// Reads a groundtruth file.
    ///
    /// File format:
    /// - 2 x u32 (little-endian): num_points, num_neighbors
    /// - num_points*num_neighbors x u32: neighbor ids
    /// - num_points*num_neighbors x f32: distances
    pub fn read(gt_filename: &Path) -> GroundTruth {
        let mut file = File::open(gt_filename)
            .expect("Failed to open ground truth file");

        // Read header: number of points and neighbors.
        let mut header = [0u8; 8];
        file.read_exact(&mut header).expect("Failed to read header");
        let num_points = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let num_neighbors = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
        let total = num_points * num_neighbors;

        // Read neighbor IDs.
        let mut neighbor_bytes = vec![0u8; total * 4];
        file.read_exact(&mut neighbor_bytes).expect("Failed to read neighbors");
        let neighbors: Vec<u32> = neighbor_bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        // Read distances.
        let mut distance_bytes = vec![0u8; total * 4];
        file.read_exact(&mut distance_bytes).expect("Failed to read distances");
        let distances: Vec<f32> = distance_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        GroundTruth {
            neighbors: neighbors.into_boxed_slice(),
            distances: distances.into_boxed_slice(),
            n: num_points,
            k: num_neighbors,
        }
    }

    /// Creates a new ground truth structure
    pub fn new(n: usize, k: usize, neighbors: Vec<u32>, distances: Vec<f32>) -> GroundTruth {
        assert_eq!(neighbors.len(), n * k, "Neighbors array size mismatch");
        assert_eq!(distances.len(), n * k, "Distances array size mismatch");
        
        GroundTruth {
            neighbors: neighbors.into_boxed_slice(),
            distances: distances.into_boxed_slice(),
            n,
            k,
        }
    }

    /// Writes ground truth to a file in the standard format.
    pub fn write(&self, gt_filename: &Path) -> Result<()> {
        let mut file = File::create(gt_filename)?;
        
        // Write header: number of points and neighbors
        file.write_all(&(self.n as u32).to_le_bytes())?;
        file.write_all(&(self.k as u32).to_le_bytes())?;
        
        // Write neighbor IDs
        for &neighbor in self.neighbors.iter() {
            file.write_all(&neighbor.to_le_bytes())?;
        }
        
        // Write distances
        for &distance in self.distances.iter() {
            file.write_all(&distance.to_le_bytes())?;
        }
        
        Ok(())
    }

    /// returns the neighbors of a point
    pub fn get_neighbors(&self, i: usize) -> &[u32] {
        let start = i * self.k;
        &self.neighbors[start..start + self.k]
    }

    /// returns the distances of a point's true nearest neighbors
    pub fn get_distances(&self, i: usize) -> &[f32] {
        let start = i * self.k;
        &self.distances[start..start + self.k]
    }
}

/// Computes exact ground truth (nearest neighbors) for a query dataset against a reference dataset
pub fn compute_ground_truth<T>(
    query_dataset: &(impl Dataset<T> + Sync), 
    ref_dataset: &(impl Dataset<T> + Sync), 
    k: usize
) -> Result<GroundTruth> 
where 
    T: Numeric + Send + Sync
{
    use rayon::prelude::*;
    
    let n = query_dataset.size();
    let ref_size = ref_dataset.size();
    
    // Parallel computation of all query points
    let results: Vec<(Vec<u32>, Vec<f32>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            // Calculate distances to all reference points
            let mut distances: Vec<(f32, u32)> = (0..ref_size)
                .map(|j| {
                    let dist = query_dataset.compare_internal(i, j) as f32;
                    (dist, j as u32)
                })
                .collect();
            
            // Sort by distance
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            
            // Take the k nearest (or fewer if dataset is smaller than k)
            let top_k = std::cmp::min(k, distances.len());
            
            // Extract neighbors and distances
            let mut neighbors = Vec::with_capacity(k);
            let mut dists = Vec::with_capacity(k);
            
            // Store the results - iterate using take() instead of range loop
            distances.iter().take(top_k).for_each(|&(dist, idx)| {
                neighbors.push(idx);
                dists.push(dist); // No cast needed, already f32
            });
            
            // Pad with zeros if needed
            for _ in top_k..k {
                neighbors.push(u32::MAX);
                dists.push(f32::MAX);
            }
            
            (neighbors, dists)
        })
        .collect();
    
    // Merge results
    let mut all_neighbors = Vec::with_capacity(n * k);
    let mut all_distances = Vec::with_capacity(n * k);
    
    for (neighbors, distances) in results {
        all_neighbors.extend(neighbors);
        all_distances.extend(distances);
    }
    
    Ok(GroundTruth::new(n, k, all_neighbors, all_distances))
}
