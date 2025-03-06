//! loads and holds ground truth from a ground truth file

use std::path::Path;
use std::fs::File;
use std::io::{Read, Result};

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
