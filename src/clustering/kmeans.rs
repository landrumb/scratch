//! basic kmeans implementation

use crate::{data_handling::dataset::VectorDataset, graph::graph::IndexT};
use rand::Rng;

pub fn kmeans_subset(dataset: &VectorDataset<f32>, k: usize, max_iter: usize, epsilon: f64, indices: &[IndexT]) -> Vec<f32> {
    let mut centroids = Vec::with_capacity(k * dataset.dim);
    let mut assignments = vec![0; indices.len()];
    let mut counts = vec![0; k];
    let mut new_centroids = vec![0.0; k * dataset.dim];
    let mut distances = vec![0.0; k];
    let mut iter = 0;

    // initialize centroids as random points in the indices
    let mut rng = rand::rng();
    for _ in 0..k {
        let centroid = dataset.get(indices[rng.random_range(0..indices.len())] as usize);
        centroids.extend_from_slice(centroid);
    }

    loop {
        // Reset assignments and counts
        for assignment in assignments.iter_mut() {
            *assignment = 0;
        }
        for count in counts.iter_mut() {
            *count = 0;
        }
        for centroid in new_centroids.iter_mut() {
            *centroid = 0.0;
        }

        // Assign points to nearest centroid
        for (i, &idx) in indices.iter().enumerate() {
            let point = dataset.get(idx as usize);
            
            // Calculate distance to each centroid
            for j in 0..k {
                let mut dist = 0.0;
                for d in 0..dataset.dim {
                    let diff = point[d] - centroids[j * dataset.dim + d];
                    dist += diff * diff;
                }
                distances[j] = dist;
            }
            
            // Find closest centroid
            let mut min_dist = distances[0];
            let mut min_idx = 0;
            for j in 1..k {
                if distances[j] < min_dist {
                    min_dist = distances[j];
                    min_idx = j;
                }
            }
            
            assignments[i] = min_idx;
            counts[min_idx] += 1;
            
            // Add point to new centroid calculation
            for d in 0..dataset.dim {
                new_centroids[min_idx * dataset.dim + d] += point[d];
            }
        }
        
        // Calculate new centroids and check for convergence
        let mut max_change = 0.0;
        for j in 0..k {
            if counts[j] > 0 {
                for d in 0..dataset.dim {
                    let idx = j * dataset.dim + d;
                    new_centroids[idx] /= counts[j] as f32;
                    let change = (new_centroids[idx] - centroids[idx]).abs() as f64;
                    if change > max_change {
                        max_change = change;
                    }
                    centroids[idx] = new_centroids[idx];
                    new_centroids[idx] = 0.0;
                }
            }
        }
        
        iter += 1;
        
        // Check termination conditions
        if max_change < epsilon || iter >= max_iter {
            break;
        }
    }
    
    centroids
}

pub fn kmeans(dataset: &VectorDataset<f32>, k: usize, max_iter: usize, epsilon: f64) -> Vec<f32> {
    let indices: Vec<IndexT> = (0..dataset.n as u32).collect();
    kmeans_subset(dataset, k, max_iter, epsilon, &indices)
}