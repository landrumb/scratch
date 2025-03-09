//! basic kmeans implementation

use crate::{data_handling::dataset::VectorDataset, graph::graph::IndexT};
use rand::Rng;
use rayon::prelude::*;

#[cfg(feature = "verbose_kmeans")]
macro_rules! verbose_println {
    ($($arg:tt)*) => {
        println!($($arg)*);
    }
}

#[cfg(not(feature = "verbose_kmeans"))]
macro_rules! verbose_println {
    ($($arg:tt)*) => {};
}

pub fn kmeans_subset(
    dataset: &VectorDataset<f32>,
    k: usize,
    max_iter: usize,
    epsilon: f64,
    indices: &[IndexT],
) -> (Vec<f32>, Vec<usize>) {
    // Initialize centroids from random indices.
    let mut centroids = Vec::with_capacity(k * dataset.dim);
    let mut rng = rand::rng();
    for _ in 0..k {
        let rand_index = rng.random_range(0..indices.len());
        let centroid = dataset.get(indices[rand_index] as usize);
        centroids.extend_from_slice(centroid);
    }

    let mut assignments = vec![0; indices.len()];
    let mut iter = 0;

    loop {
        // Assign points to centroids in parallel.
        assignments = indices
            .par_iter()
            .map(|&idx| {
                let point = dataset.get(idx as usize);
                let mut min_idx = 0;
                let mut min_dist = f32::MAX;
                for j in 0..k {
                    let base = j * dataset.dim;
                    let mut dist = 0.0;
                    for d in 0..dataset.dim {
                        let diff = point[d] - centroids[base + d];
                        dist += diff * diff;
                    }
                    if dist < min_dist {
                        min_dist = dist;
                        min_idx = j;
                    }
                }
                min_idx
            })
            .collect();

        // Parallel reduction: sum contributions to new centroids.
        let (new_centroids, counts) = indices
            .par_iter()
            .enumerate()
            .fold(
                || (vec![0.0; k * dataset.dim], vec![0usize; k]),
                |(mut local_cents, mut local_counts), (i, &idx)| {
                    let assign = assignments[i];
                    let point = dataset.get(idx as usize);
                    local_counts[assign] += 1;
                    let base = assign * dataset.dim;
                    for d in 0..dataset.dim {
                        local_cents[base + d] += point[d];
                    }
                    (local_cents, local_counts)
                },
            )
            .reduce(
                || (vec![0.0; k * dataset.dim], vec![0usize; k]),
                |(mut acc_cents, mut acc_counts), (loc_cents, loc_counts)| {
                    for j in 0..k {
                        acc_counts[j] += loc_counts[j];
                        let base = j * dataset.dim;
                        for d in 0..dataset.dim {
                            acc_cents[base + d] += loc_cents[base + d];
                        }
                    }
                    (acc_cents, acc_counts)
                },
            );

        // Update centroids and calculate maximum change.
        let mut max_change = 0.0;
        for j in 0..k {
            if counts[j] > 0 {
                let base = j * dataset.dim;
                for d in 0..dataset.dim {
                    let new_val = new_centroids[base + d] / counts[j] as f32;
                    let change = (new_val - centroids[base + d]).abs() as f64;
                    if change > max_change {
                        max_change = change;
                    }
                    centroids[base + d] = new_val;
                }
            }
        }

        verbose_println!("Iteration {}: max change = {}", iter, max_change);

        iter += 1;
        if max_change < epsilon || iter >= max_iter {
            break;
        }
    }

    (centroids, assignments)
}

pub fn kmeans(dataset: &VectorDataset<f32>, k: usize, max_iter: usize, epsilon: f64) -> (Vec<f32>, Vec<usize>) {
    let indices: Vec<IndexT> = (0..dataset.n as u32).collect();
    kmeans_subset(dataset, k, max_iter, epsilon, &indices)
}
