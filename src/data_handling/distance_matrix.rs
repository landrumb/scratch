//! A wrapper around a dataset that stores a matrix of precomputed distances

use super::dataset_traits::{Dataset, Numeric};

use rayon::prelude::*;

pub struct DistanceMatrix<T> {
    dataset: Box<dyn Dataset<T>>,
    distances: Vec<Vec<f32>>,
    n: usize,
}

impl<T: Numeric> DistanceMatrix<T> {
    pub fn new(dataset: Box<dyn Dataset<T>>) -> DistanceMatrix<T> {
        let n = dataset.size();

        // Precompute the distance matrix
        let mut distances = vec![Vec::new(); n - 1];
        for i in 0..n {
            distances[i].reserve(n - i);
        }

        for i in 0..(n - 1) {
            for j in (i + 1)..n {
                distances[i].push(dataset.compare_internal(i, j) as f32);
            }
        }

        DistanceMatrix {
            dataset,
            distances,
            n,
        }
    }

    pub fn new_with_progress_bar(dataset: Box<dyn Dataset<T>>) -> DistanceMatrix<T> {
        use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};

        let n = dataset.size();

        // Precompute the distance matrix
        let mut distances = vec![Vec::new(); n - 1];
        for i in 0..(n - 1) {
            // distances[i].resize(n - i, 0.0);
            distances[i].reserve(n - i);
        }

        let pb = ProgressBar::new((n - 1) as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg} {wide_bar:.green/gray} {pos}/{len} [{elapsed_precise}]({eta})")
                .unwrap()
                .progress_chars("█▓░"),
        );
        pb.set_message("Building distance matrix");

        distances
            .par_iter_mut()
            .progress_with(pb.clone())
            .enumerate()
            .for_each(|(i, distance_vec)| {
                distance_vec.extend(
                    (i + 1..n)
                        // .into_par_iter()
                        .map(|j| dataset.compare_internal(i, j) as f32)
                        .collect::<Vec<f32>>(),
                );
            });

        let elapsed = pb.elapsed();

        println!(
            "Distance matrix built in {}.{} seconds",
            elapsed.as_secs(),
            elapsed.subsec_millis()
        );

        DistanceMatrix {
            dataset,
            distances,
            n,
        }
    }

    pub fn get_distance(&self, i: usize, j: usize) -> f32 {
        if i == j {
            return 0.0;
        }
        if i < j {
            self.distances[i][j - i - 1]
        } else {
            self.distances[j][i - j - 1]
        }
    }
}

impl<T: Numeric> Dataset<T> for DistanceMatrix<T> {
    fn compare_internal(&self, i: usize, j: usize) -> f64 {
        self.get_distance(i, j) as f64
    }

    fn compare(&self, q: &[T], i: usize) -> f64 {
        self.dataset.compare(q, i)
    }

    fn size(&self) -> usize {
        self.n
    }
    fn get(&self, i: usize) -> &[T] {
        self.dataset.get(i)
    }
}
