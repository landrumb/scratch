use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::path::Path;

use scratch::clustering::kmeans::{kmeans, kmeans_subset};
use scratch::data_handling::dataset::VectorDataset;
use scratch::data_handling::fbin;
use scratch::graph::graph::IndexT;

// Size configurations for benchmarks
const SMALL_K: usize = 8;
const MEDIUM_K: usize = 16;
const LARGE_K: usize = 32;

// Iteration configurations
const LOW_ITER: usize = 5;
const MEDIUM_ITER: usize = 10;

// Convergence threshold
const EPSILON: f64 = 1e-4;

fn bench_kmeans_word2vec(c: &mut Criterion) {
    // Load the word2vec dataset
    let dataset_path_string =
        String::from("data/word2vec-google-news-300_50000_lowercase/base.fbin");
    let dataset_path = Path::new(&dataset_path_string);
    let dataset: VectorDataset<f32> = fbin::read_fbin(dataset_path);

    // Create benchmark group
    let mut group = c.benchmark_group("kmeans_word2vec");

    // Benchmark small subsets first (faster execution)
    // Small subset (1000 points)
    let small_subset_size = 1000.min(dataset.n);
    let small_indices: Vec<IndexT> = (0..small_subset_size as u32).collect();

    // Test configurations
    let small_configs = [
        ("small_k_low_iter", SMALL_K, LOW_ITER),
        ("medium_k_low_iter", MEDIUM_K, LOW_ITER),
    ];

    for (name, k, max_iter) in small_configs {
        group.bench_with_input(
            BenchmarkId::new("subset_1k", name),
            &(k, max_iter),
            |b, &(k, max_iter)| {
                b.iter(|| kmeans_subset(&dataset, k, max_iter, EPSILON, &small_indices))
            },
        );
    }

    // Medium subset (10000 points)
    let medium_subset_size = 10000.min(dataset.n);
    let medium_indices: Vec<IndexT> = (0..medium_subset_size as u32).collect();

    let medium_configs = [("small_k_low_iter", SMALL_K, LOW_ITER)];

    for (name, k, max_iter) in medium_configs {
        group.bench_with_input(
            BenchmarkId::new("subset_10k", name),
            &(k, max_iter),
            |b, &(k, max_iter)| {
                b.iter(|| kmeans_subset(&dataset, k, max_iter, EPSILON, &medium_indices))
            },
        );
    }

    // Finally, benchmark full dataset with small K and low iterations
    group.bench_function("full_dataset_small_k", |b| {
        b.iter(|| kmeans(&dataset, SMALL_K, LOW_ITER, EPSILON))
    });

    group.finish();
}

criterion_group!(benches, bench_kmeans_word2vec);
criterion_main!(benches);
