use criterion::{criterion_group, criterion_main, Criterion};
use rand_distr::{Distribution, Normal};

use scratch::data_handling::{dataset, fbin};
use scratch::distance::euclidean::euclidean;

fn euclidean_group(c: &mut Criterion) {
    let mut rng = rand::rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut data = vec![0.0; 10000];
    for i in 0..10000 {
        data[i] = normal.sample(&mut rng);
    }

    let mut group = c.benchmark_group("euclidean");
    group.bench_function("euclidean 8", |b| {
        b.iter(|| euclidean(&data[0..], &data[5000..], 8))
    });
    // group.bench_function("euclidean 16", |b| b.iter(|| euclidean(&data[0..], &data[5000..], 16)));
    // group.bench_function("euclidean 32", |b| b.iter(|| euclidean(&data[0..], &data[5000..], 32)));
    group.bench_function("euclidean 64", |b| {
        b.iter(|| euclidean(&data[0..], &data[5000..], 64))
    });
    // group.bench_function("euclidean 100", |b| b.iter(|| euclidean(&data[0..], &data[5000..], 100)));
    // group.bench_function("euclidean 128", |b| b.iter(|| euclidean(&data[0..], &data[5000..], 128)));
    // group.bench_function("euclidean 256", |b| b.iter(|| euclidean(&data[0..], &data[5000..], 256)));
    group.bench_function("euclidean 300", |b| {
        b.iter(|| euclidean(&data[0..], &data[5000..], 300))
    });

    group.finish();
}

fn word2vec_euclidean_group(c: &mut Criterion) {
    // load the word2vec dataset
    let dataset_path_string =
        String::from("data/word2vec-google-news-300_50000_lowercase/base.fbin");
    let dataset_path = std::path::Path::new(&dataset_path_string);
    let dataset: dataset::VectorDataset<f32> = fbin::read_fbin(&dataset_path);

    // let mut rng = rand::rng();
    let mut group = c.benchmark_group("word2vec euclidean");
    group.bench_function("word2vec euclidean 0 1", |b| {
        b.iter(|| dataset.compare_euclidean(0, 1))
    });
    group.bench_function("word2vec euclidean 0 8", |b| {
        b.iter(|| dataset.compare_euclidean(0, 8))
    });
    group.bench_function("word2vec euclidean 0 1000", |b| {
        b.iter(|| dataset.compare_euclidean(0, 1000))
    });
}

criterion_group!(benches, word2vec_euclidean_group, euclidean_group);
criterion_main!(benches);
