use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::distr::{Distribution, Uniform};
use rand::Rng;
use std::fs;
use std::path::Path;

use scratch::graph::graph::{ClassicGraph, IndexT};

// Size configurations for benchmarks
const TINY_SIZE: usize = 100;
const SMALL_SIZE: usize = 1_000;
const MEDIUM_SIZE: usize = 10_000;
// const LARGE_SIZE: usize = 50_000; // Only for fast operations

// Degree configurations
const LOW_DEGREE: usize = 8;
const MED_DEGREE: usize = 32;
// const HIGH_DEGREE: usize = 128;

// Helper to ensure benchmark directories exist
fn ensure_benchmark_dir() {
    let dir = Path::new("target/benchmark");
    if !dir.exists() {
        fs::create_dir_all(dir).expect("Failed to create benchmark directory");
    }
}

// Helper to create a graph with random connections
fn create_random_graph(n: IndexT, r: usize, avg_connections: usize) -> ClassicGraph {
    let mut graph = ClassicGraph::new(n, r);
    let mut rng = rand::rng();
    let node_dist = Uniform::new(0, n).unwrap();

    for i in 0..n {
        // Determine connections for this node (slightly random around avg_connections)
        let connections = if avg_connections > 0 {
            let variance = (avg_connections as f64 * 0.2) as usize;
            let min_conn = avg_connections.saturating_sub(variance);
            let max_conn = (avg_connections + variance).min(r);
            rng.random_range(min_conn..=max_conn)
        } else {
            0
        };

        // Generate unique random connections
        let mut neighbors = Vec::with_capacity(connections);
        while neighbors.len() < connections {
            let target = node_dist.sample(&mut rng);
            if target != i && !neighbors.contains(&target) {
                neighbors.push(target);
            }
        }

        graph.append_neighbors(i, &neighbors);
    }

    graph
}

// Benchmark graph read operations
fn bench_graph_read(c: &mut Criterion) {
    ensure_benchmark_dir();
    let mut group = c.benchmark_group("graph_read");

    // Generate benchmark graphs of different sizes
    let configs = [
        ("tiny_low", TINY_SIZE, LOW_DEGREE, LOW_DEGREE / 2),
        ("small_low", SMALL_SIZE, LOW_DEGREE, LOW_DEGREE / 2),
        ("small_med", SMALL_SIZE, MED_DEGREE, MED_DEGREE / 2),
        ("medium_low", MEDIUM_SIZE, LOW_DEGREE, LOW_DEGREE / 2),
        ("medium_med", MEDIUM_SIZE, MED_DEGREE, MED_DEGREE / 2),
    ];

    for (name, size, max_degree, avg_degree) in configs {
        let graph_path = format!("target/benchmark/bench_graph_{}.bin", name);
        let graph = create_random_graph(size as IndexT, max_degree, avg_degree);
        graph
            .save(&graph_path)
            .expect("Failed to save benchmark graph");

        // Benchmark reading this graph
        group.bench_with_input(BenchmarkId::new("read", name), &graph_path, |b, path| {
            b.iter(|| {
                let _graph = ClassicGraph::read(path).expect("Failed to read graph in benchmark");
            });
        });
    }

    group.finish();
}

// Benchmark graph write operations
fn bench_graph_write(c: &mut Criterion) {
    ensure_benchmark_dir();
    let mut group = c.benchmark_group("graph_write");

    // Generate benchmark graphs of different sizes
    let configs = [
        ("tiny_low", TINY_SIZE, LOW_DEGREE, LOW_DEGREE / 2),
        ("small_low", SMALL_SIZE, LOW_DEGREE, LOW_DEGREE / 2),
        ("small_med", SMALL_SIZE, MED_DEGREE, MED_DEGREE / 2),
        ("medium_low", MEDIUM_SIZE, LOW_DEGREE, LOW_DEGREE / 2),
        ("medium_med", MEDIUM_SIZE, MED_DEGREE, MED_DEGREE / 2),
    ];

    for (name, size, max_degree, avg_degree) in configs {
        let graph = create_random_graph(size as IndexT, max_degree, avg_degree);

        // Benchmark writing this graph
        group.bench_with_input(BenchmarkId::new("write", name), &graph, |b, g| {
            b.iter(|| {
                let graph_path = format!("target/benchmark/bench_write_{}.bin", name);
                g.save(&graph_path)
                    .expect("Failed to write graph in benchmark");
            });
        });
    }

    group.finish();
}

// Benchmark full round-trip (read + write)
fn bench_graph_roundtrip(c: &mut Criterion) {
    ensure_benchmark_dir();
    let mut group = c.benchmark_group("graph_roundtrip");

    // Generate benchmark graphs of different sizes (using smaller sizes for this more intensive benchmark)
    let configs = [
        ("tiny_low", TINY_SIZE, LOW_DEGREE, LOW_DEGREE / 2),
        ("small_low", SMALL_SIZE, LOW_DEGREE, LOW_DEGREE / 2),
        ("small_med", SMALL_SIZE, MED_DEGREE, MED_DEGREE / 2),
    ];

    for (name, size, max_degree, avg_degree) in configs {
        let input_path = format!("target/benchmark/bench_rt_in_{}.bin", name);
        let output_path = format!("target/benchmark/bench_rt_out_{}.bin", name);

        // Create the initial graph
        let graph = create_random_graph(size as IndexT, max_degree, avg_degree);
        graph
            .save(&input_path)
            .expect("Failed to save initial benchmark graph");

        // Benchmark the round-trip
        group.bench_with_input(
            BenchmarkId::new("roundtrip", name),
            &(input_path, output_path),
            |b, (in_path, out_path)| {
                b.iter(|| {
                    let graph = ClassicGraph::read(in_path)
                        .expect("Failed to read graph in roundtrip benchmark");
                    graph
                        .save(out_path)
                        .expect("Failed to write graph in roundtrip benchmark");
                });
            },
        );
    }

    group.finish();
}

// Benchmark graph creation (for comparison baseline)
fn bench_graph_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_creation");

    let configs = [
        ("tiny_low", TINY_SIZE, LOW_DEGREE, LOW_DEGREE / 2),
        ("small_low", SMALL_SIZE, LOW_DEGREE, LOW_DEGREE / 2),
        ("small_med", SMALL_SIZE, MED_DEGREE, MED_DEGREE / 2),
        ("medium_low", MEDIUM_SIZE, LOW_DEGREE, LOW_DEGREE / 2),
    ];

    for (name, size, max_degree, avg_degree) in configs {
        group.bench_with_input(
            BenchmarkId::new("create", name),
            &(size, max_degree, avg_degree),
            |b, &(size, max_degree, avg_degree)| {
                b.iter(|| create_random_graph(size as IndexT, max_degree, avg_degree));
            },
        );
    }

    group.finish();
}

// Benchmark specific operations within the graph
fn bench_graph_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_operations");

    // Create a medium-sized graph for operation benchmarks
    let n = MEDIUM_SIZE as IndexT;
    let r = MED_DEGREE;
    let graph = create_random_graph(n, r, r / 2);

    // Benchmark neighborhood access
    group.bench_function("get_neighborhood", |b| {
        let mut rng = rand::rng();
        b.iter(|| {
            let node = rng.random_range(0..n);
            graph.get_neighborhood(node)
        });
    });

    // Benchmark edge range access
    group.bench_function("get_edge_range", |b| {
        let mut rng = rand::rng();
        b.iter(|| {
            let node = rng.random_range(0..n);
            graph.get_edge_range(node)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_graph_read,
    bench_graph_write,
    bench_graph_roundtrip,
    bench_graph_creation,
    bench_graph_operations
);
criterion_main!(benches);
