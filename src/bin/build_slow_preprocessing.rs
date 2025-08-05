use clap::{Arg, Command};
use std::path::PathBuf;
use std::time::Instant;

use rand_distr::num_traits::ToPrimitive;
use rayon::prelude::*;
use scratch::constructions::neighbor_selection::{incremental_greedy, PairwiseDistancesHandler};
use scratch::constructions::slow_preprocessing::build_global_local_graph;
use scratch::data_handling::dataset::VectorDataset;
use scratch::data_handling::dataset_traits::Dataset;
use scratch::data_handling::fbin::{read_fbin, read_fbin_subset};
use scratch::graph::{beam_search, ClassicGraph, Graph, IndexT};
use scratch::util::ground_truth::compute_ground_truth;
use scratch::util::recall::recall;

// static SUBSET_SIZE: Option<&'static str> = option_env!("SUBSET_SIZE");

fn main() {
    let matches = Command::new("build_slow_preprocessing")
        .arg(
            Arg::new("dataset")
                .long("dataset")
                .short('d')
                .help("Dataset name or directory")
                .required(true),
        )
        .arg(
            Arg::new("base")
                .long("base")
                .value_name("FILE")
                .help("Path to base.fbin"),
        )
        .arg(
            Arg::new("query")
                .long("query")
                .value_name("FILE")
                .help("Path to query.fbin"),
        )
        .arg(
            Arg::new("gt")
                .long("gt")
                .value_name("FILE")
                .help("Path to ground truth file"),
        )
        .get_matches();

    let dataset = matches.get_one::<String>("dataset").unwrap();
    let inferred = scratch::util::dataset::infer_dataset_paths(dataset);

    let data_path: PathBuf = matches
        .get_one::<String>("base")
        .map(PathBuf::from)
        .unwrap_or(inferred.base);

    let query_path: PathBuf = matches
        .get_one::<String>("query")
        .map(PathBuf::from)
        .unwrap_or(inferred.query);

    let _gt_path: PathBuf = matches
        .get_one::<String>("gt")
        .map(PathBuf::from)
        .unwrap_or(inferred.gt);

    // Load dataset
    let mut start = Instant::now();
    // let dataset: VectorDataset<f32> = read_fbin(data_path);

    let dataset: VectorDataset<f32>;

    if let Ok(subset_size_str) = std::env::var("SUBSET_SIZE") {
        dataset = read_fbin_subset(&data_path, subset_size_str.parse().unwrap());
        println!("Using subset of size {subset_size_str}");
    } else {
        dataset = read_fbin(&data_path);
        println!("Using full dataset of size {}", dataset.size());
    }

    let elapsed = start.elapsed();
    println!(
        "read dataset in {}.{:03} seconds",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    // build distance matrix
    // let dataset = DistanceMatrix::new_with_progress_bar(dataset);

    // build the graph
    start = Instant::now();
    // let graph = build_slow_preprocesssing(&dataset, 1.01);

    let nested_boxed_distances = (0..dataset.size())
        .into_par_iter()
        .map(|i| {
            dataset
                .brute_force_internal(i)
                .iter()
                .map(|(j, dist)| (*j as IndexT, *dist))
                .collect::<Box<[(IndexT, f32)]>>()
        })
        .collect::<Box<[Box<[(IndexT, f32)]>]>>();
    let pairwise_distances = PairwiseDistancesHandler::new(nested_boxed_distances);

    // let graph = build_global_local_graph(&subset, |center, candidates| {
    //     robust_prune_unbounded(candidates.iter().map(|i| (*i, subset.compare_internal(*i as usize, center as usize).to_f32().unwrap())).collect(), 1.01, &subset)
    // });

    let graph = build_global_local_graph(&dataset, |center, candidates| {
        incremental_greedy(center, candidates, &dataset, 1.0, &pairwise_distances, None)
    });
    // let graph = build_global_local_graph(&subset, |center, candidates| {
    //     naive_semi_greedy_prune(center, candidates, &subset, 1.01, &pairwise_distances)
    // });

    let elapsed = start.elapsed();
    println!("built graph in {elapsed:?}");

    println!("Total edges: {}", graph.total_edges());
    println!("Average degree: {}", graph.total_edges() / dataset.size());
    println!("Max degree: {}", graph.max_degree());

    // Load queries
    let queries: VectorDataset<f32> = read_fbin(&query_path);

    // Run queries
    start = Instant::now();
    let results: Vec<Vec<u32>> = (0..queries.size())
        // let results: Vec<Vec<u32>> = (0..subset_size)
        .into_par_iter()
        .map(|i| beam_search(queries.get(i), &graph, &dataset, 0, 20, None))
        .collect();

    let elapsed = start.elapsed();
    println!(
        "ran {} queries in {:?} ({} QPS)",
        queries.size(),
        elapsed,
        queries.size().to_f64().unwrap() / elapsed.as_secs_f64()
    );

    // // Load ground truth and compute recall
    // let gt = GroundTruth::read(gt_path);

    // compute gt over the subset
    let gt = compute_ground_truth(&queries, &dataset, 10).unwrap();

    let graph_recall = (0..results.len())
        .map(|i| recall(results[i].as_slice(), gt.get_neighbors(i)))
        .sum::<f64>()
        / results.len().to_f64().unwrap();
    println!("recall: {graph_recall:.5}");

    // write the graph to disk
    let classic_graph = ClassicGraph::from(&graph);
    classic_graph
        .save(
            data_path
                .parent()
                .unwrap()
                .join("outputs")
                .join("greedy")
                .to_str()
                .unwrap(),
        )
        .unwrap();
    println!("saved graph to disk");

    // do the same querying with the dataset itself
    let dataset_results: Vec<Vec<u32>> = (0..dataset.size())
        .into_par_iter()
        .map(|i| beam_search(dataset.get(i), &graph, &dataset, 0, 1, None))
        .collect();

    let internal_gt = compute_ground_truth(&dataset, &dataset, 2).unwrap();

    let dataset_recall = (0..dataset_results.len())
        .map(|i| {
            recall(
                dataset_results[i].as_slice(),
                &internal_gt.get_neighbors(i)[..1],
            )
        })
        .sum::<f64>()
        / dataset_results.len().to_f64().unwrap();

    println!("self recall: {dataset_recall:.5}");

    let n_points_connected_to_nearest_neighbor = (0..dataset.size())
        .into_par_iter()
        .filter(|i| {
            let neighbors = graph.neighbors(*i as IndexT);
            let nearest_neighbor = internal_gt.get_neighbors(*i)[1];
            neighbors.contains(&nearest_neighbor)
        })
        .count();

    println!(
        "Fraction of points connected to their nearest neighbor: {:.5}",
        n_points_connected_to_nearest_neighbor as f64 / dataset.size() as f64
    );

    // how many points have an incoming edge from their nearest neighbor (should be all of them)
    let n_points_connected_to_inverse_nn = (0..dataset.size())
        .into_par_iter()
        .filter(|i| {
            let nearest_neighbor = internal_gt.get_neighbors(*i)[1];
            let neighbors = graph.neighbors(nearest_neighbor);
            neighbors.contains(&(*i as IndexT))
        })
        .count();
    println!(
        "Fraction of points with an incoming edge from their nearest neighbor: {:.5}",
        n_points_connected_to_inverse_nn as f64 / dataset.size() as f64
    );

    // print the results of the first 10 queries in the dataset
    // for i in 0..10 {
    //     println!("Results {}: {:?}", i, results[i]);
    //     println!("Ground truth {}: {:?}", i, gt.get_neighbors(i));
    // }
}
