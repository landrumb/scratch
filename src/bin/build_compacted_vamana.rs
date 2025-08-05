use clap::{Arg, Command};
use scratch::util::clique::{greedy_independent_cliques, maximal_cliques};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use rand_distr::num_traits::ToPrimitive;
use rayon::prelude::*;
use scratch::constructions::vamana::build_vamana_graph;
use scratch::data_handling::dataset::{Subset, VectorDataset};
use scratch::data_handling::dataset_traits::Dataset;
use scratch::data_handling::fbin::read_fbin;
use scratch::graph::{beam_search, ClassicGraph, CompactedGraphIndex, IndexT, VectorGraph};
use scratch::util::dataset::infer_dataset_paths;
use scratch::util::ground_truth::{compute_ground_truth, GroundTruth};
use scratch::util::recall::recall;

// static SUBSET_SIZE: Option<&'static str> = option_env!("SUBSET_SIZE");

fn main() {
    let matches = Command::new("build_vamana")
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
        .arg(
            Arg::new("subset_size")
                .value_parser(clap::value_parser!(usize))
                .long("subset-size")
                .value_name("SIZE")
                .help("Size of subset to use for building the graph"),
        )
        .get_matches();

    let dataset = matches.get_one::<String>("dataset").unwrap();
    let inferred = infer_dataset_paths(dataset);

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

    let dataset: Arc<VectorDataset<f32>> = Arc::new(read_fbin(&data_path));
    let mut subset_size: usize = dataset.size();

    if let Some(&subset_size_arg) = matches.get_one::<usize>("subset_size") {
        subset_size = subset_size_arg;
        println!("Using subset of size {subset_size}");
    } else {
        println!("Using full dataset of size {}", dataset.size());
    }

    let subset_indices = (0..subset_size).collect::<Vec<usize>>();
    let subset = Subset::new(dataset.clone(), subset_indices);

    let elapsed = start.elapsed();
    println!(
        "read dataset in {}.{:03} seconds",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    // build the graph, or load it from disk if it exists
    start = Instant::now();

    let graph_path = data_path.parent().unwrap().join("outputs/vamana.graph");
    let graph: VectorGraph;
    if graph_path.exists() {
        graph = ClassicGraph::read(graph_path.to_str().unwrap())
            .ok()
            .unwrap()
            .into();
        println!("loaded graph from disk");
    } else {
        println!("building graph");
        graph = build_vamana_graph(&subset, 1.01, 8, 100, Some(500), Some(0));

        let elapsed = start.elapsed();
        println!("built graph in {elapsed:?}");
    }
    println!("Total edges: {}", graph.total_edges());
    println!("Average degree: {}", graph.total_edges() / subset.size());
    println!("Max degree: {}", graph.max_degree());

    // Load queries
    let queries: VectorDataset<f32> = read_fbin(&query_path);

    // Run queries
    start = Instant::now();
    let results: Vec<Vec<u32>> = (0..queries.size())
        // let results: Vec<Vec<u32>> = (0..subset_size)
        .into_par_iter()
        .map(|i| beam_search(queries.get(i), &graph, &subset, 0, 20, None))
        .collect();

    let elapsed = start.elapsed();
    println!(
        "ran {} queries in {:?} ({} QPS)",
        queries.size(),
        elapsed,
        queries.size().to_f64().unwrap() / elapsed.as_secs_f64()
    );

    // Load ground truth and compute recall
    let gt: GroundTruth;
    if _gt_path.exists() {
        gt = GroundTruth::read(&_gt_path);
    } else {
        gt = compute_ground_truth(&queries, &subset, 10).unwrap();
    }

    let graph_recall = (0..results.len())
        .map(|i| recall(results[i].as_slice(), gt.get_neighbors(i)))
        .sum::<f64>()
        / results.len().to_f64().unwrap();
    println!("recall: {graph_recall:.5}");

    // // write the graph to disk
    // let classic_graph = ClassicGraph::from(&graph);
    // // if the output directory does not exist, create it
    // std::fs::create_dir_all(graph_path.parent().unwrap()).expect("could not create output directory");
    // classic_graph
    //     .save(graph_path.to_str().unwrap())
    //     .unwrap();

    // // do the same querying with the dataset itself
    // let dataset_results: Vec<Vec<u32>> = (0..subset_size)
    //     .into_par_iter()
    //     .map(|i| beam_search(subset.get(i), &graph, &subset, 0, 1, None))
    //     .collect();

    // let internal_gt = compute_ground_truth(&subset, &subset, 2).unwrap();

    // let dataset_recall = (0..dataset_results.len())
    //     .map(|i| {
    //         recall(
    //             dataset_results[i].as_slice(),
    //             &internal_gt.get_neighbors(i)[..1],
    //         )
    //     })
    //     .sum::<f64>()
    //     / dataset_results.len().to_f64().unwrap();

    // println!("self recall: {:.5}", dataset_recall);

    // let n_points_connected_to_nearest_neighbor = (0..subset.size())
    //     .into_par_iter()
    //     .filter(|i| {
    //         let neighbors = graph.neighbors(*i as IndexT);
    //         let nearest_neighbor = internal_gt.get_neighbors(*i)[1];
    //         neighbors.contains(&nearest_neighbor)
    //     })
    //     .count();

    // println!(
    //     "Fraction of points connected to their nearest neighbor: {:.5}",
    //     n_points_connected_to_nearest_neighbor as f64 / subset.size() as f64
    // );

    // // how many points have an incoming edge from their nearest neighbor (should be all of them)
    // let n_points_connected_to_inverse_nn = (0..subset.size())
    //     .into_par_iter()
    //     .filter(|i| {
    //         let nearest_neighbor = internal_gt.get_neighbors(*i)[1];
    //         let neighbors = graph.neighbors(nearest_neighbor);
    //         neighbors.contains(&(*i as IndexT))
    //     })
    //     .count();
    // println!(
    //     "Fraction of points with an incoming edge from their nearest neighbor: {:.5}",
    //     n_points_connected_to_inverse_nn as f64 / subset.size() as f64
    // );

    println!("---- Building compacted graph ----");
    // finding all cliques
    let cliques = maximal_cliques(&graph);
    println!("Found {} cliques", cliques.len());
    println!(
        "Largest clique: {:?}",
        cliques.iter().max_by_key(|c| c.len()).unwrap()
    );

    let independent_cliques = greedy_independent_cliques(&cliques);
    println!("Found {} independent cliques", independent_cliques.len());
    println!(
        "Largest independent clique: {:?}",
        independent_cliques.iter().max_by_key(|c| c.len()).unwrap()
    );

    let independent_cliques_box: Box<[Box<[IndexT]>]> = independent_cliques
        .into_iter()
        .map(|c| c.into_boxed_slice())
        .collect();

    let start = Instant::now();
    // build the compacted graph with the independent cliques
    let compacted_graph =
        CompactedGraphIndex::build_memory_inefficient(graph, dataset, independent_cliques_box);
    let elapsed = start.elapsed();
    println!("Built compacted graph in {elapsed:?}");

    let start = Instant::now();
    // run queries on the compacted graph
    let results: Vec<Vec<u32>> = (0..queries.size())
        .into_par_iter()
        .map(|i| compacted_graph.beam_search_post_expansion(queries.get(i), 20))
        .collect();
    let elapsed = start.elapsed();
    println!("Ran queries on compacted graph in {elapsed:?}");

    println!("Graph size: {}", compacted_graph.graph_size());

    // compute recall
    let recall = (0..results.len())
        .map(|i| recall(results[i].as_slice(), gt.get_neighbors(i)))
        .sum::<f64>()
        / results.len().to_f64().unwrap();
    println!("Recall: {recall:.5}");
}
