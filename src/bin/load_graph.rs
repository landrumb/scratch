use clap::{Arg, Command};
use std::path::PathBuf;
use std::time::Instant;

use rand_distr::num_traits::ToPrimitive;
use rayon::prelude::*;
use scratch::data_handling::dataset::VectorDataset;
use scratch::data_handling::dataset_traits::Dataset;
use scratch::data_handling::fbin::read_fbin;
use scratch::graph::beam_search;
use scratch::graph::ClassicGraph;
use scratch::util::dataset::infer_dataset_paths;
use scratch::util::ground_truth::GroundTruth;
use scratch::util::recall::recall;

fn main() {
    let matches = Command::new("load_graph")
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
            Arg::new("graph")
                .long("graph")
                .value_name("FILE")
                .help("Path to graph file"),
        )
        .arg(
            Arg::new("gt")
                .long("gt")
                .value_name("FILE")
                .help("Path to ground truth file"),
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

    let graph_path: PathBuf = matches
        .get_one::<String>("graph")
        .map(PathBuf::from)
        .unwrap_or(inferred.outputs.join("vamana"));

    let gt_path: PathBuf = matches
        .get_one::<String>("gt")
        .map(PathBuf::from)
        .unwrap_or(inferred.gt);

    // Load dataset
    let mut start = Instant::now();
    let dataset: VectorDataset<f32> = read_fbin(&data_path);
    let elapsed = start.elapsed();
    println!(
        "read dataset in {}.{:03} seconds",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    // Load graph
    start = Instant::now();
    let graph = ClassicGraph::read(graph_path.to_str().unwrap()).unwrap();
    let elapsed = start.elapsed();
    println!(
        "read graph in {}.{:03} seconds",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    // Load queries
    let queries: VectorDataset<f32> = read_fbin(&query_path);

    // Run queries
    start = Instant::now();
    let results: Vec<Vec<u32>> = (0..queries.size())
        .into_par_iter()
        .map(|i| beam_search(queries.get(i), &graph, &dataset, 0, 10, None))
        .collect();

    let elapsed = start.elapsed();
    println!(
        "ran {} queries in {}.{:03} seconds ({} QPS)",
        queries.size(),
        elapsed.as_secs(),
        elapsed.subsec_millis(),
        queries.size().to_f64().unwrap() / elapsed.as_secs_f64()
    );

    // Load ground truth and compute recall
    let gt = GroundTruth::read(&gt_path);
    let graph_recall = (0..results.len())
        .map(|i| recall(results[i].as_slice(), gt.get_neighbors(i)))
        .sum::<f64>()
        / queries.size().to_f64().unwrap();
    println!("recall: {:05}", graph_recall);
}
