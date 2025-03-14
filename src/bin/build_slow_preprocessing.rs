use std::env::args;
use std::path::Path;
use std::time::Instant;

use rand_distr::num_traits::ToPrimitive;
use rayon::prelude::*;
use scratch::constructions::slow_preprocessing::build_slow_preprocesssing;
use scratch::data_handling::dataset::VectorDataset;
use scratch::data_handling::dataset_traits::Dataset;
use scratch::data_handling::fbin::read_fbin;
use scratch::graph::beam_search;
use scratch::util::ground_truth::GroundTruth;
use scratch::util::recall::recall;

fn main() {
    let default_data_file = String::from("data/word2vec-google-news-300_50000_lowercase/base.fbin");
    let default_query_file =
        String::from("data/word2vec-google-news-300_50000_lowercase/query.fbin");
    // let default_graph_file =
    //     String::from("data/word2vec-google-news-300_50000_lowercase/outputs/vamana");
    let default_gt_file = String::from("data/word2vec-google-news-300_50000_lowercase/GT");

    // Parse arguments
    let data_path_arg = args().nth(1).unwrap_or(default_data_file);
    let data_path = Path::new(&data_path_arg);

    let query_path_arg = args().nth(2).unwrap_or(default_query_file);
    let query_path = Path::new(&query_path_arg);

    // let graph_path_arg = args().nth(3).unwrap_or(default_graph_file);

    let gt_path_arg = args().nth(4).unwrap_or(default_gt_file);
    let gt_path = Path::new(&gt_path_arg);

    // Load dataset
    let mut start = Instant::now();
    let dataset: VectorDataset<f32> = read_fbin(data_path);
    let elapsed = start.elapsed();
    println!(
        "read dataset in {}.{:03} seconds",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    // build the graph
    start = Instant::now();
    let graph = build_slow_preprocesssing(&dataset, 1.0);
    let elapsed = start.elapsed();
    println!(
        "built graph in {}.{:03} seconds",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    println!("Total edges: {}", graph.total_edges());
    println!("Average degree: {}", graph.total_edges() / dataset.n);
    println!("Max degree: {}", graph.max_degree());

    // Load queries
    let queries: VectorDataset<f32> = read_fbin(query_path);

    // Run queries
    start = Instant::now();
    let results: Vec<Vec<u32>> = (0..queries.size())
        .into_par_iter()
        .map(|i| beam_search(queries.get(i), &graph, &dataset, 0, 10))
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
    let gt = GroundTruth::read(gt_path);
    let graph_recall = (0..results.len())
        .map(|i| recall(results[i].as_slice(), gt.get_neighbors(i)))
        .sum::<f64>()
        / queries.size().to_f64().unwrap();
    println!("recall: {:.5}", graph_recall);
}
