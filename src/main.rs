#![allow(unused_variables)]

use std::env::args;
use std::path::Path;
use std::time::Instant;

use rand_distr::num_traits::ToPrimitive;
use scratch::data_handling::dataset::VectorDataset;
use scratch::data_handling::dataset_traits::Dataset;
use scratch::data_handling::fbin::read_fbin;
use scratch::graph::beam_search::beam_search;
use scratch::graph::graph::{ClassicGraph, Graph};
use scratch::util::ground_truth::GroundTruth;
use scratch::util::recall::recall;

fn main() {
    let default_data_file = String::from("data/word2vec-google-news-300_50000_lowercase/base.fbin");
    let default_query_file = String::from("data/word2vec-google-news-300_50000_lowercase/query.fbin");
    let default_graph_file = String::from("data/word2vec-google-news-300_50000_lowercase/outputs/vamana");
    let default_gt_file = String::from("data/word2vec-google-news-300_50000_lowercase/GT");

    // presumably we'll have some command line arguments here

    let data_path_arg = args().nth(1).unwrap_or(default_data_file);
    let data_path = Path::new(&data_path_arg);

    let mut start = Instant::now();

    let dataset: VectorDataset<f32> = read_fbin(&data_path);

    let elapsed = start.elapsed();
    println!(
        "read dataset in {}.{:03} seconds",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );
    start = Instant::now();

    let graph = ClassicGraph::read(&default_graph_file).unwrap();

    let elapsed = start.elapsed();
    println!(
        "read graph in {}.{:03} seconds",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );
    

    // why bother timing reading queries
    let queries: VectorDataset<f32> = read_fbin(&Path::new(&default_query_file));

    start = Instant::now();

    let results: Vec<Vec<u32>> = (0..queries.size()).map(|i| beam_search(queries.get(i), &graph, &dataset, 0, 12)).collect();

    let elapsed = start.elapsed();
    println!(
        "ran {} queries in {}.{:03} seconds ({} QPS)",
        queries.size(),
        elapsed.as_secs(),
        elapsed.subsec_millis(),
        queries.size().to_f64().unwrap() / elapsed.as_secs_f64()
    );

    // load ground truth
    let gt = GroundTruth::read(&Path::new(&default_gt_file));

    // compute recall
    let recall = (0..results.len()).map(|i| recall(results[i].as_slice(), gt.get_neighbors(i))).sum::<f64>() / queries.size().to_f64().unwrap();

    println!("recall: {}", recall);
}
