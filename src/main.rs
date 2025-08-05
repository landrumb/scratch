#![allow(unused_variables)]

use std::env::args;
use std::path::Path;
use std::time::Instant;

use rand_distr::num_traits::ToPrimitive;
use scratch::constructions::ivf::IVFIndex;
use scratch::constructions::kmeans_tree::KMeansTree;
use scratch::data_handling::dataset::VectorDataset;
use scratch::data_handling::dataset_traits::Dataset;
use scratch::data_handling::fbin::read_fbin;
use scratch::graph::beam_search;
use scratch::graph::ClassicGraph;
use scratch::util::ground_truth::GroundTruth;
use scratch::util::recall::recall;

use rayon::prelude::*;

fn main() {
    let dataset_dir = String::from("data/word2vec-google-news-300_50000_lowercase");
    let default_data_file = format!("{dataset_dir}/base.fbin");
    let default_query_file = format!("{dataset_dir}/query.fbin");
    let default_graph_file = format!("{dataset_dir}/outputs/vamana");
    let default_gt_file = format!("{dataset_dir}/GT");

    // presumably we'll have some command line arguments here

    let data_path_arg = args().nth(1).unwrap_or(default_data_file);
    let data_path = Path::new(&data_path_arg);

    let mut start = Instant::now();

    let dataset: VectorDataset<f32> = read_fbin(data_path);

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
    let queries: VectorDataset<f32> = read_fbin(Path::new(&default_query_file));

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

    // load ground truth
    let gt = GroundTruth::read(Path::new(&default_gt_file));

    // compute recall
    let graph_recall = (0..results.len())
        .map(|i| recall(results[i].as_slice(), gt.get_neighbors(i)))
        .sum::<f64>()
        / queries.size().to_f64().unwrap();

    println!("recall: {graph_recall:05}");

    start = Instant::now();

    let ivf = IVFIndex::build(&dataset, 2, 100, 0.01);

    let elapsed = start.elapsed();
    println!(
        "built IVF index in {}.{:03} seconds",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );
    start = Instant::now();

    let ivf_results: Vec<Vec<u32>> = (0..queries.size())
        .into_par_iter()
        .map(|i| {
            ivf.query(queries.get(i), 1, 10)
                .iter()
                .map(|r| r.0)
                .collect()
        })
        .collect();

    let elapsed = start.elapsed();
    println!(
        "ran {} queries in {}.{:03} seconds ({} QPS)",
        queries.size(),
        elapsed.as_secs(),
        elapsed.subsec_millis(),
        queries.size().to_f64().unwrap() / elapsed.as_secs_f64()
    );

    let ivf_recall = (0..ivf_results.len())
        .map(|i| recall(ivf_results[i].as_slice(), gt.get_neighbors(i)))
        .sum::<f64>()
        / queries.size().to_f64().unwrap();

    println!("IVF recall: {ivf_recall:05}");

    start = Instant::now();

    let kmt = KMeansTree::build_bounded_leaf(&dataset, 5, 2000, 10, 0.01);

    let elapsed = start.elapsed();
    println!(
        "built KMT index in {}.{:03} seconds, has height {}",
        elapsed.as_secs(),
        elapsed.subsec_millis(),
        kmt.get_max_height()
    );

    start = Instant::now();

    let kmt_results: Vec<Vec<u32>> = (0..queries.size())
        .into_par_iter()
        .map(|i| kmt.query(queries.get(i), 10).iter().map(|r| r.0).collect())
        .collect();

    let elapsed = start.elapsed();
    println!(
        "ran {} KMT queries in {}.{:03} seconds ({} QPS)",
        queries.size(),
        elapsed.as_secs(),
        elapsed.subsec_millis(),
        queries.size().to_f64().unwrap() / elapsed.as_secs_f64()
    );

    let kmt_recall = (0..kmt_results.len())
        .map(|i| recall(kmt_results[i].as_slice(), gt.get_neighbors(i)))
        .sum::<f64>()
        / queries.size().to_f64().unwrap();

    println!("KMT recall: {kmt_recall:05}");
}
