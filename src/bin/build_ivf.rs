use std::env::args;
use std::path::Path;
use std::time::Instant;

use rand_distr::num_traits::ToPrimitive;
use rayon::prelude::*;
use scratch::constructions::ivf::IVFIndex;
use scratch::data_handling::dataset::VectorDataset;
use scratch::data_handling::dataset_traits::Dataset;
use scratch::data_handling::fbin::read_fbin;
use scratch::util::ground_truth::GroundTruth;
use scratch::util::recall::recall;

fn main() {
    let default_data_file = String::from("data/word2vec-google-news-300_50000_lowercase/base.fbin");
    let default_query_file = String::from("data/word2vec-google-news-300_50000_lowercase/query.fbin");
    let default_gt_file = String::from("data/word2vec-google-news-300_50000_lowercase/GT");

    // Parse arguments
    let data_path_arg = args().nth(1).unwrap_or(default_data_file);
    let data_path = Path::new(&data_path_arg);
    
    let query_path_arg = args().nth(2).unwrap_or(default_query_file);
    let query_path = Path::new(&query_path_arg);
    
    let gt_path_arg = args().nth(3).unwrap_or(default_gt_file);
    let gt_path = Path::new(&gt_path_arg);
    
    // Parse IVF parameters with defaults from original code
    let clusters_arg = args().nth(4).unwrap_or("2".to_string());
    let clusters: usize = clusters_arg.parse().unwrap_or(2);
    
    let probes_arg = args().nth(5).unwrap_or("1".to_string());
    let probes: usize = probes_arg.parse().unwrap_or(1);
    
    let max_iterations_arg = args().nth(6).unwrap_or("100".to_string());
    let max_iterations: usize = max_iterations_arg.parse().unwrap_or(100);
    
    let epsilon_arg = args().nth(7).unwrap_or("0.01".to_string());
    let epsilon: f32 = epsilon_arg.parse().unwrap_or(0.01);

    // Load dataset
    let mut start = Instant::now();
    let dataset: VectorDataset<f32> = read_fbin(data_path);
    let elapsed = start.elapsed();
    println!(
        "read dataset in {}.{:03} seconds",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    // Build IVF index
    start = Instant::now();
    let ivf = IVFIndex::build(&dataset, clusters, max_iterations, epsilon);
    let elapsed = start.elapsed();
    println!(
        "built IVF index in {}.{:03} seconds",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    // Load queries
    let queries: VectorDataset<f32> = read_fbin(query_path);

    // Run queries
    start = Instant::now();
    let ivf_results: Vec<Vec<u32>> = (0..queries.size())
        .into_par_iter()
        .map(|i| ivf.query(queries.get(i), probes, 10)
            .iter()
            .map(|r| r.0)
            .collect()
        )
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
    let ivf_recall = (0..ivf_results.len()).map(|i| recall(ivf_results[i].as_slice(), gt.get_neighbors(i))).sum::<f64>() / queries.size().to_f64().unwrap();
    println!("IVF recall: {:05}", ivf_recall);
}