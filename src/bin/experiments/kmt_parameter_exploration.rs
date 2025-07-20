use clap::{Arg, Command};
use std::{fs::OpenOptions, io::Write, path::Path, time::Instant};

use rand_distr::num_traits::ToPrimitive;
use rayon::prelude::*;
use scratch::util::dataset::infer_dataset_paths;
use scratch::{
    constructions::kmeans_tree::KMeansTree,
    data_handling::{dataset::VectorDataset, dataset_traits::Dataset, fbin::read_fbin},
    util::{ground_truth::GroundTruth, recall::recall},
};

fn main() {
    let matches = Command::new("kmt_parameter_exploration")
        .arg(
            Arg::new("dataset")
                .long("dataset")
                .short('d')
                .help("Dataset name or directory")
                .required(true),
        )
        .arg(
            Arg::new("branching_factor")
                .long("branching-factor")
                .short('b')
                .required(true),
        )
        .arg(
            Arg::new("spillover")
                .long("spillover")
                .short('s')
                .required(true),
        )
        .get_matches();

    let dataset = matches.get_one::<String>("dataset").unwrap();
    let inferred = infer_dataset_paths(dataset);

    let data_path = inferred.base;
    let query_path = inferred.query;
    let gt_path = inferred.gt;

    let branching_factor: usize = matches
        .get_one::<String>("branching_factor")
        .unwrap()
        .parse()
        .expect("Invalid branching factor");

    let spillover: usize = matches
        .get_one::<String>("spillover")
        .unwrap()
        .parse()
        .expect("Invalid spillover");

    // Fixed parameters (could be made configurable if needed)
    let max_leaf_size = 200;
    let max_iterations = 10;
    let epsilon = 0.01;

    // Use a fixed output file; open it in append mode (create if not exists)
    let output_file = "outputs/kmt_experiment_results.csv";
    let file_exists = Path::new(&output_file).exists();
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(output_file)
        .expect("Failed to open output file");

    // If file did not exist, write header including construction parameters
    if !file_exists {
        writeln!(
            file,
            "branching_factor,spillover,max_leaf_size,max_iterations,epsilon,beam_size,recall,qps,leaf_node_count,total_leaf_points"
        )
        .expect("Failed to write header");
    }

    // Load dataset
    let start = Instant::now();
    let dataset: VectorDataset<f32> = read_fbin(&data_path);
    let elapsed = start.elapsed();
    println!(
        "Read dataset in {}.{:03} seconds",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    // Load queries
    let queries: VectorDataset<f32> = read_fbin(&query_path);

    // Load ground truth
    let gt = GroundTruth::read(&gt_path);

    // Build KMeansTree index once
    println!(
        "Building KMeansTree (branching_factor={}, spillover={})",
        branching_factor, spillover
    );
    let start = Instant::now();
    let kmt = KMeansTree::build_with_spillover(
        &dataset,
        branching_factor,
        max_leaf_size,
        max_iterations,
        epsilon,
        spillover,
    );
    let elapsed = start.elapsed();
    println!(
        "Built KMT index in {}.{:03} seconds, has height {}",
        elapsed.as_secs(),
        elapsed.subsec_millis(),
        kmt.get_max_height()
    );

    // Debug info for KMT structure
    println!("KMT leaf node count: {}", kmt.get_leaf_count());
    println!(
        "KMT total point count in leaf nodes: {}",
        kmt.get_total_leaf_points()
    );
    println!("Dataset size: {}", dataset.size());

    // Array of beam sizes to test
    let beam_sizes = [1, 2, 5, 10, 20, 50, 100, 200, 500];

    // Run tests for each beam size
    for &beam_size in &beam_sizes {
        println!("Testing beam size: {}", beam_size);

        // Run queries with beam search
        let start = Instant::now();
        let kmt_results: Vec<Vec<u32>> = (0..queries.size())
            .into_par_iter()
            .map(|i| {
                kmt.query_beam_search(queries.get(i), beam_size, 10)
                    .iter()
                    .map(|r| r.0)
                    .collect()
            })
            .collect();

        let elapsed = start.elapsed();
        let qps = queries.size().to_f64().unwrap() / elapsed.as_secs_f64();

        println!(
            "Ran {} queries with beam width {} in {}.{:03} seconds ({} QPS)",
            queries.size(),
            beam_size,
            elapsed.as_secs(),
            elapsed.subsec_millis(),
            qps
        );

        // Compute recall
        let kmt_recall = (0..kmt_results.len())
            .map(|i| recall(kmt_results[i].as_slice(), gt.get_neighbors(i)))
            .sum::<f64>()
            / queries.size().to_f64().unwrap();

        println!("Recall: {}", kmt_recall);

        // Write results to CSV with all parameters and the timestamp
        writeln!(
            file,
            "{},{},{},{},{},{},{},{}",
            branching_factor,
            spillover,
            max_leaf_size,
            max_iterations,
            epsilon,
            beam_size,
            kmt_recall,
            qps
        )
        .expect("Failed to write to output file");

        println!("-------------------------");
    }

    println!("Experiment complete. Results saved to {}", output_file);
}
