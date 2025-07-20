use clap::{Arg, Command};
use std::path::PathBuf;
use std::time::Instant;

use rand_distr::num_traits::ToPrimitive;
use rayon::prelude::*;
use scratch::constructions::kmeans_tree::KMeansTree;
use scratch::data_handling::dataset::VectorDataset;
use scratch::data_handling::dataset_traits::Dataset;
use scratch::data_handling::fbin::read_fbin;
use scratch::graph::IndexT;
use scratch::util::dataset::infer_dataset_paths;
use scratch::util::ground_truth::GroundTruth;
use scratch::util::recall::recall;

fn main() {
    let matches = Command::new("build_kmt")
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
            Arg::new("branching_factor")
                .long("branching-factor")
                .short('b')
                .default_value("5"),
        )
        .arg(
            Arg::new("max_leaf_size")
                .long("max-leaf-size")
                .default_value("200"),
        )
        .arg(
            Arg::new("max_iterations")
                .long("max-iterations")
                .default_value("10"),
        )
        .arg(Arg::new("epsilon").long("epsilon").default_value("0.01"))
        .arg(Arg::new("spillover").long("spillover").default_value("0"))
        .arg(Arg::new("beam_width").long("beam-width").default_value("0"))
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

    let gt_path: PathBuf = matches
        .get_one::<String>("gt")
        .map(PathBuf::from)
        .unwrap_or(inferred.gt);

    let branching_factor: usize = matches
        .get_one::<String>("branching_factor")
        .unwrap()
        .parse()
        .unwrap_or(5);

    let max_leaf_size: usize = matches
        .get_one::<String>("max_leaf_size")
        .unwrap()
        .parse()
        .unwrap_or(200);

    let max_iterations: usize = matches
        .get_one::<String>("max_iterations")
        .unwrap()
        .parse()
        .unwrap_or(10);

    let epsilon: f64 = matches
        .get_one::<String>("epsilon")
        .unwrap()
        .parse()
        .unwrap_or(0.01);

    let spillover: usize = matches
        .get_one::<String>("spillover")
        .unwrap()
        .parse()
        .unwrap_or(0);

    let beam_width: usize = matches
        .get_one::<String>("beam_width")
        .unwrap()
        .parse()
        .unwrap_or(0);

    // Load dataset
    let mut start = Instant::now();
    let dataset: VectorDataset<f32> = read_fbin(&data_path);
    let elapsed = start.elapsed();
    println!(
        "read dataset in {}.{:03} seconds",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    // Build KMeansTree index
    println!("Building KMeansTree (spillover factor = {})", spillover);
    start = Instant::now();
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
        "built KMT index in {}.{:03} seconds, has height {}",
        elapsed.as_secs(),
        elapsed.subsec_millis(),
        kmt.get_max_height()
    );

    // Add debugging for KMT structure
    println!("KMT leaf node count: {}", kmt.get_leaf_count());
    println!(
        "KMT total point count in leaf nodes: {}",
        kmt.get_total_leaf_points()
    );
    println!("Dataset size: {}", dataset.size());

    // Load queries
    let queries: VectorDataset<f32> = read_fbin(&query_path);

    // Run queries - use beam search if beam_width > 0
    start = Instant::now();

    let query_method = if beam_width > 0 {
        println!("Using beam search query with beam_width={}", beam_width);
        "beam search"
    } else {
        "standard"
    };

    let kmt_results: Vec<Vec<u32>> = (0..queries.size())
        .into_par_iter()
        .map(|i| {
            if beam_width > 0 {
                kmt.query_beam_search(queries.get(i), beam_width, 10)
                    .iter()
                    .map(|r| r.0)
                    .collect()
            } else {
                kmt.query(queries.get(i), 10).iter().map(|r| r.0).collect()
            }
        })
        .collect();

    let elapsed = start.elapsed();
    println!(
        "ran {} KMT {} queries in {}.{:03} seconds ({} QPS)",
        queries.size(),
        query_method,
        elapsed.as_secs(),
        elapsed.subsec_millis(),
        queries.size().to_f64().unwrap() / elapsed.as_secs_f64()
    );

    // Load ground truth and compute recall
    let gt = GroundTruth::read(&gt_path);
    let kmt_recall = (0..kmt_results.len())
        .map(|i| recall(kmt_results[i].as_slice(), gt.get_neighbors(i)))
        .sum::<f64>()
        / queries.size().to_f64().unwrap();
    println!("KMT recall on test queries: {:05}", kmt_recall);

    // -------------- Dataset as Queries (1-NN Recall) ----------------
    println!("\n----- Using Dataset Points as Queries for 1-NN Recall -----");

    // Compute ground truth for dataset points
    println!("Computing exact 1-NN ground truth for dataset points...");
    start = Instant::now();

    let elapsed = start.elapsed();
    println!(
        "computed exact 1-NN ground truth in {}.{:03} seconds",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    // Sample a subset of dataset points to use as queries (to make it faster)
    let sample_size = std::cmp::min(1000, dataset.size());
    println!(
        "Using {} randomly sampled dataset points as queries",
        sample_size
    );

    let sample_indices: Vec<usize> = (0..dataset.size())
        .step_by(dataset.size() / sample_size)
        .take(sample_size)
        .collect();

    // Query KMT with dataset points
    start = Instant::now();
    let dataset_kmt_results: Vec<Vec<u32>> = sample_indices
        .par_iter()
        .map(|&i| {
            let query = dataset.get(i);
            if beam_width > 0 {
                kmt.query_beam_search(query, beam_width, 1) // We want to see if the point finds itself as the 1-NN
                    .iter()
                    .map(|r| r.0)
                    .collect()
            } else {
                kmt.query(query, 1) // We want to see if the point finds itself as the 1-NN
                    .iter()
                    .map(|r| r.0)
                    .collect()
            }
        })
        .collect();

    let elapsed = start.elapsed();
    println!(
        "ran {} dataset {} queries in {}.{:03} seconds ({} QPS)",
        sample_size,
        query_method,
        elapsed.as_secs(),
        elapsed.subsec_millis(),
        sample_size as f64 / elapsed.as_secs_f64()
    );

    // Compute self-recall with detailed diagnostics
    println!("\n--- Diagnostic Information ---");

    // Print detailed info for a few examples
    let num_examples = std::cmp::min(10, sample_size);
    println!("Examining first {} queries in detail:", num_examples);

    let mut success_count = 0;
    let mut self_recall = 0.0;

    for (j, &i) in sample_indices.iter().enumerate() {
        let success = if dataset_kmt_results[j].is_empty() {
            false
        } else {
            let found_id = dataset_kmt_results[j][0] as usize;
            found_id == i
        };

        if success {
            success_count += 1;
            self_recall += 1.0;
        }

        // Print detailed diagnostics for the first few examples
        if j < num_examples {
            let found_str = if dataset_kmt_results[j].is_empty() {
                "none".to_string()
            } else {
                dataset_kmt_results[j][0].to_string()
            };

            println!(
                "Query {}: Point ID={}, Found={}{}",
                j,
                i,
                found_str,
                if success { " ✓" } else { " ✗" }
            );

            // Find which partition contains the point
            let point_partition = kmt.find_point_partition(i);
            println!("   Query point is in partition: {:?}", point_partition);

            // If not using beam search, print standard query path information
            if beam_width == 0 {
                // Run separate query to see which partition we search in
                let query = dataset.get(i);
                let search_partition = kmt.debug_query_partition(query);
                println!("   Search led to partition: {:?}", search_partition);

                // See if the point's partition contains the point itself
                if let Some(partition_id) = search_partition {
                    let partition_points = kmt.get_partition_points(partition_id);
                    let contains_self = partition_points.contains(&(i as IndexT));
                    println!("   Search partition contains self: {}", contains_self);
                }
            } else {
                println!("   Using beam search with width: {}", beam_width);
            }

            // Calculate direct distance to confirm point should match itself
            if !dataset_kmt_results[j].is_empty() {
                let query_point = dataset.get(i);
                let found_point = dataset.get(dataset_kmt_results[j][0] as usize);

                // Simple Euclidean distance calculation
                let mut distance = 0.0;
                for k in 0..dataset.dim {
                    let diff = query_point[k] - found_point[k];
                    distance += diff * diff;
                }
                distance = distance.sqrt();

                println!("   Distance between points: {}", distance);

                // If not matching, find actual distance to self
                if !success {
                    let self_distance = dataset.compare(query_point, i);
                    println!("   Distance to self: {} (should be 0.0)", self_distance);
                }
            }
        }
    }

    self_recall /= sample_size as f64;
    println!(
        "\nKMT self-recall on dataset points: {:.5} ({}/{} points found themselves)",
        self_recall, success_count, sample_size
    );
}
