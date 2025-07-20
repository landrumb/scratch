use clap::{Arg, Command};
use std::path::PathBuf;
use std::time::Instant;

use rand_distr::num_traits::ToPrimitive;
use rayon::prelude::*;
use scratch::constructions::ivf::IVFIndex;
use scratch::data_handling::dataset::VectorDataset;
use scratch::data_handling::dataset_traits::Dataset;
use scratch::data_handling::fbin::read_fbin;
use scratch::util::dataset::infer_dataset_paths;
use scratch::util::ground_truth::GroundTruth;
use scratch::util::recall::recall;

fn main() {
    let matches = Command::new("build_ivf")
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
            Arg::new("clusters")
                .long("clusters")
                .short('c')
                .default_value("2"),
        )
        .arg(
            Arg::new("probes")
                .long("probes")
                .short('p')
                .default_value("1"),
        )
        .arg(
            Arg::new("max_iterations")
                .long("max-iterations")
                .default_value("100"),
        )
        .arg(Arg::new("epsilon").long("epsilon").default_value("0.01"))
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

    let clusters: usize = matches
        .get_one::<String>("clusters")
        .unwrap()
        .parse()
        .unwrap_or(2);

    let probes: usize = matches
        .get_one::<String>("probes")
        .unwrap()
        .parse()
        .unwrap_or(1);

    let max_iterations: usize = matches
        .get_one::<String>("max_iterations")
        .unwrap()
        .parse()
        .unwrap_or(100);

    let epsilon: f32 = matches
        .get_one::<String>("epsilon")
        .unwrap()
        .parse()
        .unwrap_or(0.01);

    // Load dataset
    let mut start = Instant::now();
    let dataset: VectorDataset<f32> = read_fbin(&data_path);
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
    let queries: VectorDataset<f32> = read_fbin(&query_path);

    // Run queries
    start = Instant::now();
    let ivf_results: Vec<Vec<u32>> = (0..queries.size())
        .into_par_iter()
        .map(|i| {
            ivf.query(queries.get(i), probes, 10)
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

    // Load ground truth and compute recall
    let gt = GroundTruth::read(&gt_path);
    let ivf_recall = (0..ivf_results.len())
        .map(|i| recall(ivf_results[i].as_slice(), gt.get_neighbors(i)))
        .sum::<f64>()
        / queries.size().to_f64().unwrap();
    println!("IVF recall: {:05}", ivf_recall);
}
