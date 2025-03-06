use clap::{Arg, Command};
use std::path::Path;
use std::time::Instant;
use scratch::data_handling::dataset::VectorDataset;
use scratch::data_handling::dataset_traits::Dataset;
use scratch::util::ground_truth::compute_ground_truth;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("Compute Ground Truth")
        .version("0.1.0")
        .about("Computes exact ground truth (nearest neighbors) for a dataset")
        .arg(
            Arg::new("dataset")
                .long("dataset")
                .short('d')
                .value_name("FILE")
                .help("Dataset file path")
                .required(true),
        )
        .arg(
            Arg::new("neighbors")
                .long("neighbors")
                .short('n')
                .value_name("COUNT")
                .help("Number of nearest neighbors to compute")
                .default_value("100"),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .short('o')
                .value_name("FILE")
                .help("Output file path for ground truth")
                .required(true),
        )
        .arg(
            Arg::new("query_set")
                .long("query-set")
                .short('q')
                .value_name("FILE")
                .help("Optional query set file path. If not provided, ground truth is computed for the entire dataset."),
        )
        .get_matches();

    let dataset_path = matches.get_one::<String>("dataset").unwrap();
    let neighbors_count = matches.get_one::<String>("neighbors").unwrap().parse::<usize>()?;
    let output_path = matches.get_one::<String>("output").unwrap();
    
    println!("Loading dataset from {}", dataset_path);
    let start = Instant::now();
    let dataset = VectorDataset::<f32>::from_file(Path::new(dataset_path))?;
    println!("Dataset loaded in {:?}: {} points with {} dimensions", 
        start.elapsed(), dataset.size(), dataset.dim);
    
    let query_dataset = if let Some(query_path) = matches.get_one::<String>("query_set") {
        println!("Loading query set from {}", query_path);
        let q_start = Instant::now();
        let queries = VectorDataset::<f32>::from_file(Path::new(query_path))?;
        println!("Query set loaded in {:?}: {} points with {} dimensions", 
            q_start.elapsed(), queries.size(), queries.dim);
        Some(queries)
    } else {
        None
    };
    
    println!("Computing ground truth for {} neighbors", neighbors_count);
    let compute_start = Instant::now();
    let ground_truth = match &query_dataset {
        Some(queries) => {
            println!("Computing ground truth for {} query points", queries.size());
            compute_ground_truth(queries, &dataset, neighbors_count)?
        },
        None => {
            println!("Computing ground truth for all dataset points");
            compute_ground_truth(&dataset, &dataset, neighbors_count)?
        }
    };
    println!("Ground truth computation completed in {:?}", compute_start.elapsed());
    
    println!("Writing ground truth to {}", output_path);
    let write_start = Instant::now();
    ground_truth.write(Path::new(output_path))?;
    println!("Ground truth written in {:?}", write_start.elapsed());
    println!("Done!");
    
    Ok(())
}