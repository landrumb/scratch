use std::fs;
use std::path::Path;

use scratch::data_handling::dataset::VectorDataset;
use scratch::util::duplicates::duplicate_sets;

fn count_duplicates(dataset: &VectorDataset<f32>, radius: Option<f64>) -> (usize, usize) {
    let duplicate_sets = duplicate_sets(dataset, radius);
    let total_duplicates = duplicate_sets
        .iter()
        .filter(|set| set.len() > 1)
        .map(|set| set.len())
        .sum();
    let num_sets = duplicate_sets.len();
    (total_duplicates, num_sets)
}

fn main() {
    let data_dir = Path::new("data");
    if !data_dir.exists() {
        eprintln!("data directory not found");
        return;
    }

    for entry in fs::read_dir(data_dir).expect("unable to read data directory") {
        let entry = entry.expect("invalid entry");
        if !entry.file_type().unwrap().is_dir() {
            continue;
        }
        let dataset_path = entry.path();
        let dataset_name = dataset_path.file_name().unwrap().to_string_lossy();
        println!("Dataset: {}", dataset_name);

        let base_path = dataset_path.join("base.fbin");
        if base_path.exists() {
            println!("  loading base.fbin...");
            match VectorDataset::<f32>::from_file(&base_path) {
                Ok(dataset) => {
                    println!("  base.fbin loaded, {} vectors", dataset.n);
                    let (dup, num_sets) = count_duplicates(&dataset, None);
                    println!(
                        "  base.fbin duplicates: {} ({:.2}%, {} sets)",
                        dup,
                        dup as f32 * 100.0 / dataset.n as f32,
                        num_sets
                    );
                }
                Err(e) => println!("  failed to load base.fbin: {}", e),
            }
        } else {
            println!("  base.fbin not found");
        }

        let query_path = dataset_path.join("query.fbin");
        if query_path.exists() {
            match VectorDataset::<f32>::from_file(&query_path) {
                Ok(dataset) => {
                    let (dup, _) = count_duplicates(&dataset, None);
                    println!("  query.fbin duplicates: {}", dup);
                }
                Err(e) => println!("  failed to load query.fbin: {}", e),
            }
        } else {
            println!("  query.fbin not found");
        }
    }
}
