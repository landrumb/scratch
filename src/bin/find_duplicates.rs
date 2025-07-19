use std::fs;
use std::path::Path;

use scratch::data_handling::dataset::VectorDataset;
use scratch::util::duplicates::duplicate_sets_f32;

fn count_duplicates(dataset: &VectorDataset<f32>) -> usize {
    duplicate_sets_f32(dataset)
        .iter()
        .map(|s| s.len())
        .sum()
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
            match VectorDataset::<f32>::from_file(&base_path) {
                Ok(dataset) => {
                    let dup = count_duplicates(&dataset);
                    println!("  base.fbin duplicates: {}", dup);
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
                    let dup = count_duplicates(&dataset);
                    println!("  query.fbin duplicates: {}", dup);
                }
                Err(e) => println!("  failed to load query.fbin: {}", e),
            }
        } else {
            println!("  query.fbin not found");
        }
    }
}
