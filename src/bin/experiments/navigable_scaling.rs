use std::{path::Path, time::Instant};

use scratch::{constructions::{neighbor_selection::{incremental_greedy, PairwiseDistancesHandler}, slow_preprocessing::build_global_local_graph}, data_handling::{dataset::{Subset, VectorDataset}, dataset_traits::Dataset, fbin::read_fbin}, graph::IndexT};

use rayon::prelude::*;

fn main() {
    let default_data_file = String::from("data/word2vec-google-news-300_50000_lowercase/base.fbin");
    let output_file = String::from("outputs/scaling.csv");

    let log_file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .append(true)
        .open(output_file)
        .unwrap();

    let mut log_writer = csv::Writer::from_writer(log_file);

    // if there's no file, create it and write the header
    if log_writer.get_ref().metadata().unwrap().len() == 0 {
        log_writer.write_record(&["method", "subset_size", "dcmp_time", "construction_time", "avg_degree", "max_degree"]).unwrap();
    }
    let method = "incremental_greedy";
    
    let subset_sizes = vec![1000, 2500, 5000, 7500, 10000, 15000, 20000, 25000];
    
    let boxed_dataset: Box<VectorDataset<f32>> = Box::new(read_fbin(Path::new(&default_data_file)));

    for subset_size in subset_sizes {
        println!("Using subset of size {}", subset_size);
        let subset_indices = (0..subset_size).collect::<Vec<usize>>();
        let subset = Subset::new(boxed_dataset.clone(), subset_indices);

        // build the graph
        let start = Instant::now();

        let nested_boxed_distances = (0..subset.size())
            .into_par_iter()
            .map(|i| {
                subset.brute_force_internal(i)
                    .iter()
                    .map(|(j, dist)| (*j as IndexT, *dist))
                    .collect::<Box<[(IndexT, f32)]>>()
            })
            .collect::<Box<[Box<[(IndexT, f32)]>]>>();
        let pairwise_distances = PairwiseDistancesHandler::new( nested_boxed_distances);

        let dcmp_time = start.elapsed();
        println!(
            "brute force in {}.{:03} seconds",
            dcmp_time.as_secs(),
            dcmp_time.subsec_millis()
        );

        let start = Instant::now();
        
        let graph = build_global_local_graph(&subset, |center, candidates| {
            incremental_greedy(center, candidates, &subset, 1.0, &pairwise_distances, None)
        });

        let construction_time = start.elapsed();
        println!(
            "graph construction in {}.{:03} seconds",
            construction_time.as_secs(),
            construction_time.subsec_millis()
        );
        println!(
            "total time in {}.{:03} seconds",
            (dcmp_time + construction_time).as_secs(),
            (dcmp_time + construction_time).subsec_millis()
        );

        let avg_degree = graph.total_edges() / subset.size();
        let max_degree = graph.max_degree();

        log_writer.write_record(&[
            method.to_string(),
            subset_size.to_string(),
            dcmp_time.as_secs_f64().to_string(),
            construction_time.as_secs_f64().to_string(),
            avg_degree.to_string(),
            max_degree.to_string(),
        ]).unwrap();
        log_writer.flush().unwrap();
        println!("----------------------------------------");
    }
}