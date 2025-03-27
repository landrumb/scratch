use std::boxed;
use std::env::args;
use std::ops::Sub;
use std::path::Path;
use std::time::Instant;

use rand_distr::num_traits::ToPrimitive;
use rayon::prelude::*;
use scratch::constructions::slow_preprocessing::build_global_local_graph;
use scratch::constructions::neighbor_selection::{naive_semi_greedy_prune, PairwiseDistancesHandler};
use scratch::data_handling::dataset::{DistanceMatrix, Subset, VectorDataset};
use scratch::data_handling::dataset_traits::Dataset;
use scratch::data_handling::fbin::read_fbin;
use scratch::graph::{beam_search, ClassicGraph, Graph, IndexT};
use scratch::util::ground_truth::{compute_ground_truth, GroundTruth};
use scratch::util::recall::recall;

static SUBSET_SIZE: Option<&'static str> = option_env!("SUBSET_SIZE");

fn main() {
    let default_data_file = String::from("data/word2vec-google-news-300_50000_lowercase/base.fbin");
    let default_query_file =
        String::from("data/word2vec-google-news-300_50000_lowercase/query.fbin");
    // let default_graph_file =
    //     String::from("data/word2vec-google-news-300_50000_lowercase/outputs/vamana");
    let default_gt_file = String::from("data/word2vec-google-news-300_50000_lowercase/GT");

    // Parse arguments
    let data_path_arg = args().nth(1).unwrap_or(default_data_file);
    let data_path = Path::new(&data_path_arg);

    let query_path_arg = args().nth(2).unwrap_or(default_query_file);
    let query_path = Path::new(&query_path_arg);

    // let graph_path_arg = args().nth(3).unwrap_or(default_graph_file);

    let gt_path_arg = args().nth(4).unwrap_or(default_gt_file);
    let gt_path = Path::new(&gt_path_arg);

    // Load dataset
    let mut start = Instant::now();
    // let dataset: VectorDataset<f32> = read_fbin(data_path);
    
    let boxed_dataset: Box<VectorDataset<f32>> = Box::new(read_fbin(data_path));
    let mut subset_size: usize = boxed_dataset.size();
    
    if let Some(subset_size_arg) = SUBSET_SIZE {
        subset_size = subset_size_arg.parse::<usize>().unwrap();
    } 

    println!("Using subset of size {}", subset_size);
    let subset_indices = (0..subset_size).collect::<Vec<usize>>();
    let subset = Subset::new(boxed_dataset, subset_indices);
    
    let elapsed = start.elapsed();
    println!(
        "read dataset in {}.{:03} seconds",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    // build distance matrix
    // let dataset = DistanceMatrix::new_with_progress_bar(dataset);

    // build the graph
    start = Instant::now();
    // let graph = build_slow_preprocesssing(&dataset, 1.0);


    let nested_boxed_distances = (0..subset.size())
        .into_par_iter()
        .map(|i| {
            subset.brute_force_internal(i)
                .iter()
                .map(|(j, dist)| (*j as IndexT, *dist))
                .collect::<Box<[(IndexT, f32)]>>()
        })
        .collect::<Box<[Box<[(IndexT, f32)]>]>>();
    let pairwise_distances = PairwiseDistancesHandler::new(nested_boxed_distances);

    // let graph = build_global_local_graph(&dataset, |candidates, dataset| {
    //     robust_prune_unbounded(candidates.to_vec(), 1.0, dataset)
    // });

    let graph = build_global_local_graph(&subset, |center, candidates| {
        naive_semi_greedy_prune(center, candidates, &subset, 1.0, &pairwise_distances)
    });

    let elapsed = start.elapsed();
    println!(
        "built graph in {}.{:03} seconds",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    println!("Total edges: {}", graph.total_edges());
    println!("Average degree: {}", graph.total_edges() / subset.size());
    println!("Max degree: {}", graph.max_degree());

    // Load queries
    let queries: VectorDataset<f32> = read_fbin(query_path);
    

    // Run queries
    start = Instant::now();
    let results: Vec<Vec<u32>> = (0..queries.size())
    // let results: Vec<Vec<u32>> = (0..subset_size)
        .into_par_iter()
        .map(|i| beam_search(queries.get(i), &graph, &subset, 0, 100))
        .collect();

    let elapsed = start.elapsed();
    println!(
        "ran {} queries in {}.{:03} seconds ({} QPS)",
        queries.size(),
        elapsed.as_secs(),
        elapsed.subsec_millis(),
        queries.size().to_f64().unwrap() / elapsed.as_secs_f64()
    );

    // print the neighborhoods of the first 10 points in the dataset
    for i in 0..10 {
        println!("neighbors of {}: {:?}", i, graph.neighbors(i));
    }

    // what fraction of points have 20 as a neighbor?
    let twenty_neighbors = (0..subset_size).filter(|i| graph.get_neighborhood(*i as IndexT).contains(&20)).count();
    let fraction_twenty_neighbors = twenty_neighbors as f64 / subset.size() as f64;
    println!("Fraction of points with 20 as a neighbor: {:.5}", fraction_twenty_neighbors);


    // // Load ground truth and compute recall
    // let gt = GroundTruth::read(gt_path);

    // compute gt over the subset
    let gt = compute_ground_truth(&queries, &subset, 1).unwrap();

    let graph_recall = (0..results.len())
        .map(|i| recall(results[i].as_slice(), gt.get_neighbors(i)))
        .sum::<f64>()
        / results.len().to_f64().unwrap();
    println!("recall: {:.5}", graph_recall);

    // write the graph to disk
    let classic_graph = ClassicGraph::from(&graph);
    classic_graph.save(data_path.parent().unwrap().join("outputs").join("greedy").to_str().unwrap()).unwrap();
    println!("saved graph to disk");
}
