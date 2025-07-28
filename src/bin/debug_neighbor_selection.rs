use std::path::Path;

use scratch::{
    constructions::neighbor_selection::{incremental_greedy, PairwiseDistancesHandler},
    data_handling::{dataset::VectorDataset, dataset_traits::Dataset},
    distance::euclidean,
    graph::IndexT,
};

fn main() {
    let dataset_path = "data/random2d/base.fbin";
    let center_id: IndexT = 72;
    let alpha = 1.01;

    println!("Loading dataset from {}", dataset_path);
    let dataset =
        VectorDataset::from_file(Path::new(dataset_path)).expect("Failed to load dataset");
    println!("Dataset size: {}", dataset.size());

    println!("Dataset dimensions: {}x{}", dataset.size(), dataset.dim);
    println!("Dataset first point: {:?}", dataset.get(0));
    println!("Dataset last point: {:?}", dataset.get(dataset.size() - 1));

    // Create full candidate list (all points except center)
    let candidates: Vec<IndexT> = (0..dataset.size() as IndexT)
        .filter(|&i| i != center_id)
        .collect();

    println!("Computing pairwise distances...");
    // Compute all pairwise distances for the dataset
    let mut pairwise_distances: Vec<Box<[(IndexT, f32)]>> = Vec::with_capacity(dataset.size());
    for i in 0..dataset.size() {
        let distances = dataset
            .brute_force_internal(i)
            .iter()
            .map(|(j, dist)| (*j as IndexT, *dist))
            .collect::<Box<[(IndexT, f32)]>>();
        pairwise_distances.push(distances);
    }
    let pairwise_distances = PairwiseDistancesHandler::new(pairwise_distances.into_boxed_slice());

    let faux_nearest = pairwise_distances.nearest(center_id);

    println!(
        "'Nearest neighbor' to point {}: {:?}",
        center_id, faux_nearest
    );
    println!(
        "compare_internal of point {} with its 'nearest neighbor' {}: {}",
        center_id,
        faux_nearest.0,
        dataset.compare_internal(center_id as usize, faux_nearest.0 as usize)
    );
    println!(
        "compare of point {} with its 'nearest neighbor' {}: {}",
        center_id,
        faux_nearest.0,
        dataset.compare(dataset.get(center_id as usize), faux_nearest.1 as usize)
    );
    println!(
        "euclidean of point {} with its 'nearest neighbor' {}: {}",
        center_id,
        faux_nearest.0,
        euclidean(
            dataset.get(center_id as usize),
            dataset.get(faux_nearest.0 as usize)
        )
    );

    println!(
        "Vector for point {}: {:?}",
        center_id,
        dataset.get(center_id as usize)
    );
    println!(
        "Vector for 'nearest neighbor' {}: {:?}",
        faux_nearest.0,
        dataset.get(faux_nearest.0 as usize)
    );

    let brute_force_distances = dataset.brute_force_internal(center_id as usize);
    println!(
        "Brute force distances for point {}: {:?}",
        center_id, brute_force_distances
    );

    println!(
        "Running neighbor selection for point {} with alpha={}",
        center_id, alpha
    );
    let neighbors = incremental_greedy(
        center_id,
        &candidates,
        &dataset,
        alpha,
        &pairwise_distances,
        None,
    );

    println!("Selected {} neighbors:", neighbors.len());
    for (i, &neighbor) in neighbors.iter().enumerate() {
        let distance = dataset.compare_internal(center_id as usize, neighbor as usize);
        println!("  {}: Point {} (distance: {})", i, neighbor, distance);
    }
    println!("Done.");
}
