use clap::{Arg, Command};
use scratch::distance::get_distance_comparison_count;
use scratch::util::clique::{greedy_independent_cliques, maximal_bidirectional_cliques};
use scratch::util::edge_filter::filter_edges_percentile;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::process::Command as ProcCommand;

use rand_distr::num_traits::ToPrimitive;
use rayon::prelude::*;
use scratch::constructions::vamana::build_vamana_graph;
use scratch::data_handling::dataset::{Subset, VectorDataset};
use scratch::data_handling::dataset_traits::Dataset;
use scratch::data_handling::fbin::read_fbin;
use scratch::graph::{beam_search, ClassicGraph, CompactedGraphIndex, IndexT, VectorGraph};
use scratch::util::dataset::infer_dataset_paths;
use scratch::util::ground_truth::{compute_ground_truth, GroundTruth};
use scratch::util::recall::recall;

// static SUBSET_SIZE: Option<&'static str> = option_env!("SUBSET_SIZE");

fn main() {
    let matches = Command::new("build_vamana")
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
            Arg::new("graph")
                .long("graph")
                .value_name("FILE")
                .help("Path to graph file"),
        )
        .arg(
            Arg::new("gt")
                .long("gt")
                .value_name("FILE")
                .help("Path to ground truth file"),
        )
        .arg(
            Arg::new("subset_size")
                .value_parser(clap::value_parser!(usize))
                .long("subset-size")
                .value_name("SIZE")
                .help("Size of subset to use for building the graph"),
        )
        .get_matches();

    let dataset_name: String = matches.get_one::<String>("dataset").unwrap().clone();
    let hostname: String = ProcCommand::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let inferred = infer_dataset_paths(&dataset_name);

    let data_path: PathBuf = matches
        .get_one::<String>("base")
        .map(PathBuf::from)
        .unwrap_or(inferred.base);

    let query_path: PathBuf = matches
        .get_one::<String>("query")
        .map(PathBuf::from)
        .unwrap_or(inferred.query);

    let _gt_path: PathBuf = matches
        .get_one::<String>("gt")
        .map(PathBuf::from)
        .unwrap_or(inferred.gt);

    // Load dataset
    let mut start = Instant::now();
    // let dataset: VectorDataset<f32> = read_fbin(data_path);

    let dataset: Arc<VectorDataset<f32>> = Arc::new(read_fbin(&data_path));
    let mut subset_size: usize = dataset.size();

    if let Some(&subset_size_arg) = matches.get_one::<usize>("subset_size") {
        subset_size = subset_size_arg;
        println!("Using subset of size {subset_size}");
    } else {
        println!("Using full dataset of size {}", dataset.size());
    }

    let subset_indices = (0..subset_size).collect::<Vec<usize>>();
    let subset = Subset::new(dataset.clone(), subset_indices);

    let elapsed = start.elapsed();
    println!(
        "read dataset in {}.{:03} seconds",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    // build the graph, or load it from disk if it exists
    start = Instant::now();

    let graph_path = matches
        .get_one::<String>("graph")
        .map(PathBuf::from)
        .unwrap_or(data_path.parent().unwrap().join("outputs/vamana.graph"));
    let graph: VectorGraph;
    if graph_path.exists() {
        graph = ClassicGraph::read(graph_path.to_str().unwrap())
            .ok()
            .unwrap()
            .into();
        println!("loaded graph from disk");
    } else {
        println!("building graph");
        graph = build_vamana_graph(&subset, 1.01, 8, 100, Some(500), Some(0));

        let elapsed = start.elapsed();
        println!("built graph in {elapsed:?}");
    }
    println!("Total edges: {}", graph.total_edges());
    println!("Average degree: {}", graph.total_edges() / subset.size());
    println!("Max degree: {}", graph.max_degree());

    // Load queries
    let queries: VectorDataset<f32> = read_fbin(&query_path);

    // Run queries
    start = Instant::now();
    let prev_distance_comparisons = get_distance_comparison_count();
    let results: Vec<Vec<u32>> = (0..queries.size())
        // let results: Vec<Vec<u32>> = (0..subset_size)
        .into_par_iter()
        .map(|i| beam_search(queries.get(i), &graph, &subset, 0, 40, None))
        .collect();

    let elapsed = start.elapsed();
    println!(
        "ran {} queries in {:?} ({:.3} QPS, {:.3} comparisons/query)",
        queries.size(),
        elapsed,
        queries.size().to_f64().unwrap() / elapsed.as_secs_f64(),
        (get_distance_comparison_count() - prev_distance_comparisons) as f64
            / queries.size().to_f64().unwrap()
    );

    // Load ground truth and compute recall
    let gt: GroundTruth = if _gt_path.exists() {
        GroundTruth::read(&_gt_path)
    } else {
        compute_ground_truth(&queries, &subset, 10).unwrap()
    };

    let graph_recall = (0..results.len())
        .map(|i| recall(results[i].as_slice(), gt.get_neighbors(i)))
        .sum::<f64>()
        / results.len().to_f64().unwrap();
    println!("recall: {graph_recall:.5}");

    // // write the graph to disk
    // let classic_graph = ClassicGraph::from(&graph);
    // // if the output directory does not exist, create it
    // std::fs::create_dir_all(graph_path.parent().unwrap()).expect("could not create output directory");
    // classic_graph
    //     .save(graph_path.to_str().unwrap())
    //     .unwrap();

    // // do the same querying with the dataset itself
    // let dataset_results: Vec<Vec<u32>> = (0..subset_size)
    //     .into_par_iter()
    //     .map(|i| beam_search(subset.get(i), &graph, &subset, 0, 1, None))
    //     .collect();

    // let internal_gt = compute_ground_truth(&subset, &subset, 2).unwrap();

    // let dataset_recall = (0..dataset_results.len())
    //     .map(|i| {
    //         recall(
    //             dataset_results[i].as_slice(),
    //             &internal_gt.get_neighbors(i)[..1],
    //         )
    //     })
    //     .sum::<f64>()
    //     / dataset_results.len().to_f64().unwrap();

    // println!("self recall: {:.5}", dataset_recall);

    // let n_points_connected_to_nearest_neighbor = (0..subset.size())
    //     .into_par_iter()
    //     .filter(|i| {
    //         let neighbors = graph.neighbors(*i as IndexT);
    //         let nearest_neighbor = internal_gt.get_neighbors(*i)[1];
    //         neighbors.contains(&nearest_neighbor)
    //     })
    //     .count();

    // println!(
    //     "Fraction of points connected to their nearest neighbor: {:.5}",
    //     n_points_connected_to_nearest_neighbor as f64 / subset.size() as f64
    // );

    // // how many points have an incoming edge from their nearest neighbor (should be all of them)
    // let n_points_connected_to_inverse_nn = (0..subset.size())
    //     .into_par_iter()
    //     .filter(|i| {
    //         let nearest_neighbor = internal_gt.get_neighbors(*i)[1];
    //         let neighbors = graph.neighbors(nearest_neighbor);
    //         neighbors.contains(&(*i as IndexT))
    //     })
    //     .count();
    // println!(
    //     "Fraction of points with an incoming edge from their nearest neighbor: {:.5}",
    //     n_points_connected_to_inverse_nn as f64 / subset.size() as f64
    // );

    println!("---- Building compacted graph variants ----");

    // helper to deep-clone the posting lists for each build
    fn clone_posting_lists(src: &Box<[Box<[IndexT]>]>) -> Box<[Box<[IndexT]>]> {
        src.iter()
            .map(|b| b.to_vec().into_boxed_slice())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    // Pre-load internal GT once for the experiments section
    let gt_internal = GroundTruth::read(&_gt_path.parent().unwrap().join("internal.GT"));

    // Describe the variants we will build
    enum VariantSpec {
        MemoryInefficient,
        RobustPrune(f32),
    }

    let specs: Vec<(String, VariantSpec)> = vec![
        ("memory_inefficient".to_string(), VariantSpec::MemoryInefficient),
        ("robust_prune_alpha_1.20".to_string(), VariantSpec::RobustPrune(1.20)),
        ("robust_prune_alpha_1.05".to_string(), VariantSpec::RobustPrune(1.05)),
    ];

    // Clique computation variants
    enum CliqueSpec {
        Original,
        ShortestPercentile(f32),
    }
    let clique_specs: Vec<(String, CliqueSpec)> = vec![
        ("original".to_string(), CliqueSpec::Original),
        ("shortest_50pct".to_string(), CliqueSpec::ShortestPercentile(50.0)),
        ("shortest_10pct".to_string(), CliqueSpec::ShortestPercentile(10.0)),
    ];

    struct MethodStats {
        recall: f64,
        qps: f64,
        comparisons_per_query: f64,
    }

    struct VariantStats {
        name: String,
        clique_variant: String,
        build_secs: f64,
        graph_size: usize,
        n_primary: usize,
        n_secondary: usize,
        timestamp_secs: u64,
        post_expansion: MethodStats,
        expand_visited: MethodStats,
        primary_points: MethodStats,
        exhaustive_primary: MethodStats,
        exhaustive_secondary: MethodStats,
        primary_specific_recall: f64,
        reps_nn_in_clique: usize,
        reps_nn_pct: f64,
    }

    let mut all_stats: Vec<VariantStats> = Vec::new();

    // Need a trait-object dataset for filtering utilities
    let dataset_dyn: Arc<dyn Dataset<f32>> = dataset.clone();

    for (clique_name, clique_spec) in clique_specs.iter() {
        // Choose graph for clique computation
        let graph_for_cliques: VectorGraph = match clique_spec {
            CliqueSpec::Original => ClassicGraph::from(&graph).into(),
            CliqueSpec::ShortestPercentile(percentile) => filter_edges_percentile::<f32>(&graph, &dataset_dyn, *percentile),
        };

        // finding all cliques
        let mut cliques = maximal_bidirectional_cliques(&graph_for_cliques);
        println!("[clique:{}] Found {} cliques", clique_name, cliques.len());

        cliques.retain(|c| c.len() > 1);

        let independent_cliques = greedy_independent_cliques(&cliques);
        println!("[clique:{}] Found {} independent cliques", clique_name, independent_cliques.len());

        println!(
            "[clique:{}] compacted graph should have {} primary points and {} secondary points",
            clique_name,
            graph.n() - independent_cliques.iter().map(|c| c.len()).sum::<usize>() + independent_cliques.len(),
            independent_cliques.iter().map(|c| c.len()).sum::<usize>() - independent_cliques.len()
        );

        let independent_cliques_box: Box<[Box<[IndexT]>]> = independent_cliques
            .into_iter()
            .map(|c| c.into_boxed_slice())
            .collect();

        for (name, spec) in specs.iter() {
            // rebuild a fresh graph for each variant
            let graph_copy: VectorGraph = ClassicGraph::from(&graph).into();
            let posting_lists = clone_posting_lists(&independent_cliques_box);

        let start = Instant::now();
        let compacted_graph = match spec {
            VariantSpec::MemoryInefficient => {
                CompactedGraphIndex::build_memory_inefficient(graph_copy, dataset.clone(), posting_lists)
            }
            VariantSpec::RobustPrune(alpha) => {
                CompactedGraphIndex::build_memory_inefficient_robust_prune(
                    graph_copy,
                    dataset.clone(),
                    posting_lists,
                    *alpha,
                )
            }
        };
        let build_elapsed = start.elapsed();

        let n_primary = compacted_graph.primary_points().len();
        let n_secondary = compacted_graph.secondary_points().len();
        let graph_size = compacted_graph.graph_size();

        // Post-expansion
        let start = Instant::now();
        let prev_distance_comparisons = get_distance_comparison_count();
        let post_expansion_results: Vec<Vec<u32>> = (0..queries.size())
            .into_par_iter()
            .map(|i| compacted_graph.beam_search_post_expansion(queries.get(i), 40))
            .collect();
        let elapsed = start.elapsed();
        let post_expansion_qps = queries.size().to_f64().unwrap() / elapsed.as_secs_f64();
        let post_expansion_comps =
            (get_distance_comparison_count() - prev_distance_comparisons) as f64
                / queries.size().to_f64().unwrap();
        let post_expansion_recall = (0..post_expansion_results.len())
            .map(|i| recall(post_expansion_results[i].as_slice(), gt.get_neighbors(i)))
            .sum::<f64>()
            / post_expansion_results.len().to_f64().unwrap();

        // Expand-visited
        let start = Instant::now();
        let prev_distance_comparisons = get_distance_comparison_count();
        let expand_visited_results: Vec<Vec<u32>> = (0..queries.size())
            .into_par_iter()
            .map(|i| compacted_graph.beam_search_expand_visited(queries.get(i), 40))
            .collect();
        let elapsed = start.elapsed();
        let expand_visited_qps = queries.size().to_f64().unwrap() / elapsed.as_secs_f64();
        let expand_visited_comps =
            (get_distance_comparison_count() - prev_distance_comparisons) as f64
                / queries.size().to_f64().unwrap();
        let expand_visited_recall = (0..expand_visited_results.len())
            .map(|i| recall(expand_visited_results[i].as_slice(), gt.get_neighbors(i)))
            .sum::<f64>()
            / expand_visited_results.len().to_f64().unwrap();

        // Primary points search
        let start = Instant::now();
        let prev_distance_comparisons = get_distance_comparison_count();
        let graph_primary_results: Vec<Vec<u32>> = (0..queries.size())
            .into_par_iter()
            .map(|i| compacted_graph.beam_search_primary_points(queries.get(i), 40))
            .collect();
        let elapsed = start.elapsed();
        let primary_points_qps = queries.size().to_f64().unwrap() / elapsed.as_secs_f64();
        let primary_points_comps =
            (get_distance_comparison_count() - prev_distance_comparisons) as f64
                / queries.size().to_f64().unwrap();
        let primary_points_recall = (0..graph_primary_results.len())
            .map(|i| recall(graph_primary_results[i].as_slice(), gt.get_neighbors(i)))
            .sum::<f64>()
            / graph_primary_results.len().to_f64().unwrap();

        // Exhaustive primary
        let start = Instant::now();
        let prev_distance_comparisons = get_distance_comparison_count();
        let exhaustive_primary_results: Vec<Vec<u32>> = (0..queries.size())
            .into_par_iter()
            .map(|i| compacted_graph.exhaustive_search_primary_points(queries.get(i)))
            .collect();
        let elapsed = start.elapsed();
        let exhaustive_primary_qps = queries.size().to_f64().unwrap() / elapsed.as_secs_f64();
        let exhaustive_primary_comps =
            (get_distance_comparison_count() - prev_distance_comparisons) as f64
                / queries.size().to_f64().unwrap();
        let exhaustive_primary_recall = (0..exhaustive_primary_results.len())
            .map(|i| recall(exhaustive_primary_results[i].as_slice(), gt.get_neighbors(i)))
            .sum::<f64>()
            / exhaustive_primary_results.len().to_f64().unwrap();

        // Primary points specific recall (vs top-10 exhaustive primary)
        let primary_points_specific_recall = (0..graph_primary_results.len())
            .map(|i| recall(graph_primary_results[i].as_slice(), &exhaustive_primary_results[i][..10]))
            .sum::<f64>()
            / graph_primary_results.len().to_f64().unwrap();

        // Exhaustive secondary
        let start = Instant::now();
        let prev_distance_comparisons = get_distance_comparison_count();
        let exhaustive_secondary_results: Vec<Vec<u32>> = (0..queries.size())
            .into_par_iter()
            .map(|i| compacted_graph.exhaustive_search_secondary_points(queries.get(i)))
            .collect();
        let elapsed = start.elapsed();
        let exhaustive_secondary_qps = queries.size().to_f64().unwrap() / elapsed.as_secs_f64();
        let exhaustive_secondary_comps =
            (get_distance_comparison_count() - prev_distance_comparisons) as f64
                / queries.size().to_f64().unwrap();
        let exhaustive_secondary_recall = (0..exhaustive_secondary_results.len())
            .map(|i| recall(exhaustive_secondary_results[i].as_slice(), gt.get_neighbors(i)))
            .sum::<f64>()
            / exhaustive_secondary_results.len().to_f64().unwrap();

        // Experiment: representatives with nearest neighbor in clique
        let mut reps_with_nn_in_clique = 0usize;
        for (i, list) in compacted_graph.get_posting_lists().iter() {
            if list.len() == 1 {
                continue;
            }
            let nearest_neighbor = gt_internal.get_neighbors(*i as usize)[1];
            if list.contains(&nearest_neighbor) {
                reps_with_nn_in_clique += 1;
            }
        }
        let reps_pct = reps_with_nn_in_clique as f64
            / compacted_graph.get_posting_lists().len() as f64
            * 100.0;

        let now_secs: u64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        all_stats.push(VariantStats {
            name: name.clone(),
            clique_variant: clique_name.clone(),
            build_secs: build_elapsed.as_secs_f64(),
            graph_size,
            n_primary,
            n_secondary,
            timestamp_secs: now_secs,
            post_expansion: MethodStats {
                recall: post_expansion_recall,
                qps: post_expansion_qps,
                comparisons_per_query: post_expansion_comps,
            },
            expand_visited: MethodStats {
                recall: expand_visited_recall,
                qps: expand_visited_qps,
                comparisons_per_query: expand_visited_comps,
            },
            primary_points: MethodStats {
                recall: primary_points_recall,
                qps: primary_points_qps,
                comparisons_per_query: primary_points_comps,
            },
            exhaustive_primary: MethodStats {
                recall: exhaustive_primary_recall,
                qps: exhaustive_primary_qps,
                comparisons_per_query: exhaustive_primary_comps,
            },
            exhaustive_secondary: MethodStats {
                recall: exhaustive_secondary_recall,
                qps: exhaustive_secondary_qps,
                comparisons_per_query: exhaustive_secondary_comps,
            },
            primary_specific_recall: primary_points_specific_recall,
            reps_nn_in_clique: reps_with_nn_in_clique,
            reps_nn_pct: reps_pct,
        });
        }
    }

    // Markdown summary table
    println!("\n### CompactedGraphIndex benchmarks\n");
    println!("| Name | Clique | Dataset | Host | Build (s) | n | Primary | Secondary | PostExp Recall | PostExp QPS | PostExp Comp/Q | ExpandVis Recall | ExpandVis QPS | ExpandVis Comp/Q | Primary Recall | Primary QPS | Primary Comp/Q | Exhaustive Primary Recall | Exhaustive Secondary Recall | Primary Specific Recall | Reps NN in Clique | Reps NN (%) |");
    println!("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for s in &all_stats {
        println!(
            "| {} | {} | {} | {} | {:.3} | {} | {} | {} | {:.5} | {:.3} | {:.3} | {:.5} | {:.3} | {:.3} | {:.5} | {:.3} | {:.3} | {:.5} | {:.5} | {:.5} | {} | {:.2} |",
            s.name,
            s.clique_variant,
            dataset_name,
            hostname,
            s.build_secs,
            s.graph_size,
            s.n_primary,
            s.n_secondary,
            s.post_expansion.recall,
            s.post_expansion.qps,
            s.post_expansion.comparisons_per_query,
            s.expand_visited.recall,
            s.expand_visited.qps,
            s.expand_visited.comparisons_per_query,
            s.primary_points.recall,
            s.primary_points.qps,
            s.primary_points.comparisons_per_query,
            s.exhaustive_primary.recall,
            s.exhaustive_secondary.recall,
            s.primary_specific_recall,
            s.reps_nn_in_clique,
            s.reps_nn_pct,
        );
    }

    let csv_dir = Path::new("outputs");
    let csv_path = csv_dir.join("compacted_benchmarks.csv");
    if let Some(parent) = csv_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut csv_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&csv_path)
        .expect("Failed to open CSV file for append");
    // Write header only if file is new or empty
    let need_header = csv_path
        .metadata()
        .map(|m| m.len() == 0)
        .unwrap_or(true);
    if need_header {
        writeln!(
            csv_file,
            "timestamp,dataset,host,name,clique,n,primary,secondary,build_secs,postexp_recall,postexp_qps,postexp_comp_q,expandvis_recall,expandvis_qps,expandvis_comp_q,primary_recall,primary_qps,primary_comp_q,exhaustive_primary_recall,exhaustive_secondary_recall,primary_specific_recall,reps_nn_in_clique,reps_nn_pct"
        )
        .ok();
    }
    for s in &all_stats {
        writeln!(
            csv_file,
            "{},{} ,{} ,{} ,{} ,{},{} ,{},{} ,{:.5},{:.3},{:.3},{:.5},{:.3},{:.3},{:.5},{:.3},{:.3},{:.5},{:.5},{:.5},{} ,{:.2}",
            s.timestamp_secs,
            dataset_name,
            hostname,
            s.name,
            s.clique_variant,
            s.graph_size,
            s.n_primary,
            s.n_secondary,
            s.build_secs,
            s.post_expansion.recall,
            s.post_expansion.qps,
            s.post_expansion.comparisons_per_query,
            s.expand_visited.recall,
            s.expand_visited.qps,
            s.expand_visited.comparisons_per_query,
            s.primary_points.recall,
            s.primary_points.qps,
            s.primary_points.comparisons_per_query,
            s.exhaustive_primary.recall,
            s.exhaustive_secondary.recall,
            s.primary_specific_recall,
            s.reps_nn_in_clique,
            s.reps_nn_pct
        )
        .ok();
    }
    println!("Saved CSV to {}", csv_path.display());
}
