use std::collections::HashMap;
use std::env;
use std::process;
use std::time::Instant;

use scratch::graph::graph::ClassicGraph;

fn main() {
    // Get the filename from command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <graph_file>", args[0]);
        process::exit(1);
    }

    let graph_file = &args[1];
    println!("Reading graph from file: {}", graph_file);

    // Measure the time it takes to read the graph
    let start = Instant::now();
    let graph = match ClassicGraph::read(graph_file) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error reading graph file: {}", e);
            process::exit(1);
        }
    };
    let elapsed = start.elapsed();

    // Basic graph statistics
    println!("Graph loaded in {:.3} seconds", elapsed.as_secs_f64());
    println!("Number of nodes: {}", graph.size());
    println!("Maximum degree: {}", graph.max_degree());

    // Calculate actual statistics
    let mut total_edges = 0;
    let mut min_degree = graph.max_degree();
    let mut max_degree = 0;
    let mut degree_distribution = HashMap::new();

    for i in 0..graph.n {
        let degree = graph.degree(i);
        total_edges += degree;

        min_degree = min_degree.min(degree);
        max_degree = max_degree.max(degree);

        *degree_distribution.entry(degree).or_insert(0) += 1;
    }

    println!("Total edges: {}", total_edges);
    println!(
        "Average degree: {:.2}",
        total_edges as f64 / graph.size() as f64
    );
    println!("Minimum degree: {}", min_degree);
    println!("Maximum actual degree: {}", max_degree);

    // Show degree distribution (limit to 10 most common degrees)
    println!("\nDegree distribution (top 10):");
    let mut distribution: Vec<_> = degree_distribution.into_iter().collect();
    distribution.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by frequency (descending)

    for (i, (degree, count)) in distribution.iter().take(10).enumerate() {
        println!(
            "  {}: {} nodes with degree {} ({:.2}%)",
            i + 1,
            count,
            degree,
            (*count as f64 / graph.size() as f64) * 100.0
        );
    }

    // Sample some nodes and their neighborhoods
    println!("\nSample of neighborhoods:");
    let sample_size = 5.min(graph.size());

    for i in 0..sample_size as u32 {
        let neighborhood = graph.get_neighborhood(i);
        println!(
            "Node {} (degree {}): {:?}",
            i,
            neighborhood.len(),
            if neighborhood.len() <= 10 {
                format!("{:?}", neighborhood)
            } else {
                format!(
                    "{:?}...(and {} more)",
                    &neighborhood[..10],
                    neighborhood.len() - 10
                )
            }
        );
    }

    // Check for any potential issues
    let mut has_self_loops = false;
    let mut nodes_with_self_loops = 0;

    for i in 0..5.min(graph.n) {
        let neighborhood = graph.get_neighborhood(i);
        if neighborhood.contains(&i) {
            has_self_loops = true;
            nodes_with_self_loops += 1;
        }
    }

    if has_self_loops {
        println!(
            "\nNote: Detected self-loops in the graph (at least {} nodes)",
            nodes_with_self_loops
        );
    }

    println!("\nGraph info summary complete");
}
