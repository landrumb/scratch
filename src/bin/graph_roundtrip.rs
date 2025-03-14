use std::env;
use std::path::Path;
use std::process;
use std::time::Instant;

use scratch::graph::ClassicGraph;

/// This utility demonstrates reading a graph from one file and writing it to another
/// It helps test the roundtrip functionality and provides information about both processes
fn main() {
    // Get the filename from command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input_graph_file> <output_graph_file>", args[0]);
        process::exit(1);
    }

    let input_file = &args[1];
    let output_file = &args[2];

    if !Path::new(input_file).exists() {
        eprintln!("Input file does not exist: {}", input_file);
        process::exit(1);
    }

    println!("Reading graph from: {}", input_file);

    // Measure the time it takes to read the graph
    let start_read = Instant::now();
    let graph = match ClassicGraph::read(input_file) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error reading graph file: {}", e);
            process::exit(1);
        }
    };
    let read_elapsed = start_read.elapsed();

    // Basic graph statistics
    println!("Graph loaded in {:.3} seconds", read_elapsed.as_secs_f64());
    println!("Number of nodes: {}", graph.size());
    println!("Maximum degree: {}", graph.max_degree());

    // Calculate edge statistics
    let mut total_edges = 0;
    let mut min_degree = usize::MAX;
    let mut max_degree = 0;

    for i in 0..graph.n {
        let degree = graph.degree(i);
        total_edges += degree;

        if degree < min_degree {
            min_degree = degree;
        }
        if degree > max_degree {
            max_degree = degree;
        }
    }

    println!("Total edges: {}", total_edges);
    println!(
        "Average degree: {:.2}",
        total_edges as f64 / graph.size() as f64
    );
    println!("Minimum degree: {}", min_degree);
    println!("Maximum actual degree: {}", max_degree);

    // Now write the graph to the output file
    println!("\nWriting graph to: {}", output_file);
    let start_write = Instant::now();
    match graph.save(output_file) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Error writing graph file: {}", e);
            process::exit(1);
        }
    }
    let write_elapsed = start_write.elapsed();

    println!(
        "Graph written in {:.3} seconds",
        write_elapsed.as_secs_f64()
    );
    println!("Roundtrip complete!");

    // Optional: verify the written file
    if args.len() > 3 && args[3] == "--verify" {
        println!("\nVerifying written file...");

        let start_verify = Instant::now();
        let verified_graph = match ClassicGraph::read(output_file) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Error reading written graph file: {}", e);
                process::exit(1);
            }
        };
        let verify_elapsed = start_verify.elapsed();

        // Compare original and written graphs
        let mut is_equal = true;
        if verified_graph.n != graph.n {
            println!(
                "❌ Node count mismatch: {} vs {}",
                graph.n, verified_graph.n
            );
            is_equal = false;
        }

        if verified_graph.r != graph.r {
            println!(
                "❌ Max degree mismatch: {} vs {}",
                graph.r, verified_graph.r
            );
            is_equal = false;
        }

        let mut edge_mismatch = false;
        for i in 0..graph.n {
            let orig_neighbors = graph.get_neighborhood(i);
            let new_neighbors = verified_graph.get_neighborhood(i);

            if orig_neighbors.len() != new_neighbors.len() {
                println!(
                    "❌ Degree mismatch for node {}: {} vs {}",
                    i,
                    orig_neighbors.len(),
                    new_neighbors.len()
                );
                is_equal = false;
                break;
            }

            for j in 0..orig_neighbors.len() {
                if orig_neighbors[j] != new_neighbors[j] {
                    println!("❌ Edge mismatch for node {} at position {}", i, j);
                    edge_mismatch = true;
                    is_equal = false;
                    break;
                }
            }

            if edge_mismatch {
                break;
            }
        }

        if is_equal {
            println!(
                "✅ Verification successful in {:.3} seconds",
                verify_elapsed.as_secs_f64()
            );
            println!("Original and written graphs are identical");
        } else {
            println!("❌ Verification failed! Graphs are different");
            process::exit(1);
        }
    }
}
