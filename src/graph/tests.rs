#[cfg(test)]
mod tests {
    use crate::graph::graph::ClassicGraph;
    use std::fs;
    use std::io::Write;
    use std::path::Path;

    // Helper function to create a temporary file path
    fn temp_file_path(name: &str) -> String {
        format!("target/{}", name)
    }

    // Helper to clean up temporary files
    fn cleanup_temp_file(path: &str) {
        if Path::new(path).exists() {
            fs::remove_file(path).unwrap_or_else(|_| {
                println!("Warning: Failed to remove temporary file: {}", path);
            });
        }
    }

    // Helper to create a simple test graph
    fn create_test_graph() -> ClassicGraph {
        let n = 10;
        let r = 5;
        let mut graph = ClassicGraph::new(n, r);

        // Add some edges
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 3);
        graph.add_edge(1, 0);
        graph.add_edge(1, 4);
        graph.add_edge(2, 0);
        graph.add_edge(2, 5);
        graph.add_edge(3, 0);
        graph.add_edge(3, 6);
        graph.add_edge(4, 1);
        graph.add_edge(4, 7);
        graph.add_edge(5, 2);
        graph.add_edge(5, 8);
        graph.add_edge(6, 3);
        graph.add_edge(6, 9);

        graph
    }

    // Test saving and loading an empty graph
    #[test]
    fn test_empty_graph() {
        let path = temp_file_path("empty_graph.bin");
        cleanup_temp_file(&path);

        let n = 5;
        let r = 3;
        let graph = ClassicGraph::new(n, r);

        // Save
        assert!(graph.save(&path).is_ok());

        // Load
        let loaded_graph = ClassicGraph::read(&path).expect("Failed to read empty graph");

        // Verify
        assert_eq!(loaded_graph.n, n);
        assert_eq!(loaded_graph.r, r as usize);
        for i in 0..n {
            assert_eq!(loaded_graph.degree(i), 0);
        }

        cleanup_temp_file(&path);
    }

    // Test saving and loading a simple graph
    #[test]
    fn test_simple_graph() {
        let path = temp_file_path("simple_graph.bin");
        cleanup_temp_file(&path);

        let graph = create_test_graph();

        // Save
        assert!(graph.save(&path).is_ok());

        // Load
        let loaded_graph = ClassicGraph::read(&path).expect("Failed to read simple graph");

        // Verify
        assert_eq!(loaded_graph.n, graph.n);
        assert_eq!(loaded_graph.r, graph.r);

        for i in 0..graph.n {
            let original_neighbors = graph.get_neighborhood(i);
            let loaded_neighbors = loaded_graph.get_neighborhood(i);

            assert_eq!(
                original_neighbors.len(),
                loaded_neighbors.len(),
                "Node {} has different number of neighbors",
                i
            );

            for j in 0..original_neighbors.len() {
                assert_eq!(
                    original_neighbors[j], loaded_neighbors[j],
                    "Neighbors don't match for node {} at position {}",
                    i, j
                );
            }
        }

        cleanup_temp_file(&path);
    }

    // Test saving and loading a larger graph with different degrees
    #[test]
    fn test_varying_degrees() {
        let path = temp_file_path("varying_degrees.bin");
        cleanup_temp_file(&path);

        let n = 20;
        let r = 10;
        let mut graph = ClassicGraph::new(n, r);

        // Add varying number of edges to each node
        for i in 0..n {
            let num_edges = (i % (r as u32)) as usize + 1; // 1 to r edges
            for j in 0..num_edges {
                let target = (i + j as u32 + 1) % n;
                graph.add_edge(i, target);
            }
        }

        // Save
        assert!(graph.save(&path).is_ok());

        // Load
        let loaded_graph = ClassicGraph::read(&path).expect("Failed to read varying degrees graph");

        // Verify
        assert_eq!(loaded_graph.n, graph.n);
        assert_eq!(loaded_graph.r, graph.r);

        for i in 0..graph.n {
            let expected_degree = (i % (r as u32)) as usize + 1;
            assert_eq!(
                loaded_graph.degree(i),
                expected_degree,
                "Node {} has wrong degree",
                i
            );

            let original_neighbors = graph.get_neighborhood(i);
            let loaded_neighbors = loaded_graph.get_neighborhood(i);

            for j in 0..original_neighbors.len() {
                assert_eq!(
                    original_neighbors[j], loaded_neighbors[j],
                    "Neighbors don't match for node {} at position {}",
                    i, j
                );
            }
        }

        cleanup_temp_file(&path);
    }

    // Test with a graph at max capacity
    #[test]
    fn test_max_capacity() {
        let path = temp_file_path("max_capacity.bin");
        cleanup_temp_file(&path);

        let n = 10;
        let r = 4;
        let mut graph = ClassicGraph::new(n, r);

        // Fill every node to max capacity
        for i in 0..n {
            for j in 0..r as u32 {
                let target = (i + j + 1) % n;
                graph.add_edge(i, target);
            }
        }

        // Save
        assert!(graph.save(&path).is_ok());

        // Load
        let loaded_graph = ClassicGraph::read(&path).expect("Failed to read max capacity graph");

        // Verify
        assert_eq!(loaded_graph.n, graph.n);
        assert_eq!(loaded_graph.r, graph.r);

        for i in 0..graph.n {
            assert_eq!(
                loaded_graph.degree(i),
                r,
                "Node {} doesn't have max degree",
                i
            );

            let original_neighbors = graph.get_neighborhood(i);
            let loaded_neighbors = loaded_graph.get_neighborhood(i);

            for j in 0..r {
                assert_eq!(
                    original_neighbors[j], loaded_neighbors[j],
                    "Neighbors don't match for node {} at position {}",
                    i, j
                );
            }
        }

        cleanup_temp_file(&path);
    }

    // Test edge case with a single node graph
    #[test]
    fn test_single_node_graph() {
        let path = temp_file_path("single_node.bin");
        cleanup_temp_file(&path);

        let n = 1;
        let r = 1;
        let graph = ClassicGraph::new(n, r);

        // Save
        assert!(graph.save(&path).is_ok());

        // Load
        let loaded_graph = ClassicGraph::read(&path).expect("Failed to read single node graph");

        // Verify
        assert_eq!(loaded_graph.n, n);
        assert_eq!(loaded_graph.r, r);
        assert_eq!(loaded_graph.degree(0), 0);

        cleanup_temp_file(&path);
    }

    // Test adding a large batch of neighbors at once
    #[test]
    fn test_append_neighbors() {
        let path = temp_file_path("append_neighbors.bin");
        cleanup_temp_file(&path);

        let n = 10;
        let r = 8;
        let mut graph = ClassicGraph::new(n, r);

        // Add a batch of neighbors to node 0
        let neighbors = [1, 2, 3, 4, 5];
        graph.append_neighbors(0, &neighbors);

        // Save
        assert!(graph.save(&path).is_ok());

        // Load
        let loaded_graph =
            ClassicGraph::read(&path).expect("Failed to read graph with appended neighbors");

        // Verify
        assert_eq!(loaded_graph.degree(0), neighbors.len());
        let loaded_neighbors = loaded_graph.get_neighborhood(0);
        for (i, &neighbor) in neighbors.iter().enumerate() {
            assert_eq!(loaded_neighbors[i], neighbor);
        }

        cleanup_temp_file(&path);
    }

    // Test round-trip with multiple saves and loads
    #[test]
    fn test_multiple_round_trips() {
        let path1 = temp_file_path("round_trip_1.bin");
        let path2 = temp_file_path("round_trip_2.bin");
        cleanup_temp_file(&path1);
        cleanup_temp_file(&path2);

        // Create initial graph
        let mut graph = create_test_graph();

        // First round trip
        assert!(graph.save(&path1).is_ok());
        let mut loaded_graph1 = ClassicGraph::read(&path1).expect("Failed first load");

        // Modify the loaded graph
        loaded_graph1.add_edge(7, 8);
        loaded_graph1.add_edge(8, 9);
        loaded_graph1.add_edge(9, 7);

        // Second round trip
        assert!(loaded_graph1.save(&path2).is_ok());
        let loaded_graph2 = ClassicGraph::read(&path2).expect("Failed second load");

        // Verify the twice-loaded graph
        assert_eq!(loaded_graph2.n, loaded_graph1.n);
        assert_eq!(loaded_graph2.r, loaded_graph1.r);

        for i in 0..loaded_graph1.n {
            let neighbors1 = loaded_graph1.get_neighborhood(i);
            let neighbors2 = loaded_graph2.get_neighborhood(i);

            assert_eq!(neighbors1.len(), neighbors2.len());
            for j in 0..neighbors1.len() {
                assert_eq!(neighbors1[j], neighbors2[j]);
            }
        }

        cleanup_temp_file(&path1);
        cleanup_temp_file(&path2);
    }

    // Test error handling for invalid files
    #[test]
    fn test_read_invalid_file() {
        let result = ClassicGraph::read("nonexistent_file.bin");
        assert!(result.is_err(), "Reading nonexistent file should fail");
    }

    // Test for a file with invalid header size
    #[test]
    fn test_read_invalid_header() {
        let path = temp_file_path("invalid_header.bin");
        cleanup_temp_file(&path);

        // Create a file with just 4 bytes
        {
            let mut file = fs::File::create(&path).unwrap();
            file.write_all(&[1, 2, 3, 4]).unwrap();
        }

        let result = ClassicGraph::read(&path);
        assert!(
            result.is_err(),
            "Reading file with invalid header should fail"
        );

        cleanup_temp_file(&path);
    }
}
