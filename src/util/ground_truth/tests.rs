use super::*;
use crate::data_handling::dataset_traits::Dataset;
use tempfile::tempdir;

// A simple mock dataset for testing
struct MockVectorDataset {
    data: Vec<Vec<f32>>,
}

impl MockVectorDataset {
    fn new(data: Vec<Vec<f32>>) -> Self {
        MockVectorDataset { data }
    }
}

impl Dataset<f32> for MockVectorDataset {
    fn compare_internal(&self, i: usize, j: usize) -> f64 {
        let v1 = &self.data[i];
        let v2 = &self.data[j];

        // Calculate Euclidean distance
        let sum_sq: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| (a - b).powi(2)).sum();

        sum_sq as f64
    }

    fn compare(&self, q: &[f32], i: usize) -> f64 {
        let v = &self.data[i];

        // Calculate Euclidean distance
        let sum_sq: f32 = q.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum();

        sum_sq as f64
    }

    fn size(&self) -> usize {
        self.data.len()
    }
}

#[test]
fn test_ground_truth_computation() {
    // Create a simple 2D dataset
    // Point 0: (0, 0)
    // Point 1: (1, 0)
    // Point 2: (0, 1)
    // Point 3: (1, 1)
    // Point 4: (2, 2)
    let dataset = MockVectorDataset::new(vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
    ]);

    // Compute ground truth for k=2
    let gt = compute_ground_truth(&dataset, &dataset, 2).unwrap();

    // Verify the number of points and neighbors
    assert_eq!(gt.n, 5);
    assert_eq!(gt.k, 2);

    // Test point 0: (0,0)
    // Nearest should be itself, then (1,0) or (0,1)
    let neighbors = gt.get_neighbors(0);
    assert_eq!(neighbors[0], 0); // Self
    assert!(neighbors[1] == 1 || neighbors[1] == 2); // (1,0) or (0,1)

    // Test point 3: (1,1)
    // Nearest should be itself, then either (1,0) or (0,1)
    let neighbors = gt.get_neighbors(3);
    assert_eq!(neighbors[0], 3); // Self
    assert!(neighbors[1] == 1 || neighbors[1] == 2); // (1,0) or (0,1)

    // Test point 4: (2,2)
    // Nearest should be itself, then (1,1)
    let neighbors = gt.get_neighbors(4);
    assert_eq!(neighbors[0], 4); // Self
    assert_eq!(neighbors[1], 3); // (1,1)

    // Test that distances are correct
    let distances = gt.get_distances(0);
    assert_eq!(distances[0], 0.0); // Self = 0

    let distances = gt.get_distances(4);
    assert_eq!(distances[0], 0.0); // Self = 0
    assert_eq!(distances[1], 2.0); // Distance to (1,1) = √((2-1)² + (2-1)²) = √2 = 2
}

#[test]
fn test_ground_truth_file_io() {
    // Create a simple dataset
    let dataset = MockVectorDataset::new(vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]]);

    // Compute ground truth
    let ground_truth = compute_ground_truth(&dataset, &dataset, 2).unwrap();

    // Create a temporary directory to store the file
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("test_gt.bin");

    // Write ground truth to file
    ground_truth.write(&file_path).unwrap();

    // Read ground truth back from file
    let read_gt = GroundTruth::read(&file_path);

    // Verify the contents
    assert_eq!(read_gt.n, ground_truth.n);
    assert_eq!(read_gt.k, ground_truth.k);

    for i in 0..read_gt.n {
        // Check that all neighbors match
        let original_neighbors = ground_truth.get_neighbors(i);
        let read_neighbors = read_gt.get_neighbors(i);

        for j in 0..read_gt.k {
            assert_eq!(original_neighbors[j], read_neighbors[j]);
        }

        // Check that all distances match
        let original_distances = ground_truth.get_distances(i);
        let read_distances = read_gt.get_distances(i);

        for j in 0..read_gt.k {
            assert!((original_distances[j] - read_distances[j]).abs() < 1e-6);
        }
    }
}

#[test]
fn test_large_k() {
    // Test case where k is larger than dataset size
    let dataset = MockVectorDataset::new(vec![vec![0.0, 0.0], vec![1.0, 0.0]]);

    // Compute ground truth with k > dataset size
    let gt = compute_ground_truth(&dataset, &dataset, 5).unwrap();

    // Verify the number of points and neighbors
    assert_eq!(gt.n, 2);
    assert_eq!(gt.k, 5);

    // Check that valid neighbors are first, followed by padding
    let neighbors = gt.get_neighbors(0);
    assert_eq!(neighbors[0], 0); // Self
    assert_eq!(neighbors[1], 1); // Other point

    // The rest should be padded with u32::MAX
    for i in 2..gt.k {
        assert_eq!(neighbors[i], u32::MAX);
    }

    // Check distances too
    let distances = gt.get_distances(0);
    assert_eq!(distances[0], 0.0); // Self
    assert_eq!(distances[1], 1.0); // Other point at distance 1

    // The rest should be padded with f32::MAX
    for i in 2..gt.k {
        assert_eq!(distances[i], f32::MAX);
    }
}
