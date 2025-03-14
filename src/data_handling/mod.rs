pub mod dataset;
pub mod dataset_traits;
pub mod fbin;

#[cfg(test)]
mod tests {
    use super::dataset::VectorDataset;

    #[test]
    fn test_brute_force() {
        let data: Box<[f32]> = vec![
            1.0, 0.0, 0.0, // Vector 0
            0.0, 1.0, 0.0, // Vector 1
            0.0, 0.0, 1.0, // Vector 2
            0.5, 0.5, 0.0, // Vector 3
        ]
        .into_boxed_slice();

        let dataset = VectorDataset::new(data, 4, 3);
        let query = vec![1.0, 0.0, 0.0];
        let subset = vec![0, 1, 2, 3];

        let results = dataset.brute_force(&query, &subset);

        // Vector 0 should be closest to query
        assert_eq!(results[0].0, 0);
        assert_eq!(results[0].1, 0.0);

        // Vector 3 should be second closest
        assert_eq!(results[1].0, 3);
        assert_eq!(results[1].1, f32::sqrt(0.5));
    }

    #[test]
    fn test_closest() {
        let data: Box<[f32]> = vec![
            1.0, 0.0, 0.0, // Vector 0
            0.0, 1.0, 0.0, // Vector 1
            0.0, 0.0, 1.0, // Vector 2
            0.5, 0.5, 0.0, // Vector 3
        ]
        .into_boxed_slice();

        let dataset = VectorDataset::new(data, 4, 3);

        // Test with query matching vector 0
        let query1 = vec![1.0, 0.0, 0.0];
        let candidates1 = vec![0, 1, 2, 3];
        let closest_idx1 = dataset.closest(&query1, &candidates1);
        assert_eq!(closest_idx1, 0);

        // Test with query closest to vector 3
        let query2 = vec![0.6, 0.4, 0.0];
        let candidates2 = vec![0, 1, 2, 3];
        let closest_idx2 = dataset.closest(&query2, &candidates2);
        assert_eq!(closest_idx2, 3);

        // Test with empty candidates
        let empty_candidates: Vec<usize> = vec![];
        let closest_idx3 = dataset.closest(&query1, &empty_candidates);
        assert_eq!(closest_idx3, 0);
    }
}
