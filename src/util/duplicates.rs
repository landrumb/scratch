use std::collections::{HashMap, HashSet};

use crate::data_handling::dataset::VectorDataset;

/// Finds duplicate vectors in a dataset and returns the sets of ids for each group.
/// Each set contains the indices of vectors that are identical.
pub fn duplicate_sets_f32(dataset: &VectorDataset<f32>) -> Vec<HashSet<usize>> {
    let mut map: HashMap<Vec<u32>, HashSet<usize>> = HashMap::new();
    for i in 0..dataset.n {
        let bits: Vec<u32> = dataset.get(i).iter().map(|&v| v.to_bits()).collect();
        map.entry(bits).or_default().insert(i);
    }
    map.into_iter()
        .filter_map(|(_, set)| if set.len() > 1 { Some(set) } else { None })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_duplicate_sets() {
        let data: Vec<f32> = vec![
            1.0, 0.0,
            0.5, 0.5,
            1.0, 0.0,
            0.5, 0.5,
            2.0, 1.0,
        ];
        let dataset = VectorDataset::new(data.into_boxed_slice(), 5, 2);
        let sets = duplicate_sets_f32(&dataset);
        // Expect two duplicate sets: {0,2} and {1,3}
        assert_eq!(sets.len(), 2);
        for set in sets {
            assert!(set == [0usize, 2].iter().cloned().collect::<HashSet<_>>()
                || set == [1usize, 3].iter().cloned().collect::<HashSet<_>>());
        }
    }
}
