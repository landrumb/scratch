//! function(s) for evaluating recall
use std::cmp::min;

/// returns the recall of an output relative to the groundtruth
///
/// the $k$ for the purposes of recall@k is the lesser of the lengths of the output and groundtruth
pub fn recall<T: Eq>(output: &[T], groundtruth: &[T]) -> f64 {
    let mut matches = 0;
    let length = min(output.len(), groundtruth.len());

    for neighbor in output.iter().take(length) {
        if groundtruth[..length].contains(neighbor) {
            matches += 1;
        }
    }

    matches as f64 / length as f64
}
