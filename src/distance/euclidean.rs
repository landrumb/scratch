use std::ops::Sub;

/// calculate the euclidean distance between two vectors
///
/// Example:
/// ```
/// let a = [1.0, 2.0, 3.0];
/// let b = [4.0, 5.0, 6.0];
/// let distance = scratch::distance::euclidean::euclidean(&a, &b, 3);
/// assert!((distance - 5.196152).abs() < 1e-6);
/// ```
pub fn euclidean<T>(a: &[T], b: &[T], length: usize) -> f64
where
    T: Copy + Into<f64> + Sub<Output = T>,
{
    assert!(a.len() >= length && b.len() >= length);
    let sum = a
        .iter()
        .zip(b.iter())
        .take(length)
        .map(|(&x, &y)| {
            let diff = (x - y).into();
            diff * diff
        })
        .sum::<f64>();
    sum.sqrt()
}

pub fn sq_euclidean<T>(a: &[T], b: &[T], length: usize) -> f64
where
    T: Copy + Into<f64>,
{
    assert!(a.len() >= length && b.len() >= length);
    let sum = a
        .iter()
        .zip(b.iter())
        .take(length)
        .map(|(&x, &y)| {
            let diff = x.into() - y.into();
            diff * diff
        })
        .sum::<f64>();
    sum
}
