use std::ops::Sub;

pub fn euclidean<T>(a: &[T], b: &[T], length: usize) -> f64
where
    T: Copy + Into<f64> + Sub<Output = T>,
{
    assert!(a.len() >= length && b.len() >= length);
    let sum = a.iter()
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
    let sum = a.iter()
        .zip(b.iter())
        .take(length)
        .map(|(&x, &y)| {
            let diff = x.into() - y.into();
            diff * diff
        })
        .sum::<f64>();
    sum
}