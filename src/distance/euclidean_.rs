use std::ops::Sub;

use rand_distr::num_traits::ToPrimitive;

/// calculate the euclidean distance between two vectors
///
/// Example:
/// ```
/// let a = [1.0, 2.0, 3.0];
/// let b = [4.0, 5.0, 6.0];
/// let distance = crate::scratch::distance::euclidean(&a, &b);
/// assert!((distance as f64 - f64::sqrt(27.0)).abs() < 1e-6);
///
/// let c = [1.0, 0.0, 0.0, 0.0];
/// let d = [0.0, 1.0, 0.0, 0.0];
/// let distance2 = crate::scratch::distance::euclidean(&c, &d);
/// assert!((distance2 as f64 - f64::sqrt(2.0)).abs() < 1e-6);
/// 
/// let e = [3.0, 4.0];
/// let f = [7.0, 1.0];
/// let distance3 = crate::scratch::distance::euclidean(&e, &f);
/// assert!((distance3 - 5.0).abs() < 1e-6);
/// ```
pub fn euclidean<T>(a: &[T], b: &[T]) -> f32
where
    T: Copy + Into<f64> + Sub<Output = T> + SqEuclidean,
{
    // assert_eq!(a.len(), b.len());
    // let sum = a
    //     .iter()
    //     .zip(b.iter())
    //     .map(|(&x, &y)| {
    //         let diff = (x - y).into();
    //         diff * diff
    //     })
    //     .sum::<f64>();
    // sum.sqrt() as f32
    T::euclidean(a, b)
}

pub fn sq_euclidean<T>(a: &[T], b: &[T]) -> f32
where
    T: Copy + Into<f64> + SqEuclidean,
{
    // assert_eq!(a.len(), b.len());
    // let sum = a
    //     .iter()
    //     .zip(b.iter())
    //     .map(|(&x, &y)| {
    //         let diff = x.into() - y.into();
    //         diff * diff
    //     })
    //     .sum::<f64>();
    // sum as f32
    T::sq_euclidean(a, b)
}

pub trait SqEuclidean {
    fn sq_euclidean(a: &[Self], b: &[Self]) -> f32
    where
        Self: Sized;
    fn euclidean(a: &[Self], b: &[Self]) -> f32
    where
        Self: Sized,
        {
        f32::sqrt(Self::sq_euclidean(a, b))
    }
}

impl SqEuclidean for f32 {
    fn sq_euclidean(a: &[Self], b: &[Self]) -> f32 {
        use simsimd::SpatialSimilarity;
        f32::sqeuclidean(a, b).unwrap().to_f32().unwrap()
    }
}

impl SqEuclidean for f64 {
    fn sq_euclidean(a: &[Self], b: &[Self]) -> f32 {
        use simsimd::SpatialSimilarity;
        f64::sqeuclidean(a, b).unwrap().to_f32().unwrap()
    }
}

impl SqEuclidean for i32 {
    fn sq_euclidean(a: &[Self], b: &[Self]) -> f32 {
        let a: Vec<f32> = a.iter().map(|&x| x as f32).collect();
        let b: Vec<f32> = b.iter().map(|&x| x as f32).collect();
        sq_euclidean(&a, &b)
    }
}

impl SqEuclidean for i8 {
    fn sq_euclidean(a: &[Self], b: &[Self]) -> f32 {
        use simsimd::SpatialSimilarity;
        i8::sqeuclidean(a, b).unwrap().to_f32().unwrap()
    }
}
