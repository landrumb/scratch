use rand_distr::num_traits::ToPrimitive;

#[cfg(feature = "dcmp")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "dcmp")]
static DIST_CMP_COUNT: AtomicU64 = AtomicU64::new(0);

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
    T: DenseVector,
{
    #[cfg(feature = "dcmp")]
    {
        DIST_CMP_COUNT.fetch_add(1, Ordering::Relaxed);
    }
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
    T: DenseVector,
{
    #[cfg(feature = "dcmp")]
    {
        DIST_CMP_COUNT.fetch_add(1, Ordering::Relaxed);
    }
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

use crate::data_handling::dataset_traits::Numeric;

pub trait DenseVector: Numeric + 'static {
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

impl DenseVector for f32 {
    fn sq_euclidean(a: &[Self], b: &[Self]) -> f32 {
        use simsimd::SpatialSimilarity;
        f32::sqeuclidean(a, b).unwrap().to_f32().unwrap()
    }
}

impl DenseVector for f64 {
    fn sq_euclidean(a: &[Self], b: &[Self]) -> f32 {
        use simsimd::SpatialSimilarity;
        f64::sqeuclidean(a, b).unwrap().to_f32().unwrap()
    }
}

impl DenseVector for i32 {
    fn sq_euclidean(a: &[Self], b: &[Self]) -> f32 {
        let a: Vec<f32> = a.iter().map(|&x| x as f32).collect();
        let b: Vec<f32> = b.iter().map(|&x| x as f32).collect();
        sq_euclidean(&a, &b)
    }
}

impl DenseVector for i8 {
    fn sq_euclidean(a: &[Self], b: &[Self]) -> f32 {
        use simsimd::SpatialSimilarity;
        i8::sqeuclidean(a, b).unwrap().to_f32().unwrap()
    }
}

/// Returns the number of distance comparisons performed (only if 'dcmp' feature is enabled)
#[inline]
pub fn get_distance_comparison_count() -> u64 {
    #[cfg(feature = "dcmp")]
    {
        DIST_CMP_COUNT.load(Ordering::Relaxed)
    }
    #[cfg(not(feature = "dcmp"))]
    {
        0
    }
}
