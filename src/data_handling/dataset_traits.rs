//! traits for generic handling of collections of items

use std::ops::Sub;

pub trait Dataset<T>: Send + Sync {
    fn compare_internal(&self, i: usize, j: usize) -> f64; // this probably should take reference args(?)
    fn compare(&self, q: &[T], i: usize) -> f64;
    fn get(&self, i: usize) -> &[T];
    fn size(&self) -> usize;

    fn brute_force(&self, q: &[T]) -> Box<[(usize, f32)]> {
        let mut result = Vec::with_capacity(self.size());
        for i in 0..self.size() {
            result.push((i, self.compare(q, i) as f32));
        }
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        result.into_boxed_slice()
    }

    fn brute_force_internal(&self, q: usize) -> Box<[(usize, f32)]> {
        let mut result = Vec::with_capacity(self.size());
        for i in 0..self.size() {
            result.push((i, self.compare_internal(q, i) as f32));
        }
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        result.into_boxed_slice()
    }

    fn brute_force_subset(&self, q: &[T], subset: &[usize]) -> Box<[(usize, f32)]> {
        let mut result = Vec::with_capacity(subset.len());
        for &i in subset {
            result.push((i, self.compare(q, i) as f32));
        }
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        result.into_boxed_slice()
    }

    // Rewritten with slice instead of iterator
    fn brute_force_subset_internal(&self, q: usize, subset: &[usize]) -> Box<[(usize, f32)]> {
        let mut result = Vec::with_capacity(subset.len());
        for &i in subset {
            result.push((i, self.compare_internal(q, i) as f32));
        }
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        result.into_boxed_slice()
    }
}



pub trait Numeric: Copy + Into<f64> + Sub<Output = Self> + Default + Send + Sync {}

impl Numeric for f32 {}
impl Numeric for f64 {}
impl Numeric for i32 {}
impl Numeric for i8 {}
