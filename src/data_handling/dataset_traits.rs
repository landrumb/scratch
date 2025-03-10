//! traits for generic handling of collections of items

use std::ops::Sub;

pub trait Dataset<T> {
    fn compare_internal(&self, i: usize, j: usize) -> f64; // this probably should take reference args(?)
    fn compare(&self, q: &[T], i: usize) -> f64;
    fn size(&self) -> usize;
}

pub trait Numeric: Copy + Into<f64> + Sub<Output = Self> + Default + Send + Sync {}

impl Numeric for f32 {}
impl Numeric for f64 {}
impl Numeric for i32 {}
impl Numeric for i8 {}
