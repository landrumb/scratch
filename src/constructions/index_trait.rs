//! a trait for generic index evaluations

use std::{any::Any, collections::HashMap};

use crate::graph::IndexT;

// having a struct that you have to change every time you need a new parameter is obviously bad
pub struct Parameters {
    values: HashMap<String, Box<dyn Any>>,
}

impl Default for Parameters {
    fn default() -> Self {
        Self::new()
    }
}

impl Parameters {
    pub fn new() -> Self {
        Parameters {
            values: HashMap::new(),
        }
    }

    pub fn set<T: Any>(&mut self, key: &str, value: T) {
        self.values.insert(key.to_string(), Box::new(value));
    }

    pub fn get<T: Any>(&self, key: &str) -> Option<&T> {
        self.values.get(key).and_then(|v| v.downcast_ref::<T>())
    }
}

pub trait VectorIndex<T> {
    /// Does k-NN search on the index
    fn query(&self, query: &[T], parameters: Parameters) -> Vec<IndexT>;
}
