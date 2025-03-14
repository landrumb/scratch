//! Traits for graph interfaces

pub type IndexT = u32;

pub trait Graph {
    fn neighbors(&self, i: IndexT) -> &[IndexT];
}

pub trait MutableGraph {
    fn add_neighbor(&mut self, from: IndexT, to: IndexT);
    fn set_neighborhood(&mut self, i: IndexT, neighborhood: &[IndexT]);
}
