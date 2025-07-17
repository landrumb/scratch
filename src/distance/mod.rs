mod euclidean_;

pub use self::euclidean_::{euclidean, sq_euclidean, SqEuclidean};

pub enum Distance {
    Euclidean,
    SquaredEuclidean,
    Cosine,
    InnerProduct,
}
impl Distance {
    pub fn from_str(s: &str) -> Self {
        match s {
            "euclidean" => Distance::Euclidean,
            "squared_euclidean" => Distance::SquaredEuclidean,
            "cosine" => Distance::Cosine,
            "inner_product" => Distance::InnerProduct,
            _ => panic!("Unknown distance: {}", s),
        }
    }
}