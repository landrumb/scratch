mod euclidean_;

pub use self::euclidean_::{euclidean, sq_euclidean};

pub enum Distance {
    Euclidean,
    Cosine,
    InnerProduct,
}
impl Distance {
    pub fn from_str(s: &str) -> Self {
        match s {
            "euclidean" => Distance::Euclidean,
            "cosine" => Distance::Cosine,
            "inner_product" => Distance::InnerProduct,
            _ => panic!("Unknown distance: {}", s),
        }
    }
}