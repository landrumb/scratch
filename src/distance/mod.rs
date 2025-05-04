mod euclidean_;

pub use self::euclidean_::{euclidean, sq_euclidean};

pub enum Distance {
    Euclidean,
    Cosine,
    InnerProduct,
}