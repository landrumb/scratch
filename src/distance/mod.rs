mod euclidean_;

pub use self::euclidean_::{euclidean, get_distance_comparison_count, sq_euclidean, SqEuclidean};

pub enum Distance {
    Euclidean,
    SquaredEuclidean,
    Cosine,
    InnerProduct,
}
use std::str::FromStr;

impl FromStr for Distance {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "euclidean" => Ok(Distance::Euclidean),
            "squared_euclidean" => Ok(Distance::SquaredEuclidean),
            "cosine" => Ok(Distance::Cosine),
            "inner_product" => Ok(Distance::InnerProduct),
            _ => Err(()),
        }
    }
}
