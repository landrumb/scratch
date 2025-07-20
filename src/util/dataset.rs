use std::path::{Path, PathBuf};

pub struct DatasetPaths {
    pub base: PathBuf,
    pub query: PathBuf,
    pub gt: PathBuf,
    pub outputs: PathBuf,
}

pub fn infer_dataset_paths(dataset: &str) -> DatasetPaths {
    let dataset_root = if Path::new(dataset).exists() {
        PathBuf::from(dataset)
    } else {
        PathBuf::from("data").join(dataset)
    };
    DatasetPaths {
        base: dataset_root.join("base.fbin"),
        query: dataset_root.join("query.fbin"),
        gt: dataset_root.join("GT"),
        outputs: dataset_root.join("outputs"),
    }
}
