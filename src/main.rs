#![allow(unused_variables)]

use std::env::args;
use std::path::Path;
use std::time::Instant;

use scratch::data_handling::dataset::VectorDataset;
use scratch::data_handling::fbin::read_fbin;

fn main() {
    let default_data_file = String::from("data/word2vec-google-news-300_50000_lowercase/base.fbin");

    // presumably we'll have some command line arguments here

    let data_path_arg = args().nth(1).unwrap_or(default_data_file);
    let data_path = Path::new(&data_path_arg);

    let start = Instant::now();

    let dataset: VectorDataset<f32> = read_fbin(&data_path);

    let elapsed = start.elapsed();
    println!(
        "read dataset in {}.{:03} seconds",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );
}
