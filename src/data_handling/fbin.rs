//! functions for reading and writing fbin files

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use memmap2::Mmap;

use crate::data_handling::dataset::{Numeric, VectorDataset};

/// read a dataset from an fbin file
pub fn read_fbin<T: Numeric>(path: &Path) -> VectorDataset<T> {
    let file = File::open(path).expect("could not open file");
    let mmap = unsafe { Mmap::map(&file).expect("could not mmap file") };
    let mut reader = BufReader::new(&*mmap);

    let n = u32::from_le_bytes(mmap[0..4].try_into().expect("could not read n"));
    let dim = u32::from_le_bytes(mmap[4..8].try_into().expect("could not read dim"));

    let element_size = std::mem::size_of::<T>();
    let expected_size = n as usize * dim as usize * element_size;
    assert!(
        mmap.len() == expected_size + 8,
        "expected {} bytes, got {}",
        expected_size + 8,
        mmap.len()
    );

    let data = {
        let mut data = vec![T::default(); n as usize * dim as usize];
        reader
            .read_exact(unsafe {
                std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, expected_size)
            })
            .expect("could not read data");
        Box::from(data)
    };

    VectorDataset::new(data, n as usize, dim as usize)
}
