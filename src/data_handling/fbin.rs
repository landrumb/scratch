//! functions for reading and writing fbin files

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{self, Path};

use memmap2::Mmap;

use super::dataset::VectorDataset;
use super::dataset_traits::Numeric;

/// read a dataset from an fbin file
pub fn read_fbin<T: Numeric>(path: &Path) -> VectorDataset<T> {
    let file = File::open(path).expect("could not open file");
    let mmap = unsafe { Mmap::map(&file).expect("could not mmap file") };
    let mut reader = BufReader::new(&*mmap);

    let mut n_buf = [0u8; 4];
    reader.read_exact(&mut n_buf).expect("could not read n");
    let n = u32::from_le_bytes(n_buf);

    let mut dim_buf = [0u8; 4];
    reader.read_exact(&mut dim_buf).expect("could not read dim");
    let dim = u32::from_le_bytes(dim_buf);

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

/// reads only a subset of the dataset from an fbin file, but is otherwise the same as `read_fbin`
pub fn read_fbin_subset<T: Numeric>(path: &Path, subset_size: usize) -> VectorDataset<T> {
    let file = File::open(path).expect("could not open file");
    let mmap = unsafe { Mmap::map(&file).expect("could not mmap file") };
    let mut reader = BufReader::new(&*mmap);

    let mut n_buf = [0u8; 4];
    reader.read_exact(&mut n_buf).expect("could not read n");
    let n = u32::from_le_bytes(n_buf);

    assert!(
        subset_size <= n as usize,
        "subset size {} is larger than dataset size {}",
        subset_size,
        n
    );

    let n = subset_size as u32;

    let mut dim_buf = [0u8; 4];
    reader.read_exact(&mut dim_buf).expect("could not read dim");
    let dim = u32::from_le_bytes(dim_buf);
    let element_size = std::mem::size_of::<T>();
    let expected_size = subset_size * dim as usize * element_size;

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