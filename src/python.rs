//! Python bindings for the scratch library
//! This module is only compiled when the "python" feature is enabled

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray2, PyReadonlyArray1};
use crate::data_handling::dataset::VectorDataset;
use crate::data_handling::dataset_traits::Dataset;

#[pyclass]
pub struct PyVectorDataset {
    dataset: VectorDataset<f32>,
}

#[pymethods]
impl PyVectorDataset {
    #[new]
    fn new(data: PyReadonlyArray2<f32>) -> PyResult<Self> {
        let array = data.as_array();
        let n = array.shape()[0];
        let dim = array.shape()[1];
        
        // Create a contiguous copy of the data
        let data_vec: Vec<f32> = array.iter().copied().collect();
        
        Ok(PyVectorDataset {
            dataset: VectorDataset::new(data_vec.into_boxed_slice(), n, dim),
        })
    }
    
    #[getter]
    fn get_n(&self) -> usize {
        self.dataset.n
    }
    
    #[getter]
    fn get_dim(&self) -> usize {
        self.dataset.dim
    }
    
    fn get_vector(&self, idx: usize, py: Python<'_>) -> PyResult<PyObject> {
        if idx >= self.dataset.n {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                format!("Index {} out of bounds for dataset with {} elements", idx, self.dataset.n)
            ));
        }
        
        let vector = self.dataset.get(idx);
        Ok(PyArray1::from_slice(py, vector).to_object(py))
    }
    
    fn compare_internal(&self, i: usize, j: usize) -> f64 {
        Dataset::compare_internal(&self.dataset, i, j)
    }
    
    fn compare(&self, query: PyReadonlyArray1<f32>, i: usize) -> PyResult<f64> {
        if i >= self.dataset.n {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                format!("Index {} out of bounds for dataset with {} elements", i, self.dataset.n)
            ));
        }
        
        let query_slice = query.as_slice()?;
        if query_slice.len() != self.dataset.dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Query dimension {} does not match dataset dimension {}", 
                        query_slice.len(), self.dataset.dim)
            ));
        }
        
        Ok(Dataset::compare(&self.dataset, query_slice, i))
    }
    
    fn size(&self) -> usize {
        Dataset::size(&self.dataset)
    }
    
    fn brute_force(&self, query: PyReadonlyArray1<f32>, _py: Python<'_>) -> PyResult<Vec<(usize, f32)>> {
        let query_slice = query.as_slice()?;
        if query_slice.len() != self.dataset.dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Query dimension {} does not match dataset dimension {}", 
                        query_slice.len(), self.dataset.dim)
            ));
        }
        
        let results = Dataset::brute_force(&self.dataset, query_slice);
        Ok(results.into_vec())
    }
    
    fn brute_force_internal(&self, q: usize, _py: Python<'_>) -> PyResult<Vec<(usize, f32)>> {
        if q >= self.dataset.n {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                format!("Index {} out of bounds for dataset with {} elements", q, self.dataset.n)
            ));
        }
        
        let results = Dataset::brute_force_internal(&self.dataset, q);
        Ok(results.into_vec())
    }
    
    fn brute_force_subset(
        &self, 
        query: PyReadonlyArray1<f32>, 
        subset: Vec<usize>, 
        _py: Python<'_>
    ) -> PyResult<Vec<(usize, f32)>> {
        let query_slice = query.as_slice()?;
        if query_slice.len() != self.dataset.dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Query dimension {} does not match dataset dimension {}", 
                        query_slice.len(), self.dataset.dim)
            ));
        }
        
        // Validate subset indices
        for &idx in &subset {
            if idx >= self.dataset.n {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    format!("Subset index {} out of bounds for dataset with {} elements", idx, self.dataset.n)
                ));
            }
        }
        
        let results = Dataset::brute_force_subset(&self.dataset, query_slice, &subset);
        Ok(results.into_vec())
    }
    
    fn brute_force_subset_internal(
        &self, 
        q: usize, 
        subset: Vec<usize>,
        _py: Python<'_>
    ) -> PyResult<Vec<(usize, f32)>> {
        if q >= self.dataset.n {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                format!("Index {} out of bounds for dataset with {} elements", q, self.dataset.n)
            ));
        }
        
        // Validate subset indices
        for &idx in &subset {
            if idx >= self.dataset.n {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    format!("Subset index {} out of bounds for dataset with {} elements", idx, self.dataset.n)
                ));
            }
        }
        
        let results = Dataset::brute_force_subset_internal(&self.dataset, q, &subset);
        Ok(results.into_vec())
    }
}

/// Python module for scratch library
#[pymodule]
pub fn scratch(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyVectorDataset>()?;
    Ok(())
}