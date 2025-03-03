//! implementation of a regular degree-limited graph

use std::io::{Read, Write, BufReader};
use std::fs::File;

pub type IndexT = u32;

pub struct ClassicGraph {
    neighborhoods: Box<[Box<[IndexT]>]>,
    degrees: Box<[IndexT]>,
    pub n: IndexT, // number of nodes
    pub r: usize, // degree limit
}

impl ClassicGraph {
    pub fn new(n: IndexT, r: usize) -> ClassicGraph {
        let neighborhoods: Box<[Box<[IndexT]>]> = (0..n).map(|_| vec![0; r].into_boxed_slice()).collect();
        let degrees = vec![0; n as usize].into_boxed_slice();

        ClassicGraph { neighborhoods, degrees, n, r }
    }
     
    /// save the graph to a file
    /// uses the parlayANN format for compatibility
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        println!("Writing graph with {} nodes and max degree {}", self.n, self.r);

        // Write header: n and r (as u32)
        file.write_all(&self.n.to_le_bytes())?;
        file.write_all(&(self.r as u32).to_le_bytes())?;

        // Write degrees for each node
        for &deg in self.degrees.iter() {
            file.write_all(&deg.to_le_bytes())?;
        }

        // Write edge data sequentially for each node in blocks (similar to the C++ impl)
        const BLOCK_SIZE: usize = 1_000_000;
        let mut node_index = 0;
        
        while node_index < self.n as usize {
            let block_start = node_index;
            let block_end = (block_start + BLOCK_SIZE).min(self.n as usize);
            
            // Gather all edges for this block
            let mut block_edges = Vec::new();
            for i in block_start..block_end {
                let degree = self.degrees[i] as usize;
                for j in 0..degree {
                    block_edges.push(self.neighborhoods[i][j]);
                }
            }
            
            // Write all edges in this block
            for &edge in &block_edges {
                file.write_all(&edge.to_le_bytes())?;
            }
            
            node_index = block_end;
        }
        
        Ok(())
    }

    /// read a graph from a file in parlayANN format
    pub fn read(path: &str) -> std::io::Result<ClassicGraph> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        
        // Read header: n and r
        let mut header_buf = [0u8; 8];
        reader.read_exact(&mut header_buf)?;
        let n = u32::from_le_bytes(header_buf[0..4].try_into().unwrap());
        let r = u32::from_le_bytes(header_buf[4..8].try_into().unwrap()) as usize;
        
        println!("Reading graph with {} nodes and max degree {}", n, r);
        
        // Read all degrees at once
        let mut degrees = vec![0u32; n as usize];
        for i in 0..n as usize {
            let mut degree_buf = [0u8; 4];
            reader.read_exact(&mut degree_buf)?;
            degrees[i] = u32::from_le_bytes(degree_buf);
        }
        
        // Calculate total edges
        let total_edges: usize = degrees.iter().map(|&d| d as usize).sum();
        println!("Total edges in graph: {}", total_edges);
        
        // Create neighborhoods
        let mut neighborhoods = Vec::with_capacity(n as usize);
        for _ in 0..n {
            neighborhoods.push(vec![0u32; r].into_boxed_slice());
        }
        
        // Read edges in blocks (like the C++ implementation)
        const BLOCK_SIZE: usize = 1_000_000;
        let mut node_index = 0;
        // To reuse code from the analysis tool in an Artifact, you must rewrite the code in its entirety in the Artifact.a
        let mut _total_size_read = 0;
        
        while node_index < n as usize {
            let block_start = node_index;
            let block_end = (block_start + BLOCK_SIZE).min(n as usize);
            
            // Calculate how many edges to read for this block
            let block_edges: usize = degrees[block_start..block_end].iter().map(|&d| d as usize).sum();
            
            // Read all edges for this block
            let mut edges_buf = vec![0u8; block_edges * 4];
            reader.read_exact(&mut edges_buf)?;
            
            // Distribute edges to neighborhoods
            let mut edge_offset = 0;
            for i in block_start..block_end {
                let degree = degrees[i] as usize;
                for j in 0..degree {
                    let start_byte = edge_offset * 4;
                    let end_byte = start_byte + 4;
                    
                    neighborhoods[i][j] = u32::from_le_bytes(
                        edges_buf[start_byte..end_byte].try_into().unwrap()
                    );
                    
                    edge_offset += 1;
                }
            }
            
            _total_size_read += block_edges;
            node_index = block_end;
        }
        
        Ok(ClassicGraph {
            neighborhoods: neighborhoods.into_boxed_slice(),
            degrees: degrees.into_boxed_slice(),
            n,
            r,
        })
    }

    /// returns a view of the neighborhood of a node
    pub fn get_neighborhood(&self, i: IndexT) -> &[IndexT] {
        assert!(i < self.n);
        &self.neighborhoods[i as usize][..self.degrees[i as usize] as usize]
    }

    /// overwrites the neighborhood of a node
    pub fn set_neighborhood(&mut self, i: IndexT, neighborhood: &[IndexT]) {
        assert!(i < self.n);
        assert!(neighborhood.len() <= self.r); // neighborhood must be smaller than the degree limit
        self.degrees[i as usize] = neighborhood.len() as IndexT;
        self.neighborhoods[i as usize][..neighborhood.len()].copy_from_slice(neighborhood);
    }

    /// adds a neighbor to a node
    pub fn add_edge(&mut self, from: IndexT, to: IndexT) {
        let from_index = from as usize;
        assert!(from < self.n && to < self.n);
        assert!(self.degrees[from_index] < self.r as IndexT);

        self.neighborhoods[from_index][self.degrees[from_index] as usize] = to;
        self.degrees[from_index] += 1;
    }
    
    /// adds an undirected edge (adds in both directions)
    pub fn add_undirected_edge(&mut self, a: IndexT, b: IndexT) {
        self.add_edge(a, b);
        self.add_edge(b, a);
    }
    
    /// returns the number of nodes in the graph
    pub fn size(&self) -> usize {
        self.n as usize
    }
    
    /// returns the maximum degree of the graph
    pub fn max_degree(&self) -> usize {
        self.r
    }
    
    /// returns the current degree of a node
    pub fn degree(&self, i: IndexT) -> usize {
        assert!(i < self.n);
        self.degrees[i as usize] as usize
    }
    
    /// clears the neighborhood of a node
    pub fn clear_neighborhood(&mut self, i: IndexT) {
        assert!(i < self.n);
        self.degrees[i as usize] = 0;
    }
    
    /// appends a list of neighbors to a node
    pub fn append_neighbors(&mut self, i: IndexT, neighbors: &[IndexT]) {
        assert!(i < self.n);
        let i_usize = i as usize;
        let current_degree = self.degrees[i_usize] as usize;
        assert!(current_degree + neighbors.len() <= self.r, 
                "Cannot exceed max degree {} for node {}", self.r, i);
        
        for (j, &neighbor) in neighbors.iter().enumerate() {
            self.neighborhoods[i_usize][current_degree + j] = neighbor;
        }
        self.degrees[i_usize] += neighbors.len() as IndexT;
    }
    
    /// sorts the neighborhood of a node according to a comparator
    pub fn sort_neighborhood<F>(&mut self, i: IndexT, comparator: F)
    where
        F: FnMut(&IndexT, &IndexT) -> std::cmp::Ordering
    {
        assert!(i < self.n);
        let i_usize = i as usize;
        let degree = self.degrees[i_usize] as usize;
        self.neighborhoods[i_usize][..degree].sort_by(comparator);
    }
}

/// A wrapper around a slice of neighbors, providing similar functionality to the C++ edgeRange
pub struct EdgeRange<'a> {
    neighbors: &'a [IndexT],
    id: IndexT,
}

impl<'a> EdgeRange<'a> {
    pub fn new(neighbors: &'a [IndexT], id: IndexT) -> Self {
        EdgeRange { neighbors, id }
    }
    
    pub fn size(&self) -> usize {
        self.neighbors.len()
    }
    
    pub fn id(&self) -> IndexT {
        self.id
    }
    
    /// prefetch the neighborhood into cache (similar to C++ implementation)
    pub fn prefetch(&self) {
        // This is a no-op in Rust since we don't have direct cache control
        // The C++ version uses __builtin_prefetch
    }
}

impl<'a> std::ops::Index<usize> for EdgeRange<'a> {
    type Output = IndexT;
    
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.neighbors.len(), 
                "index exceeds degree while accessing neighbors");
        &self.neighbors[index]
    }
}

impl ClassicGraph {
    /// returns an EdgeRange view of the neighborhood of a node
    /// this is equivalent to the [] operator in the C++ implementation
    pub fn get_edge_range(&self, i: IndexT) -> EdgeRange {
        assert!(i < self.n, "graph index out of range: {}", i);
        EdgeRange::new(self.get_neighborhood(i), i)
    }
}

/// Implement indexing for ClassicGraph to be similar to the C++ Graph
impl std::ops::Index<IndexT> for ClassicGraph {
    type Output = [IndexT];
    
    fn index(&self, index: IndexT) -> &Self::Output {
        self.get_neighborhood(index)
    }
}