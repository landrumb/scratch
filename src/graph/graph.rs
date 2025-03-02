//! implementation of a regular degree-limited graph

use std::io::{Read, Write};


type IndexT = u32;

struct ClassicGraph {
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

        // Write header: n and r (as u32)
        file.write_all(&self.n.to_le_bytes())?;
        file.write_all(&(self.r as u32).to_le_bytes())?;

        // Write degrees for each node
        for &deg in self.degrees.iter() {
            file.write_all(&deg.to_le_bytes())?;
        }

        // Write edge data sequentially for each node
        for i in 0..(self.n as usize) {
            let d = self.degrees[i] as usize;
            for j in 0..d {
                file.write_all(&self.neighborhoods[i][j].to_le_bytes())?;
            }
        }
        Ok(())
    }

    pub fn read(path: &str) -> ClassicGraph {
        let file = std::fs::File::open(path).expect("could not open file");

        let mut reader = std::io::BufReader::new(file);
        let mut buffer = [0 as u8; 8];

        // Read header: n and r (as u32)
        reader.read_exact(&mut buffer).expect("could not read n");
        let n = u32::from_le_bytes(buffer[0..4].try_into().expect("could not read n"));
        let r = u32::from_le_bytes(buffer[4..8].try_into().expect("could not read r")) as usize;

        let mut degrees = vec![0; n as usize].into_boxed_slice();
        let mut neighborhoods: Box<[Box<[IndexT]>]> = (0..n).map(|_| vec![0; r].into_boxed_slice()).collect();


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
        assert!(self.degrees[from as usize] < self.r as IndexT);


        self.neighborhoods[from_index][self.degrees[from_index] as usize] = to;
        self.degrees[from_index] += 1;
    }


}