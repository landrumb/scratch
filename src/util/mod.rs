use std::collections::{HashMap, HashSet};
pub mod dataset;
pub mod duplicates;
pub mod ground_truth;
pub mod recall;

pub struct DSU {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl DSU {
    /// Create `n` singleton sets: 0, 1, …, n-1.
    pub fn new(n: usize) -> Self {
        DSU {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    /// Find the set representative for `x`, compressing paths.
    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let root = self.find(self.parent[x]);
            self.parent[x] = root;
        }
        self.parent[x]
    }

    /// Union the sets containing `x` and `y`.
    pub fn union(&mut self, x: usize, y: usize) {
        let mut a = self.find(x);
        let mut b = self.find(y);
        if a == b {
            return;
        }

        // attach smaller rank tree under the higher-rank one
        if self.rank[a] < self.rank[b] {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b] = a;
        if self.rank[a] == self.rank[b] {
            self.rank[a] += 1;
        }
    }

    /// Check if `x` and `y` are in the same set.
    pub fn same(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }

    pub fn components(&mut self) -> Vec<HashSet<usize>> {
        let n = self.parent.len();
        let mut map: HashMap<usize, HashSet<usize>> = HashMap::new();
        for i in 0..n {
            let root = self.find(i);
            map.entry(root).or_default().insert(i);
        }
        map.into_values().collect()
    }
}
