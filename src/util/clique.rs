use crate::graph::{IndexT, VectorGraph};
use rayon::prelude::*;
use std::cmp::Ordering;

/// Computes all maximal cliques in the given `VectorGraph`.
///
/// The algorithm uses a parallelized variant of the Bron–Kerbosch
/// algorithm with pivoting. Each vertex is treated as the smallest
/// element of a potential clique and explored in parallel, ensuring
/// cliques are enumerated exactly once.
pub fn maximal_cliques(graph: &VectorGraph) -> Vec<Vec<IndexT>> {
    let n = graph.n();

    // sort neighborhoods for efficient set operations
    let neighbors: Vec<Vec<IndexT>> = (0..n)
        .map(|i| {
            let mut neigh = graph.get_neighborhood(i as IndexT).to_vec();
            neigh.sort_unstable();
            neigh
        })
        .collect();

    (0..n)
        .into_par_iter()
        .map(|i| {
            let v = i as IndexT;
            let mut p: Vec<IndexT> = neighbors[i].iter().cloned().filter(|&u| u > v).collect();
            p.sort_unstable();
            let mut x: Vec<IndexT> = neighbors[i].iter().cloned().filter(|&u| u < v).collect();
            x.sort_unstable();
            let r = vec![v];
            let mut cliques = Vec::new();
            bron_kerbosch_pivot(&neighbors, r, p, x, &mut cliques);
            cliques
        })
        .reduce(
            Vec::new,
            |mut acc, mut cliques| {
                acc.append(&mut cliques);
                acc
            },
        )
}

fn bron_kerbosch_pivot(
    neighbors: &[Vec<IndexT>],
    r: Vec<IndexT>,
    mut p: Vec<IndexT>,
    mut x: Vec<IndexT>,
    cliques: &mut Vec<Vec<IndexT>>,
) {
    if p.is_empty() && x.is_empty() {
        cliques.push(r);
        return;
    }

    let u = choose_pivot(&p, &x, neighbors);
    let candidates: Vec<IndexT> = p
        .iter()
        .cloned()
        .filter(|&v| !contains(&neighbors[u as usize], v))
        .collect();

    for v in candidates {
        let neigh = &neighbors[v as usize];
        let mut r_new = r.clone();
        r_new.push(v);

        let p_new = intersect(&p, neigh);
        let x_new = intersect(&x, neigh);

        bron_kerbosch_pivot(neighbors, r_new, p_new, x_new, cliques);

        p.retain(|&n| n != v);
        insert_sorted(&mut x, v);
    }
}

fn intersect(a: &[IndexT], b: &[IndexT]) -> Vec<IndexT> {
    let mut result = Vec::new();
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    result
}

fn contains(slice: &[IndexT], value: IndexT) -> bool {
    slice.binary_search(&value).is_ok()
}

fn choose_pivot(p: &[IndexT], x: &[IndexT], neighbors: &[Vec<IndexT>]) -> IndexT {
    *p.iter()
        .chain(x.iter())
        .max_by_key(|&&v| neighbors[v as usize].len())
        .expect("P and X cannot both be empty")
}

fn insert_sorted(vec: &mut Vec<IndexT>, value: IndexT) {
    match vec.binary_search(&value) {
        Ok(pos) | Err(pos) => vec.insert(pos, value),
    }
}

/// Returns a set of pairwise disjoint cliques chosen greedily by size.
///
/// The input is expected to be the result of `maximal_cliques`.
/// Cliques are considered in descending order of their cardinality; a clique is
/// selected if none of its vertices has appeared in a previously selected
/// clique. The selected cliques are returned in the order they were accepted.
pub fn greedy_independent_cliques(cliques: &[Vec<IndexT>]) -> Vec<Vec<IndexT>> {
    use std::collections::HashSet;

    // Sort clique indices by descending size
    let mut indices: Vec<usize> = (0..cliques.len()).collect();
    indices.sort_by_key(|&i| std::cmp::Reverse(cliques[i].len()));

    let mut used: HashSet<IndexT> = HashSet::new();
    let mut result = Vec::new();

    for i in indices {
        if cliques[i].iter().all(|v| !used.contains(v)) {
            for &v in &cliques[i] {
                used.insert(v);
            }
            result.push(cliques[i].clone());
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::MutableGraph;

    #[test]
    fn finds_maximal_cliques() {
        let mut g = VectorGraph::empty(4);
        // triangle between 0,1,2
        g.add_neighbor(0, 1);
        g.add_neighbor(1, 0);
        g.add_neighbor(0, 2);
        g.add_neighbor(2, 0);
        g.add_neighbor(1, 2);
        g.add_neighbor(2, 1);
        // edge 2-3
        g.add_neighbor(2, 3);
        g.add_neighbor(3, 2);

        let mut cliques = maximal_cliques(&g);
        for c in cliques.iter_mut() {
            c.sort_unstable();
        }
        cliques.sort();
        assert_eq!(cliques, vec![vec![0, 1, 2], vec![2, 3]]);
    }

    #[test]
    fn handles_empty_graph() {
        let g = VectorGraph::empty(3);
        let mut cliques = maximal_cliques(&g);
        for c in cliques.iter_mut() {
            c.sort_unstable();
        }
        cliques.sort();
        assert_eq!(cliques, vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn finds_clique_in_complete_graph() {
        let mut g = VectorGraph::empty(4);
        for i in 0..4 {
            for j in (i + 1)..4 {
                g.add_neighbor(i, j);
                g.add_neighbor(j, i);
            }
        }
        let mut cliques = maximal_cliques(&g);
        for c in cliques.iter_mut() {
            c.sort_unstable();
        }
        cliques.sort();
        assert_eq!(cliques, vec![vec![0, 1, 2, 3]]);
    }
}
