"""Plot a histogram of maximal clique sizes in a Vamana graph."""

import argparse
from pathlib import Path
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import scratch
from utils import graph_file_to_list_of_lists


def plot_clique_histogram(graph_file: Path, output_dir: Path) -> None:
    """Plot a histogram of maximal clique sizes in ``graph_file``."""

    neighborhoods = graph_file_to_list_of_lists(graph_file)
    graph = scratch.PyVectorGraph(neighborhoods)
    cliques = graph.maximal_bidirectional_cliques()
    cliques = [c for c in cliques if len(c) > 1]
    sizes = [len(c) for c in cliques]
    print(f"Maximal cliques: {len(cliques):,}")
    print(f"\tLargest clique: {max(sizes):,}")
    print(f"\tSmallest clique: {min(sizes):,}")
    print(f"\tAverage clique size: {sum(sizes) / len(sizes):.2f}")
    maximal_clique_coverage = set()
    for c in tqdm(cliques, desc="computing maximal clique coverage"):
        maximal_clique_coverage |= set(c)
    maximal_clique_coverage = len(maximal_clique_coverage)
    print(f"\tpoints covered: {maximal_clique_coverage:,} ({maximal_clique_coverage / graph.n * 100:.2f}%)")
    
    for c in tqdm(cliques, desc="validating that the maximal cliques actually exist"):
        for i in range(len(c)):
            point = c[i]
            others = c[:i] + c[i+1:]
            neighbors = graph.get_neighborhood(point)
            for other in others:
                assert other in neighbors, f"edge {point}-{other} is not in clique {c} (point {point} has neighbors {neighbors}, {other} has neighbors {graph.get_neighborhood(other)})"
    print("done")

    bins = np.arange(1, max(sizes) + 2) - 0.5
    plt.hist(sizes, bins=bins, edgecolor="black")
    plt.xlabel("Clique size")
    plt.ylabel("Frequency")
    plt.title("Histogram of maximal clique sizes")
    plt.xticks(range(1, max(sizes) + 1))
    plt.savefig(output_dir / "maximal_clique_histogram.png", bbox_inches="tight", dpi=300)
    plt.close()
    
    independent_cliques = graph.maximal_independent_bidirectional_cliques()
    independent_cliques = [c for c in independent_cliques if len(c) > 1]
    sizes = [len(c) for c in independent_cliques]
    print(f"Maximal independent cliques: {len(independent_cliques):,}")
    print(f"\tLargest clique: {max(sizes):,}")
    print(f"\tSmallest clique: {min(sizes):,}")
    print(f"\tAverage clique size: {sum(sizes) / len(sizes):.2f}")
    independent_clique_coverage = set()
    for c in tqdm(independent_cliques, desc="computing independent clique coverage"):
        independent_clique_coverage |= set(c)
    independent_clique_coverage = len(independent_clique_coverage)
    print(f"\tpoints covered: {independent_clique_coverage:,} ({independent_clique_coverage / graph.n * 100:.2f}%)")
    
    for c in tqdm(independent_cliques, desc="validating that the maximal independent cliques actually exist"):
        for i in range(len(c)):
            point = c[i]
            others = c[:i] + c[i+1:]
            neighbors = graph.get_neighborhood(point)
            for other in others:
                assert other in neighbors, f"edge {point}-{other} is not in clique {c} (point {point} has neighbors {neighbors}, {other} has neighbors {graph.get_neighborhood(other)})"
    print("done")
    
    bins = np.arange(1, max(sizes) + 2) - 0.5
    plt.hist(sizes, bins=bins, edgecolor="black")
    plt.xlabel("Clique size")
    plt.ylabel("Frequency")
    plt.title("Histogram of maximal independent clique sizes")
    plt.xticks(range(1, max(sizes) + 1))
    plt.savefig(output_dir / "maximal_independent_clique_histogram.png", bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"points covered by independent cliques: {sum(sizes):,} ({sum(sizes) / graph.n * 100:.2f}%)")
    print(f"points not covered by independent cliques: {graph.n - sum(sizes):,} ({100 - sum(sizes) / graph.n * 100:.2f}%)")
    print(f"total nodes in a graph where independent cliques are contracted: {graph.n - sum(sizes) + len(independent_cliques):,} ({100 - (graph.n - sum(sizes) + len(independent_cliques)) / graph.n * 100:.2f}% reduction)")
    
    # histogram of size of independent clique each point is in
    clique_sizes = [1] * (graph.n - independent_clique_coverage)
    for s in tqdm(sizes, desc="computing independent clique size histogram"):
        clique_sizes.extend([s] * s)
    plt.hist(clique_sizes, bins=np.arange(1, max(sizes) + 2) - 0.5, edgecolor="black")
    plt.xlabel("Clique size")
    plt.ylabel("Frequency")
    plt.title("Histogram of size of independent clique each point is in")
    plt.xticks(range(1, max(sizes) + 1))
    plt.savefig(output_dir / "independent_clique_size_weighted_histogram.png", bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a histogram of maximal clique sizes in a Vamana graph"
    )
    parser.add_argument("--graph", type=Path, required=True, help="Path to graph file")
    parser.add_argument(
        "--output-dir", type=Path, help="Path to output image directory", default="../outputs/plots"
    )
    args = parser.parse_args()

    plot_clique_histogram(args.graph, args.output_dir)

