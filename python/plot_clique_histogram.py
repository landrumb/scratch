"""Plot a histogram of maximal clique sizes in a Vamana graph."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import scratch
from utils import graph_file_to_list_of_lists


def plot_clique_histogram(graph_file: Path, output_file: Path) -> None:
    """Plot a histogram of maximal clique sizes in ``graph_file``."""

    neighborhoods = graph_file_to_list_of_lists(graph_file)
    graph = scratch.PyVectorGraph(neighborhoods)
    cliques = graph.maximal_cliques()
    sizes = [len(c) for c in cliques]

    bins = np.arange(1, max(sizes) + 2) - 0.5
    plt.hist(sizes, bins=bins, edgecolor="black")
    plt.xlabel("Clique size")
    plt.ylabel("Frequency")
    plt.title("Histogram of maximal clique sizes")
    plt.xticks(range(1, max(sizes) + 1))
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a histogram of maximal clique sizes in a Vamana graph"
    )
    parser.add_argument("--graph", type=Path, required=True, help="Path to graph file")
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to output image file"
    )
    args = parser.parse_args()

    plot_clique_histogram(args.graph, args.output)

