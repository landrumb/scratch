import matplotlib.pyplot as plt
import numpy as np
from scratch import PyVectorDataset, PyVectorGraph
from utils import fbin_to_numpy, graph_file_to_list_of_lists


def main():
    data_path = "../data/word2vec-google-news-300_50000_lowercase/base.fbin"
    graph_path = "../data/word2vec-google-news-300_50000_lowercase/outputs/vamana"

    vectors = fbin_to_numpy(data_path)
    dataset = PyVectorDataset(vectors)

    neighborhoods = graph_file_to_list_of_lists(graph_path)
    graph = PyVectorGraph(neighborhoods)

    query = dataset.get_vector(0)

    _frontier, _visited, steps = graph.transparent_beam_search(
        query, dataset, 0, 10, None
    )

    # Precompute the rank of each dataset point based on distance to the query
    brute_force = dataset.brute_force(query)
    ordered_ids = [idx for idx, _ in brute_force]
    rank = {idx: i for i, idx in enumerate(ordered_ids)}

    log_ranks = [np.log(rank[step[1]] + 1) for step in steps]

    plt.figure()
    plt.plot(range(len(log_ranks)), log_ranks)
    plt.xlabel("Step")
    plt.ylabel("Log Rank")
    plt.title("Transparent Beam Search Rank Progress")
    plt.tight_layout()
    plt.savefig("beam_search_progress.png")


if __name__ == "__main__":
    main()
