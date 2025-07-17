import matplotlib.pyplot as plt
import numpy as np
from scratch import PyVectorDataset, PyVectorGraph
from utils import fbin_to_numpy, graph_file_to_list_of_lists
import os


def main():
    data_path = "../data/word2vec-google-news-300_50000_lowercase/base.fbin"
    graph_path = "../data/word2vec-google-news-300_50000_lowercase/outputs/vamana"

    vectors = fbin_to_numpy(data_path)
    dataset = PyVectorDataset(vectors)

    neighborhoods = graph_file_to_list_of_lists(graph_path)
    graph = PyVectorGraph(neighborhoods)

    query = dataset.get_vector(10_000)

    _frontier, _visited, steps = graph.transparent_beam_search(
        query, dataset, 0, 10, None
    )

    # Precompute the rank of each dataset point based on distance to the query
    brute_force = dataset.brute_force(query)
    ordered_ids = [idx for idx, _ in brute_force]
    rank = {idx: i for i, idx in enumerate(ordered_ids)}

    log_ranks = [np.log(rank[step[1]] + 1) for step in steps]
    # print("Steps:", steps)
    # print("Log Ranks:", log_ranks)
    # print("Distances:", [dataset.compare(query, step[1]) for step in steps])

    plt.figure()
    plt.plot(range(len(log_ranks)), log_ranks)
    plt.xlabel("Step")
    plt.ylabel("Log Rank")
    plt.title("Transparent Beam Search Rank Progress")
    plt.tight_layout()
    
    # ensure the output directory exists
    output_dir = "../outputs/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "transparent_beam_search_rank_progress.png"))
    plt.show()


if __name__ == "__main__":
    main()
