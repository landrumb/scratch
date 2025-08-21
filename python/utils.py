"""misc utility functions"""

import numpy as np
from pathlib import Path
from contextlib import contextmanager
import time

DATA_DIR = Path("data")


def numpy_to_fbin(vectors, fbin_path):
    """writes a 2d numpy array to a .fbin file"""
    n, d = vectors.shape
    with open(fbin_path, "wb") as fbin_file:
        np.array([n, d], dtype=np.int32).tofile(fbin_file)
        vectors.astype(np.float32).tofile(fbin_file)


def fbin_to_numpy(fbin_path):
    """reads a 2d numpy array from a .fbin file"""
    with open(fbin_path, "rb") as fbin_file:
        n, d = np.fromfile(fbin_file, dtype=np.int32, count=2)
        return np.fromfile(fbin_file, dtype=np.float32).reshape(n, d)


def gt_file_to_numpy(gt_file):
    """reads a ground truth file and returns a 2d numpy array"""
    with open(gt_file, "rb") as f:
        num_points, num_neighbors = np.fromfile(f, dtype=np.int32, count=2)
        print(f"reading {num_points} points with {num_neighbors} neighbors each")

        neighbors = np.fromfile(
            f, dtype=np.int32, count=num_points * num_neighbors
        ).reshape(num_points, num_neighbors)
        distances = np.fromfile(
            f, dtype=np.float32, count=num_points * num_neighbors
        ).reshape(num_points, num_neighbors)
        return neighbors, distances


def graph_file_to_list_of_lists(graph_file):
    """reads a parlay graph file and returns a list of lists representing out neighborhoods"""
    with open(graph_file, "rb") as f:
        num_points, max_degree = np.fromfile(f, dtype=np.int32, count=2)
        print(f"reading {num_points} points with max degree {max_degree}")

        degrees = np.fromfile(f, dtype=np.int32, count=num_points)
        out_neighborhoods = []
        for degree in degrees:
            out_neighborhoods.append(np.fromfile(f, dtype=np.int32, count=degree))

        remaining = np.fromfile(f, dtype=np.int32)
        assert len(remaining) == 0, (
            f"file has {len(remaining)} remaining values after reading, expected 0"
        )

        return out_neighborhoods


def list_of_lists_to_graph_file(graph, graph_file):
    """writes a list of lists representing out neighborhoods to a parlay graph file"""
    with open(graph_file, "wb") as f:
        num_points = len(graph)
        max_degree = max(len(neighbors) for neighbors in graph)
        np.array([num_points, max_degree], dtype=np.int32).tofile(f)

        degrees = np.array([len(neighbors) for neighbors in graph], dtype=np.int32)
        degrees.tofile(f)

        for neighbors in graph:
            np.array(neighbors, dtype=np.int32).tofile(f)


def sort_neighbors_by_distance(graph, vectors):
    """sorts the neighbors of each point by the length of the edge inplace"""
    for idx, neighbors in enumerate(graph):
        lengths = -np.dot(vectors[neighbors], vectors[idx])
        graph[idx] = [neighbor for _, neighbor in sorted(zip(lengths, neighbors))]


def read_vocab(vocab_dir):
    """returns a word_to_idx dict and idx_to_word list from a vocab directory"""
    if type(vocab_dir) == str:
        vocab_dir = Path(vocab_dir)

    word_to_idx = {}
    idx_to_word = []

    with open(vocab_dir) as file:
        for idx, line in enumerate(file):
            word = line.strip()
            word_to_idx[word] = idx
            idx_to_word.append(word)

    return word_to_idx, idx_to_word


@contextmanager
def time_block(name="task"):
    start = time.perf_counter()
    print(name, end="", flush=True)
    yield
    end = time.perf_counter()
    print(f": {end - start:.6f}s")
