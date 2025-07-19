import os
from typing import Optional, Sequence, Tuple, Dict, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scratch import PyVectorDataset, PyVectorGraph
from utils import fbin_to_numpy, graph_file_to_list_of_lists


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def compute_rank_cache(
    dataset: PyVectorDataset,
    query_vec: np.ndarray,
    brute_force_result: Optional[Sequence[Tuple[int, float]]] = None,
) -> Dict[int, int]:
    """
    Return {point_id -> rank} for all dataset points w.r.t. query_vec.
    If you already have `dataset.brute_force(query_vec)` results, pass them in.
    """
    if brute_force_result is None:
        brute_force_result = dataset.brute_force(query_vec)
    ordered_ids = [idx for idx, _ in brute_force_result]
    return {idx: i for i, idx in enumerate(ordered_ids)}


def _compute_recall_at_k(
    visited_ids: Sequence[int],
    brute_force_order: Sequence[int],
    k: int,
) -> float:
    """Recall@k = |visited ∩ top_k| / k."""
    topk = set(brute_force_order[:k])
    hits = sum(1 for vid in set(visited_ids) if vid in topk)
    return hits / k if k > 0 else 0.0


# ------------------------------------------------------------------
# Core plotting helper
# ------------------------------------------------------------------
def plot_beam_search_rank_progress(
    graph: PyVectorGraph,
    dataset: PyVectorDataset,
    query: Union[int, np.ndarray],
    entry_point: int,
    beam_width: int,
    filter_mask: Optional[np.ndarray] = None,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,         # kept for backward compat
    add_axis_labels: bool = True,        # <-- NEW: suppress when doing grids
    brute_force_cache: Optional[Dict[int, int]] = None,
    brute_force_order: Optional[Sequence[int]] = None,
    log_base: float = np.e,
    line_kwargs: Optional[dict] = None,
    recall_k: Optional[int] = None,
    recall_cmap=plt.cm.viridis,
) -> Tuple[plt.Line2D, Sequence[Tuple[int, int]], np.ndarray, Optional[float]]:
    """
    Run transparent_beam_search and plot log(rank) progression.

    If `add_axis_labels=False`, the function will *not* set per-axes x/y labels,
    allowing you to manage figure-level labels in the caller.
    """
    # Resolve query vector
    if isinstance(query, (int, np.integer)):
        query_vec = dataset.get_vector(int(query))
    else:
        query_vec = np.asarray(query)

    # Run search
    _frontier, _visited, steps = graph.transparent_beam_search(
        query_vec, dataset, entry_point, beam_width, filter_mask
    )

    # If user didn't supply rank cache or order, compute them.
    if brute_force_cache is None or brute_force_order is None:
        brute_force_result = dataset.brute_force(query_vec)
        if brute_force_cache is None:
            brute_force_cache = compute_rank_cache(dataset, query_vec, brute_force_result)
        if brute_force_order is None:
            brute_force_order = [idx for idx, _ in brute_force_result]

    # Extract dataset ids from steps. Adjust if your tuple structure differs.
    ids = [s[1] for s in steps]

    # Compute log ranks
    max_rank = len(brute_force_cache)
    ranks = np.array([brute_force_cache.get(i, max_rank) for i in ids], dtype=np.int64)
    if log_base == np.e:
        log_ranks = np.log(ranks + 1)
        log_label = "log Rank"
    else:
        log_ranks = np.log(ranks + 1) / np.log(log_base)
        log_label = f"log$_{{{log_base:g}}}$ Rank"

    # Determine recall color (unless user specified their own color)
    recall = None
    if recall_k is not None:
        recall = _compute_recall_at_k(ids, brute_force_order, recall_k)
    color = None
    if line_kwargs is not None and "color" in line_kwargs:
        color = line_kwargs["color"]
    elif recall_k is not None:
        color = recall_cmap(recall)  # map 0..1 to colormap

    # Plot
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True

    if line_kwargs is None:
        line_kwargs = {}
    if color is not None:
        line_kwargs = {**line_kwargs, "color": color}

    line = ax.plot(range(len(log_ranks)), log_ranks, **line_kwargs)[0]

    if add_axis_labels:
        ax.set_xlabel("Step")
        ax.set_ylabel(log_label)

    if title is not None:
        ax.set_title(title)

    if created_fig:
        plt.tight_layout()

    return line, steps, log_ranks, recall


# ------------------------------------------------------------------
# Example "main" showing: NO subplot titles; UNIFORM limits; FIGURE-LEVEL labels;
# colorbar placed cleanly at right.
# ------------------------------------------------------------------
def main():
    data_path = "../data/word2vec-google-news-300_50000_lowercase/base.fbin"
    graph_path = "../data/word2vec-google-news-300_50000_lowercase/outputs/vamana"

    vectors = fbin_to_numpy(data_path)
    dataset = PyVectorDataset(vectors)

    neighborhoods = graph_file_to_list_of_lists(graph_path)
    graph = PyVectorGraph(neighborhoods)
    
    rows, cols = 3, 3
    n = rows * cols

    # Choose some queries to illustrate
    query_ids = np.random.choice(dataset.size(), size=n, replace=False)  # example ids
    entry_point = 0
    beam_widths = [10] * n    # example per-subplot params
    recall_k = 10                       # color scale

    # Use constrained_layout to help with spacing;
    # we'll still reserve extra room for colorbar.
    fig, axes = plt.subplots(
        rows, cols, figsize=(8, 6), sharex=True, sharey=True, constrained_layout=False
    )
    axes = axes.ravel()

    global_max_x = 0
    global_min_y = np.inf
    global_max_y = -np.inf

    recalls = []
    for ax, qid, bw in zip(axes, query_ids, beam_widths):
        qvec = dataset.get_vector(qid)
        bf_result = dataset.brute_force(qvec)
        bf_cache = compute_rank_cache(dataset, qvec, bf_result)
        bf_order = [idx for idx, _ in bf_result]

        _, _, log_ranks, recall = plot_beam_search_rank_progress(
            graph,
            dataset,
            qvec,          # could also pass qid
            entry_point,
            bw,
            None,
            ax=ax,
            title=None,            # no per-subplot titles
            add_axis_labels=False, # suppress per-axes labels
            brute_force_cache=bf_cache,
            brute_force_order=bf_order,
            recall_k=recall_k,
        )
        recalls.append(recall)

        # track limits
        global_max_x = max(global_max_x, len(log_ranks) - 1)
        global_min_y = min(global_min_y, float(np.min(log_ranks)))
        global_max_y = max(global_max_y, float(np.max(log_ranks)))

    # Clamp y>=0 (log(rank+1) is >=0) unless you want actual min
    global_min_y = max(0.0, global_min_y)

    # Apply uniform limits
    for ax in axes:
        ax.set_xlim(0, global_max_x)
        ax.set_ylim(global_min_y, global_max_y)

    # Figure-level axis labels (centered, not scaled to subplot size)
    fig.supxlabel("Step", fontsize="large")
    fig.supylabel("log Rank", fontsize="large")

    # Reserve space on right for colorbar *before* creating it.
    # Adjust as needed if labels spill; 0.85 leaves ~15% width for cbar.
    fig.subplots_adjust(right=0.86)

    # Colorbar keyed to recall
    sm = mpl.cm.ScalarMappable(cmap=plt.cm.viridis, norm=mpl.colors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height] in fig frac
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"Recall@{recall_k}")

    # Optional overall title
    fig.suptitle("Transparent Beam Search Rank Progress", fontsize="x-large", y=0.98)

    # Save
    output_dir = "../outputs/plots"
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "transparent_beam_search_rank_progress_grid.png"))
    plt.show()

    return recalls


if __name__ == "__main__":
    main()