import os
from typing import Optional, Sequence, Tuple, Dict, Union, List

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
# Pass-1 search runner (no plotting)
# ------------------------------------------------------------------
def run_beam_search_collect(
    graph: PyVectorGraph,
    dataset: PyVectorDataset,
    query: Union[int, np.ndarray],
    entry_point: int,
    beam_width: int,
    filter_mask: Optional[np.ndarray],
    *,
    recall_k: Optional[int],
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Run transparent_beam_search ONCE and return:
        log_ranks  : np.ndarray (log(rank+1))
        recall     : float (Recall@k) or None
    This is a *data collection* helper used in pass 1.
    """
    # Resolve query vector
    if isinstance(query, (int, np.integer)):
        query_vec = dataset.get_vector(int(query))
    else:
        query_vec = np.asarray(query)

    # Brute force once (we'll need for ranks and recall)
    bf_result = dataset.brute_force(query_vec)
    bf_cache = compute_rank_cache(dataset, query_vec, bf_result)
    bf_order = [idx for idx, _ in bf_result]

    # Transparent search
    _frontier, _visited, steps = graph.transparent_beam_search(
        query_vec, dataset, entry_point, beam_width, filter_mask
    )

    ids = [s[1] for s in steps]
    max_rank = len(bf_cache)
    ranks = np.array([bf_cache.get(i, max_rank) for i in ids], dtype=np.int64)
    log_ranks = np.log(ranks + 1)  # natural log; change if you want base control

    recall = None
    if recall_k is not None:
        recall = _compute_recall_at_k(ids, bf_order, recall_k)

    return log_ranks, recall


# ------------------------------------------------------------------
# Pass-2 plotting helper (plots precomputed arrays)
# ------------------------------------------------------------------
def _plot_log_ranks_trace(
    ax: plt.Axes,
    log_ranks: np.ndarray,
    *,
    color,
    line_kwargs: Optional[dict] = None,
):
    if line_kwargs is None:
        line_kwargs = {}
    line_kwargs = {**line_kwargs, "color": color}
    return ax.plot(range(len(log_ranks)), log_ranks, **line_kwargs)[0]


# ------------------------------------------------------------------
# Example "main" — TWO PASS VERSION
# ------------------------------------------------------------------
def main():
    data_path = "../data/word2vec-google-news-300_50000_lowercase/base.fbin"
    graph_path = "../data/word2vec-google-news-300_50000_lowercase/outputs/vamana"

    vectors = fbin_to_numpy(data_path)
    dataset = PyVectorDataset(vectors)

    neighborhoods = graph_file_to_list_of_lists(graph_path)
    graph = PyVectorGraph(neighborhoods)

    rows, cols = 10, 10
    n = rows * cols

    query_ids = np.random.choice(dataset.size(), size=n, replace=False)
    entry_point = 0
    beam_widths = [10] * n
    recall_k = 10

    # --------------------------------------------------------------
    # PASS 1: run all searches, cache data, compute global ranges
    # --------------------------------------------------------------
    collected_log_ranks: List[np.ndarray] = []
    collected_recalls: List[Optional[float]] = []

    global_max_x = 0
    global_min_y = np.inf
    global_max_y = -np.inf

    for qid, bw in zip(query_ids, beam_widths):
        log_ranks, recall = run_beam_search_collect(
            graph,
            dataset,
            qid,
            entry_point,
            bw,
            None,  # filter_mask
            recall_k=recall_k,
        )
        collected_log_ranks.append(log_ranks)
        collected_recalls.append(recall)

        global_max_x = max(global_max_x, len(log_ranks) - 1)
        global_min_y = min(global_min_y, float(np.min(log_ranks)))
        global_max_y = max(global_max_y, float(np.max(log_ranks)))

    # Clamp y >= 0
    global_min_y = max(0.0, global_min_y)

    # Observed recall range
    recalls_arr = np.array([r for r in collected_recalls if r is not None], dtype=float)
    if recalls_arr.size == 0:
        rmin, rmax = 0.0, 1.0
    else:
        rmin = float(recalls_arr.min())
        rmax = float(recalls_arr.max())
        if np.isclose(rmin, rmax):
            # pad to avoid degenerate color range
            pad = max(1e-6, 0.01 * max(rmax, 1.0))
            rmin = max(0.0, rmin - pad)
            rmax = min(1.0, rmax + pad)

    recall_norm = mpl.colors.Normalize(vmin=rmin, vmax=rmax)
    cmap = plt.cm.viridis

    # --------------------------------------------------------------
    # PASS 2: build figure & plot colored traces
    # --------------------------------------------------------------
    fig, axes = plt.subplots(
        rows, cols, figsize=(8, 6), sharex=True, sharey=True, constrained_layout=False
    )
    axes = axes.ravel()

    lines = []
    for ax, log_ranks, r in zip(axes, collected_log_ranks, collected_recalls):
        color = cmap(recall_norm(r)) if r is not None else "gray"
        line = _plot_log_ranks_trace(ax, log_ranks, color=color)
        lines.append(line)

    # Uniform limits
    for ax in axes:
        ax.set_xlim(0, global_max_x)
        ax.set_ylim(global_min_y, global_max_y)

    # Figure-level axis labels (centered, not per subplot)
    fig.supxlabel("Step", fontsize="large")
    fig.supylabel("log Rank", fontsize="large")

    # Reserve space on right for colorbar
    fig.subplots_adjust(right=0.86)

    # Colorbar using observed recall range
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=recall_norm)
    sm.set_array([])  # needed for older Matplotlib
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"Recall@{recall_k}")

    # Optional overall title
    # fig.suptitle("Transparent Beam Search Rank Progress", fontsize="x-large", y=0.98)

    # Save
    output_dir = "../outputs/plots"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "transparent_beam_search_rank_progress_grid.png")
    fig.savefig(out_path)
    plt.show()

    return collected_recalls


if __name__ == "__main__":
    main()