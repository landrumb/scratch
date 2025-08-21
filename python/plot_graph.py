import matplotlib.pyplot as plt
from matplotlib import animation
from utils import graph_file_to_list_of_lists, fbin_to_numpy
import numpy as np


def plot_edges(edges, points, ax, color="blue", alpha=0.5):
    for edge in edges:
        start, end = edge
        ax.plot(
            [points[start][0], points[end][0]],
            [points[start][1], points[end][1]],
            color=color,
            alpha=alpha,
        )


def plot_graph(graph_file, points_file, output_file):
    # Read the graph from the file
    graph = graph_file_to_list_of_lists(graph_file)

    # Read the points from the fbin file
    points = fbin_to_numpy(points_file)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], color="red", s=15)

    # Plot the edges
    for i, neighbors in enumerate(graph):
        edges = [(i, neighbor) for neighbor in neighbors]
        plot_edges(edges, points, ax, color="k", alpha=0.25)

    # remove axes
    ax.set_axis_off()
    fig.tight_layout()

    # Save the plot to a file
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()


def plot_graph_bidirectional(graph_file, points_file, output_file):
    # Read the graph from the file
    graph = graph_file_to_list_of_lists(graph_file)

    # Read the points from the fbin file
    points = fbin_to_numpy(points_file)

    # Precompute neighbor sets for fast bidirectional checks
    neighbor_sets = [set(neighbors.tolist()) for neighbors in graph]

    # Collect bidirectional edges without duplicates (i < j)
    bidir_edges = []
    for i, neighbors in enumerate(graph):
        for j in neighbors:
            if i < j and i in neighbor_sets[j]:
                bidir_edges.append((i, j))

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], color="red", s=15)

    # Plot only bidirectional edges
    plot_edges(bidir_edges, points, ax, color="k", alpha=0.6)

    # remove axes
    ax.set_axis_off()
    fig.tight_layout()

    # Save the plot to a file
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()


def _choose_extreme_points(points):
    # Normalize coordinates to [0,1] for robust quadrant selection
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()
    norm_x = (points[:, 0] - min_x) / max(1e-9, (max_x - min_x))
    norm_y = (points[:, 1] - min_y) / max(1e-9, (max_y - min_y))

    # Upper-left: low x, high y -> maximize (norm_y - norm_x)
    start_idx = np.argmax(norm_y - norm_x)
    # Bottom-right: high x, low y -> maximize (norm_x - norm_y)
    target_idx = np.argmax(norm_x - norm_y)
    return int(start_idx), int(target_idx)


def _compute_greedy_path(graph, points, start_idx, target_idx, max_steps=10000):
    target = points[target_idx]

    def dist2(i):
        v = points[i] - target
        return float(v[0] * v[0] + v[1] * v[1])

    neighbor_sets = [set(neighbors.tolist()) for neighbors in graph]
    current = start_idx
    path_edges = []
    visited = set([current])
    current_d2 = dist2(current)

    for _ in range(max_steps):
        # If the target is directly connected, step there and finish
        if target_idx in neighbor_sets[current]:
            path_edges.append((current, target_idx))
            break

        neighbors = graph[current]
        if len(neighbors) == 0:
            break
        # Choose neighbor closest to target
        best = None
        best_d2 = current_d2
        for nb in neighbors:
            d2 = dist2(int(nb))
            if d2 < best_d2:
                best_d2 = d2
                best = int(nb)
        if best is None or best in visited:
            # No improvement or stuck in loop
            break
        path_edges.append((current, best))
        current = best
        visited.add(current)
        current_d2 = best_d2
        if current == target_idx:
            break

    return path_edges, start_idx, target_idx


def animate_greedy_search(graph_file, points_file, output_file):
    # Load data
    graph = graph_file_to_list_of_lists(graph_file)
    points = fbin_to_numpy(points_file)

    # Pick start (upper-left) and target (bottom-right)
    start_idx, target_idx = _choose_extreme_points(points)

    # Compute greedy path edges
    path_edges, start_idx, target_idx = _compute_greedy_path(
        graph, points, start_idx, target_idx
    )

    # Figure setup
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(points[:, 0], points[:, 1], color="lightcoral", s=12, zorder=1)

    # Markers for start and target
    ax.scatter(
        points[start_idx, 0],
        points[start_idx, 1],
        marker="*",
        s=200,
        color="green",
        edgecolor="black",
        zorder=3,
    )
    ax.scatter(
        points[target_idx, 0],
        points[target_idx, 1],
        marker="X",
        s=160,
        color="black",
        edgecolor="white",
        zorder=3,
    )

    # Pre-create Line2D objects for each path edge (hidden initially)
    path_lines = []
    for i, j in path_edges:
        (line,) = ax.plot(
            [points[i, 0], points[j, 0]],
            [points[i, 1], points[j, 1]],
            color="blue",
            alpha=0.9,
            linewidth=2.0,
            zorder=3,
            visible=False,
        )
        path_lines.append(line)

    # For each step, pre-create the out-edges of the newly encountered node (hidden initially)
    out_edges_lines_per_step = []  # list[list[Line2D]] aligned with path steps
    seen_out_edges = set()
    for _, v in path_edges:
        step_lines = []
        neighbors_v = graph[v]
        for nb in neighbors_v:
            e = (int(v), int(nb))
            if e in seen_out_edges:
                continue
            seen_out_edges.add(e)
            (ln,) = ax.plot(
                [points[v, 0], points[int(nb), 0]],
                [points[v, 1], points[int(nb), 1]],
                color="gray",
                alpha=0.7,
                linewidth=1.0,
                zorder=2,
                visible=False,
            )
            step_lines.append(ln)
        out_edges_lines_per_step.append(step_lines)

    ax.set_axis_off()
    fig.tight_layout()

    def init():
        for ln in path_lines:
            ln.set_visible(False)
        for step_lines in out_edges_lines_per_step:
            for ln in step_lines:
                ln.set_visible(False)
        return path_lines

    def update(frame):
        # Reveal path edges up to current frame
        upto = min(frame, len(path_lines))
        for idx in range(upto):
            path_lines[idx].set_visible(True)
        # Reveal out-edges for each completed step (show out-edges of the node we just reached)
        for step in range(upto):
            for ln in out_edges_lines_per_step[step]:
                ln.set_visible(True)
        return path_lines

    frames = max(1, len(path_lines) + 1)
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=200,
        blit=True,
        repeat=False,
    )

    # Try saving as GIF via PillowWriter, fall back to mp4 if unavailable
    try:
        from matplotlib.animation import PillowWriter

        anim.save(output_file, writer=PillowWriter(fps=5))
    except Exception:
        anim.save(output_file, writer="ffmpeg", fps=5)
    plt.close(fig)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Plot a graph from a graph file and points file"
    )
    parser.add_argument("--graph", type=str, help="Path to the graph file")
    parser.add_argument("--points", type=str, help="Path to the points file")
    parser.add_argument("--output", type=str, help="Path to the output image file")
    parser.add_argument(
        "--bidirectional-output",
        type=str,
        default=None,
        help="Path to the output image file for bidirectional edges only",
    )
    parser.add_argument(
        "--animate",
        type=str,
        default=None,
        help="Path to an animated output (gif/mp4) showing a greedy search from upper-left to bottom-right",
    )
    args = parser.parse_args()

    plot_graph(args.graph, args.points, args.output)
    if args.bidirectional_output is not None:
        plot_graph_bidirectional(args.graph, args.points, args.bidirectional_output)
    if args.animate is not None:
        animate_greedy_search(args.graph, args.points, args.animate)
