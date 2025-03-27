import matplotlib.pyplot as plt
from utils import graph_file_to_list_of_lists, fbin_to_numpy
import numpy as np

def plot_edges(edges, points, ax, color='blue', alpha=0.5):
    for edge in edges:
        start, end = edge
        ax.plot([points[start][0], points[end][0]], [points[start][1], points[end][1]], color=color, alpha=alpha)

def plot_graph(graph_file, points_file, output_file):
    # Read the graph from the file
    graph = graph_file_to_list_of_lists(graph_file)
    
    # Read the points from the fbin file
    points = fbin_to_numpy(points_file)
    
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], color='red', s=15)
    
    # Plot the edges
    for i, neighbors in enumerate(graph):
        edges = [(i, neighbor) for neighbor in neighbors]
        plot_edges(edges, points, ax, color='k', alpha=0.25)
    
    
    # remove axes
    ax.set_axis_off()
    fig.tight_layout()
    
    # Save the plot to a file
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Plot a graph from a graph file and points file")
    parser.add_argument("--graph", type=str, help="Path to the graph file")
    parser.add_argument("--points", type=str, help="Path to the points file")
    parser.add_argument("--output", type=str, help="Path to the output image file")
    args = parser.parse_args()
    
    plot_graph(args.graph, args.points, args.output)