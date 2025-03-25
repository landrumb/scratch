#!/usr/bin/env python3
"""
Visualize a graph constructed with the build_global_local_graph method.
"""

import numpy as np
import matplotlib.pyplot as plt
from scratch import PyVectorDataset, PySubset, PyVectorGraph
import random

# Set random seed for reproducibility
np.random.seed(42)

# Generate random 2D points
n_points = 1000
dim = 2
points = np.random.rand(n_points, dim).astype(np.float32)

# Create a VectorDataset from the points
print(f"Creating VectorDataset with {n_points} points of dimension {dim}...")
dataset = PyVectorDataset(points)

# Create a subset with a smaller number of points for visualization
subset_size = 100
subset_indices = random.sample(range(n_points), subset_size)
subset = PySubset(dataset, subset_indices)
print(f"Created subset with {subset.size} points")

# Build the graph using the build_global_local_graph method with alpha=1.01
print("Building graph...")
alpha = 1.01
graph = subset.build_global_local_graph(alpha)

print(f"Graph built with {graph.n} nodes")
print(f"Total edges: {graph.total_edges()}")
print(f"Average degree: {graph.total_edges() / graph.n:.2f}")
print(f"Max degree: {graph.max_degree()}")

# Get the positions of the subset points for visualization
subset_points = np.array([subset.get_vector(i) for i in range(subset.size)])

# Visualize the graph
plt.figure(figsize=(10, 8))

# Draw all points
plt.scatter(subset_points[:, 0], subset_points[:, 1], 
            c='lightgray', alpha=0.5, s=30, edgecolor='k', linewidth=0.5)

# Choose some center nodes to highlight
center_indices = random.sample(range(subset.size), 5)
center_colors = ['red', 'blue', 'green', 'purple', 'orange']

# For each center, draw its neighborhood
for idx, center_idx in enumerate(center_indices):
    center = subset_points[center_idx]
    color = center_colors[idx]
    
    # Get neighborhood
    neighborhood = graph.get_neighborhood(center_idx)
    
    # Draw edges from center to neighbors
    for neighbor_idx in neighborhood:
        neighbor = subset_points[neighbor_idx]
        plt.plot([center[0], neighbor[0]], [center[1], neighbor[1]], 
                 color=color, alpha=0.6, linewidth=1.0)
    
    # Draw neighbors
    neighbor_positions = subset_points[neighborhood]
    plt.scatter(neighbor_positions[:, 0], neighbor_positions[:, 1], 
                color=color, s=30, alpha=0.8, edgecolor='k', linewidth=0.5)
    
    # Highlight center node
    plt.scatter(center[0], center[1], color=color, s=100, edgecolor='k', 
                linewidth=1.5, label=f'Center {center_idx}')

plt.title(f'Graph Neighborhoods (alpha={alpha})')
plt.legend()
plt.tight_layout()
plt.savefig('graph_visualization.png')
print("Visualization saved to 'graph_visualization.png'")

# Create another visualization showing all graph edges
plt.figure(figsize=(10, 8))

# Draw all points
plt.scatter(subset_points[:, 0], subset_points[:, 1], 
            c='lightgray', alpha=0.5, s=30, edgecolor='k', linewidth=0.5)

# Draw all edges
print("Drawing all graph edges...")
edge_count = 0
for i in range(graph.n):
    neighbors = graph.get_neighborhood(i)
    for j in neighbors:
        plt.plot([subset_points[i][0], subset_points[j][0]], 
                 [subset_points[i][1], subset_points[j][1]], 
                 color='darkgray', alpha=0.2, linewidth=0.5)
        edge_count += 1
        
print(f"Drew {edge_count} edges")

plt.title(f'Complete Graph Visualization (alpha={alpha})')
plt.tight_layout()
plt.savefig('complete_graph_visualization.png')
print("Complete visualization saved to 'complete_graph_visualization.png'")

# Create a visualization showing points within alpha-distance of neighbors
# Pick one center to focus on
center_idx = center_indices[0]  # Use the first highlighted center
center = subset_points[center_idx]
neighbors = graph.get_neighborhood(center_idx)

plt.figure(figsize=(10, 8))

# Draw all points as background
plt.scatter(subset_points[:, 0], subset_points[:, 1], 
            c='lightgray', alpha=0.5, s=30, edgecolor='k', linewidth=0.5)

# Draw center point
plt.scatter(center[0], center[1], color='red', s=100, edgecolor='k',
            linewidth=1.5, label=f'Center {center_idx}')

# For each neighbor, draw points that are alpha times closer to the neighbor than to the center
for i, neighbor_idx in enumerate(neighbors):
    neighbor = subset_points[neighbor_idx]
    
    # Calculate distances from all points to center and to neighbor
    distances_to_center = np.sqrt(np.sum((subset_points - center)**2, axis=1))
    distances_to_neighbor = np.sqrt(np.sum((subset_points - neighbor)**2, axis=1))
    
    # Find points where dist_to_neighbor < dist_to_center / alpha
    alpha_close_mask = distances_to_neighbor < (distances_to_center / alpha)
    alpha_close_points = subset_points[alpha_close_mask]
    
    # Skip if no points are found to avoid empty plot errors
    if len(alpha_close_points) == 0:
        continue
    
    # Get a unique color for this neighbor
    color = plt.cm.tab10(i % 10)
    
    # Draw alpha-close points
    plt.scatter(alpha_close_points[:, 0], alpha_close_points[:, 1],
                color=color, alpha=0.4, s=20)
    
    # Draw edge from center to neighbor
    plt.plot([center[0], neighbor[0]], [center[1], neighbor[1]],
             color=color, linewidth=2.0)
    
    # Draw neighbor
    plt.scatter(neighbor[0], neighbor[1], color=color, s=80, edgecolor='black',
                linewidth=1.0, label=f'Neighbor {neighbor_idx}')
    
    # Draw accurate boundary of alpha region using contour plot
    # The boundary is where dist_to_neighbor = dist_to_center / alpha
    if i < 3:  # Only draw for first few neighbors to avoid clutter
        # Create a grid for contour plot
        x_min, x_max = np.min(subset_points[:, 0]) - 0.1, np.max(subset_points[:, 0]) + 0.1
        y_min, y_max = np.min(subset_points[:, 1]) - 0.1, np.max(subset_points[:, 1]) + 0.1
        grid_size = 100
        xs = np.linspace(x_min, x_max, grid_size)
        ys = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(xs, ys)
        
        # Calculate the contour function: dist_to_neighbor - dist_to_center/alpha
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T
        dist_to_center_grid = np.sqrt(np.sum((grid_points - center)**2, axis=1)).reshape(X.shape)
        dist_to_neighbor_grid = np.sqrt(np.sum((grid_points - neighbor)**2, axis=1)).reshape(X.shape)
        
        # The boundary is where this function equals zero
        Z = dist_to_neighbor_grid - dist_to_center_grid / alpha
        
        # Plot the contour at level 0
        contour = plt.contour(X, Y, Z, levels=[0], colors=[color], linestyles='dashed', 
                    linewidths=1.5)
        # Add a custom label in the legend
        plt.plot([], [], color=color, linestyle='dashed', linewidth=1.5, 
                label=f'Alpha boundary for {neighbor_idx}')

plt.title(f'Alpha Coverage Visualization (alpha={alpha})')
plt.legend()
plt.tight_layout()
plt.savefig('alpha_coverage_visualization.png')
print("Alpha coverage visualization saved to 'alpha_coverage_visualization.png'")