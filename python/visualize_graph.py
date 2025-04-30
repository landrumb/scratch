#!/usr/bin/env python3
"""
Visualize a graph constructed with the build_global_local_graph method.
"""

import numpy as np
import matplotlib.pyplot as plt
from scratch import PyVectorDataset, PySubset, PyVectorGraph
import random
from utils import fbin_to_numpy

# Set random seed for reproducibility
np.random.seed(42)

# Generate random 2D points
points = fbin_to_numpy('../data/random2d/base.fbin')

n_points, dim = points.shape

# Create a VectorDataset from the points
print(f"Creating VectorDataset with {n_points} points of dimension {dim}...")
dataset = PyVectorDataset(points)

# Build the graph using the build_global_local_graph method with alpha=1.01
print("Building graph...")
alpha = 1.0

graph = dataset.build_global_local_graph(alpha)

print(f"Graph built with {graph.n} nodes")
print(f"Total edges: {graph.total_edges()}")
print(f"Average degree: {graph.total_edges() / graph.n:.2f}")
print(f"Max degree: {graph.max_degree()}")

# Get the positions of the subset points for visualization
subset_points = np.array([dataset.get_vector(i) for i in range(dataset.size())])

# Visualize the graph
plt.figure(figsize=(10, 8))

# Draw all points
plt.scatter(subset_points[:, 0], subset_points[:, 1], 
            c='lightgray', alpha=0.5, s=30, edgecolor='k', linewidth=0.5)

# Choose some center nodes to highlight
center_indices = [72] + random.sample(range(dataset.size()), 4)
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
# plt.legend()
plt.tight_layout()
plt.savefig('../outputs/plots/graph_visualization.png')
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
                 color='darkgray', alpha=0.7, linewidth=0.5)
        edge_count += 1
        
print(f"Drew {edge_count} edges")

plt.title(f'Complete Graph Visualization (alpha={alpha})')
plt.tight_layout()
plt.savefig('../outputs/plots/complete_graph_visualization.png')
print("Complete visualization saved to 'complete_graph_visualization.png'")

# Create a visualization showing points within alpha-distance of neighbors
# Pick one center to focus on
center_idx = center_indices[0]  # Use the first highlighted center
center = subset_points[center_idx]
neighbors = graph.get_neighborhood(center_idx)

plt.figure(figsize=(10, 10))

# Draw all points as background
# plt.scatter(subset_points[:, 0], subset_points[:, 1], 
            # c='lightgray', alpha=0.5, s=30, edgecolor='k', linewidth=0.5)
            
# draw all points as text giving their index
for i, point in enumerate(subset_points):
    plt.text(point[0], point[1], str(i), fontsize=8, ha='center', va='center')

# Draw center point
plt.scatter(center[0], center[1], color='red', s=250, edgecolor='k',
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
    # if len(alpha_close_points) == 0:
    #     continue
    
    # Get a unique color for this neighbor
    color = plt.cm.tab10(i % 10)
    
    # Draw alpha-close points
    # plt.scatter(alpha_close_points[:, 0], alpha_close_points[:, 1],
                # color=color, alpha=0.4, s=20)
    
    # Draw edge from center to neighbor
    plt.plot([center[0], neighbor[0]], [center[1], neighbor[1]],
             color=color, linewidth=2.0)
    
    # Draw neighbor
    plt.scatter(neighbor[0], neighbor[1], color=color, s=250, edgecolor='black',
                linewidth=1.0, label=f'Neighbor {i + 1} ({neighbor_idx})')
    
    # Draw accurate boundary of alpha region using contour plot
    # The boundary is where dist_to_neighbor = dist_to_center / alpha
    
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

# plt.title(f'Alpha Coverage Visualization (alpha={alpha})')
plt.legend()
plt.tight_layout()
plt.axis('equal')
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.gca().set_axis_off()

plt.savefig('../outputs/plots/alpha_coverage_visualization.svg')
plt.savefig('../outputs/plots/alpha_coverage_visualization.png', dpi=300)
print("Alpha coverage visualization saved to 'alpha_coverage_visualization.svg'")

# Add a visualization of greedy search for a random query point
print("Creating greedy search visualization...")

# Generate a random query point
query_point = np.random.rand(dim).astype(np.float32)

# Helper function to compute Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

# Find the exact nearest neighbor (brute force)
def find_exact_nearest_neighbor(dataset, query_point):
    min_distance = float('inf')
    nearest_idx = -1
    
    for i in range(dataset.size()):
        point = dataset.get_vector(i)
        distance = euclidean_distance(point, query_point)
        
        if distance < min_distance:
            min_distance = distance
            nearest_idx = i
    
    return nearest_idx, min_distance

# Function to perform greedy search
def greedy_search(graph, dataset, query_point, start_idx=None, max_steps=20):
    # If no start index is provided, choose a random one
    if start_idx is None:
        start_idx = random.randint(0, graph.n - 1)
    
    path = [start_idx]
    current_idx = start_idx
    
    for step in range(max_steps):
        # Get current neighborhood
        neighbors = graph.get_neighborhood(current_idx)
        
        # If no neighbors, stop
        if len(neighbors) == 0:
            print(f"Stopped at step {step}: No neighbors found")
            break
        
        # Find distances from each neighbor to the query
        best_distance = float('inf')
        best_neighbor = None
        
        for neighbor_idx in neighbors:
            neighbor_point = dataset.get_vector(neighbor_idx)
            distance = euclidean_distance(neighbor_point, query_point)
            
            if distance < best_distance:
                best_distance = distance
                best_neighbor = neighbor_idx
        
        # If we're not making progress (best neighbor is already in path), stop
        if best_neighbor in path:
            print(f"Stopped at step {step}: Cycle detected")
            break
        
        # Move to the best neighbor
        current_idx = best_neighbor
        path.append(current_idx)
        
        # Check if we've reached a point sufficiently close to the query
        current_distance = euclidean_distance(dataset.get_vector(current_idx), query_point)
        if current_distance < 0.01:  # Threshold for "close enough"
            print(f"Stopped at step {step}: Reached close proximity to query")
            break
    
    # Calculate final distance to query
    final_point = dataset.get_vector(path[-1])
    final_distance = euclidean_distance(final_point, query_point)
    
    print(f"Greedy search completed with {len(path)} steps")
    print(f"Final distance to query: {final_distance:.4f}")
    
    return path, final_distance

# Function to perform greedy search and record each step for animation
def greedy_search_with_steps(graph, dataset, query_point, start_idx=None, max_steps=20):
    # If no start index is provided, choose a random one
    if start_idx is None:
        start_idx = random.randint(0, graph.n - 1)
    
    path = [start_idx]
    current_idx = start_idx
    steps = []  # List to store step information
    
    # Store initial state
    steps.append({
        'current_idx': current_idx,
        'path': path.copy(),
        'explored_neighbors': [],
        'best_neighbor': None,
        'distances': {}
    })
    
    for step in range(max_steps):
        # Get current neighborhood
        neighbors = graph.get_neighborhood(current_idx)
        
        # If no neighbors, stop
        if len(neighbors) == 0:
            print(f"Stopped at step {step}: No neighbors found")
            break
        
        # Find distances from each neighbor to the query
        best_distance = float('inf')
        best_neighbor = None
        distances = {}
        
        for neighbor_idx in neighbors:
            # Skip neighbors that are already in the path
            if neighbor_idx in path:
                continue
                
            neighbor_point = dataset.get_vector(neighbor_idx)
            distance = euclidean_distance(neighbor_point, query_point)
            distances[neighbor_idx] = distance
            
            if distance < best_distance:
                best_distance = distance
                best_neighbor = neighbor_idx
        
        # No unvisited neighbors left
        if best_neighbor is None:
            print(f"Stopped at step {step}: No unvisited neighbors")
            break
        
        # Store this step's information
        steps.append({
            'current_idx': current_idx,
            'path': path.copy(),
            'explored_neighbors': list(neighbors),
            'best_neighbor': best_neighbor,
            'distances': distances
        })
        
        # Move to the best neighbor
        current_idx = best_neighbor
        path.append(current_idx)
        
        # Check if we've reached a point sufficiently close to the query
        current_distance = euclidean_distance(dataset.get_vector(current_idx), query_point)
        if current_distance < 0.01:  # Threshold for "close enough"
            print(f"Stopped at step {step}: Reached close proximity to query")
            break
    
    # Calculate final distance to query
    final_point = dataset.get_vector(path[-1])
    final_distance = euclidean_distance(final_point, query_point)
    
    print(f"Greedy search completed with {len(path)} steps")
    print(f"Final distance to query: {final_distance:.4f}")
    
    return path, final_distance, steps

# Find the exact nearest neighbor
exact_nn_idx, exact_nn_distance = find_exact_nearest_neighbor(dataset, query_point)
exact_nn_point = dataset.get_vector(exact_nn_idx)
print(f"Exact nearest neighbor: index {exact_nn_idx}, distance {exact_nn_distance:.4f}")

# Run greedy search
search_path, greedy_distance = greedy_search(graph, dataset, query_point)

# Calculate the ratio of greedy search distance to exact NN distance
distance_ratio = greedy_distance / exact_nn_distance if exact_nn_distance > 0 else float('inf')
print(f"Greedy search distance ratio: {distance_ratio:.2f}x the optimal")

# Create the visualization
plt.figure(figsize=(10, 8))

# Draw all points
plt.scatter(subset_points[:, 0], subset_points[:, 1], 
            c='lightgray', alpha=0.5, s=30, edgecolor='k', linewidth=0.5)

# Draw the query point
plt.scatter(query_point[0], query_point[1], color='red', s=150, marker='*',
            edgecolor='k', linewidth=1.5, label='Query Point')

# Draw the exact nearest neighbor
plt.scatter(exact_nn_point[0], exact_nn_point[1], color='magenta', s=120, marker='X',
            edgecolor='k', linewidth=1.5, label='Exact NN')

# Draw a line from query to exact NN
plt.plot([query_point[0], exact_nn_point[0]], [query_point[1], exact_nn_point[1]], 
         'magenta', linestyle='-', linewidth=1.5)

# Draw the search path with arrows to show direction
path_points = np.array([dataset.get_vector(i) for i in search_path])

# Draw arrows between consecutive points in the path
for i in range(len(path_points) - 1):
    plt.arrow(path_points[i, 0], path_points[i, 1],
              path_points[i+1, 0] - path_points[i, 0],
              path_points[i+1, 1] - path_points[i, 1],
              head_width=0.02, head_length=0.03, fc='blue', ec='blue',
              length_includes_head=True, alpha=0.7)

# Draw the points in the search path
plt.scatter(path_points[:, 0], path_points[:, 1], color='blue', s=80, 
            edgecolor='k', linewidth=1.0)

# Emphasize the start and end points
plt.scatter(path_points[0, 0], path_points[0, 1], color='green', s=100, 
            edgecolor='k', linewidth=1.5, label='Start Point')
plt.scatter(path_points[-1, 0], path_points[-1, 1], color='purple', s=100, 
            edgecolor='k', linewidth=1.5, label='End Point')

# Add numbers to indicate the order of visited points
for i, (x, y) in enumerate(path_points):
    plt.text(x+0.01, y+0.01, str(i), fontsize=10, bbox=dict(facecolor='white', alpha=0.7))


# Add text annotation with performance statistics
plt.text(0.02, 0.02, 
         f"Greedy search: {len(search_path)} steps, distance {greedy_distance:.4f}\n"
         f"Exact NN distance: {exact_nn_distance:.4f}\n"
         f"Ratio: {distance_ratio:.2f}x optimal",
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8),
         fontsize=9)

plt.title('Greedy Search Visualization')
plt.legend()
plt.tight_layout()
plt.savefig('../outputs/plots/greedy_search_visualization.png')
print("Greedy search visualization saved to 'greedy_search_visualization.png'")

# Add an animated visualization of greedy search
print("Creating greedy search animation...")

# Import required modules
import matplotlib.animation as animation
import os

# Make sure output directory exists
os.makedirs('../outputs/plots', exist_ok=True)

# Create a specific query point in the bottom left
query_point_animated = np.array([0.2, 0.2], dtype=np.float32)

# Find a start point in the top right
# First get points in the top right quadrant
top_right_mask = (subset_points[:, 0] > 0.75) & (subset_points[:, 1] > 0.75)
top_right_indices = np.where(top_right_mask)[0]
if len(top_right_indices) > 0:
    start_idx = top_right_indices[0]
else:
    # Fallback to the point closest to top right corner
    distances_to_top_right = np.sqrt(
        (subset_points[:, 0] - 1.0)**2 + (subset_points[:, 1] - 1.0)**2
    )
    start_idx = np.argmin(distances_to_top_right)

print(f"Starting greedy search at index {start_idx} (top right)")
print(f"Query point (bottom left): [{query_point_animated[0]:.2f}, {query_point_animated[1]:.2f}]")

# Find the exact nearest neighbor for the animated search
exact_nn_idx_animated, exact_nn_distance_animated = find_exact_nearest_neighbor(dataset, query_point_animated)
exact_nn_point_animated = dataset.get_vector(exact_nn_idx_animated)
print(f"Exact nearest neighbor: index {exact_nn_idx_animated}, distance {exact_nn_distance_animated:.4f}")

# Run greedy search with steps
search_path_animated, greedy_distance_animated, steps = greedy_search_with_steps(
    graph, dataset, query_point_animated, start_idx=start_idx
)

# Calculate the ratio of greedy search distance to exact NN distance
distance_ratio_animated = greedy_distance_animated / exact_nn_distance_animated if exact_nn_distance_animated > 0 else float('inf')
print(f"Greedy search distance ratio: {distance_ratio_animated:.2f}x the optimal")

# Create a figure for the animation
fig, ax = plt.subplots(figsize=(12, 10))

# We'll update this function for each frame of the animation
def update_frame(frame_num):
    ax.clear()
    
    # Draw all points
    ax.scatter(subset_points[:, 0], subset_points[:, 1], 
               c='lightgray', alpha=0.5, s=30, edgecolor='k', linewidth=0.5,
               zorder=1)
    
    # Draw the query point (target)
    ax.scatter(query_point_animated[0], query_point_animated[1], color='red', s=200, marker='*',
               edgecolor='k', linewidth=2.0, label='Query Point (Target)', zorder=10)
    
    # Draw the exact nearest neighbor (optimal solution)
    ax.scatter(exact_nn_point_animated[0], exact_nn_point_animated[1], color='magenta', s=140, marker='X',
               edgecolor='k', linewidth=1.5, label='Exact NN', zorder=9)
    
    # Draw a dashed line from query to exact NN
    ax.plot([query_point_animated[0], exact_nn_point_animated[0]], 
            [query_point_animated[1], exact_nn_point_animated[1]], 
            'magenta', linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)
    
    # Get the current step data
    if frame_num >= len(steps):
        step_data = steps[-1]  # Use the last step if we've gone past the end
        final_frame = True
    else:
        step_data = steps[frame_num]
        final_frame = False
    
    # Get the path up to this step
    current_path = step_data['path']
    path_points = np.array([dataset.get_vector(i) for i in current_path])
    
    # Draw the current graph position
    current_idx = step_data['current_idx']
    current_pos = dataset.get_vector(current_idx)
    
    # Draw the full path with arrows to show direction
    for i in range(len(path_points) - 1):
        ax.arrow(path_points[i, 0], path_points[i, 1],
                 path_points[i+1, 0] - path_points[i, 0],
                 path_points[i+1, 1] - path_points[i, 1],
                 head_width=0.02, head_length=0.03, fc='blue', ec='blue',
                 length_includes_head=True, alpha=0.7, zorder=3)
    
    # Draw the points in the path
    ax.scatter(path_points[:, 0], path_points[:, 1], color='blue', s=80, 
               edgecolor='k', linewidth=1.0, zorder=4)
    
    # Draw the starting point
    ax.scatter(path_points[0, 0], path_points[0, 1], color='green', s=120, 
               edgecolor='k', linewidth=1.5, label='Start Point', zorder=8)
    
    # If we're at the final step, emphasize the end point
    if final_frame:
        ax.scatter(path_points[-1, 0], path_points[-1, 1], color='purple', s=120, 
                   edgecolor='k', linewidth=1.5, label='End Point', zorder=8)
    
    # Highlight the current point with a yellow glow
    ax.scatter(current_pos[0], current_pos[1], color='yellow', s=140, 
               edgecolor='orange', linewidth=2.0, label='Current Position', zorder=7)
    
    # Add numbers to indicate the order of visited points
    for i, (x, y) in enumerate(path_points):
        ax.text(x+0.01, y+0.01, str(i), fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7), zorder=11)
    
    # If we're not at the final frame, show the exploration process
    if not final_frame and frame_num > 0:
        # Get the next step data for exploration visualization
        next_step = steps[frame_num]
        
        # Draw explored neighbors and their distances
        for neighbor_idx in next_step['explored_neighbors']:
            if neighbor_idx in current_path:
                continue  # Skip already visited nodes
                
            neighbor_pos = dataset.get_vector(neighbor_idx)
            
            # Get distance if available, otherwise show as '?'
            distance_text = f"{next_step['distances'].get(neighbor_idx, '?'):.4f}"
            
            # Draw line from current position to neighbor
            ax.plot([current_pos[0], neighbor_pos[0]], [current_pos[1], neighbor_pos[1]], 
                    'gray', linestyle=':', linewidth=1.0, alpha=0.6, zorder=5)
            
            # Draw the neighbor with size proportional to how good it is (smaller distance = larger point)
            if neighbor_idx in next_step['distances']:
                # Normalize distance for visualization (closer = bigger)
                max_dist = max(next_step['distances'].values()) if next_step['distances'] else 1.0
                min_dist = min(next_step['distances'].values()) if next_step['distances'] else 0.0
                dist_range = max_dist - min_dist if max_dist != min_dist else 1.0
                normalized_dist = (max_dist - next_step['distances'][neighbor_idx]) / dist_range
                size = 40 + normalized_dist * 60  # Size between 40 and 100 based on distance
                
                # Color: red for bad (far), green for good (close)
                color_val = 1 - normalized_dist  # 0 = good (green), 1 = bad (red)
                color = (color_val, normalized_dist, 0)  # (R,G,B) from (1,0,0) to (0,1,0)
                
                ax.scatter(neighbor_pos[0], neighbor_pos[1], color=color, s=size, 
                           edgecolor='black', linewidth=1.0, alpha=0.7, zorder=6)
                
                # Show distance as text
                ax.text(neighbor_pos[0], neighbor_pos[1] - 0.03, 
                        f"{next_step['distances'][neighbor_idx]:.4f}", 
                        fontsize=8, ha='center', va='top', 
                        bbox=dict(facecolor='white', alpha=0.7, pad=1), zorder=11)
        
        # Highlight best neighbor if there is one
        if next_step['best_neighbor'] is not None:
            best_neighbor_pos = dataset.get_vector(next_step['best_neighbor'])
            ax.scatter(best_neighbor_pos[0], best_neighbor_pos[1], color='lightgreen', s=120, 
                       edgecolor='darkgreen', linewidth=2.0, label='Best Neighbor', zorder=7)
    
    # Add step counter and distance information
    current_distance = euclidean_distance(current_pos, query_point_animated)
    step_info = f"Step: {frame_num}/{len(steps)-1}\n"
    step_info += f"Current distance to query: {current_distance:.4f}\n"
    if exact_nn_distance_animated > 0:
        step_info += f"Distance ratio: {current_distance/exact_nn_distance_animated:.2f}x optimal"
    
    ax.text(0.02, 0.02, step_info,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            fontsize=10, zorder=12)
    
    # Set title with frame number
    if final_frame:
        ax.set_title(f'Greedy Search Animation - Final State (Step {len(steps)-1})')
    else:
        ax.set_title(f'Greedy Search Animation - Step {frame_num}')
    
    ax.legend(loc='upper right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    return []

# Create the animation
num_frames = len(steps) + 5  # Add a few extra frames to let viewer see the final state
ani = animation.FuncAnimation(fig, update_frame, frames=num_frames, interval=800, blit=True, repeat=True)

# Save the animation
animation_path = '../outputs/plots/greedy_search_animation.gif'
ani.save(animation_path, writer='pillow', fps=1)
print(f"Greedy search animation saved to '{animation_path}'")

# Display final frame as a static image
plt.figure(figsize=(12, 10))
update_frame(len(steps) - 1)
plt.tight_layout()
plt.savefig('../outputs/plots/greedy_search_final_step.png')
print("Final step visualization saved to 'greedy_search_final_step.png'")