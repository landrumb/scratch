#!/usr/bin/env python3
"""
Test the VectorDataset and VectorGraph functionality using random 2D points.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from scratch import PyVectorDataset, PyVectorGraph

# Set random seed for reproducibility
np.random.seed(42)

# Generate 500 random 2D points
n_points = 500
dim = 2
points = np.random.rand(n_points, dim).astype(np.float32)  # Make sure to use float32

# Create a VectorDataset from the points
print(f"Creating VectorDataset with {n_points} points of dimension {dim}...")
dataset = PyVectorDataset(points)

# Verify dataset properties
print(f"Dataset size: {dataset.n}")
print(f"Dataset dimension: {dataset.dim}")

# Test get_vector
print("\nTesting get_vector...")
point_10 = dataset.get_vector(10)
print(f"Point at index 10: {point_10}")
print(f"Original point at index 10: {points[10]}")
assert np.allclose(point_10, points[10]), "get_vector doesn't match original data"

# Test compare_internal
print("\nTesting compare_internal...")
random_indices = np.random.randint(0, n_points, size=5)
for i, j in zip(random_indices[:-1], random_indices[1:]):
    rust_distance = dataset.compare_internal(i, j)
    numpy_distance = np.sqrt(np.sum((points[i] - points[j]) ** 2))
    print(f"Distance between points {i} and {j}:")
    print(f"  Rust: {rust_distance}")
    print(f"  NumPy: {numpy_distance}")
    assert np.isclose(rust_distance, numpy_distance), "compare_internal gives incorrect results"

# Test compare
print("\nTesting compare...")
query = np.random.rand(dim).astype(np.float32)  # Make sure to use float32
for i in random_indices:
    rust_distance = dataset.compare(query, i)
    numpy_distance = np.sqrt(np.sum((query - points[i]) ** 2))
    print(f"Distance between query and point {i}:")
    print(f"  Rust: {rust_distance}")
    print(f"  NumPy: {numpy_distance}")
    assert np.isclose(rust_distance, numpy_distance), "compare gives incorrect results"

# Test size
print("\nTesting size...")
assert dataset.size() == n_points, "size doesn't match expected value"
print(f"Size returns: {dataset.size()}")

# Test brute_force
print("\nTesting brute_force...")
results = dataset.brute_force(query)
# Check first 5 results
print(f"Top 5 closest points to query: {results[:5]}")
# Verify ordering
for i in range(len(results) - 1):
    assert results[i][1] <= results[i+1][1], "brute_force results not properly sorted"

# Test brute_force_internal
print("\nTesting brute_force_internal...")
query_idx = 0
results = dataset.brute_force_internal(query_idx)
# Check first 5 results
print(f"Top 5 closest points to point {query_idx}: {results[:5]}")
# Verify ordering
for i in range(len(results) - 1):
    assert results[i][1] <= results[i+1][1], "brute_force_internal results not properly sorted"

# Test brute_force_subset
print("\nTesting brute_force_subset...")
subset = list(np.random.randint(0, n_points, size=50))
results = dataset.brute_force_subset(query, subset)
# Check first 5 results
print(f"Top 5 closest points to query (from subset): {results[:5]}")
# Verify ordering
for i in range(len(results) - 1):
    assert results[i][1] <= results[i+1][1], "brute_force_subset results not properly sorted"

# Test brute_force_subset_internal
print("\nTesting brute_force_subset_internal...")
results = dataset.brute_force_subset_internal(query_idx, subset)
# Check first 5 results
print(f"Top 5 closest points to point {query_idx} (from subset): {results[:5]}")
# Verify ordering
for i in range(len(results) - 1):
    assert results[i][1] <= results[i+1][1], "brute_force_subset_internal results not properly sorted"

# Visualize the results
print("\nGenerating visualization...")
plt.figure(figsize=(10, 8))

# Plot all points
plt.scatter(points[:, 0], points[:, 1], alpha=0.3, color='blue', label='All points')

# Plot query point
plt.scatter(query[0], query[1], color='red', s=100, label='Query point')

# Plot closest points from brute_force
closest_indices = [idx for idx, _ in results[:10]]
plt.scatter(
    points[closest_indices, 0], 
    points[closest_indices, 1], 
    color='green', 
    s=50, 
    label='Top 10 closest (all)'
)

# Plot subset points
plt.scatter(
    points[subset, 0], 
    points[subset, 1], 
    alpha=0.5, 
    color='orange', 
    s=30, 
    label='Subset points'
)

plt.title('Vector Dataset Nearest Neighbor Search')
plt.legend()
plt.tight_layout()
plt.savefig('vector_dataset_test.png')
print("Visualization saved to 'vector_dataset_test.png'")

# Test VectorGraph functionality
print("\n-----------------------------------")
print("Testing VectorGraph functionality...")
print("-----------------------------------")

# Create an empty graph with 10 nodes
n_nodes = 10
print(f"\nCreating empty graph with {n_nodes} nodes...")
graph = PyVectorGraph.empty(n_nodes)

# Verify the number of nodes
print(f"Graph size: {graph.n}")
assert graph.n == n_nodes, "Graph size doesn't match"

# Test the total_edges method (should be 0 for an empty graph)
print(f"Total edges (should be 0): {graph.total_edges()}")
assert graph.total_edges() == 0, "Empty graph should have 0 total edges"

# Test the max_degree method (should be 0 for an empty graph)
print(f"Max degree (should be 0): {graph.max_degree()}")
assert graph.max_degree() == 0, "Empty graph should have max degree 0"

# Add some edges
print("\nAdding edges to the graph...")
graph.add_neighbor(0, 1)
graph.add_neighbor(0, 2)
graph.add_neighbor(0, 3)
graph.add_neighbor(1, 0)
graph.add_neighbor(1, 4)
graph.add_neighbor(2, 0)
graph.add_neighbor(2, 5)

# Test the total_edges method
print(f"Total edges (should be 7): {graph.total_edges()}")
assert graph.total_edges() == 7, "Graph should have 7 total edges"

# Test the max_degree method
print(f"Max degree (should be 3): {graph.max_degree()}")
assert graph.max_degree() == 3, "Graph should have max degree 3"

# Test get_neighborhood
print("\nTesting get_neighborhood...")
neighbors_0 = graph.get_neighborhood(0)
print(f"Neighbors of node 0: {neighbors_0}")
assert len(neighbors_0) == 3, "Node 0 should have 3 neighbors"
assert set(neighbors_0) == {1, 2, 3}, "Node 0 should have neighbors 1, 2, 3"

# Test set_neighborhood
print("\nTesting set_neighborhood...")
graph.set_neighborhood(0, [5, 6, 7, 8])
neighbors_0 = graph.get_neighborhood(0)
print(f"New neighbors of node 0: {neighbors_0}")
assert len(neighbors_0) == 4, "Node 0 should have 4 neighbors after set_neighborhood"
assert set(neighbors_0) == {5, 6, 7, 8}, "Node 0 should have neighbors 5, 6, 7, 8"

# Create a graph with predefined neighborhoods
print("\nCreating graph with predefined neighborhoods...")
neighborhoods = [
    [1, 2, 3],  # Node 0 connected to 1, 2, 3
    [0, 4, 5],  # Node 1 connected to 0, 4, 5
    [0, 6, 7],  # Node 2 connected to 0, 6, 7
    [0, 8, 9],  # Node 3 connected to 0, 8, 9
    [1],        # Node 4 connected to 1
    [1],        # Node 5 connected to 1
    [2],        # Node 6 connected to 2
    [2],        # Node 7 connected to 2
    [3],        # Node 8 connected to 3
    [3],        # Node 9 connected to 3
]
graph2 = PyVectorGraph(neighborhoods)

# Verify graph properties
print(f"Graph2 size: {graph2.n}")
assert graph2.n == 10, "Graph2 should have 10 nodes"

print(f"Graph2 total edges: {graph2.total_edges()}")
assert graph2.total_edges() == 18, "Graph2 should have 18 total edges"

print(f"Graph2 max degree: {graph2.max_degree()}")
assert graph2.max_degree() == 3, "Graph2 should have max degree 3"

print("\nAll tests passed successfully!")