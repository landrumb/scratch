#!/usr/bin/env python3
"""
Test the VectorDataset functionality using random 2D points.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from scratch import PyVectorDataset

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

print("\nAll tests passed successfully!")