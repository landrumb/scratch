"""Takes pairs of points as inputs and prints the distance between them."""

import numpy as np
import sys
from utils import fbin_to_numpy
from scipy.spatial import distance

if __name__ == "__main__":
    points_file = sys.argv[1]

    # Load points and pairs
    points = fbin_to_numpy(points_file)
    n, d = points.shape
    print(f"Loaded {n} points with {d} dimensions")
    
    print("give indices of 2 points to compare (return to exit)")
    while True:
        try:
            line = input()
            if not line.strip():
                break
            i, j = map(int, line.split())
            if i < 0 or i >= n or j < 0 or j >= n:
                print("Indices out of bounds")
                continue
            dist = distance.euclidean(points[i], points[j])
            print(dist, points[i], points[j])
        except Exception as e:
            print(f"Error: {e}")
    