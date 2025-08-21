from utils import fbin_to_numpy, numpy_to_fbin
import numpy as np
import os, sys


def rot(s, x, y, rx, ry):
    """
    Rotate/flip a quadrant appropriately.
    s: current sub-square size
    (x, y): coordinates
    rx, ry: quadrant bits
    """
    if ry == 0:
        if rx == 1:
            x = s - 1 - x
            y = s - 1 - y
        # Swap x and y
        x, y = y, x
    return x, y


def hilbert_index(x, y, order):
    """
    Compute the Hilbert index for coordinates (x, y) on a 2^order x 2^order grid.
    """
    n = 2**order
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = rot(s, x, y, rx, ry)
        s //= 2
    return d


def hilbert_project(x, y, order):
    """
    Computes Hilbert curve indices for arrays x and y.
    Scales the coordinates to the integer grid [0, 2^order - 1] if they are not already.
    """
    n = 2**order
    # Check if scaling is needed (if not integer or out of bounds)
    if (
        (not np.issubdtype(x.dtype, np.integer))
        or (np.min(x) < 0)
        or (np.max(x) > n - 1)
    ):
        x_min, x_max = np.min(x), np.max(x)
        x = ((x - x_min) / (x_max - x_min) * (n - 1)).astype(int)
    if (
        (not np.issubdtype(y.dtype, np.integer))
        or (np.min(y) < 0)
        or (np.max(y) > n - 1)
    ):
        y_min, y_max = np.min(y), np.max(y)
        y = ((y - y_min) / (y_max - y_min) * (n - 1)).astype(int)

    indices = np.empty_like(x, dtype=np.int64)
    for i in range(len(x)):
        indices[i] = hilbert_index(x[i], y[i], order)
    return indices


def sort_by_hilbert(points, order=10):
    """
    Sorts the points by their Hilbert curve index.
    """
    # Extract x and y coordinates from the points
    x = points[:, 0]
    y = points[:, 1]

    # Compute Hilbert indices
    hilbert_indices = hilbert_project(x, y, order)

    # Sort points by Hilbert indices
    sorted_indices = np.argsort(hilbert_indices)
    sorted_points = points[sorted_indices]

    return sorted_points


if __name__ == "__main__":
    path = sys.argv[1]

    points = fbin_to_numpy(path)
    point_46 = points[77]

    order = 10  # Order of the Hilbert curve
    sorted_points = sort_by_hilbert(points, order)

    # Find the new index of point 46
    new_index = np.where(np.all(sorted_points == point_46, axis=1))[0]
    if len(new_index) == 0:
        print("Point 46 not found in sorted dataset.")
    else:
        print(f"new index of point 77: {new_index[0]}")

    # Save the sorted points to a new file
    sorted_path = os.path.splitext(path)[0] + "_sorted.fbin"
    numpy_to_fbin(sorted_points, sorted_path)
    print(f"Sorted points saved to {sorted_path}")
