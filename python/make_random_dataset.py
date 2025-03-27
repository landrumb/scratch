import argparse
from utils import *
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make a random dataset")
    parser.add_argument("-n", type=int, default=1000, help="number of points in the dataset")
    parser.add_argument("-d", type=int, default=128, help="dimension of the dataset")
    parser.add_argument("-o", type=str, default="data", help="filename to save the dataset")
    args = parser.parse_args()

    # Create random dataset
    dataset = np.random.rand(args.n, args.d).astype(np.float32)

    # Save the dataset to a .fbin file
    numpy_to_fbin(dataset, args.o)