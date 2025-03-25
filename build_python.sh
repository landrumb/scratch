#!/bin/bash
# Script to build Python bindings for the scratch library

# Exit immediately if a command exits with a non-zero status
set -e

echo "Building Python bindings for scratch library..."

# Unset CONDA_PREFIX if it exists (to avoid conflicts with virtual environments)
if [ -n "$CONDA_PREFIX" ]; then
    echo "Unsetting CONDA_PREFIX to avoid conflicts..."
    unset CONDA_PREFIX
fi

# Check if virtual environment exists and activate it
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "No .venv directory found. Creating a new virtual environment..."
    uv venv
    source .venv/bin/activate
    
    echo "Installing required packages..."
    uv pip install numpy matplotlib maturin
fi

# Build the Python extension with maturin
echo "Building extension module with maturin..."
maturin develop --release --features python

echo "Python bindings built successfully!"
echo "You can now import the module with: from scratch import PyVectorDataset"
echo
echo "To test the bindings, run: python test_vector_dataset.py"