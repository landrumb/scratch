#!/bin/bash

# this script unpacks the data files and computes groundtruth for the dataset

# unzipping data.tar.gz
if [ -f "data.tar.gz" ]; then
    echo "Unpacking data.tar.gz..."
    tar -xzf data.tar.gz
else
    echo "data.tar.gz not found. Please ensure it is in the current directory."
    exit 1
fi

# for every subdir of data, compute groundtruth
for dir in data/*/; do
    if [ -d "$dir" ]; then
        echo "Computing groundtruth for $dir..."
        cargo run -r --bin compute_groundtruth -- -d ${dir}base.fbin -o ${dir}GT -q ${dir}query.fbin
        if [ $? -ne 0 ]; then
            echo "Failed to compute groundtruth for $dir"
            exit 1
        fi
    fi
done
