#!/bin/bash

# this script unpacks the data files and computes groundtruth for the dataset

# unzipping data.tar.gz
if [ -f "word2vec-google-news-300_50000_lowercase.tar.gz" ]; then
    echo "Unpacking word2vec-google-news-300_50000_lowercase.tar.gz..."
    tar -xzf word2vec-google-news-300_50000_lowercase.tar.gz
else
    echo "word2vec-google-news-300_50000_lowercase.tar.gz not found. Please ensure it is in the current directory."
    exit 1
fi

# for every subdir of data, compute groundtruth if it doesn't already exist
for dir in data/*/; do
    if [ -d "$dir" ] && [ ! -f "${dir}GT" ]; then
        echo "Computing groundtruth for $dir..."
        cargo run -r --bin compute_groundtruth -- -d ${dir}base.fbin -o ${dir}GT -q ${dir}query.fbin
        if [ $? -ne 0 ]; then
            echo "Failed to compute groundtruth for $dir"
            exit 1
        fi
    fi
done
