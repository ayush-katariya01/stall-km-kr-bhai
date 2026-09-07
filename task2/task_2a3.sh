#!/bin/bash

B=128
PREFETCH_DISTANCE=64
CACHE_FILL_LEVEL=1

OUTPUT="results/task2a3_with_prefetcher.csv"

SIZES=(256 512 1024 1536 2048 3072 4096)

echo "H,W,K,B,prefetch_distance,cache_fill_level,time" > "$OUTPUT"

for N in "${SIZES[@]}"
do
    output=$(./bin/matmul prefetch "$N" "$N" "$N")

    echo "$output"

    time=$(echo "$output" | tail -n 1 | awk '{print $(NF-2)}')

    echo "$N,$N,$N,$B,$PREFETCH_DISTANCE,$CACHE_FILL_LEVEL,${time}" >> "$OUTPUT"
done