#!/bin/bash

B=128
PREFETCH_DISTANCE=64
CACHE_FILL_LEVEL=1

OUTPUT="results/task2_prefetch.csv"

SIZES=(256 512 1024 1536 2048 3072 4096)

echo "H,W,K,B,prefetch_distance,cache_fill_level,speedup" > "$OUTPUT"

for N in "${SIZES[@]}"
do
    output=$(./bin/matmul prefetch "$N" "$N" "$N")

    echo "$output"

    speedup=$(echo "$output" | tail -n 1 | awk '{print $NF}')

    echo "$N,$N,$N,$B,$PREFETCH_DISTANCE,$CACHE_FILL_LEVEL,${speedup}" >> "$OUTPUT"
done