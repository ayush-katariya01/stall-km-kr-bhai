#!/bin/bash

BLOCK_SIZE=(32 64 128 256 512 1024)

SIZES=(256 512 1024 1536 2048)
DISTANCES=(8 16 32 64 128)
CACHE_LEVELS=(1 2 3)

OUTPUT="results/task2a2.csv"

echo "H,W,K,B,prefetch_distance,cache_fill_level,L1D_misses,L2_misses,LLC_misses" > "$OUTPUT"

for N in "${SIZES[@]}"
do
    for DIST in "${DISTANCES[@]}"
    do
        for LEVEL in "${CACHE_LEVELS[@]}"
        do
            for B in "${BLOCK_SIZE[@]}"; do
            if [[ $B -ge $N ]]; then
                continue
            fi
            if [[ $DIST -ge $B ]]; then
                continue
            fi
            echo "Size=$N  B=$B  Distance=$DIST  CacheLevel=$LEVEL"

            # Measure L1D misses
            perf_output=$(sudo perf stat \
    -e l1-dcache-load-misses,l2_cache_misses_from_dc_misses,cache-misses \
    ./bin/matmul prefetch "$N" "$N" "$N" \
    2>&1 >/dev/null)

L1D=$(echo "$perf_output" | grep "l1-dcache-load-misses" | \
      awk '{gsub(",", "", $1); print $1}')

L2=$(echo "$perf_output" | grep "l2_cache_misses_from_dc_misses" | \
     awk '{gsub(",", "", $1); print $1}')

LLC=$(echo "$perf_output" | grep "cache-misses" | \
      awk '{gsub(",", "", $1); print $1}')

            echo "$N,$N,$N,$B,$DIST,$LEVEL,$L1D,$L2,$LLC" >> "$OUTPUT"
            done
        done
    done
done

echo
echo "=========================================="
echo "Results:"
echo "=========================================="
cat results/task2a2.csv