#!/bin/bash

CSV_FILE="matmul_results.csv"

# Create CSV header
echo "version,matrix_size,naivetime,simd256_time,simd256_speedup" > "$CSV_FILE"

for N in 512 1024 2048 4096 8192
do
    echo "======================================"
    echo "Running matrix size: $N"
    echo "======================================"

    # Run program and capture output
    OUTPUT=$(./bin/matmul simd "$N" "$N" "$N")

    # Print output on screen
    echo "$OUTPUT"

    # Extract values
    NAIVE_TIME=$(echo "$OUTPUT" | awk '/^naive \(ref\)/ {print $4}')
    SIMD_TIME=$(echo "$OUTPUT" | awk '/^simd[[:space:]]/ {print $3}')
    SIMD_SPEEDUP=$(echo "$OUTPUT" | awk '/^simd[[:space:]]/ {print $5}' | sed 's/x//')

    # Add to CSV
    echo "simd256,$N,$NAIVE_TIME,$SIMD_TIME,$SIMD_SPEEDUP" >> "$CSV_FILE"

    echo
done

echo "======================================"
echo "Results saved to $CSV_FILE"
echo "======================================"

cat "$CSV_FILE"