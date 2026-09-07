#!/bin/bash

OUTPUT="results.csv"

echo "Matrix_Size,K,Naive_Time_ms,SIMD256_Time_ms,SIMD256_Speedup" > "$OUTPUT"

for SIZE in 512 1024 2048 4096 8192
do
    for K in 3 5 7 9 11 13 15 17 19 21
    do
        echo "Running SIZE=$SIZE K=$K"

        RESULT=$(./bin/conv simd $SIZE $SIZE $K)

        # Extract naive time
        NAIVE_TIME=$(echo "$RESULT" | awk '/naive \(ref\)/ {print $4}')

        # Extract SIMD time
        SIMD_TIME=$(echo "$RESULT" | awk '/simd[[:space:]]/ {print $3}')

        # Calculate speedup
        SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $NAIVE_TIME / $SIMD_TIME}")

        echo "$SIZE,$K,$NAIVE_TIME,$SIMD_TIME,$SPEEDUP" >> "$OUTPUT"
    done
done

echo
echo "========================================"
echo "FINAL RESULTS"
echo "========================================"

cat "$OUTPUT"