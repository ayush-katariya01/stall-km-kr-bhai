#!/bin/bash

OUTPUT="test_results.csv"

echo "Cleaning previous build..."
make clean

echo "Building..."
make

if [ $? -ne 0 ]; then
    echo "Build failed. Exiting."
    exit 1
fi

echo
echo "Build successful."
echo

echo "K,size,stage,correct,time_ms,gflops,speedup" > "$OUTPUT"

SIZES=(256 512 1024 2048 4096 8192 16384)
KS=(3 5)

for K in "${KS[@]}"; do
    for SIZE in "${SIZES[@]}"; do

        echo "Running K=$K SIZE=${SIZE}x${SIZE}"

        RESULT=$(./bin/conv all "$SIZE" "$SIZE" "$K" 2>/dev/null)

        NAIVE=$(echo "$RESULT" | grep -E "^[[:space:]]*naive[[:space:]]+\(ref\)")
        OPTIMIZED=$(echo "$RESULT" | grep -E "^[[:space:]]*optimized[[:space:]]")

        NAIVE_CORRECT=$(echo "$NAIVE" | awk '{print $3}')
        NAIVE_TIME=$(echo "$NAIVE" | awk '{print $4}')
        NAIVE_GFLOPS=$(echo "$NAIVE" | awk '{print $5}')
        NAIVE_SPEEDUP=$(echo "$NAIVE" | awk '{print $6}')

        OPTIMIZED_CORRECT=$(echo "$OPTIMIZED" | awk '{print $2}')
        OPTIMIZED_TIME=$(echo "$OPTIMIZED" | awk '{print $3}')
        OPTIMIZED_GFLOPS=$(echo "$OPTIMIZED" | awk '{print $4}')
        OPTIMIZED_SPEEDUP=$(echo "$OPTIMIZED" | awk '{print $5}')

        echo "$K,$SIZE,naive,$NAIVE_CORRECT,$NAIVE_TIME,$NAIVE_GFLOPS,$NAIVE_SPEEDUP" >> "$OUTPUT"

        echo "$K,$SIZE,optimized,$OPTIMIZED_CORRECT,$OPTIMIZED_TIME,$OPTIMIZED_GFLOPS,$OPTIMIZED_SPEEDUP" >> "$OUTPUT"

    done
done

echo
echo "======================================"
echo "All tests completed."
echo "Results saved to: $OUTPUT"
echo "======================================"