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
KS=(3)

# Add or remove stages here
STAGES=(naive simd optimized)

for K in "${KS[@]}"; do
    for SIZE in "${SIZES[@]}"; do

        echo "Running K=$K SIZE=${SIZE}x${SIZE}"

        RESULT=$(./bin/conv all "$SIZE" "$SIZE" "$K" 2>/dev/null)

        for STAGE in "${STAGES[@]}"; do

            if [ "$STAGE" = "naive" ]; then

                LINE=$(echo "$RESULT" | grep -E "^[[:space:]]*naive[[:space:]]+\(ref\)")

                CORRECT=$(echo "$LINE" | awk '{print $3}')
                TIME=$(echo "$LINE" | awk '{print $4}')
                GFLOPS=$(echo "$LINE" | awk '{print $5}')
                SPEEDUP=$(echo "$LINE" | awk '{print $6}')

            else

                LINE=$(echo "$RESULT" | grep -E "^[[:space:]]*$STAGE[[:space:]]")

                CORRECT=$(echo "$LINE" | awk '{print $2}')
                TIME=$(echo "$LINE" | awk '{print $3}')
                GFLOPS=$(echo "$LINE" | awk '{print $4}')
                SPEEDUP=$(echo "$LINE" | awk '{print $5}')

            fi

            echo "$K,$SIZE,$STAGE,$CORRECT,$TIME,$GFLOPS,$SPEEDUP" >> "$OUTPUT"

        done

    done
done

echo
echo "======================================"
echo "All tests completed."
echo "Results saved to: $OUTPUT"
echo "======================================"