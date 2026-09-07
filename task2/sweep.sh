#!/bin/bash

BLOCK_SIZES=(32 64 96 128 160 192 256 512 1024 2000)
PREFETCH_DISTANCES=(8 16 32 64 128)
CACHE_LEVELS=(1 2 3)

OUTPUT="prefetchnew_sweep.csv"

# CSV header
echo "block size,prefetch distance,cache level,speedup" > "$OUTPUT"

for b in "${BLOCK_SIZES[@]}"
do
    for dist in "${PREFETCH_DISTANCES[@]}"
    do
        for level in "${CACHE_LEVELS[@]}"
        do
            if [[ $dist -ge $b ]]; then
                continue
            fi
            echo "Testing: b=$b distance=$dist cache_level=$level"

            make clean > /dev/null

            make \
                BLOCK_SIZE=$b \
                PREFETCH_DISTANCE=$dist \
                CACHE_FILL_LEVEL=$level \
                > /dev/null

            if [ $? -ne 0 ]; then
                echo "BUILD FAILED: b=$b dist=$dist level=$level"
                continue
            fi

            # Run benchmark and select ONLY the prefetch result
            #./bin/matmul prefetch | awk '{for(i=1;i<=NF;i++) print $i}' | while IFS= read -r line
            ./bin/matmul prefetch | tail -n 1 | awk -v OFS="," -v b="$b" -v d="$dist" -v L="$level" '{print b, d, L, $NF}' | while IFS= read -r line
            do
                echo "$line" >> "$OUTPUT"
            done

        done
    done
done

echo ""
echo "======================================"
echo "Sweep complete"
echo "Results saved to $OUTPUT"
echo "======================================"