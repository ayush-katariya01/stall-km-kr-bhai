#!/bin/bash

VERSION=$1
CSV="perf_report.csv"

if [ -z "$VERSION" ]; then
    echo "Usage: ./run_perf.sh <version>"
    echo "Example: ./run_perf.sh naive"
    echo "         ./run_perf.sh simd"
    exit 1
fi

if [ ! -f "$CSV" ]; then
    echo "Version,Size,K,Cycles,Instructions,Cache-Misses,Branches,Branch-Misses,Execution-Time-ms" > "$CSV"
fi

for size in 512 1024 2048 4096 8192
do
    for k in 3 5
    do
        echo "Running: $VERSION $size $size $k"

        # Run program and capture its complete output
        output=$(perf stat \
            ./bin/conv "$VERSION" "$size" "$size" "$k" \
            2>&1)

        # Extract perf counters
        cycles=$(echo "$output" | sed -n 's/^[[:space:]]*\([0-9,]*\)[[:space:]].*cpu-cycles.*/\1/p' | head -1 | tr -d ',')

        instructions=$(echo "$output" | sed -n 's/^[[:space:]]*\([0-9,]*\)[[:space:]].*instructions.*/\1/p' | head -1 | tr -d ',')

        cache_misses=$(echo "$output" | sed -n 's/^[[:space:]]*\([0-9,]*\)[[:space:]].*cache-misses.*/\1/p' | head -1 | tr -d ',')

        branches=$(echo "$output" | sed -n 's/^[[:space:]]*\([0-9,]*\)[[:space:]]*branches.*/\1/p' | head -1 | tr -d ',')

        branch_misses=$(echo "$output" | sed -n 's/^[[:space:]]*\([0-9,]*\)[[:space:]].*branch-misses.*/\1/p' | head -1 | tr -d ',')

        # Extract the selected stage's time
        #
        # For VERSION=naive:
        # naive (ref)     yes          39.501
        #
        # For VERSION=simd:
        # simd            yes           5.531

        if [ "$VERSION" = "naive" ]; then
            elapsed=$(echo "$output" | awk '$1=="naive" && $2=="(ref)" {print $4; exit}')
        else
            elapsed=$(echo "$output" | awk -v v="$VERSION" '$1==v {print $4; exit}')
        fi

        echo "Cycles:         $cycles"
        echo "Instructions:   $instructions"
        echo "Cache misses:   $cache_misses"
        echo "Branches:       $branches"
        echo "Branch misses:  $branch_misses"
        echo "Execution time: $elapsed ms"
        echo

        # Append to CSV
        echo "$VERSION,$size,$k,$cycles,$instructions,$cache_misses,$branches,$branch_misses,$elapsed" >> "$CSV"
    done
done

echo "================ CSV REPORT ================"
column -s, -t < "$CSV"