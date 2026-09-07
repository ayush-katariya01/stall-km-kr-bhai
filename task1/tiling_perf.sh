#!/bin/bash

OUT="tile_dif_matrix3.csv"

echo "K,H,W,b,speedup,L1D_misses,instructions,MPKI" > "$OUT"

for K in 3
do
    #for SIZE in 512 1024 2048 4096 8192 16384
    for SIZE in 16384
    do
        H=$SIZE
        W=$SIZE

        for B in 128 256 512 1024 2048 4096 8192
        do
            # Skip tile sizes >= problem size
            if [[ $B -ge $SIZE ]]; then
                continue
            fi

            echo "Running K=$K H=$H W=$W b=$B"

            # Build with this tile size
            make clean >/dev/null 2>&1
            make BLOCK_SIZE=$B >/dev/null 2>&1

            SPEEDUPS=0
            L1D_MISSES_TOTAL=0
            INSTRUCTIONS_TOTAL=0

            VALID_RUNS=0

            for loop_idx in {1..1}
            do
                # -------------------------
                # Run for speedup
                # -------------------------
                RESULT=$(./bin/conv tile $H $W $K 42 2>/dev/null)

                SPEEDUP=$(echo "$RESULT" | tail -n 1 | awk '{print $NF}')

                if [[ "$SPEEDUP" =~ ^[0-9]+\.[0-9]+x$ ]]; then

                    # Remove x
                    SPEEDUP_NUM=${SPEEDUP%x}

                    # Add floating-point value
                    SPEEDUPS=$(awk -v a="$SPEEDUPS" -v b="$SPEEDUP_NUM" \
                        'BEGIN {printf "%.6f", a+b}')

                    ((VALID_RUNS++))

                else
                    echo "Warning: could not read speedup"
                    echo "$RESULT"
                fi


                # -------------------------
                # Run perf
                # -------------------------
                PERF=$(sudo perf stat \
                    -e instructions,L1-dcache-load-misses \
                    ./bin/conv tile $H $W $K 42 \
                    2>&1 >/dev/null)

                INSTRUCTIONS=$(echo "$PERF" |
                    grep "instructions" |
                    awk '{gsub(",","",$1); print $1}')

                L1D_MISSES=$(echo "$PERF" |
                    grep "L1-dcache-load-misses" |
                    awk '{gsub(",","",$1); print $1}')


                # Check perf values
                if [[ "$INSTRUCTIONS" =~ ^[0-9]+$ ]] &&
                   [[ "$L1D_MISSES" =~ ^[0-9]+$ ]]; then

                    (( INSTRUCTIONS_TOTAL += INSTRUCTIONS ))
                    (( L1D_MISSES_TOTAL += L1D_MISSES ))

                else
                    echo "Warning: perf measurement failed"
                fi

            done


            # -------------------------
            # Calculate averages
            # -------------------------

            if [[ $VALID_RUNS -gt 0 ]]; then

                AVG_SPEEDUP=$(awk \
                    -v total="$SPEEDUPS" \
                    -v n="$VALID_RUNS" \
                    'BEGIN {printf "%.3fx", total/n}')

                AVG_INSTRUCTIONS=$(awk \
                    -v total="$INSTRUCTIONS_TOTAL" \
                    -v n="$VALID_RUNS" \
                    'BEGIN {printf "%.0f", total/n}')

                AVG_L1D_MISSES=$(awk \
                    -v total="$L1D_MISSES_TOTAL" \
                    -v n="$VALID_RUNS" \
                    'BEGIN {printf "%.0f", total/n}')

                MPKI=$(awk \
                    -v m="$AVG_L1D_MISSES" \
                    -v i="$AVG_INSTRUCTIONS" \
                    'BEGIN {printf "%.3f", (m/i)*1000}')

            else
                AVG_SPEEDUP="NA"
                AVG_INSTRUCTIONS="NA"
                AVG_L1D_MISSES="NA"
                MPKI="NA"
            fi


            # -------------------------
            # Save CSV
            # -------------------------

            echo "$K,$H,$W,$B,$AVG_SPEEDUP,$AVG_L1D_MISSES,$AVG_INSTRUCTIONS,$MPKI" \
                >> "$OUT"

        done
    done
done

echo
echo "Done."
echo "Results saved to $OUT"