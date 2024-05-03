#!/usr/bin/env sh

# This shell script autotunes the BM, BN, BK, TM, TN paramters for kenel_9
# Sweep range:
# 1. BM: 32, 64, 128, 256, 512, 1024 (if applicable)
# 2. BN: 32, 64, 128, 256, 512, 1024 (if applicable)
# 3. BK: 32, 64, 128, 256, 512, 1024 (if applicable)
# 4. TM: range(4, BM, 4) (if applicable)
# 5. TN: range(4, BN, 4) (if applicable)

run_autotune() {
    BM=$1
    BN=$2
    BK=$3
    TM=$4
    TN=$5

    # Replace values in autotune.cu file
    sed -i "s/#define BM [0-9]\+/#define BM $BM/g" ./autotune.cu
    sed -i "s/#define BN [0-9]\+/#define BN $BN/g" ./autotune.cu
    sed -i "s/#define BK [0-9]\+/#define BK $BK/g" ./autotune.cu
    sed -i "s/#define TM [0-9]\+/#define TM $TM/g" ./autotune.cu
    sed -i "s/#define TN [0-9]\+/#define TN $TN/g" ./autotune.cu

    # Run cmake
    echo "Running cmake with BM=$BM, BN=$BN, BK=$BK, TM=$TM, TN=$TN"
    cmake --build ./build --target autotune > /dev/null 2>&1

    # Run autotune
    if [ $? -eq 0 ]; then
        ./autotune 9 > ./test/autotune/autotune_${BM}_${BN}_${BK}_${TM}_${TN}.txt
        if [ $? -eq 0 ]; then
            echo "Autotune run successful"
        else
            echo "Autotune run failed"
            rm ./test/autotune/autotune_${BM}_${BN}_${BK}_${TM}_${TN}.txt
        fi
    else
        echo "CMake failed. Autotune not executed."
    fi
}

# Usage example:
# Hierarchical search of BM, BN, BK, TM, TN
for BM in 256 512 1024; do
    for BN in 256 512 1024; do
        for BK in 8 16 32 128 256 512 1024; do
            for TM in 4 8 16 32; do
                for TN in 4 8 16 32; do
                    run_autotune $BM $BN $BK $TM $TN
                done
            done
        done
    done
done