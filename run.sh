#!/bin/bash

export PYTHONUNBUFFERED=1

N="${1:-10}"

run_training() {
    local ch=$1
    local scaling=$2
    local encoders=$3
    local scaling_str="${scaling// /}"
    local encoders_str="${encoders// /}"
    local tag="${ch}_${scaling_str}e${encoders_str}"

    {
    for i in $(seq 1 "$N"); do
        echo "***********************************************************************"
        echo "Run ${i}  (ch=${ch}  scaling=${scaling}  encoders=${encoders})"
        echo "***********************************************************************"
        python train.py --ch "$ch" --scaling $scaling --encoders $encoders --no-pbar
        python test.py cabinet_D50_100/imx219_ag12_test --weights checkpoints/best.pth --quick
        python train.py --filter imx219 ag12 --ch "$ch" --scaling $scaling --encoders $encoders \
            --resume checkpoints/best.pth --lr 1e-4 --lr-min 1e-6 --no-pbar
        python test.py cabinet_D50_100/imx219_ag12_test --weights checkpoints/best.pth --quick
        mv checkpoints/best.pth "checkpoints/run_${tag}_${i}.pth"
    done
    } > >(tee "run_${tag}.txt") 2>&1
}

# Example: run_training 20 "1 2 2" "1 1 2"

run_training 16 "2 2 1" "1 2 2"
run_training 20 "2 2" "1 2"
run_training 16 "2 2" "2 2"
run_training 32 "2 2 2 1" "1 1 2 4"
run_training 20 "2 2 1" "1 1 2"
