#!/bin/bash

export PYTHONUNBUFFERED=1

N="${1:-5}"

run_training() {
    local ch=$1
    local scaling=$2
    local encoders=$3
    local num=$4
    local scaling_str="${scaling// /}"
    local encoders_str="${encoders// /}"
    local gmin=$5
    local gmax=$6
    local tag="${ch}_${scaling_str}e${encoders_str}_${gmin}_${gmax}"

    {
    for i in $(seq 1 "$num"); do
        echo "***********************************************************************"
        echo "Run ${i}  (ch=${ch}  scaling=${scaling}  encoders=${encoders})"
        echo "***********************************************************************"
        python train_temporal.py --ch "$ch" --scaling "$scaling" --encoders $encoders --no-pbar \
            --lr-min 2e-6 --gain-min "${gmin}" --gain-max "${gmax}" --seq-len 10 --window-stride 10 --val-frac 0.2
        python test_temporal.py /opt/datasets/IMX662_temporal/cabinet_D50_100_static_0 \
            --checkpoint checkpoints/best_temporal.pth --gain 64 --still-checkpoint checkpoints/imx662_64_2e11.pth
        cp checkpoints/best_temporal.pth "checkpoints/run_${tag}_${i}.pth"
    done
    } > >(tee "run_${tag}.txt") 2>&1
}

run_training 64 "2" "1 1" 3 32 128
run_training 64 "2" "1 1" 3 16 256
