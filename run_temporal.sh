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
    local diffweight=$7
    local temporalweight=$8
    local tag="${ch}_${scaling_str}e${encoders_str}_g${gmin}-${gmax}_d${diffweight}_t${temporalweight}"

    {
    for i in $(seq 1 "$num"); do
        echo "***********************************************************************"
        echo "Run ${i}  (ch=${ch}  scaling=${scaling}  encoders=${encoders})"
        echo "***********************************************************************"
        python train_temporal.py --ch "$ch" --scaling "$scaling" --encoders $encoders --no-pbar \
            --lr-min 1e-5 --gain-min "${gmin}" --gain-max "${gmax}" --seq-len 15 --window-stride 15 \
            --val-frac 0.3 --diff-weight ${diffweight} \
            --flicker-scale 4 --temporal-weight ${temporalweight} --temporal-start 3
        python test_temporal.py /opt/datasets/IMX662_temporal/cabinet_D50_36_static_0 \
            --checkpoint checkpoints/best_temporal.pth --gain 64 --still-checkpoint checkpoints/imx662_64_2e11.pth
        cp checkpoints/best_temporal.pth "checkpoints/run_${tag}_${i}.pth"
    done
    } > >(tee "run_${tag}.txt") 2>&1
}

run_training 64 "2" "1 1" 3 32 128 50 10
run_training 64 "2" "1 1" 3 32 128 40 40
