#!/bin/bash

exec > >(tee run.txt) 2>&1

export PYTHONUNBUFFERED=1

for i in $(seq 1 "${1:-5}"); do
    echo "***********************************************************************"
    echo "Run ${i}"
    echo "***********************************************************************"
    python train.py --no-pbar
    python test.py cabinet_D50_100/imx219_ag12_test --weight checkpoints/best.pth --quick
    python train.py --filter imx219 ag12 --resume checkpoints/best.pth --lr 1e-4 --lr-min 1e-6 --no-pbar
    python test.py cabinet_D50_100/imx219_ag12_test --weight checkpoints/best.pth --quick
    cp checkpoints/best.pth "checkpoints/run_${i}.pth"
done
