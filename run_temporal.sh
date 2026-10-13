#!/bin/bash
#
# Train and test the temporal DenoiseNet several times per configuration.
#
# Usage: ./run_temporal.sh [RUNS]     (RUNS per configuration, default 3)
#
# Each run_training call below is one configuration, given as key=value
# options; anything not given uses the default in DEFAULTS. For each run the
# checkpoint is tested on TEST_CLIPS, then kept as
# checkpoints/run_<tag>_<i>.pth, and everything is logged to run_<tag>.txt.
#
# Options:
#   runs=N           runs of this configuration (default: RUNS)
#   ch=64 scaling="2" encoders="1 1"    network size
#   gmin=32 gmax=128                    synthetic noise gain range
#   diff=0           --diff-weight
#   temporal=0       --temporal-weight
#   blend=1          --blend-output
#   state=S          --state-ch S
#   gated=1          --state-gated
#   feed=0           --no-feed-image
#   resume=PTH       fine-tune from this checkpoint (copied first, so every
#                    run starts from the same weights)
#   name=STR         extra suffix for the tag, to tell configurations apart
#   extra="..."      any other train_temporal.py arguments, passed as is

export PYTHONUNBUFFERED=1

RUNS="${1:-3}"

DATASET=/opt/datasets/IMX662_temporal
STILL_CHECKPOINT=checkpoints/imx662_64_2e11.pth
TEST_GAIN=64
CHANGE_THRESHOLD=0.05
TEST_CLIPS=(cabinet_D50_36_static_0 cabinet_D50_96_static_0 cabinet_D65_100_local_0)

declare -A DEFAULTS=(
    [ch]=64 [scaling]="2" [encoders]="1 1" [gmin]=32 [gmax]=128
    [diff]=0 [temporal]=0 [blend]=0 [state]=0 [gated]=0 [feed]=1
    [resume]="" [name]="" [extra]=""
)

run_training() {
    local -A o
    local kv
    for kv in "${!DEFAULTS[@]}"; do o[$kv]="${DEFAULTS[$kv]}"; done
    o[runs]=$RUNS
    for kv in "$@"; do
        if [[ "$kv" != *=* || -z "${o[${kv%%=*}]+set}" ]]; then
            echo "run_training: unknown option '$kv'" >&2
            return 1
        fi
        o[${kv%%=*}]="${kv#*=}"
    done

    local tag="${o[ch]}_${o[scaling]// /}e${o[encoders]// /}_g${o[gmin]}-${o[gmax]}_d${o[diff]}_t${o[temporal]}"
    local args=(--ch "${o[ch]}" --scaling "${o[scaling]}" --encoders ${o[encoders]}
                --gain-min "${o[gmin]}" --gain-max "${o[gmax]}"
                --diff-weight "${o[diff]}" --temporal-weight "${o[temporal]}")
    local test_args=()
    if [[ "${o[blend]}" == 1 ]]; then
        tag+="_blend"; args+=(--blend-output); test_args+=(--save-alpha)
    fi
    if [[ "${o[state]}" != 0 ]]; then
        tag+="_s${o[state]}"; args+=(--state-ch "${o[state]}"); test_args+=(--save-state)
    fi
    if [[ "${o[gated]}" == 1 ]]; then
        tag+="g"; args+=(--state-gated)
    fi
    if [[ "${o[feed]}" == 0 ]]; then
        tag+="_nofeed"; args+=(--no-feed-image)
    fi
    local resume_copy=""
    if [[ -n "${o[resume]}" ]]; then
        tag+="_ft"
        resume_copy=$(mktemp --suffix=.pth)
        cp "${o[resume]}" "$resume_copy" || return 1
        args+=(--resume "$resume_copy")
    fi
    [[ -n "${o[name]}" ]] && tag+="_${o[name]}"
    if [[ -e "run_${tag}.txt" ]]; then
        echo "run_training: run_${tag}.txt already exists, skipping (use name=... to" \
             "run it again, or remove the old results)" >&2
        [[ -n "$resume_copy" ]] && rm -f "$resume_copy"
        return 1
    fi

    local stamp
    stamp=$(mktemp)
    {
    echo "Configuration: $*"
    [[ -n "${o[resume]}" ]] && echo "Fine-tuning from ${o[resume]}"
    for i in $(seq 1 "${o[runs]}"); do
        echo "***********************************************************************"
        echo "Run ${i}/${o[runs]}  ${tag}"
        echo "***********************************************************************"
        touch "$stamp"
        if ! python train_temporal.py "${args[@]}" --no-pbar \
                --lr-min 1e-5 --seq-len 15 --window-stride 15 --val-frac 0.3 \
                --flicker-scale 4 --temporal-start 3 ${o[extra]}; then
            echo "Run ${i}: training failed, skipping its test."
            continue
        fi
        if [[ ! checkpoints/best_temporal.pth -nt "$stamp" ]]; then
            echo "Run ${i}: no new checkpoint was saved, skipping its test."
            continue
        fi
        cp checkpoints/best_temporal.pth "checkpoints/run_${tag}_${i}.pth"
        for clip in "${TEST_CLIPS[@]}"; do
            echo "----- test: ${clip} -----"
            python test_temporal.py "${DATASET}/${clip}" \
                --checkpoint "checkpoints/run_${tag}_${i}.pth" --gain "$TEST_GAIN" --seed 0 \
                --still-checkpoint "$STILL_CHECKPOINT" --change-threshold "$CHANGE_THRESHOLD" \
                --output "output_temporal/${clip}" "${test_args[@]}"
        done
    done
    } > >(tee "run_${tag}.txt") 2>&1
    rm -f "$stamp" ${resume_copy:+"$resume_copy"}
}

# Experiments (see the plan): A baseline, A0 blend (fine-tuned), B/C state,
# D gated, E blend + gated state. Comment out the ones you don't want.
# A (baseline) already exists as run_64_2e11_g32-128_d50_t10_*.pth
#run_training diff=50 temporal=10                                   # A
#run_training diff=50 temporal=10 blend=1 resume=checkpoints/run_64_2e11_g32-128_d50_t10_1.pth runs=1  # A0
#run_training diff=50 temporal=10 blend=1                            # A0, from scratch
#run_training diff=50 temporal=10 state=4                            # B
#run_training diff=50 temporal=10 state=8 feed=0                     # C
#run_training diff=50 temporal=10 state=4 gated=1                    # B-gated
#run_training diff=50 temporal=10 state=8 gated=1 feed=0             # C-gated
#run_training runs=2 diff=50 temporal=10 blend=1 state=4 gated=1            # E
run_training ch=32 runs=3 diff=50 temporal=10 blend=1
