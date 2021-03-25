#!/bin/bash
set -x

export LC_NUMERIC="en_US.UTF-8"
export PYTHONHASHSEED=1234

# Parameters:
SEED=$1
seed_mod = "global/seed=${SEED}"

#OS-experiment
paiprun experiments/configs/main/os-baseline.yaml --output_path "s3://lpepino-datasets2/paper_experiments/os-baseline/${SEED}" --mods $seed_mod

