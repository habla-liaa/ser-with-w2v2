#!/bin/sh
set -x

OUTPUT_PATH=$1

for seed in 0123 1234 2345 3456 4567; do
    ./run_paper_experiments.sh $seed $OUTPUT_PATH
    done
