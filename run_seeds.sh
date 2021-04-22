#!/bin/sh
set -x

OUTPUT_PATH=$1

for seed in 0123 1234 2345 3456 4567 5678 6789 7890 8901 9012; do
    ./run_paper_experiments.sh $seed $OUTPUT_PATH
    done