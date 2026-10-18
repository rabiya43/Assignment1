#!/usr/bin/env bash
# Reproduces every Task 2 run in the report. Console output of each run is saved in logs/.
# Usage (from the Assignment1 folder):  bash run_img_experiments.sh [python executable]
set -u
PY=${1:-python}
mkdir -p logs checkpoints

run() {  # run <run name> <args...>
    name=$1; shift
    echo "=== $name: $*"
    "$PY" img_classifier.py --run_name "$name" --output "checkpoints/$name.pth" "$@" 2>&1 | tee "logs/$name.log" \
        | grep -E "accuracy|Error"
}

# Tasks 2.1-2.3 and 2.5: default hyperparameters (5 epochs, batch 8, SGD lr 0.001), colour images
run neural-the-narwhal
# Task 2.4: same model and hyperparameters on single-channel grayscale images
run grayscale-base --grayscale
# Task 2.6: original MLP vs the new convolutional model, same hyperparameters
run base-model --model base
run new-model  --model new
