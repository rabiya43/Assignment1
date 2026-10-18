#!/usr/bin/env bash
# Reproduces every Task 3 run in the report. Console output of each run is saved in logs/.
# Usage (from the Assignment1 folder):  bash run_txt_experiments.sh [python executable]
set -u
PY=${1:-python}
mkdir -p logs

run() {  # run <log/run name> <args...>
    name=$1; shift
    echo "=== $name: $*"
    "$PY" txt_classifier.py --run_name "$name" --output "checkpoints/$name.pth" "$@" 2>&1 | tee "logs/$name.log" \
        | grep -E "^Epoch|^Test|runtime|Error"
}
mkdir -p checkpoints

# Task 3.1: raw length histogram (figures/txt_length_histogram.png)
"$PY" txt_classifier.py --batch_size 1 --max_len -1 --length_histogram | tee logs/length_histogram.log

# Task 3.2.a: runtime comparison (starter SGD optimizer, lr 5, 5 epochs)
run sgd-bs1-maxlen-none  --optimizer sgd --batch_size 1  --max_len -1
run sgd-bs1-maxlen-1024  --optimizer sgd --batch_size 1  --max_len 1024
run sgd-bs32-maxlen-1024 --optimizer sgd --batch_size 32 --max_len 1024

# Task 3.2.b: expected to fail (variable-length articles cannot be stacked into one batch)
"$PY" txt_classifier.py --optimizer sgd --batch_size 32 --max_len -1 --wandb_mode disabled \
    > logs/sgd-bs32-maxlen-none-error.log 2>&1
tail -n 1 logs/sgd-bs32-maxlen-none-error.log

# Task 3.3: Adam (default lr 0.001) in the same two batching setups as the SGD runs above
run adam-bs1-maxlen-none  --optimizer adam --batch_size 1  --max_len -1
run adam-bs32-maxlen-1024 --optimizer adam --batch_size 32 --max_len 1024

# Task 3.4: baseline vs LSTM, both with Adam, batch 32, max_len 1024, 10 epochs
run baseline-adam-10ep --model baseline --batch_size 32 --max_len 1024 --n_epochs 10
run lstm-10ep          --model lstm     --batch_size 32 --max_len 1024 --n_epochs 10
# Ablation for 3.4.c: same LSTM but it reads and pools over <PAD> tokens
run lstm-pad-ablation-10ep --model lstm --batch_size 32 --max_len 1024 --n_epochs 10 --no_ignore_padding
