# Assignment 1 — PyTorch Foundations: Image and Text Classification

Generative AI & LLM, Fall 2026 — Rabiya Tahir

- `txt_classifier.py` — Task 3: real vs. LLM-generated news classifier (EmbeddingBag baseline and LSTM).
- `img_classifier.py` — Task 2: parrot / narwhal / axolotl image classifier (MLP baseline and CNN).
- `run_txt_experiments.sh`, `run_img_experiments.sh` — reproduce every run in the report.
- `logs/` — console output of every reported run. `figures/` — plots used in the report.
- `Report.pdf` — results and analysis. `Assignment_1.pdf` — the assignment handout.

**Model checkpoints (Google Drive):** _link to be added_

**Weights & Biases projects:** `txt_classifier` and `img_classifier`.

## Setup

Python 3.10–3.12 is required for PyTorch 2.5.1.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt   # CPU-only machines can first install torch==2.5.1 torchvision==0.20.1
                                  # from https://download.pytorch.org/whl/cpu to save download size
wandb login                       # paste the API key from https://wandb.ai/authorize
```

Pass `--wandb_mode disabled` to any command to run without wandb (or `offline` to sync later).
Both scripts use CUDA automatically when available; all reported runs used CPU.

## Task 3: text classification (`txt_classifier.py`)

| Argument | Default | Meaning |
|---|---|---|
| `--data_dir` | `data` | Folder containing `txt_train.csv`, `txt_val.csv`, `txt_test.csv` |
| `--batch_size` | 1 | Batch size |
| `--max_len` | -1 | Truncate/pad every article to this many tokens; -1 disables truncation and padding |
| `--n_epochs` | 5 | Training epochs |
| `--model` | `baseline` | `baseline` (EmbeddingBag + Linear) or `lstm` |
| `--optimizer` | `adam` | `adam` or `sgd` (SGD was the starter optimizer) |
| `--learning_rate` | 5.0 for SGD, 0.001 for Adam | Learning rate |
| `--embed_dim` | 64 | Word embedding size |
| `--hidden_dim`, `--num_layers`, `--dropout` | 64, 1, 0.3 | LSTM size, depth and dropout |
| `--unidirectional` | off | Use a one-direction LSTM instead of a bidirectional one |
| `--no_ignore_padding` | off | Ablation: let the LSTM read `<PAD>` tokens and pool over them |
| `--length_histogram` | off | Only plot the raw training-article length histogram (Task 3.1) and exit |
| `--run_name`, `--wandb_project`, `--wandb_mode` | | wandb settings |
| `--output` | `txt_model.pth` | Checkpoint path |

Examples:

```bash
python txt_classifier.py --batch_size 1 --max_len -1 --length_histogram     # 3.1 -> figures/txt_length_histogram.png
python txt_classifier.py --optimizer sgd --batch_size 32 --max_len 1024     # 3.2 / 3.3 SGD baseline
python txt_classifier.py --optimizer adam --batch_size 32 --max_len 1024    # 3.3 Adam baseline
python txt_classifier.py --model lstm --batch_size 32 --max_len 1024 --n_epochs 10   # 3.4 LSTM
bash run_txt_experiments.sh .venv/Scripts/python                            # every Task 3 run
```

The LSTM needs equal-length batches, so use a positive `--max_len` (or `--batch_size 1`).
`--batch_size 32 --max_len -1` fails on purpose (Task 3.2.b).

## Task 2: image classification (`img_classifier.py`)

| Argument | Default | Meaning |
|---|---|---|
| `--data_dir` | `data` | Folder containing the image CSVs and image folders |
| `--n_epochs`, `--batch_size`, `--learning_rate` | 5, 8, 0.001 | Training settings |
| `--optimizer` | `sgd` | `sgd` or `adam` |
| `--model` | `base` | `base` (original MLP) or `new` (Task 2.6 CNN) |
| `--grayscale` | off | Convert images to one grayscale channel (Task 2.4) |
| `--run_name`, `--wandb_project`, `--wandb_mode` | | wandb settings |
| `--output` | `img_model.pth` | Checkpoint path |

```bash
python img_classifier.py --run_name neural-the-narwhal          # 2.1-2.3, 2.5
python img_classifier.py --grayscale --run_name grayscale-base  # 2.4
python img_classifier.py --model new --run_name new-model       # 2.6
bash run_img_experiments.sh .venv/Scripts/python                # every Task 2 run
```

The original MLP has about 101M parameters (≈400 MB of weights), so it needs several GB of RAM.

## Data

Text: 358 / 76 / 74 train / val / test articles, balanced real (label 0) and fake (label 1).
Images: 540 / 116 / 115 train / val / test images, labels parrot (0), narwhal (1), axolotl (2).
Only the `article` column is used as model input.
