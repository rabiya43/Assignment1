# Assignment 1 — Student Starter Package

Generative AI & LLM, Fall 2026. Read Assignment_1.pdf for questions, marking,
submission rules and deadline. This package supplies data and baseline code;
it does not contain completed experiment answers or an LSTM solution.

## Contents

- Assignment_1.pdf — your assignment handout.
- img_classifier.py — working RGB image MLP baseline for Task 2.
- txt_classifier.py — working EmbeddingBag text baseline for Task 3.
- data/img_train.csv, img_val.csv, img_test.csv and the referenced JPG images.
- data/txt_train.csv, txt_val.csv, txt_test.csv — UTF-8 text files.
- requirements.txt — environment dependencies.
- SOURCE_NOTES.md — source attribution and scaffold adjustments.
- DATA_CHECKS.json — text split counts and duplicate-removal record.

## Setup

Use Python 3.10–3.12. Extract the ZIP completely. Open a terminal inside
Assignment1_Student_Package, where the scripts and data folder are located.

```bash
python -m pip install -r requirements.txt
```

PyTorch 2.5.1 and torchvision 0.20.1 are the paired versions used for this
package. For a GPU installation, use the appropriate installation command
from https://pytorch.org/get-started/locally/ for your platform, keeping these
versions together. Then install the remaining requirements.
The scripts automatically use CUDA when available. CPU also works; image
training can be slower and its baseline MLP needs substantial memory.

## Task 2: image classification

```bash
python img_classifier.py --n_epochs 5 --batch_size 8 --learning_rate 0.001
```

The baseline uses training and validation data during training and evaluates
the test set at the end. It saves img_model.pth. Implement the required wandb
logging, experiments, grayscale modifications, optimizer experiment and new
architecture yourself. Do not select model settings using test results.

Image splits: 540 training images, 116 validation images, 115 test images.

## Task 3: text classification

Start with the runnable baseline, without truncation or padding:

```bash
python txt_classifier.py --batch_size 1 --max_len -1 --n_epochs 5
```

It uses SGD with learning rate 5 and saves txt_model.pth. The vocabulary is
built from training text only. Tokenization uses lowercase words and
punctuation, without torchtext or external downloads.

Student implementation points are marked TODO:

1. Task 3.1: implement the raw article length histogram. Use training articles,
   count tokens before truncation/padding, label axes, and identify the split
   and length unit in your report. The --length_histogram flag intentionally
   raises NotImplementedError until you implement this part.
2. Task 3.2: implement TruncateToMaxLen and PadToMaxLen. A positive --max_len
   intentionally raises NotImplementedError until these are implemented.
3. Task 3.3: compare the supplied SGD baseline with Adam at its default learning
   rate; retain Adam for subsequent experiments.
4. Task 3.4: implement your own LSTM using nn.Embedding and pooling as specified.
5. Task 3.5: examine the paired real/generated texts and discuss your results.

After completing the relevant TODOs, run the commands from the PDF:

```bash
python txt_classifier.py --batch_size 1 --max_len -1 --length_histogram
python txt_classifier.py --batch_size 1 --max_len -1
python txt_classifier.py --batch_size 1 --max_len 1024
python txt_classifier.py --batch_size 32 --max_len 1024
python txt_classifier.py --batch_size 32 --max_len -1
```

The collator stacks the sequences produced by the transforms. It does not
silently pad variable-length batches. Keep this behaviour for Task 3.2.
Underscore and hyphen forms of --batch_size and --max_len are accepted.
Keep epoch count, seed, device and dataset constant for runtime comparisons.

## Text data

| Split | Articles | Real | Fake/generated |
|---|---:|---:|---:|
| Train | 358 | 179 | 179 |
| Validation | 76 | 38 | 38 |
| Test | 74 | 37 | 37 |

Columns: article, source, label, label_idx. Label mapping: real=0, fake=1.
The first line of article is the title. Each real/generated pair is adjacent
and remains entirely within one split. The cleaned split is approximately
70% / 15% / 15%. Do not re-split the supplied CSVs.

Here, real/fake are the original dataset's source labels: published text
versus generated text. They are not independent factual verification of
every claim. The source column contains a URL for published articles and a
generation prompt for generated articles. Only article is a model input;
source, label and label_idx must never be included as input features.

## Submission

Follow Assignment_1.pdf. Submit your own .py implementation and report,
maintain the required private GitHub repository, and provide the required
checkpoint and AI-conversation links. Do not include any supplied scaffold
as evidence of code you independently implemented.
