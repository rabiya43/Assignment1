"""Builds the report figures from the runs logged to Weights & Biases.

Usage: python make_figures.py [--entity ENTITY] [--figure_dir figures]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import wandb

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']  # categorical slots 1-4, fixed order
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID,
    'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2,
    'font.size': 10, 'legend.frameon': False,
})


def get_runs(api, entity, project):
    return {r.name: r for r in api.runs(f'{entity}/{project}')}


def epoch_history(run, keys):
    rows = [r for r in run.history(keys=['epoch'] + keys, samples=100000, pandas=False)
            if all(r.get(k) is not None for k in ['epoch'] + keys)]
    rows.sort(key=lambda r: r['epoch'])
    return {k: np.array([r[k] for r in rows]) for k in ['epoch'] + keys}


def batch_history(run):
    rows = [r for r in run.history(keys=['examples_seen', 'batch_loss'], samples=100000, pandas=False)
            if r.get('examples_seen') is not None and r.get('batch_loss') is not None]
    rows.sort(key=lambda r: r['examples_seen'])
    return np.array([r['examples_seen'] for r in rows]), np.array([r['batch_loss'] for r in rows])


def smooth(y, window):
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    return np.convolve(np.pad(y, (window - 1, 0), mode='edge'), kernel, mode='valid')


def train_val_plot(runs, labels, metric, ylabel, title, out):
    """Train (solid) and validation (dashed) curves per run on one axis; colour = run."""
    fig, ax = plt.subplots(figsize=(7, 4.2 + 0.25 * len(runs)))
    for i, (name, label) in enumerate(zip(runs, labels)):
        h = epoch_history(runs[name], [f'train_{metric}', f'val_{metric}'])
        ax.plot(h['epoch'], h[f'train_{metric}'], color=SERIES[i], marker='o', ms=4,
                label=f'{label}: train')
        ax.plot(h['epoch'], h[f'val_{metric}'], color=SERIES[i], marker='s', ms=4, ls='--',
                label=f'{label}: validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if metric == 'acc':
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print('wrote', out)


def batch_loss_plot(runs, labels, title, out, window=1, ymax=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (name, label) in enumerate(zip(runs, labels)):
        x, y = batch_history(runs[name])
        if window > 1:
            ax.plot(x, y, color=SERIES[i], lw=0.6, alpha=0.25)
        ax.plot(x, smooth(y, window), color=SERIES[i],
                label=label + (f' ({window}-batch moving average)' if window > 1 else ''))
    ax.set_xlabel('Examples seen by the optimizer')
    ax.set_ylabel('Training loss (mean over batch)')
    if ymax is not None:
        ax.set_ylim(0, ymax)
        title += f' (y-axis cut at {ymax})'
    ax.set_title(title, loc='left')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print('wrote', out)


def image_grid(run, key, out, media_dir):
    """Task 2.5: grid of the logged first test batch with '<predicted> / <true>' captions."""
    value = run.history(keys=[key], pandas=False)[0][key]
    if isinstance(value, str):
        value = json.loads(value)
    files, captions = value['filenames'], value['captions']
    fig, axes = plt.subplots(2, (len(files) + 1) // 2, figsize=(2.2 * ((len(files) + 1) // 2), 5))
    for ax, fn, cap in zip(axes.flat, files, captions):
        path = run.file(fn).download(root=media_dir, replace=True).name
        ax.imshow(plt.imread(path))
        pred, true = cap.split(' / ')
        ax.set_title(cap, fontsize=9, color=INK if pred == true else '#d03b3b',
                     fontweight='normal' if pred == true else 'bold')
        ax.axis('off')
    for ax in list(axes.flat)[len(files):]:
        ax.axis('off')
    fig.suptitle('First test batch: predicted / true (bold red = error)', x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print('wrote', out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity', default=None)
    parser.add_argument('--figure_dir', default='figures')
    args = parser.parse_args()
    api = wandb.Api()
    entity = args.entity or api.default_entity
    out = Path(args.figure_dir)
    out.mkdir(exist_ok=True)

    txt = get_runs(api, entity, 'txt_classifier')
    batch_loss_plot({k: txt[k] for k in ['sgd-bs1-maxlen-none', 'adam-bs1-maxlen-none']},
                    ['SGD (lr 5)', 'Adam (lr 0.001)'],
                    'Text: per-batch training loss, batch size 1', out / 'txt_batch_loss_sgd_vs_adam.png',
                    window=50, ymax=3)
    train_val_plot({k: txt[k] for k in ['sgd-bs1-maxlen-none', 'adam-bs1-maxlen-none']},
                   ['SGD', 'Adam'], 'acc', 'Accuracy',
                   'Text: SGD vs Adam (batch 1, no truncation)', out / 'txt_sgd_vs_adam_acc.png')
    train_val_plot({k: txt[k] for k in ['sgd-bs32-maxlen-1024', 'adam-bs32-maxlen-1024']},
                   ['SGD', 'Adam'], 'acc', 'Accuracy',
                   'Text: SGD vs Adam (batch 32, max_len 1024)', out / 'txt_sgd_vs_adam_bs32_acc.png')
    lstm_runs = [k for k in ['baseline-adam-10ep', 'lstm-10ep', 'lstm-pad-ablation-10ep'] if k in txt]
    lstm_labels = {'baseline-adam-10ep': 'Baseline', 'lstm-10ep': 'LSTM',
                   'lstm-pad-ablation-10ep': 'LSTM reading <PAD>'}
    for metric, ylabel in [('acc', 'Accuracy'), ('loss', 'Cross-entropy loss')]:
        train_val_plot({k: txt[k] for k in lstm_runs}, [lstm_labels[k] for k in lstm_runs],
                       metric, ylabel, 'Text: baseline vs LSTM (Adam, batch 32, max_len 1024)',
                       out / f'txt_baseline_vs_lstm_{metric}.png')

    img = get_runs(api, entity, 'img_classifier')
    batch_loss_plot({'neural-the-narwhal': img['neural-the-narwhal']}, ['neural-the-narwhal'],
                    'Image 2.1: per-batch training loss', out / 'img_2_1_batch_loss.png')
    train_val_plot({'neural-the-narwhal': img['neural-the-narwhal']}, ['neural-the-narwhal'],
                   'loss', 'Cross-entropy loss', 'Image 2.2: train vs validation loss',
                   out / 'img_2_2_loss.png')
    train_val_plot({'neural-the-narwhal': img['neural-the-narwhal']}, ['neural-the-narwhal'],
                   'acc', 'Accuracy', 'Image 2.3: train vs validation accuracy', out / 'img_2_3_acc.png')
    train_val_plot({k: img[k] for k in ['neural-the-narwhal', 'grayscale-base']}, ['Colour', 'Grayscale'],
                   'acc', 'Accuracy', 'Image 2.4: colour vs grayscale input', out / 'img_2_4_gray_acc.png')
    if 'new-model' in img and img['new-model'].state == 'finished':
        train_val_plot({k: img[k] for k in ['base-model', 'new-model']}, ['base-model', 'new-model'],
                       'acc', 'Accuracy', 'Image 2.6: original MLP vs new CNN', out / 'img_2_6_acc.png')
    image_grid(img['neural-the-narwhal'], 'test_first_batch', out / 'img_2_5_test_batch.png',
               out / 'wandb_media')


if __name__ == '__main__':
    main()
