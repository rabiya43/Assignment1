"""Task 3 baseline scaffold. Complete the marked student tasks yourself.

Data: CMU 10-423/10-623 Fall 2024 public PyTorch Primer handout.
Baseline architecture follows the PyTorch text classification tutorial.
This replacement scaffold avoids the obsolete torchtext binary dependency.
"""
import argparse
import csv
import re
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def tokenize(text):
    """Lowercase word/punctuation tokenizer; no downloads required."""
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


class Vocabulary:
    def __init__(self, texts):
        counts = Counter(token for text in texts for token in tokenize(text))
        self.stoi = {'<UNK>': 0, '<PAD>': 1}
        for token in sorted(counts, key=lambda t: (-counts[t], t)):
            if token not in self.stoi:
                self.stoi[token] = len(self.stoi)

    def __getitem__(self, token):
        return self.stoi.get(token, 0)

    def __len__(self):
        return len(self.stoi)


class CsvTextDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        with open(csv_file, encoding='utf-8', newline='') as f:
            self.rows = list(csv.DictReader(f))
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text = row['article']
        if self.transform is not None:
            text = self.transform(text)
        return text, int(row['label_idx'])


class CorpusInfo:
    def __init__(self, dataset):
        self.tokenizer = tokenize
        self.vocab = Vocabulary(row['article'] for row in dataset.rows)
        self.oov_token, self.pad_token = '<UNK>', '<PAD>'
        self.oov_idx, self.pad_idx = 0, 1
        self.vocab_size = len(self.vocab)
        self.num_labels = len({row['label_idx'] for row in dataset.rows})


class TextTransform:
    def __init__(self, tokenizer, vocab):
        self.tokenizer, self.vocab = tokenizer, vocab

    def __call__(self, text):
        return [self.vocab[token] for token in self.tokenizer(text)]


class TruncateToMaxLen:
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, tokens):
        # Keep only the first max_len tokens; shorter articles pass through unchanged.
        return tokens[:self.max_len]


class PadToMaxLen:
    def __init__(self, max_len, pad_idx):
        self.max_len, self.pad_idx = max_len, pad_idx

    def __call__(self, tokens):
        # Append pad_idx until the article is exactly max_len tokens long.
        return tokens + [self.pad_idx] * (self.max_len - len(tokens))


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, value):
        for transform in self.transforms:
            value = transform(value)
        return value


class PadSequence:
    """Stack sequences as supplied by the transforms (no automatic padding).

    Retains the batching behaviour needed for the experiment in Task 3.2.
    """
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        text = torch.stack([torch.tensor(x, dtype=torch.long) for x, _ in batch])
        labels = torch.tensor([y for _, y in batch], dtype=torch.long)
        return text, labels


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, pad_idx):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode='mean', padding_idx=pad_idx)
        self.fc = nn.Linear(embed_dim, num_class)
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)
        with torch.no_grad():
            self.embedding.weight[pad_idx].zero_()
        nn.init.uniform_(self.fc.weight, -0.5, 0.5)
        nn.init.zeros_(self.fc.bias)

    def forward(self, text):
        return self.fc(self.embedding(text))


def plot_length_histogram(dataset, tokenizer, figure_dir):
    """Task 3.1: histogram of raw (untruncated, unpadded) training article lengths in tokens."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lengths = {'real': [], 'fake': []}
    for row in dataset.rows:
        lengths[row['label']].append(len(tokenizer(row['article'])))
    all_lengths = lengths['real'] + lengths['fake']
    print(f'Train articles: {len(all_lengths)}  min={min(all_lengths)}  '
          f'median={sorted(all_lengths)[len(all_lengths) // 2]}  max={max(all_lengths)}  '
          f'longer than 1024: {sum(n > 1024 for n in all_lengths)}')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = range(0, max(all_lengths) + 50, 50)
    ax.hist([lengths['real'], lengths['fake']], bins=bins, stacked=True,
            label=['real', 'fake (LLM-generated)'], color=['#3b6ea8', '#e08a3c'], edgecolor='white')
    ax.axvline(1024, color='black', linestyle='--', linewidth=1, label='max_len = 1024')
    ax.set_xlabel('Article length (tokens, before truncation/padding)')
    ax.set_ylabel('Number of articles')
    ax.set_title('Raw article lengths, training split (n = %d)' % len(all_lengths))
    ax.legend()
    fig.tight_layout()
    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    out = Path(figure_dir) / 'txt_length_histogram.png'
    fig.savefig(out, dpi=150)
    print(f'Saved histogram to {out}')


def evaluate(loader, model, criterion, device):
    model.eval()
    correct = count = 0
    loss_sum = 0.0
    with torch.no_grad():
        for text, labels in loader:
            text, labels = text.to(device), labels.to(device)
            logits = model(text)
            loss_sum += criterion(logits, labels).item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
    return loss_sum / count, correct / count


def main():
    parser = argparse.ArgumentParser(description='Task 3 text baseline')
    parser.add_argument('--data_dir', '--data-dir', default='data')
    parser.add_argument('--batch_size', '--batch-size', type=int, default=1)
    parser.add_argument('--max_len', '--max-len', type=int, default=-1)
    parser.add_argument('--n_epochs', '--epochs', type=int, default=5)
    parser.add_argument('--learning_rate', '--learning-rate', type=float, default=5.0)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    parser.add_argument('--length_histogram', '--length-histogram', action='store_true')
    parser.add_argument('--output', default='txt_model.pth')
    parser.add_argument('--figure_dir', '--figure-dir', default='figures')
    args = parser.parse_args()
    if args.batch_size < 1 or args.n_epochs < 1 or args.max_len == 0 or args.max_len < -1:
        parser.error('Use positive batch size/epochs and max_len=-1 or a positive integer.')
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = Path(args.data_dir)
    raw_train = CsvTextDataset(data_dir / 'txt_train.csv')
    corpus_info = CorpusInfo(raw_train)
    if args.length_histogram:
        plot_length_histogram(raw_train, corpus_info.tokenizer, args.figure_dir)
        return
    transforms = [TextTransform(corpus_info.tokenizer, corpus_info.vocab)]
    if args.max_len != -1:
        transforms += [TruncateToMaxLen(args.max_len), PadToMaxLen(args.max_len, corpus_info.pad_idx)]
    transform = Compose(transforms)
    loaders = {}
    for split in ['train', 'val', 'test']:
        dataset = CsvTextDataset(data_dir / f'txt_{split}.csv', transform)
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size,
                                   shuffle=(split == 'train'),
                                   collate_fn=PadSequence(corpus_info.pad_idx))
    model = TextClassificationModel(corpus_info.vocab_size, args.embed_dim,
                                    corpus_info.num_labels, corpus_info.pad_idx).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    # TODO Task 3.3: switch to Adam with its default learning rate for the second run.
    # TODO Task 3.4: replace the baseline with your own LSTM classifier.
    start = time.perf_counter()
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        for text, labels in loaders['train']:
            text, labels = text.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(text), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
        train_loss, train_acc = evaluate(loaders['train'], model, criterion, device)
        val_loss, val_acc = evaluate(loaders['val'], model, criterion, device)
        print(f'Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} '
              f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}')
    test_loss, test_acc = evaluate(loaders['test'], model, criterion, device)
    if device == 'cuda':
        torch.cuda.synchronize()
    print(f'Test loss={test_loss:.4f} test accuracy={test_acc:.4f}')
    print(f'Training and evaluation runtime: {time.perf_counter() - start:.3f} seconds')
    torch.save({'model_state_dict': model.state_dict(), 'vocab': corpus_info.vocab.stoi,
                'pad_idx': corpus_info.pad_idx, 'args': vars(args)}, args.output)


if __name__ == '__main__':
    main()
