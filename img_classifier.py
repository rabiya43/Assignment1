# Adapted from CMU Fall 2024 public PyTorch Primer handout.
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import argparse
from pathlib import Path
import wandb

img_size = (256,256)
num_labels = 3
label_names = ['parrot', 'narwhal', 'axolotl']  # index = label_idx
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class CsvImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.data_root = Path(csv_file).resolve().parent
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if idx >= self.__len__(): raise IndexError()
        img_name = self.data_frame.loc[idx, "image"]
        parts = Path(img_name).parts
        relative = Path(*parts[parts.index("data") + 1:]) if "data" in parts else Path(img_name)
        with Image.open(self.data_root / relative) as im:
            image = im.convert("RGB")
        label = self.data_frame.loc[idx, "label_idx"]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_norm_stats(grayscale):
    if grayscale:
        # gray = 0.299R + 0.587G + 0.114B, so mix the imagenet stats the same way
        weights = [0.299, 0.587, 0.114]
        mean = sum(w * m for w, m in zip(weights, normalize_mean))
        std = sum(w * s for w, s in zip(weights, normalize_std))
        return [mean], [std]
    return normalize_mean, normalize_std

def get_data(batch_size, data_dir, grayscale=False):
    mean, std = get_norm_stats(grayscale)
    transforms = [
        T.ToTensor(),
        T.Resize(min(img_size[0], img_size[1]), antialias=True),  # Resize the smallest side to 256 pixels
        T.CenterCrop(img_size),  # Center crop to 256x256
    ]
    if grayscale:
        transforms.append(T.Grayscale(num_output_channels=1))  # Task 2.4
    transforms.append(T.Normalize(mean=mean, std=std))  # Normalize each color dimension
    transform_img = T.Compose(transforms)
    train_data = CsvImageDataset(
        csv_file=Path(data_dir) / 'img_train.csv',
        transform=transform_img,
    )
    test_data = CsvImageDataset(
        csv_file=Path(data_dir) / 'img_test.csv',
        transform=transform_img,
    )

    val_data = CsvImageDataset(Path(data_dir) / "img_val.csv", transform_img)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in train_dataloader:
        print(f"Shape of X [B, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return train_dataloader, val_dataloader, test_dataloader

class NeuralNetwork(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.flatten = nn.Flatten()
        # First layer input size must be the dimension of the image
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(img_size[0] * img_size[1] * in_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# layer norm over the channels at each pixel (nn.LayerNorm wants channels last)
class ChannelLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

# Task 2.6 model from the computation graph
class ConvNetwork(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=4, stride=4),  # -> 128 x 64 x 64
            ChannelLayerNorm(128),
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),  # depthwise conv
            ChannelLayerNorm(128),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.AvgPool2d(kernel_size=2),  # -> 128 x 32 x 32
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, num_labels),
        )

    def forward(self, x):
        return self.layers(x)

def train_one_epoch(dataloader, model, loss_fn, optimizer, t, examples_seen):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        examples_seen += len(y)
        wandb.log({"batch_loss": loss, "examples_seen": examples_seen})
        if batch % 10 == 0:
            print(f"Train loss = {loss:>7f}  [{examples_seen - t * size:>5d}/{size:>5d}]")
    return examples_seen

def caption_first_batch(X, y, pred, grayscale):
    # Task 2.5 - undo the normalization and caption as "predicted / true"
    mean, std = get_norm_stats(grayscale)
    denormalize = T.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std])
    images = []
    for i in range(len(y)):
        Xi_denorm = torch.clamp(denormalize(X[i].cpu()), min=0, max=1)
        caption = f"{label_names[pred[i]]} / {label_names[y[i]]}"
        # wandb 0.31 casts float tensors straight to uint8 (everything turns black), so scale to 0-255 first
        pixels = (Xi_denorm.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        if grayscale:
            images.append(wandb.Image(pixels[:, :, 0], caption=caption, mode="L"))
        else:
            images.append(wandb.Image(pixels, caption=caption, mode="RGB"))
    return images

def evaluate(dataloader, dataname, model, loss_fn, log_images=False, grayscale=False):
    size = len(dataloader.dataset)
    model.eval()
    avg_loss, correct = 0, 0
    images = None
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            avg_loss += loss_fn(pred, y).item() * len(y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            if log_images and batch == 0:
                images = caption_first_batch(X, y.tolist(), pred.argmax(1).tolist(), grayscale)
    avg_loss /= size
    correct /= size
    print(f"{dataname} accuracy = {(100*correct):>0.1f}%, {dataname} avg loss = {avg_loss:>8f}")
    return avg_loss, correct, images

def build_model(model_name, in_channels):
    return (NeuralNetwork if model_name == 'base' else ConvNetwork)(in_channels)

def main(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    print(f"Using {device} device")
    train_dataloader, val_dataloader, test_dataloader = get_data(args.batch_size, args.data_dir, args.grayscale)
    in_channels = 1 if args.grayscale else 3

    model = build_model(args.model, in_channels).to(device)
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    loss_fn = nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    run = wandb.init(project=args.wandb_project, name=args.run_name, mode=args.wandb_mode,
                     config={**vars(args), 'num_params': num_params, 'device': device})
    wandb.define_metric("epoch")
    wandb.define_metric("train_*", step_metric="epoch")
    wandb.define_metric("val_*", step_metric="epoch")

    examples_seen = 0
    for t in range(args.n_epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        last_epoch = t == args.n_epochs - 1
        examples_seen = train_one_epoch(train_dataloader, model, loss_fn, optimizer, t, examples_seen)
        train_loss, train_acc, train_images = evaluate(train_dataloader, "Train", model, loss_fn,
                                                       last_epoch, args.grayscale)
        val_loss, val_acc, val_images = evaluate(val_dataloader, "Validation", model, loss_fn,
                                                 last_epoch, args.grayscale)
        metrics = {"epoch": t + 1, "train_loss": train_loss, "train_acc": train_acc,
                   "val_loss": val_loss, "val_acc": val_acc}
        if last_epoch:
            metrics.update({"train_first_batch": train_images, "val_first_batch": val_images})
        wandb.log(metrics)
    test_loss, test_acc, test_images = evaluate(test_dataloader, "Test", model, loss_fn,
                                                True, args.grayscale)
    wandb.log({"test_first_batch": test_images})
    run.summary.update({"test_loss": test_loss, "test_acc": test_acc,
                        "final_train_acc": train_acc, "final_val_acc": val_acc})
    print("Done!")

    # Save the model
    torch.save(model.state_dict(), args.output)
    print(f"Saved PyTorch Model State to {args.output}")

    # Load the model (just for the sake of example)
    model = build_model(args.model, in_channels).to(device)
    model.load_state_dict(torch.load(args.output, map_location=device, weights_only=True))
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Image Classifier')
    parser.add_argument('--n_epochs', default=5, help='The number of training epochs', type=int)
    parser.add_argument('--batch_size', default=8, help='The batch size', type=int)
    parser.add_argument('--learning_rate', default=1e-3, help='The learning rate for the optimizer', type=float)
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('--model', choices=['base', 'new'], default='base',
                        help='base = original MLP, new = Task 2.6 convolutional model')
    parser.add_argument('--grayscale', action='store_true', help='Task 2.4: single-channel grayscale input')

    parser.add_argument('--data_dir', '--data-dir', default='data')
    parser.add_argument('--output', default='img_model.pth')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb_project', default='img_classifier')
    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online')
    parser.add_argument('--run_name', default=None, help='wandb run name')
    args = parser.parse_args()

    main(args)
