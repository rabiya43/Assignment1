# Adapted from CMU Fall 2024 public PyTorch Primer handout.
# Starter baseline only: complete Task 2 experiments yourself.
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import argparse
from pathlib import Path

img_size = (256,256)
num_labels = 3

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

def get_data(batch_size, data_dir):
    transform_img = T.Compose([
        T.ToTensor(), 
        T.Resize(min(img_size[0], img_size[1]), antialias=True),  # Resize the smallest side to 256 pixels
        T.CenterCrop(img_size),  # Center crop to 256x256
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize each color dimension
        ])
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
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # First layer input size must be the dimension of the image
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(img_size[0] * img_size[1] * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_one_epoch(dataloader, model, loss_fn, optimizer, t):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        current = (batch + 1) * dataloader.batch_size
        if batch % 10 == 0:
            print(f"Train loss = {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
def evaluate(dataloader, dataname, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    avg_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            avg_loss += loss_fn(pred, y).item() * len(y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    avg_loss /= size
    correct /= size
    print(f"{dataname} accuracy = {(100*correct):>0.1f}%, {dataname} avg loss = {avg_loss:>8f}")
    
def main(n_epochs, batch_size, learning_rate, data_dir, output, seed):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    print(f"Using {device} device")
    train_dataloader, val_dataloader, test_dataloader = get_data(batch_size, data_dir)
    
    model = NeuralNetwork().to(device)
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    for t in range(n_epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        train_one_epoch(train_dataloader, model, loss_fn, optimizer, t)
        evaluate(train_dataloader, "Train", model, loss_fn)
        evaluate(val_dataloader, "Validation", model, loss_fn)
    evaluate(test_dataloader, "Test", model, loss_fn)
    print("Done!")

    # Save the model
    torch.save(model.state_dict(), output)
    print(f"Saved PyTorch Model State to {output}")

    # Load the model (just for the sake of example)
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(output, map_location=device, weights_only=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Image Classifier')
    parser.add_argument('--n_epochs', default=5, help='The number of training epochs', type=int)
    parser.add_argument('--batch_size', default=8, help='The batch size', type=int)
    parser.add_argument('--learning_rate', default=1e-3, help='The learning rate for the optimizer', type=float)

    parser.add_argument('--data_dir', '--data-dir', default='data')
    parser.add_argument('--output', default='img_model.pth')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
        
    main(args.n_epochs, args.batch_size, args.learning_rate, args.data_dir, args.output, args.seed)