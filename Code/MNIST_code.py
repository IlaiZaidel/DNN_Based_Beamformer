
# 25/12/2025
# Bar Ilan University
# Advisor: Ilai Zaidel

"""
MNIST ConvNet exercise 

What students should do:
1) Fill in the missing layers in SimpleConvNet (marked "TODO").
2) Choose LR / epochs / batch size.
3) Run training and inspect the plots + final test metrics.

Dependencies:
  pip install torch torchvision matplotlib
"""

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt


# ----------------
# Config
# ----------------
@dataclass
class CFG:
    batch_size: int = 64
    lr: float = 0.05              # students can change
    epochs: int = 5               # students can change
    momentum: float = 0.9
    weight_decay: float = 0.0
    num_workers: int = 2
    seed: int = 0


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------
# Data loading (from internet)
# ----------------
def get_mnist_loaders(cfg: CFG):
    """
    Downloads MNIST automatically via torchvision (if not already present).
    CPU-friendly transforms.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
    ])

    train_ds = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    return train_loader, test_loader


# ----------------
# Network configuration and setup
# ----------------
class SimpleConvNet(nn.Module):
    """
    Students: fill in the TODO parts (conv layers / pooling / dropout etc.)

    You already get:
    - First layer (a Conv2d) so you know how to start
    - Last layer (a Linear to 10 classes) so you know how to end
    """

    def __init__(self):
        super().__init__()

        # FIRST LAYER (given)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=14, kernel_size=3, padding=1)
        # TODO: add more layers here

        
        # IMPORTANT: if you use pooling twice, the spatial size 28x28 -> 14x14 -> 7x7

        # LAST LAYER (given) — you may need to change "in_features" to match your design!
        self.fc_last = nn.Linear(in_features=16 * 28 * 28, out_features=10)

    def forward(self, x):
        # x: [B, 1, 28, 28] - Original shape

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.tanh(x)
        # TODO: add the rest of your forward pass here



        x = torch.flatten(x, start_dim=1)  # [B, features]
        logits = self.fc_last(x)           # [B, 10]
        return logits


# ----------------
# Training / Evaluation
# ----------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")  # sum so we can average later
        total_loss += loss.item()

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    wrong = total - correct
    return avg_loss, acc, correct, wrong, total


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    wrong = total - correct
    return avg_loss, acc, correct, wrong, total


# ----------------
# Plotting
# ----------------
def plot_curves(history):
    epochs = list(range(1, len(history["train_acc"]) + 1))

    # Accuracy plot
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train acc")
    plt.plot(epochs, history["test_acc"], label="Test acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.grid(True)

    # Loss plot
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["test_loss"], label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.grid(True)

    plt.show()


# ----------------
# Main
# ----------------
def main():
    cfg = CFG()
    set_seed(cfg.seed)

    device = torch.device("cpu")  # CPU-friendly by design

    train_loader, test_loader = get_mnist_loaders(cfg)

    model = SimpleConvNet().to(device)

    # Gradient descent optimizer (SGD). Students can switch to Adam if you want.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_correct": [],
        "train_wrong": [],
        "test_loss": [],
        "test_acc": [],
        "test_correct": [],
        "test_wrong": [],
    }

    print("Starting training on CPU...")
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        ep_t0 = time.time()

        train_loss, train_acc, train_correct, train_wrong, train_total = train_one_epoch(
            model, train_loader, optimizer, device
        )
        test_loss, test_acc, test_correct, test_wrong, test_total = evaluate(
            model, test_loader, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_correct"].append(train_correct)
        history["train_wrong"].append(train_wrong)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["test_correct"].append(test_correct)
        history["test_wrong"].append(test_wrong)

        ep_time = time.time() - ep_t0
        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train: loss={train_loss:.4f}, acc={train_acc*100:.2f}% "
            f"(correct={train_correct}, wrong={train_wrong}) | "
            f"test: loss={test_loss:.4f}, acc={test_acc*100:.2f}% "
            f"(correct={test_correct}, wrong={test_wrong}) | "
            f"time={ep_time:.1f}s"
        )

    total_time = time.time() - t0
    print(f"\nDone. Total time: {total_time:.1f}s")

    # Final summary (how many classified correctly vs not on test)
    print(
        f"Final TEST: acc={history['test_acc'][-1]*100:.2f}% | "
        f"correct={history['test_correct'][-1]} | wrong={history['test_wrong'][-1]}"
    )

    plot_curves(history)


if __name__ == "__main__":
    main()
