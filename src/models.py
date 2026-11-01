import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SEED = 0              # fixed randomness, so every run gives the same results
BATCH_SIZE = 64       # how many images the model studies at once
EPOCHS = 3            # how many times it goes through all 60,000 images
LEARNING_RATE = 1e-3  # how big each weight adjustment is
DATA_DIR = "data"                     # MNIST downloads here the first time
MODEL_PATH = "results/mnist_fp32.pt"  # where the trained weights are saved


class DigitClassifier(nn.Module):
    """A small fully connected network: 784 pixels -> 128 neurons -> 10 digit scores."""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()          # (1, 28, 28) image -> 784 numbers in a row
        self.hidden = nn.Linear(784, 128)    # 128 neurons, each with a weight per pixel
        self.relu = nn.ReLU()                # keep positive signals, zero out negative ones
        self.output = nn.Linear(128, 10)     # 10 scores, one per digit

    def forward(self, x):
        """Run images through the network and return 10 scores per image."""
        x = self.flatten(x)
        x = self.relu(self.hidden(x))
        return self.output(x)


def load_mnist():
    """Download MNIST if needed and return the training and test sets."""
    to_tensor = transforms.ToTensor()   # image -> grid of numbers from 0 to 1
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=to_tensor)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=to_tensor)
    return train_set, test_set


def train_one_epoch(model, train_set):
    """Study all 60,000 training images once, adjusting the weights as we go."""
    loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()                     # measures how wrong the scores are
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # adjusts the weights

    model.train()
    for images, labels in loader:
        scores = model(images)            # 1. answer a batch of 64 questions
        loss = loss_fn(scores, labels)    # 2. check the answers: how wrong were we?
        optimizer.zero_grad()             # 3. clear the previous batch's corrections
        loss.backward()                   # 4. work out how each weight contributed to the mistakes
        optimizer.step()                  # 5. nudge every weight to be a little less wrong


def evaluate(model, test_set):
    """Return the share of test images the model gets right (0.0 to 1.0)."""
    loader = DataLoader(test_set, batch_size=1000)
    model.eval()
    correct = 0
    with torch.no_grad():                  # just answering, no learning
        for images, labels in loader:
            predictions = model(images).argmax(dim=1)   # highest score = the guess
            correct += (predictions == labels).sum().item()
    return correct / len(test_set)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    train_set, test_set = load_mnist()
    model = DigitClassifier()

    print(f"accuracy before training: {evaluate(model, test_set):.2%}")
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, train_set)
        print(f"accuracy after epoch {epoch}:   {evaluate(model, test_set):.2%}")

    os.makedirs("results", exist_ok=True)          # create the folder if it's missing
    torch.save(model.state_dict(), MODEL_PATH)     # write all 101,770 weights to a file
    print(f"saved trained weights to {MODEL_PATH}")