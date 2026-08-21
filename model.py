import torch
import torch.nn as nn

from data import NUM_CLASSES

learning_rate = 0.005

# 32x32 -> 16 -> 8 -> 4 after the three pools
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(128 * 4 * 4, 256),
    nn.ReLU(),
    nn.Linear(256, NUM_CLASSES),
)

# CrossEntropyLoss takes raw logits, so the net ends at the linear layer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    out = model(torch.zeros(8, 1, 32, 32))
    print(tuple(out.shape), sum(p.numel() for p in model.parameters()), "params")
