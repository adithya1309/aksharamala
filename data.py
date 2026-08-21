from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import TensorDataset

DATA_DIR = Path(__file__).parent / "handwritten" / "handwritten"
NUM_CLASSES = 85


def split(name):
    raw = pd.read_csv(DATA_DIR / f"Handwritten_V2_{name}.csv", header=None).to_numpy("float32")
    x = torch.from_numpy(raw[:, 1:]).view(-1, 1, 32, 32)
    y = torch.from_numpy(raw[:, 0]).long() - 1
    return TensorDataset(x, y)


def train():
    return split("train")
