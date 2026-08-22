from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torchvision.transforms import v2

DATA_DIR = Path(__file__).parent / "handwritten" / "handwritten"
NUM_CLASSES = 85

degrees = 8            
translate = 0.08
scale = (0.9, 1.1)
shear = 5
elastic_alpha = 20.0   
elastic_sigma = 4.0    
stroke_p = 0.25       

def split(name):
    raw = pd.read_csv(DATA_DIR / f"Handwritten_V2_{name}.csv", header=None).to_numpy("float32")
    x = torch.from_numpy(raw[:, 1:]).view(-1, 1, 32, 32).transpose(2, 3)
    y = torch.from_numpy(raw[:, 0]).long() - 1 
    return TensorDataset(x, y)


def train():
    return split("train")

augment = v2.Compose([
    v2.RandomAffine(degrees=degrees, translate=(translate, translate), scale=scale, shear=shear),
    v2.ElasticTransform(alpha=elastic_alpha, sigma=elastic_sigma),
    v2.RandomApply([v2.Lambda(lambda t: F.max_pool2d(t, 3, 1, 1))], p=stroke_p),
    v2.RandomApply([v2.Lambda(lambda t: -F.max_pool2d(-t, 3, 1, 1))], p=stroke_p),
    v2.Lambda(lambda t: (t > 0.5).float()), 
])



