import json

import numpy as np
import onnxruntime
import torch
from torch.utils.data import DataLoader

from data import split, NUM_CLASSES
from glyphs import CHARACTERS
from model import model

model.load_state_dict(torch.load("model.pt"))
model.eval()  

torch.onnx.export(
    model,
    torch.zeros(1, 1, 32, 32),
    "model.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    verbose=False,  
    external_data=False,  
)


assert len(CHARACTERS) == NUM_CLASSES
COMBINING = "\u0D4D\u0D3E\u0D3F\u0D40\u0D41\u0D42\u0D44\u0D46\u0D47\u0D4C\u0D02"
labels = [
    {
        "index": i,
        "char": char,
        "transliteration": name,
        
        "display": "\u25CC" + char if char[0] in COMBINING else char,
    }
    for i, (char, name) in enumerate(CHARACTERS)
]
with open("labels.json", "w", encoding="utf-8") as f:
    json.dump(labels, f, indent=1, ensure_ascii=False)


ds = split("test")
x, y = next(iter(DataLoader(ds, batch_size=256, shuffle=True)))
with torch.no_grad():
    torch_out = model(x).numpy()
onnx_out = onnxruntime.InferenceSession("model.onnx").run(None, {"input": x.numpy()})[0]

print("max abs diff", float(np.abs(torch_out - onnx_out).max()))
print("argmax agreement", float((torch_out.argmax(1) == onnx_out.argmax(1)).mean()))
print("onnx accuracy on this batch", float((onnx_out.argmax(1) == y.numpy()).mean()))
assert np.allclose(torch_out, onnx_out, atol=1e-5), "ONNX output drifted from PyTorch"
assert (torch_out.argmax(1) == onnx_out.argmax(1)).all(), "ONNX predicts a different class"
print("verified")
