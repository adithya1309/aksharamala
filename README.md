# Malayalam Handwritten Character Recognition

A CNN that recognises 85 handwritten Malayalam characters, trained in PyTorch and
run entirely in the browser via ONNX Runtime Web. Draw a character on the canvas,
submit, and the model tells you what you wrote and how confident it is.

**94.51% test accuracy** on handwriting from 18 writers the model never trained on.

## How it works

```
CSV -> data.py -> model.py -> train.py -> model.pt -> export.py -> model.onnx -> app/
```

No backend. The static site ships `model.onnx` (2.8 MB) and `labels.json` to the
browser, and every prediction runs on the visitor's machine: a drawing never
leaves it.

## Running it

Training:

```
python train.py     # writes model.pt, curves.png, confusion.png
python export.py    # writes model.onnx and labels.json, verified against PyTorch
```

The app:

```
cp model.onnx labels.json app/
cd app && python -m http.server 8000
```

Then open <http://127.0.0.1:8000>. Some HTTP server is required, because opening
`index.html` as a `file://` URL fails: the WASM runtime and the model are
loaded with `fetch()`.

## Results

| | accuracy |
|---|---|
| train | 99.1% |
| validation | 98.1% |
| **test (held-out writers)** | **94.5%** |
| test, top-5 | 99.6% |

The validation-to-test gap is the number worth watching. Both are unseen samples;
the difference is that test writers never appear in training, so that gap measures
how well the model generalises across *people* rather than across samples. Quote
the test figure.

Getting there: baseline 88.3% → augmentation 92.7% → dropout 94.3-95.2%.
Run-to-run variation with a fixed configuration is about ±1 point, so treat
differences smaller than that as noise.

Remaining errors concentrate in visually similar character pairs, which is a
property of the script rather than a bug. Top-5 accuracy of 99.6% says the model
nearly always has the right answer in mind, it just cannot always pick between
near-identical shapes.

## Preprocessing must match on both sides

Training images are cropped to the ink bounding box, stretched to 32×32 without
preserving aspect ratio, and binarised. `app/app.js` does the same to the canvas
before inference. Changing one side without the other is the most common way this
kind of app scores well in the notebook and feels broken in the browser.

One canvas-specific adjustment: scanned ballpoint strokes are thicker than a
downsampled mouse line, so if ink coverage falls below `MIN_INK` the image is
dilated once. That constant and `PEN` are the knobs to tune if predictions feel
wrong on real drawings.

## Dataset

Amrita_MalCharDb. 85 character classes, 29,302 images from 77 native Malayalam
writers, scanned ballpoint on paper, segmented with active contour models and
binarised to 32×32.

> Manjusha K, Malayalam Character Image Database (Amrita_MalCharDb)

K. Manjusha, M. Anand Kumar, K.P. Soman, "On developing handwritten character
image database for Malayalam language script", *Engineering Science and
Technology, an International Journal* 22 (2019) 637-645.
<https://doi.org/10.1016/j.jestch.2018.10.011>

The dataset is **not** in this repository. It is available from the authors on
request (see the paper). Place the three CSVs in `handwritten/handwritten/`.

Two things the CSVs do not tell you, both of which cost a debugging session:

- Pixels are vectorised **column-wise**. A row-major `reshape` loads every image
  transposed.
- The class label order is **not** the order of Table 1 in the paper. The mapping
  in `glyphs.py` was recovered by matching that paper's Table 3 per-class counts
  against the CSVs; 16 indices are anchored exactly, and the conjunct block's
  interior follows Table 1 order without individual verification.

## Known limitations

- The writer pool skews heavily female (60 of 77) and is adults-only with at least
  graduate education, so generalisation to children's or less practised handwriting
  is untested.
- Training data is scanned pen-on-paper; the app feeds it uniform digital strokes.
  The preprocessing above narrows that gap but does not close it.
- 26% of the model's mistakes are made above 0.9 confidence. High confidence is not
  a guarantee, and the UI should not present it as one.
