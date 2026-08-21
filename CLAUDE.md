# CLAUDE.md

Malayalam handwritten character recognition: PyTorch CNN -> ONNX -> browser-only
demo app. Portfolio project. Full plan in `malayalam_cnn_webapp_plan.txt` — read it
before making architecture decisions.

## Code style

- No comments. No docstrings. Names carry the meaning.
- No unnecessary functions, methods, or classes. One helper with one caller is not a
  helper — inline it.
- Simple methods only: short, single-purpose, no clever indirection.
- No abstractions until there are two real users of them. No config for constants,
  no interface for one implementation, no factory, no wrapper layers.
- Robustness is not optional: validate at trust boundaries (file loads, user input,
  model I/O), assert invariants that would otherwise fail silently, and let errors
  surface loudly rather than be swallowed.
- Stdlib and already-installed deps first. No new dependency for what a few lines do.

## Project facts

- Dataset: Amrita_MalCharDb, 85 classes, 32x32 pre-binarized, CSV rows of
  `label + 1024 pixels`. CSV labels are 1-based; code uses 0-84.
- Split is writer-independent and predefined (`train`/`valid`/`test`). Never
  reshuffle across it.
- `handwritten/` and `*.zip` are third-party data — never commit them.
- `data.py` is the only loader. Datasets come from `data.split(name)`.

## Invariants to preserve

- Train/serve preprocessing must match exactly: crop to ink bbox, resize 32x32,
  binarize. Any change to one side changes the other.
- `labels.json` class ordering is generated from the training class ordering — never
  hand-written.
- ONNX export must be verified against PyTorch outputs on real test samples in eval
  mode before it is trusted.
