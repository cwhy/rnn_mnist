# rnn-mnist autoresearch

Autoregressive pixel prediction on MNIST using a stacked LSTM in JAX + Optax.
Follows the `program.md` experiment loop.

## Task

Each 28×28 MNIST image is treated as a sequence of 784 tokens (pixel values 0–255).
The model predicts pixel[t] given pixel[0..t-1].
Metric: **val_bpb** (bits per pixel on the test split — lower is better).

## Setup

```bash
# 1. Install dependencies (JAX CUDA + Optax)
uv sync

# 2. Download MNIST and write data shards to ~/.cache/autoresearch/
uv run prepare.py
```

## Running an experiment

```bash
uv run train.py > run.log 2>&1
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

## Files

| File | Role |
|------|------|
| `prepare.py` | **Fixed.** Data download, tokenizer, dataloader, `evaluate_bpb`. |
| `train.py` | **Modify this.** Model, optimiser, training loop. |
| `results.tsv` | Experiment log (untracked by git). |
| `run.log` | Last training run output (untracked by git). |

## Autoresearch

See `program.md` for the full experiment loop specification.
