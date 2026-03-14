"""prepare.py — Fixed. Do not modify.

Downloads MNIST, creates data shards at ~/.cache/autoresearch/,
provides the tokenizer, dataloader, and the evaluation function.

Constants
---------
SEQ_LEN       : 784   pixels per image (28×28)
VOCAB_SIZE    : 256   pixel values 0–255
TRAIN_SECONDS : 300   five-minute wall-clock training budget
CACHE_DIR     : str   path where shards are stored
"""

import gzip
import math
import os
import struct
import urllib.request
from pathlib import Path
from typing import Iterator

import numpy as np
import jax
import jax.numpy as jnp

# ── Fixed constants ────────────────────────────────────────────────────────────

SEQ_LEN       = 784   # 28 × 28 pixels per MNIST image
VOCAB_SIZE    = 256   # pixel values 0–255
TRAIN_SECONDS = 300   # five-minute wall-clock training budget
CACHE_DIR     = os.path.expanduser("~/.cache/autoresearch")

# ── MNIST download + parse ─────────────────────────────────────────────────────

_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
}


def _fetch_gz(url: str, dest: str) -> None:
    print(f"Downloading {url} ...", flush=True)
    urllib.request.urlretrieve(url, dest)


def _parse_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)


def _load_mnist_raw(cache_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (train, test) arrays of shape [N, 784] uint8."""
    os.makedirs(cache_dir, exist_ok=True)
    arrays = {}
    for key, filename in _FILES.items():
        npy = os.path.join(cache_dir, key + ".npy")
        if os.path.exists(npy):
            arrays[key] = np.load(npy)
            continue
        gz = os.path.join(cache_dir, filename)
        if not os.path.exists(gz):
            _fetch_gz(_BASE_URL + filename, gz)
        arrays[key] = _parse_images(gz)
        np.save(npy, arrays[key])
        print(f"  saved {npy}  shape={arrays[key].shape}", flush=True)
    return arrays["train_images"], arrays["test_images"]


def prepare(cache_dir: str = CACHE_DIR) -> None:
    """Download MNIST and write data shards. Safe to call multiple times."""
    train, test = _load_mnist_raw(cache_dir)
    for name, arr in [("train_shard", train), ("val_shard", test)]:
        path = os.path.join(cache_dir, name + ".npy")
        if not os.path.exists(path):
            np.save(path, arr)
            print(f"  saved {path}  shape={arr.shape}", flush=True)
    print("Data ready.", flush=True)


# ── Tokenizer ──────────────────────────────────────────────────────────────────
# Pixel values are their own token ids — no lookup table needed.

def encode(pixels: np.ndarray) -> np.ndarray:
    """uint8 pixels → int32 token ids (identity mapping)."""
    return pixels.astype(np.int32)


def decode(tokens: np.ndarray) -> np.ndarray:
    """int32 token ids → uint8 pixels (identity mapping)."""
    return np.clip(tokens, 0, 255).astype(np.uint8)


# ── Data loading ───────────────────────────────────────────────────────────────

_cache: dict[str, np.ndarray] = {}


def _shard(split: str) -> np.ndarray:
    key = "train_shard" if split == "train" else "val_shard"
    if key not in _cache:
        path = os.path.join(CACHE_DIR, key + ".npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found — run:  uv run prepare.py"
            )
        _cache[key] = np.load(path)
    return _cache[key]


def make_batches(
    split: str,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[tuple[jax.Array, jax.Array]]:
    """Yield (inputs, targets) JAX arrays of shape [batch_size, SEQ_LEN].

    The task is autoregressive pixel prediction:
      inputs [b, t]  = pixel[t-1]  (0 for t=0, acts as start-of-image token)
      targets[b, t]  = pixel[t]
    """
    data = _shard(split)
    n = len(data)
    idx = rng.permutation(n)
    for start in range(0, n - batch_size + 1, batch_size):
        batch = encode(data[idx[start : start + batch_size]])  # [B, 784] int32
        # Shift right: prepend a zero column, drop the last column
        start_col = np.zeros((batch_size, 1), dtype=np.int32)
        inputs  = np.concatenate([start_col, batch[:, :-1]], axis=1)
        targets = batch
        yield jnp.array(inputs), jnp.array(targets)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_bpb(
    model_apply,          # (params, inputs[B,T]) -> logits[B,T,VOCAB_SIZE]
    params,
    batch_size: int = 128,
) -> float:
    """Bits per byte (= bits per pixel) on the validation split. Lower is better."""
    rng = np.random.default_rng(0)
    total_nll    = 0.0
    total_tokens = 0

    for inputs, targets in make_batches("val", batch_size, rng):
        logits   = model_apply(params, inputs)           # [B, T, V]
        B, T, V  = logits.shape
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # [B, T, V]
        # Gather log-prob of the correct pixel at every position
        flat_lp  = log_probs.reshape(-1, V)              # [B*T, V]
        flat_tgt = targets.reshape(-1)                   # [B*T]
        nll = -flat_lp[jnp.arange(B * T), flat_tgt].sum()
        total_nll    += float(nll)
        total_tokens += B * T

    return (total_nll / total_tokens) / math.log(2)


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prepare()
