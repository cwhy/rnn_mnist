"""train.py — Modify this file to experiment.

Model architecture, optimizer, and training loop for autoregressive
pixel prediction on MNIST. The goal is to minimise val_bpb.

Constraints
-----------
- Only this file may be edited.
- prepare.py is the fixed evaluation harness.
- No new packages; only what is in pyproject.toml.
"""

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from prepare import (
    VOCAB_SIZE, SEQ_LEN, TRAIN_SECONDS,
    make_batches, evaluate_bpb,
)

# ── Hyperparameters ────────────────────────────────────────────────────────────

BATCH_SIZE = 256
D_EMBED    = 256   # token embedding dimension
D_MODEL    = 256   # LSTM hidden size
N_LAYERS   = 2     # stacked LSTM layers
LR         = 3e-3
WEIGHT_DECAY = 1e-4

# ── Parameter initialisation ───────────────────────────────────────────────────

def init_embed(key):
    # Small normal init keeps initial logit variance reasonable
    return jax.random.normal(key, (VOCAB_SIZE, D_EMBED)) * 0.02


def init_lstm_cell(key, in_dim: int, hidden: int) -> dict:
    k1, k2 = jax.random.split(key)
    scale = (1.0 / hidden) ** 0.5
    # forget-gate bias initialised to 1 for better gradient flow
    bias = jnp.zeros(4 * hidden).at[hidden : 2 * hidden].set(1.0)
    return {
        "Wih": jax.random.uniform(k1, (4 * hidden, in_dim),   minval=-scale, maxval=scale),
        "Whh": jax.random.uniform(k2, (4 * hidden, hidden),   minval=-scale, maxval=scale),
        "b":   bias,
    }


def init_head(key) -> dict:
    scale = (2.0 / D_MODEL) ** 0.5
    return {
        "W": jax.random.normal(key, (D_MODEL, VOCAB_SIZE)) * scale,
        "b": jnp.zeros(VOCAB_SIZE),
    }


def init_params(key) -> dict:
    keys = jax.random.split(key, 2 + N_LAYERS)
    lstm = [
        init_lstm_cell(keys[i], D_EMBED if i == 0 else D_MODEL, D_MODEL)
        for i in range(N_LAYERS)
    ]
    return {"embed": init_embed(keys[-2]), "lstm": lstm, "head": init_head(keys[-1])}


# ── Model (pure functions) ─────────────────────────────────────────────────────

def lstm_cell(p, h, c, x):
    """Single LSTM step → (h_new, c_new).  All shapes: [D_MODEL]."""
    gates = p["Wih"] @ x + p["Whh"] @ h + p["b"]   # [4*D]
    i, f, g, o = jnp.split(gates, 4)
    c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
    h = jax.nn.sigmoid(o) * jnp.tanh(c)
    return h, c


def lstm_sequence(p, xs):
    """Run one LSTM layer over xs [T, D_in] → ys [T, D_MODEL]."""
    zeros = jnp.zeros(D_MODEL)
    def step(carry, x):
        h, c = carry
        h, c = lstm_cell(p, h, c, x)
        return (h, c), h
    _, ys = jax.lax.scan(step, (zeros, zeros), xs)
    return ys


def forward_single(params, xi):
    """xi: [T] int32 tokens → logits: [T, VOCAB_SIZE]."""
    x = params["embed"][xi]            # [T, D_EMBED]
    for cell_p in params["lstm"]:
        x = lstm_sequence(cell_p, x)  # [T, D_MODEL]
    return x @ params["head"]["W"] + params["head"]["b"]  # [T, V]


def model_apply(params, inputs):
    """inputs: [B, T] int32 → logits: [B, T, VOCAB_SIZE]."""
    return jax.vmap(lambda xi: forward_single(params, xi))(inputs)


# ── Loss ───────────────────────────────────────────────────────────────────────

def loss_fn(params, inputs, targets):
    """Mean cross-entropy over every pixel in the batch."""
    logits   = model_apply(params, inputs)   # [B, T, V]
    B, T, V  = logits.shape
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    flat_lp   = log_probs.reshape(-1, V)
    flat_tgt  = targets.reshape(-1)
    return -flat_lp[jnp.arange(B * T), flat_tgt].mean()


# ── Optimiser + training step ──────────────────────────────────────────────────

optimizer = optax.adamw(LR, weight_decay=WEIGHT_DECAY)


@jax.jit
def train_step(params, opt_state, inputs, targets):
    loss, grads = jax.value_and_grad(loss_fn)(params, inputs, targets)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# ── Utilities ──────────────────────────────────────────────────────────────────

def count_params(params) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def peak_vram_mb() -> float:
    try:
        stats = jax.devices()[0].memory_stats()
        return stats["peak_bytes_in_use"] / 1024 / 1024
    except Exception:
        return 0.0


def estimate_mfu(num_steps: int, elapsed: float) -> float:
    """Rough MFU estimate assuming an A100-80GB (312 TFLOPS FP32)."""
    # LSTM FLOPs per step (multiply-adds × 2 for mul+add):
    # 4 gates × 2 matmuls (Wih, Whh) × D_MODEL rows × D_MODEL cols × T × B
    lstm_flops = 8 * D_MODEL * (D_EMBED + D_MODEL) * SEQ_LEN * BATCH_SIZE * N_LAYERS
    head_flops  = 2 * D_MODEL * VOCAB_SIZE * SEQ_LEN * BATCH_SIZE
    flops_total = (lstm_flops + head_flops) * num_steps
    peak_tflops = 312e12  # A100 FP32
    return (flops_total / elapsed) / peak_tflops * 100


# ── Main training loop ─────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(42)
    key = jax.random.PRNGKey(0)

    params    = init_params(key)
    opt_state = optimizer.init(params)
    n_params  = count_params(params)

    print(f"num_params_M: {n_params / 1e6:.2f}M  depth: {N_LAYERS}", flush=True)

    # Warm-up: compile train_step and eval_apply
    print("Compiling ...", flush=True)
    dummy_in, dummy_tgt = next(make_batches("train", BATCH_SIZE, np.random.default_rng(0)))
    params, opt_state, _ = train_step(params, opt_state, dummy_in, dummy_tgt)
    eval_apply = jax.jit(model_apply)
    dummy_eval, _ = next(make_batches("val", 128, np.random.default_rng(0)))
    _ = eval_apply(params, dummy_eval)  # compile eval trace
    print("Done. Training ...", flush=True)

    t0            = time.perf_counter()
    step          = 0
    total_tokens  = 0
    last_loss     = float("nan")

    while True:
        for inputs, targets in make_batches("train", BATCH_SIZE, rng):
            if time.perf_counter() - t0 >= TRAIN_SECONDS:
                break
            params, opt_state, loss = train_step(params, opt_state, inputs, targets)
            step         += 1
            total_tokens += inputs.size
            last_loss     = float(loss)
            if step % 100 == 0:
                elapsed = time.perf_counter() - t0
                print(f"step={step:5d}  loss={last_loss:.4f}  t={elapsed:.0f}s", flush=True)
        if time.perf_counter() - t0 >= TRAIN_SECONDS:
            break

    training_seconds = time.perf_counter() - t0

    print("Evaluating ...", flush=True)
    val_bpb       = evaluate_bpb(eval_apply, params)
    total_seconds = time.perf_counter() - t0
    vram          = peak_vram_mb()
    mfu           = estimate_mfu(step, training_seconds)

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {vram:.1f}")
    print(f"mfu_percent:      {mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {n_params / 1e6:.1f}")
    print(f"depth:            {N_LAYERS}")


if __name__ == "__main__":
    main()
