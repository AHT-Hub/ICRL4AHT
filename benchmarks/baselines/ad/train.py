#!/usr/bin/env python3
"""AD Training Script for Overcooked V2.

This script trains an Algorithm Distillation model on collected learning histories.
Follows the AD training approach with:
- Cross-entropy loss on actions with causal masking
- Contiguous sequence sampling from learning histories
- Step-level token format: (obs, prev_action, prev_reward) -> action

Usage:
    # Basic training (obs_shape auto from index)
    python -m benchmarks.baselines.ad.train \
        --h5_path datasets/histories.h5 \
        --index_path datasets/histories_index.jsonl \
        --out_dir outputs/ad_runs

    # With evaluation and debug mode
    python -m benchmarks.baselines.ad.train \
        --h5_path datasets/histories.h5 \
        --index_path datasets/histories_index.jsonl \
        --out_dir outputs/ad_runs \
        --eval_every 1000 \
        --debug

    # Run on specific GPU
    python -m benchmarks.baselines.ad.train \
        --h5_path datasets/histories.h5 \
        --index_path datasets/histories_index.jsonl \
        --out_dir outputs/ad_runs \
        --gpu 0

    # Quick smoke test
    python -m benchmarks.baselines.ad.train \
        --h5_path datasets/histories.h5 \
        --index_path datasets/histories_index.jsonl \
        --out_dir outputs/ad_debug \
        --num_steps 50 \
        --batch_size 16 \
        --seq_len 128 \
        --debug

Performance Optimizations (enabled by default):
    - Threaded data prefetching: Loads batches in background while GPU trains
      (--prefetch_buffer_size 3, set to 0 to disable)
    - Async device transfer: Overlaps CPU-GPU data transfer with computation
      (--no-async-transfer to disable)
    - Large HDF5 cache: 512MB cache for faster data access
      (--hdf5_cache_mb 512, increase for larger datasets)
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import threading
import queue
from functools import partial
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Parse GPU argument early before JAX import (JAX needs CUDA_VISIBLE_DEVICES set before import)
def _get_gpu_arg():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith('--gpu='):
            return arg.split('=', 1)[1]
    return None

_gpu_id = _get_gpu_arg()
if _gpu_id is not None:
    if _gpu_id == "-1":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_id

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state, checkpoints

from benchmarks.baselines.ad.model import ADConfig, ADModel, create_ad_model, count_parameters
from runners.history_adapter import HistoryStore, ADBatch

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)


def get_obs_shape_from_index(index_path: str) -> Tuple[int, ...]:
    """Read obs_shape from the first entry of an index file."""
    with open(index_path, "r") as f:
        for line in f:
            entry = line.strip()
            if not entry:
                continue
            record = json.loads(entry)
            if "obs_shape" in record:
                return tuple(record["obs_shape"])
            if "obs_dim" in record:
                return (record["obs_dim"],)
            # Keep scanning in case later lines contain obs_shape
            continue
    raise ValueError("obs_shape not found in index file")


class CSVMetricsLogger:
    """Helper class to write metrics to CSV file."""

    def __init__(self, filepath: str, fieldnames: List[str]):
        """Initialize CSV logger.

        Args:
            filepath: Path to CSV file.
            fieldnames: List of column names.
        """
        self.filepath = filepath
        self.fieldnames = fieldnames
        self._initialized = False

    def _init_file(self):
        """Initialize CSV file with header."""
        if self._initialized:
            return

        # If file already exists and is non-empty, assume header is present and keep appending
        if os.path.isfile(self.filepath) and os.path.getsize(self.filepath) > 0:
            self._initialized = True
            return

        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
        self._initialized = True

    def log(self, metrics: Dict[str, Any]):
        """Write a row of metrics to CSV.

        Args:
            metrics: Dictionary of metric name to value.
        """
        self._init_file()
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            # Only write fields that exist in fieldnames
            row = {k: metrics.get(k, '') for k in self.fieldnames}
            writer.writerow(row)


@dataclass
class TrainConfig:
    """Training configuration."""
    # Data paths
    h5_path: str = ""
    index_path: str = ""

    # Model hyperparameters (obs_shape is auto-calculated from index)
    obs_shape: Tuple[int, ...] = field(default_factory=lambda: (9, 7, 26))  # (H, W, C)
    num_actions: int = 6
    embedding_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    seq_len: int = 512  # Context length (steps)

    # Dropout
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1

    # Training hyperparameters
    num_steps: int = 100000
    batch_size: int = 128
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0

    # Loss masking options
    ignore_loss_after_done: bool = False  # Ignore loss on steps after done

    # Teammate action conditioning (optional feature)
    use_teammate_actions: bool = False  # If True, include teammate actions in token

    # Evaluation
    eval_every: int = 500
    eval_batch_size: int = 128
    eval_batches: int = 1

    # Checkpointing
    save_every: int = 10000
    out_dir: str = "outputs/ad_runs"

    # Misc
    seed: int = 0
    log_every: int = 100
    gpu: Optional[str] = None  # GPU device ID(s) to use
    debug: bool = False  # Enable debug mode (prints batch examples, deterministic)

    # CSV logging
    csv_log: bool = False  # Enable CSV logging
    csv_interval: int = 100  # Log to CSV every N steps

    # Performance optimization
    prefetch_buffer_size: int = 3  # Number of batches to prefetch (0 to disable)
    hdf5_cache_mb: float = 512.0  # HDF5 chunk cache size in MB (default: 512MB)
    use_async_transfer: bool = True  # Use async device transfer with JAX

    # Resume training
    resume: bool = False  # Resume from latest checkpoint in out_dir

    def to_model_config(self) -> ADConfig:
        """Convert to model config."""
        return ADConfig(
            obs_shape=self.obs_shape,
            num_actions=self.num_actions,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            attention_dropout=self.attention_dropout,
            residual_dropout=self.residual_dropout,
            embedding_dropout=self.embedding_dropout,
            use_teammate_actions=self.use_teammate_actions,
        )


class PrefetchDataLoader:
    """Threaded data prefetcher for overlapping data loading with training.

    Loads batches in a background thread while the GPU is training,
    providing significant speedup by overlapping I/O with computation.
    """

    def __init__(
        self,
        store: HistoryStore,
        rng: np.random.Generator,
        batch_size: int,
        seq_len: int,
        use_teammate_actions: bool,
        buffer_size: int = 3,
    ):
        """Initialize prefetch loader.

        Args:
            store: HistoryStore for data access
            rng: Random number generator
            batch_size: Batch size
            seq_len: Sequence length
            use_teammate_actions: Whether to include teammate actions
            buffer_size: Number of batches to prefetch (default: 3)
        """
        self.store = store
        self.rng = rng
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.use_teammate_actions = use_teammate_actions
        self.buffer_size = buffer_size

        self.queue = queue.Queue(maxsize=buffer_size)
        self.thread = None
        self.stop_event = threading.Event()
        self.exception = None

    def _worker(self):
        """Background worker thread that loads batches."""
        try:
            while not self.stop_event.is_set():
                # Sample batch (this is the slow I/O operation)
                batch = self.store.sample_ad_batch(
                    rng=self.rng,
                    batch_size=self.batch_size,
                    seq_len=self.seq_len,
                    include_teammate_actions=self.use_teammate_actions,
                )

                # Put batch in queue (blocks if queue is full)
                # Keep retrying until we can put the batch or stop is requested
                while not self.stop_event.is_set():
                    try:
                        self.queue.put(batch, timeout=0.1)
                        break  # Successfully put batch, exit inner loop
                    except queue.Full:
                        # Queue is full, retry after checking stop_event
                        continue
        except Exception as e:
            self.exception = e

    def start(self):
        """Start the prefetch thread."""
        if self.thread is not None:
            raise RuntimeError("Prefetch thread already started")
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def get_batch(self, timeout: float = 300.0) -> ADBatch:
        """Get next prefetched batch.

        Args:
            timeout: Maximum time to wait for a batch (seconds)

        Returns:
            ADBatch

        Raises:
            RuntimeError: If worker thread encountered an error or is not running
            queue.Empty: If timeout expires
        """
        # Check for exceptions first
        if self.exception is not None:
            raise RuntimeError(f"Prefetch worker failed: {self.exception}")

        # Check if thread is alive
        if self.thread is None or not self.thread.is_alive():
            if self.exception is not None:
                raise RuntimeError(f"Prefetch worker failed: {self.exception}")
            raise RuntimeError("Prefetch worker thread is not running")

        try:
            batch = self.queue.get(timeout=timeout)
            return batch
        except queue.Empty:
            # Check again for exception after timeout
            if self.exception is not None:
                raise RuntimeError(f"Prefetch worker failed: {self.exception}")
            # Check if thread died during wait
            if self.thread is None or not self.thread.is_alive():
                raise RuntimeError("Prefetch worker thread died unexpectedly")
            raise RuntimeError(
                f"Timeout waiting for batch after {timeout}s. "
                f"Queue size: {self.queue.qsize()}, Thread alive: {self.thread.is_alive() if self.thread else False}"
            )

    def stop(self):
        """Stop the prefetch thread."""
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join(timeout=2.0)
            self.thread = None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        del exc_type, exc_val, exc_tb  # Unused but required for context manager protocol
        self.stop()
        return False


class TrainState(train_state.TrainState):
    """Custom train state with additional tracking."""
    step: int


def create_train_state(
    config: TrainConfig,
    model: ADModel,
    params: Any,
) -> TrainState:
    """Create training state with optimizer.

    Args:
        config: Training configuration
        model: AD model
        params: Initialized parameters

    Returns:
        TrainState with optimizer
    """
    # Learning rate schedule: warmup + cosine decay
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=config.learning_rate,
        transition_steps=config.warmup_steps,
    )
    decay_fn = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=max(1, config.num_steps - config.warmup_steps),
    )
    lr_schedule = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[config.warmup_steps],
    )

    # Optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
        ),
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


@partial(jax.jit, static_argnames=("num_actions", "label_smoothing", "use_teammate_actions"))
def _train_step_impl(
    state: TrainState,
    batch_data: Tuple,
    dropout_rng: jax.Array,
    num_actions: int,
    label_smoothing: float,
    use_teammate_actions: bool,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """Single training step implementation (JIT-compiled).

    Args:
        state: Training state
        batch_data: Tuple of (obs, prev_actions, prev_rewards, target_actions, attention_mask, [prev_teammate_actions])
        dropout_rng: RNG key for dropout
        num_actions: Number of actions
        label_smoothing: Label smoothing factor
        use_teammate_actions: Whether to use teammate actions

    Returns:
        Tuple of (updated state, metrics)
    """
    if use_teammate_actions:
        obs, prev_actions, prev_rewards, target_actions, attention_mask, prev_teammate_actions = batch_data
    else:
        obs, prev_actions, prev_rewards, target_actions, attention_mask = batch_data
        prev_teammate_actions = None

    def loss_fn(params):
        # Forward pass
        logits = state.apply_fn(
            params,
            obs,
            prev_actions,
            prev_rewards,
            attention_mask=attention_mask,
            prev_teammate_actions=prev_teammate_actions,
            train=True,
            rngs={'dropout': dropout_rng},
        )  # (batch, seq_len, num_actions)

        batch_size, seq_len = target_actions.shape

        # Cross-entropy loss on all positions (causal masking handled in attention)
        # Flatten for loss computation
        logits_flat = logits.reshape(-1, num_actions)  # (batch * seq, num_actions)
        targets_flat = target_actions.reshape(-1)  # (batch * seq,)

        # One-hot targets
        targets_onehot = jax.nn.one_hot(targets_flat, num_actions)

        # Apply label smoothing
        if label_smoothing > 0:
            targets_onehot = targets_onehot * (1 - label_smoothing) + label_smoothing / num_actions

        # Cross-entropy
        log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
        per_token_loss = -jnp.sum(targets_onehot * log_probs, axis=-1)

        # Apply attention mask to loss (ignore padding)
        mask_flat = attention_mask.reshape(-1)  # (batch * seq,)
        masked_loss = per_token_loss * mask_flat
        loss = jnp.sum(masked_loss) / jnp.maximum(jnp.sum(mask_flat), 1.0)

        # Compute accuracy (on non-padded positions)
        predicted = jnp.argmax(logits_flat, axis=-1)
        correct = (predicted == targets_flat) * mask_flat
        accuracy = jnp.sum(correct) / jnp.maximum(jnp.sum(mask_flat), 1.0)

        return loss, {"loss": loss, "accuracy": accuracy}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    return state, metrics


# Global RNG for train_step
_train_rng = jax.random.PRNGKey(0)


def train_step(
    state: TrainState,
    batch_data: Tuple,
    config_dict: Dict,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """Single training step (dispatches to JIT-compiled implementation)."""
    global _train_rng
    _train_rng, dropout_rng = jax.random.split(_train_rng)

    return _train_step_impl(
        state,
        batch_data,
        dropout_rng,
        num_actions=config_dict["num_actions"],
        label_smoothing=config_dict["label_smoothing"],
        use_teammate_actions=config_dict["use_teammate_actions"],
    )


@partial(jax.jit, static_argnames=("num_actions", "use_teammate_actions"))
def _eval_step_impl(
    state: TrainState,
    batch_data: Tuple,
    num_actions: int,
    use_teammate_actions: bool,
) -> Dict[str, jnp.ndarray]:
    """Single evaluation step implementation (JIT-compiled).

    Unlike _train_step_impl, this:
    - Uses train=False to disable dropout
    - Does not compute gradients
    - Does not update state

    Args:
        state: Training state
        batch_data: Tuple of (obs, prev_actions, prev_rewards, target_actions, attention_mask, [prev_teammate_actions])
        num_actions: Number of actions
        use_teammate_actions: Whether to use teammate actions

    Returns:
        Metrics dictionary
    """
    if use_teammate_actions:
        obs, prev_actions, prev_rewards, target_actions, attention_mask, prev_teammate_actions = batch_data
    else:
        obs, prev_actions, prev_rewards, target_actions, attention_mask = batch_data
        prev_teammate_actions = None

    # Forward pass with train=False (no dropout)
    logits = state.apply_fn(
        state.params,
        obs,
        prev_actions,
        prev_rewards,
        attention_mask=attention_mask,
        prev_teammate_actions=prev_teammate_actions,
        train=False,  # Disable dropout during evaluation
    )  # (batch, seq_len, num_actions)

    # Flatten for loss computation
    logits_flat = logits.reshape(-1, num_actions)  # (batch * seq, num_actions)
    targets_flat = target_actions.reshape(-1)  # (batch * seq,)

    # One-hot targets (no label smoothing during eval)
    targets_onehot = jax.nn.one_hot(targets_flat, num_actions)

    # Cross-entropy
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    per_token_loss = -jnp.sum(targets_onehot * log_probs, axis=-1)

    # Apply attention mask to loss (ignore padding)
    mask_flat = attention_mask.reshape(-1)  # (batch * seq,)
    masked_loss = per_token_loss * mask_flat
    loss = jnp.sum(masked_loss) / jnp.maximum(jnp.sum(mask_flat), 1.0)

    # Compute accuracy (on non-padded positions)
    predicted = jnp.argmax(logits_flat, axis=-1)
    correct = (predicted == targets_flat) * mask_flat
    accuracy = jnp.sum(correct) / jnp.maximum(jnp.sum(mask_flat), 1.0)

    return {"loss": loss, "accuracy": accuracy}


def eval_step(
    state: TrainState,
    batch_data: Tuple,
    config_dict: Dict,
) -> Dict[str, jnp.ndarray]:
    """Single evaluation step (dispatches to JIT-compiled implementation)."""
    return _eval_step_impl(
        state,
        batch_data,
        num_actions=config_dict["num_actions"],
        use_teammate_actions=config_dict["use_teammate_actions"],
    )


def batch_to_tuple(batch: ADBatch, use_teammate_actions: bool = False) -> Tuple:
    """Convert ADBatch to tuple for JIT compatibility."""
    base = (
        jnp.array(batch.obs),
        jnp.array(batch.prev_actions),
        jnp.array(batch.prev_rewards),
        jnp.array(batch.target_actions),
        jnp.array(batch.attention_mask, dtype=jnp.float32),
    )
    if use_teammate_actions and batch.prev_teammate_actions is not None:
        return base + (jnp.array(batch.prev_teammate_actions),)
    return base


def print_batch_debug(batch: ADBatch, prefix: str = ""):
    """Print debug information about a batch."""
    print(f"\n{prefix}Batch Debug Info:")
    print(f"  obs shape: {batch.obs.shape}, dtype: {batch.obs.dtype}")
    print(f"  prev_actions shape: {batch.prev_actions.shape}, dtype: {batch.prev_actions.dtype}")
    print(f"  prev_rewards shape: {batch.prev_rewards.shape}, dtype: {batch.prev_rewards.dtype}")
    print(f"  target_actions shape: {batch.target_actions.shape}, dtype: {batch.target_actions.dtype}")
    print(f"  attention_mask shape: {batch.attention_mask.shape}")
    print(f"  dones shape: {batch.dones.shape}")
    print(f"  history_ids: {batch.history_ids[:5]}...")
    print(f"  start_ts: {batch.start_ts[:5]}...")

    # Sample values
    print(f"\n  Sample (first seq, first 5 steps):")
    print(f"    prev_actions: {batch.prev_actions[0, :5]}")
    print(f"    prev_rewards: {batch.prev_rewards[0, :5]}")
    print(f"    target_actions: {batch.target_actions[0, :5]}")
    print(f"    attention_mask: {batch.attention_mask[0, :5]}")
    print(f"    dones: {batch.dones[0, :5]}")

    # Check for NaNs
    has_nan_obs = np.any(np.isnan(batch.obs))
    has_nan_rewards = np.any(np.isnan(batch.prev_rewards))
    print(f"\n  NaN check: obs={has_nan_obs}, rewards={has_nan_rewards}")

    # Mask statistics
    valid_ratio = np.mean(batch.attention_mask)
    print(f"  Attention mask valid ratio: {valid_ratio:.3f}")


def evaluate(
    state: TrainState,
    store: HistoryStore,
    config: TrainConfig,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Evaluate on held-out batches.

    Args:
        state: Current training state
        store: History store for sampling
        config: Training config
        rng: Random generator

    Returns:
        Evaluation metrics
    """
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    config_dict = {
        "num_actions": config.num_actions,
        "label_smoothing": 0.0,  # No label smoothing during eval
        "use_teammate_actions": config.use_teammate_actions,
    }

    for _ in range(config.eval_batches):
        batch = store.sample_ad_batch(
            rng=rng,
            batch_size=config.eval_batch_size,
            seq_len=config.seq_len,
            include_teammate_actions=config.use_teammate_actions,
        )
        batch_data = batch_to_tuple(batch, use_teammate_actions=config.use_teammate_actions)

        # Forward pass only (no dropout during eval)
        metrics = eval_step(state, batch_data, config_dict)

        total_loss += float(metrics["loss"])
        total_acc += float(metrics["accuracy"])
        num_batches += 1

    return {
        "eval_loss": total_loss / num_batches,
        "eval_accuracy": total_acc / num_batches,
    }


def train(config: TrainConfig):
    """Main training loop.

    Args:
        config: Training configuration
    """
    global _train_rng

    # Setup output directory (must be absolute for orbax checkpointing)
    out_dir = Path(config.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    log.info(f"Training AD model")
    log.info(f"  Output: {out_dir}")
    log.info(f"  H5 path: {config.h5_path}")
    log.info(f"  Obs shape: {config.obs_shape}")
    log.info(f"  Batch size: {config.batch_size}")
    log.info(f"  Seq len: {config.seq_len}")
    log.info(f"  Num steps: {config.num_steps}")
    log.info(f"  Use teammate actions: {config.use_teammate_actions}")
    log.info(f"  Debug mode: {config.debug}")

    # Initialize RNG
    _train_rng = jax.random.PRNGKey(config.seed)

    # Initialize model
    model_config = config.to_model_config()
    model, params = create_ad_model(model_config)

    num_params = count_parameters(params)
    log.info(f"  Model parameters: {num_params:,}")

    # Create training state
    state = create_train_state(config, model, params)

    # Resume from checkpoint if requested
    start_step = 0
    ckpt_dir = out_dir / "checkpoints"
    if config.resume and ckpt_dir.exists():
        state = checkpoints.restore_checkpoint(ckpt_dir=str(ckpt_dir), target=state)
        start_step = int(state.step)
        if start_step > 0:
            log.info(f"Resumed from checkpoint at step {start_step}")
            # Advance RNG to match the resumed step
            _train_rng = jax.random.PRNGKey(config.seed)
            for _ in range(start_step):
                _train_rng, _ = jax.random.split(_train_rng)
        else:
            log.info("No checkpoint found, starting from scratch")
    elif config.resume:
        log.info("Resume requested but no checkpoint directory found, starting from scratch")

    # Load dataset
    log.info("Loading dataset...")
    store = HistoryStore(config.h5_path, config.index_path, cache_size_mb=config.hdf5_cache_mb)
    log.info(f"  Histories: {len(store)}")
    log.info(f"  Tasks: {len(store.get_task_ids())}")
    log.info(f"  HDF5 cache: {config.hdf5_cache_mb:.1f} MB")

    # Get obs_shape from data if not specified or mismatched
    if len(store) > 0:
        meta = store.get_history_meta(0)
        data_obs_shape = tuple(meta["obs_shape"])
        if data_obs_shape != config.obs_shape:
            log.warning(f"  obs_shape mismatch: config={config.obs_shape}, data={data_obs_shape}")
            log.warning(f"  Using data obs_shape: {data_obs_shape}")
            # Recreate model with correct obs_shape
            config.obs_shape = data_obs_shape
            model_config = config.to_model_config()
            model, params = create_ad_model(model_config)
            state = create_train_state(config, model, params)

    # RNG for sampling
    rng = np.random.default_rng(config.seed)

    # Config dict for JIT
    config_dict = {
        "num_actions": config.num_actions,
        "label_smoothing": config.label_smoothing,
        "use_teammate_actions": config.use_teammate_actions,
    }

    # Setup CSV logging
    csv_logger = None
    if config.csv_log:
        csv_path = out_dir / "training_metrics.csv"
        csv_fieldnames = [
            "step", "loss", "accuracy", "eval_loss", "eval_accuracy",
            "steps_per_sec", "elapsed_time"
        ]
        csv_logger = CSVMetricsLogger(str(csv_path), csv_fieldnames)
        log.info(f"CSV logging enabled: {csv_path}")

    # Debug: print a sample batch
    if config.debug:
        log.info("\n=== DEBUG: Sample batch ===")
        debug_batch = store.sample_ad_batch(
            rng=np.random.default_rng(config.seed),
            batch_size=min(4, config.batch_size),
            seq_len=min(32, config.seq_len),
            include_teammate_actions=config.use_teammate_actions,
        )
        print_batch_debug(debug_batch, prefix="[DEBUG] ")

    # Load existing metrics if resuming
    all_metrics = []
    metrics_path = out_dir / "metrics.json"
    if config.resume and metrics_path.exists() and start_step > 0:
        try:
            with open(metrics_path, "r") as f:
                all_metrics = json.load(f)
            # Filter metrics to only include steps before start_step
            all_metrics = [m for m in all_metrics if m.get("step", 0) < start_step]
            log.info(f"Loaded {len(all_metrics)} existing metric entries")
        except Exception as e:
            log.warning(f"Failed to load existing metrics: {e}")
            all_metrics = []

    # Training loop
    log.info("Starting training...")
    if start_step > 0:
        log.info(f"  Resuming from step {start_step}")
    log.info(f"  Prefetch buffer: {config.prefetch_buffer_size} batches")
    log.info(f"  Async device transfer: {config.use_async_transfer}")
    start_time = time.time()

    # Setup data loader with prefetching
    if config.prefetch_buffer_size > 0:
        log.info("Using threaded data prefetching for faster I/O")
        data_loader = PrefetchDataLoader(
            store=store,
            rng=rng,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            use_teammate_actions=config.use_teammate_actions,
            buffer_size=config.prefetch_buffer_size,
        )
        data_loader.start()
    else:
        data_loader = None

    try:
        for step in range(start_step, config.num_steps):
            # Sample batch (prefetched or synchronous)
            if data_loader is not None:
                batch = data_loader.get_batch()
            else:
                batch = store.sample_ad_batch(
                    rng=rng,
                    batch_size=config.batch_size,
                    seq_len=config.seq_len,
                    include_teammate_actions=config.use_teammate_actions,
                )

            batch_data = batch_to_tuple(batch, use_teammate_actions=config.use_teammate_actions)

            # Async device transfer (overlaps host-to-device copy with previous computation)
            if config.use_async_transfer:
                # Transfer data to device asynchronously (JAX will choose the appropriate device)
                batch_data = jax.tree_map(jax.device_put, batch_data)

            # Training step
            state, metrics = train_step(state, batch_data, config_dict)

            # Check for NaN
            if jnp.isnan(metrics["loss"]):
                log.error(f"NaN loss at step {step}!")
                if config.debug:
                    print_batch_debug(batch, prefix="[NaN DEBUG] ")
                break

            # Logging
            if step % config.log_every == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0

                log.info(
                    f"Step {step:6d} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Acc: {metrics['accuracy']:.4f} | "
                    f"Steps/s: {steps_per_sec:.1f}"
                )

                all_metrics.append({
                    "step": step,
                    "loss": float(metrics["loss"]),
                    "accuracy": float(metrics["accuracy"]),
                })

                # CSV logging at specified intervals
                if csv_logger is not None and (step % config.csv_interval == 0):
                    csv_logger.log({
                        "step": step,
                        "loss": float(metrics["loss"]),
                        "accuracy": float(metrics["accuracy"]),
                        "eval_loss": "",
                        "eval_accuracy": "",
                        "steps_per_sec": steps_per_sec,
                        "elapsed_time": elapsed,
                    })

            # Evaluation
            if config.eval_every > 0 and step > 0 and step % config.eval_every == 0:
                eval_rng = np.random.default_rng(config.seed + step)
                eval_metrics = evaluate(state, store, config, eval_rng)
                log.info(
                    f"  Eval | "
                    f"Loss: {eval_metrics['eval_loss']:.4f} | "
                    f"Acc: {eval_metrics['eval_accuracy']:.4f}"
                )
                all_metrics[-1].update(eval_metrics)

                # CSV logging for evaluation
                if csv_logger is not None:
                    elapsed = time.time() - start_time
                    steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
                    csv_logger.log({
                        "step": step,
                        "loss": float(metrics["loss"]),
                        "accuracy": float(metrics["accuracy"]),
                        "eval_loss": eval_metrics["eval_loss"],
                        "eval_accuracy": eval_metrics["eval_accuracy"],
                        "steps_per_sec": steps_per_sec,
                        "elapsed_time": elapsed,
                    })

            # Save checkpoint and metrics
            if config.save_every > 0 and step > 0 and step % config.save_every == 0:
                ckpt_dir = out_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)
                checkpoints.save_checkpoint(
                    ckpt_dir=str(ckpt_dir),
                    target=state,
                    step=step,
                    keep=1e4,
                )
                # Save metrics at regular intervals
                with open(out_dir / "metrics.json", "w") as f:
                    json.dump(all_metrics, f, indent=2)
                log.info(f"  Saved checkpoint and metrics at step {step}")

    finally:
        # Stop prefetch thread
        if data_loader is not None:
            data_loader.stop()

    # Final save
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    checkpoints.save_checkpoint(
        ckpt_dir=str(ckpt_dir),
        target=state,
        step=config.num_steps,
        keep=1e4,
    )

    # Save metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    elapsed = time.time() - start_time
    log.info(f"Training complete! Elapsed: {elapsed:.1f}s")

    store.close()


def parse_args() -> TrainConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train AD model")

    # Data
    parser.add_argument("--h5_path", type=str, required=True,
                       help="Path to HDF5 history file")
    parser.add_argument("--index_path", type=str, required=True,
                       help="Path to index JSONL file")

    # Model
    parser.add_argument("--num_actions", type=int, default=6,
                       help="Number of actions")
    parser.add_argument("--embedding_dim", type=int, default=64,
                       help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Transformer hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4,
                       help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4,
                       help="Number of attention heads")
    parser.add_argument("--seq_len", type=int, default=500,
                       help="Sequence length (context window)")

    # Dropout
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--residual_dropout", type=float, default=0.1)
    parser.add_argument("--embedding_dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--num_steps", type=int, default=20000,
                       help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=1024,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=None,
                       help="Warmup steps (default: 5% of num_steps)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                       help="Weight decay")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                       help="Label smoothing factor")

    # Eval
    parser.add_argument("--eval_every", type=int, default=500,
                       help="Evaluate every N steps (0 to disable)")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                       help="Evaluation batch size (default: same as batch_size)")
    parser.add_argument("--eval_batches", type=int, default=3,
                       help="Number of batches for evaluation")

    # Output
    parser.add_argument("--save_every", type=int, default=100,
                       help="Save checkpoint every N steps")
    parser.add_argument("--out_dir", type=str, default="outputs/ad_runs",
                       help="Output directory")
    parser.add_argument("--log_every", type=int, default=100,
                       help="Log every N steps")

    # Teammate actions (optional feature)
    parser.add_argument("--use_teammate_actions", action="store_true",
                       help="Include teammate actions in token (optional)")

    # Misc
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--gpu", type=str, default=None,
                       help="GPU device ID(s) to use (e.g., '0', '0,1'). Use '-1' for CPU. If not specified, uses all available GPUs.")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--no-csv-log", action="store_false", dest="csv_log", default=True,
                       help="Disable CSV logging of metrics (enabled by default)")
    parser.add_argument("--csv_interval", type=int, default=100,
                       help="Log to CSV every N steps (default: 100)")

    # Performance optimization
    parser.add_argument("--prefetch_buffer_size", type=int, default=5,
                       help="Number of batches to prefetch in background thread (0 to disable, default: 5)")
    parser.add_argument("--hdf5_cache_mb", type=float, default=512.0,
                       help="HDF5 chunk cache size in MB (default: 512)")
    parser.add_argument("--no-async-transfer", action="store_false", dest="use_async_transfer", default=True,
                       help="Disable async device transfer (enabled by default)")

    # Resume training
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from latest checkpoint in out_dir")

    args = parser.parse_args()

    # Default eval_batch_size to training batch_size when not provided
    resolved_eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size

    # Default warmup_steps to 5% of num_steps when not provided
    resolved_warmup_steps = args.warmup_steps if args.warmup_steps is not None else max(1, int(args.num_steps * 0.05))

    # Auto-calculate obs_shape from index
    obs_shape = get_obs_shape_from_index(args.index_path)
    log.info(f"Index '{args.index_path}' -> obs_shape: {obs_shape}")

    return TrainConfig(
        h5_path=args.h5_path,
        index_path=args.index_path,
        obs_shape=obs_shape,
        num_actions=args.num_actions,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        attention_dropout=args.attention_dropout,
        residual_dropout=args.residual_dropout,
        embedding_dropout=args.embedding_dropout,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=resolved_warmup_steps,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        use_teammate_actions=args.use_teammate_actions,
        eval_every=args.eval_every,
        eval_batch_size=resolved_eval_batch_size,
        eval_batches=args.eval_batches,
        save_every=args.save_every,
        out_dir=args.out_dir,
        log_every=args.log_every,
        seed=args.seed,
        gpu=args.gpu,
        debug=args.debug,
        csv_log=args.csv_log,
        csv_interval=args.csv_interval,
        prefetch_buffer_size=args.prefetch_buffer_size,
        hdf5_cache_mb=args.hdf5_cache_mb,
        use_async_transfer=args.use_async_transfer,
        resume=args.resume,
    )


def main():
    """Main entry point."""
    config = parse_args()

    # Log GPU info
    if config.gpu is not None:
        if config.gpu == "-1":
            log.info("Using CPU")
        else:
            log.info(f"Using GPU: {config.gpu}")
    else:
        log.info(f"Using all available devices: {jax.devices()}")

    try:
        train(config)
        return 0
    except Exception as e:
        log.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
