#!/usr/bin/env python3
"""DPT Training Script for Overcooked V2.

This script trains a Decision Pretrained Transformer on collected learning histories.
Follows the DPT training approach with:
- Cross-entropy loss on expert actions
- Random context sampling from learning histories
- Optional evaluation during training

Note: Expert actions must be included in the histories.h5 file.
Run build_index.py with --relabel flag to add expert actions.

Usage:
    # Basic training
    python -m benchmarks.baselines.dpt.train \
        --h5_path datasets/histories.h5 \
        --index_path datasets/histories_index.jsonl \
        --out_dir outputs/dpt_runs

    # With evaluation
    python -m benchmarks.baselines.dpt.train \
        --h5_path datasets/histories.h5 \
        --index_path datasets/histories_index.jsonl \
        --out_dir outputs/dpt_runs \
        --eval_every 1000

    # Run on specific GPU
    python -m benchmarks.baselines.dpt.train \
        --h5_path datasets/histories.h5 \
        --index_path datasets/histories_index.jsonl \
        --out_dir outputs/dpt_runs \
        --gpu 0

    # Quick debug run
    python -m benchmarks.baselines.dpt.train \
        --h5_path datasets/histories.h5 \
        --index_path datasets/histories_index.jsonl \
        --out_dir outputs/dpt_debug \
        --num_steps 100 \
        --batch_size 32 \
        --seq_len 64

Performance Optimizations (enabled by default):
    - Threaded data prefetching: Loads batches in background while GPU trains
      (--prefetch_buffer_size 5, set to 0 to disable)
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
from typing import Any, Dict, Iterator, List, Optional, Tuple

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

from benchmarks.baselines.dpt.model import DPTConfig, DPTModel, create_dpt_model, count_parameters
from runners.history_adapter import DPTBatch, DPTDataset

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
    obs_shape: Tuple[int, ...] = field(default_factory=lambda: (9, 7, 26))
    num_actions: int = 6
    embedding_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    seq_len: int = 512  # Context length

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

    # Training options
    with_prior: bool = True  # Include query in loss

    # Teammate action conditioning (optional feature)
    use_teammate_actions: bool = False  # If True, include teammate actions in transition embedding

    # Evaluation
    eval_every: int = 500
    eval_batch_size: int = 128
    eval_batches: int = 1

    # Checkpointing
    save_every: int = 10000
    out_dir: str = "outputs/dpt_runs"

    # Misc
    seed: int = 0
    log_every: int = 100
    gpu: Optional[str] = None  # GPU device ID(s) to use

    # CSV logging
    csv_log: bool = False  # Enable CSV logging
    csv_interval: int = 100  # Log to CSV every N steps

    # Performance optimization
    prefetch_buffer_size: int = 5  # Number of batches to prefetch (0 to disable)
    hdf5_cache_mb: float = 512.0  # HDF5 chunk cache size in MB (default: 512MB)
    use_async_transfer: bool = True  # Use async device transfer with JAX

    # Resume training
    resume: bool = False  # Resume from latest checkpoint in out_dir

    def to_model_config(self) -> DPTConfig:
        """Convert to model config."""
        return DPTConfig(
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
    """Threaded data prefetcher for DPT training.

    Loads batches in a background thread while the GPU is training,
    providing significant speedup by overlapping I/O with computation.
    """

    def __init__(
        self,
        dataset: DPTDataset,
        rng: np.random.Generator,
        batch_size: int,
        use_teammate_actions: bool,
        buffer_size: int = 5,
    ):
        """Initialize prefetch loader.

        Args:
            dataset: DPTDataset for data access
            rng: Random number generator
            batch_size: Batch size
            use_teammate_actions: Whether to include teammate actions
            buffer_size: Number of batches to prefetch (default: 5)
        """
        self.dataset = dataset
        self.rng = rng
        self.batch_size = batch_size
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
                batch = self.dataset.sample_batch(
                    batch_size=self.batch_size,
                    rng=self.rng,
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

    def get_batch(self, timeout: float = 600.0) -> DPTBatch:
        """Get next prefetched batch.

        Args:
            timeout: Maximum time to wait for a batch (seconds)

        Returns:
            DPTBatch

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
    model: DPTModel,
    params: Any,
) -> TrainState:
    """Create training state with optimizer.

    Args:
        config: Training configuration
        model: DPT model
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
        decay_steps=config.num_steps - config.warmup_steps,
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


def compute_loss(
    params: Any,
    model: DPTModel,
    batch: DPTBatch,
    config: TrainConfig,
    train: bool = True,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute cross-entropy loss on expert actions.

    Args:
        params: Model parameters
        model: DPT model
        batch: Training batch
        config: Training config
        train: Whether in training mode

    Returns:
        Tuple of (loss, metrics dict)
    """
    # Forward pass
    context_teammate_actions = None
    if config.use_teammate_actions and batch.context_teammate_actions is not None:
        context_teammate_actions = jnp.array(batch.context_teammate_actions)

    logits = model.apply(
        params,
        jnp.array(batch.query_obs),
        jnp.array(batch.context_obs),
        jnp.array(batch.context_actions),
        jnp.array(batch.context_next_obs),
        jnp.array(batch.context_rewards),
        context_teammate_actions=context_teammate_actions,
        train=train,
    )  # (batch, seq_len + 1, num_actions)

    batch_size = batch.query_obs.shape[0]
    seq_len = batch.context_obs.shape[1]
    num_actions = config.num_actions

    # Target: expert action for query (replicated across all positions for DPT-style loss)
    target_actions = jnp.array(batch.query_target)  # (batch,)
    
    if config.with_prior:
        # DPT-style: compute loss on all positions predicting query target
        # Replicate target across all positions
        targets_expanded = jnp.tile(target_actions[:, None], (1, seq_len + 1))  # (batch, seq_len+1)
        targets_onehot = jax.nn.one_hot(targets_expanded, num_actions)  # (batch, seq_len+1, num_actions)

        # Flatten for cross-entropy
        logits_flat = logits.reshape(-1, num_actions)
        targets_flat = targets_onehot.reshape(-1, num_actions)
    else:
        # Only compute loss on query position (position 0)
        logits_flat = logits[:, 0, :]  # (batch, num_actions)
        targets_flat = jax.nn.one_hot(target_actions, num_actions)

    # Cross-entropy loss
    if config.label_smoothing > 0:
        # Apply label smoothing
        targets_flat = targets_flat * (1 - config.label_smoothing) + config.label_smoothing / num_actions

    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    loss = -jnp.sum(targets_flat * log_probs, axis=-1).mean()

    # Compute accuracy (on query position only for interpretability)
    query_logits = logits[:, 0, :]  # (batch, num_actions)
    predicted = jnp.argmax(query_logits, axis=-1)
    accuracy = jnp.mean(predicted == target_actions)

    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }

    return loss, metrics


@partial(jax.jit, static_argnames=("num_actions", "with_prior", "label_smoothing", "use_teammate_actions"))
def _train_step_impl(
    state: TrainState,
    batch_data: Tuple,
    dropout_rng: jax.Array,
    num_actions: int,
    with_prior: bool,
    label_smoothing: float,
    use_teammate_actions: bool,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """Single training step implementation (JIT-compiled).

    Args:
        state: Training state
        batch_data: Tuple of batch arrays
        dropout_rng: RNG key for dropout
        num_actions: Number of actions
        with_prior: Whether to use prior
        label_smoothing: Label smoothing factor
        use_teammate_actions: Whether to use teammate actions

    Returns:
        Tuple of (updated state, metrics)
    """
    # Unpack batch (can't pass dataclass through JIT boundary cleanly)
    if use_teammate_actions:
        (query_obs, query_target, context_obs, context_actions,
         context_next_obs, context_rewards, context_teammate_actions) = batch_data
    else:
        (query_obs, query_target, context_obs, context_actions,
         context_next_obs, context_rewards) = batch_data
        context_teammate_actions = None

    def loss_fn(params):
        # Manual forward pass with dropout RNG
        logits = state.apply_fn(
            params,
            query_obs,
            context_obs,
            context_actions,
            context_next_obs,
            context_rewards,
            context_teammate_actions=context_teammate_actions,
            train=True,
            rngs={'dropout': dropout_rng},
        )

        seq_len = context_obs.shape[1]
        target_actions = query_target

        if with_prior:
            targets_expanded = jnp.tile(target_actions[:, None], (1, seq_len + 1))
            targets_onehot = jax.nn.one_hot(targets_expanded, num_actions)
            logits_flat = logits.reshape(-1, num_actions)
            targets_flat = targets_onehot.reshape(-1, num_actions)
        else:
            logits_flat = logits[:, 0, :]
            targets_flat = jax.nn.one_hot(target_actions, num_actions)

        if label_smoothing > 0:
            targets_flat = (targets_flat * (1 - label_smoothing) +
                          label_smoothing / num_actions)

        log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
        loss = -jnp.sum(targets_flat * log_probs, axis=-1).mean()

        # Accuracy
        query_logits = logits[:, 0, :]
        predicted = jnp.argmax(query_logits, axis=-1)
        accuracy = jnp.mean(predicted == target_actions)

        return loss, {"loss": loss, "accuracy": accuracy}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    return state, metrics


# Global RNG for train_step (will be updated each call)
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
        with_prior=config_dict["with_prior"],
        label_smoothing=config_dict["label_smoothing"],
        use_teammate_actions=config_dict["use_teammate_actions"],
    )


def batch_to_tuple(batch: DPTBatch, use_teammate_actions: bool = False) -> Tuple:
    """Convert DPTBatch to tuple for JIT compatibility."""
    base = (
        jnp.array(batch.query_obs),
        jnp.array(batch.query_target),
        jnp.array(batch.context_obs),
        jnp.array(batch.context_actions),
        jnp.array(batch.context_next_obs),
        jnp.array(batch.context_rewards),
    )
    if use_teammate_actions and batch.context_teammate_actions is not None:
        return base + (jnp.array(batch.context_teammate_actions),)
    return base


def evaluate(
    state: TrainState,
    dataset: DPTDataset,
    config: TrainConfig,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Evaluate on held-out batches.

    Args:
        state: Current training state
        dataset: Dataset for sampling
        config: Training config
        rng: Random generator

    Returns:
        Evaluation metrics
    """
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for _ in range(config.eval_batches):
        batch = dataset.sample_batch(config.eval_batch_size, rng, include_teammate_actions=config.use_teammate_actions)
        batch_data = batch_to_tuple(batch, use_teammate_actions=config.use_teammate_actions)

        config_dict = {
            "num_actions": config.num_actions,
            "with_prior": config.with_prior,
            "label_smoothing": 0.0,  # No label smoothing during eval
            "use_teammate_actions": config.use_teammate_actions,
        }

        # Forward pass only (no gradients)
        _, metrics = train_step(state, batch_data, config_dict)

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
    # Setup output directory (must be absolute for orbax checkpointing)
    out_dir = Path(config.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    log.info(f"Training DPT model")
    log.info(f"  Output: {out_dir}")
    log.info(f"  H5 path: {config.h5_path}")
    log.info(f"  Obs shape: {config.obs_shape}")
    log.info(f"  Batch size: {config.batch_size}")
    log.info(f"  Seq len: {config.seq_len}")
    log.info(f"  Num steps: {config.num_steps}")
    log.info(f"  Use teammate actions: {config.use_teammate_actions}")

    # Initialize model
    model_config = config.to_model_config()
    model, params = create_dpt_model(model_config)

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
        else:
            log.info("No checkpoint found, starting from scratch")
    elif config.resume:
        log.info("Resume requested but no checkpoint directory found, starting from scratch")

    # Load dataset
    log.info("Loading dataset...")
    dataset = DPTDataset(
        h5_path=config.h5_path,
        index_path=config.index_path,
        seq_len=config.seq_len,
        seed=config.seed,
        use_expert_actions=True,  # Required for DPT training
        cache_size_mb=config.hdf5_cache_mb,
    )
    log.info(f"  Tasks: {dataset.num_tasks}")
    log.info(f"  Histories: {len(dataset.store)}")
    log.info(f"  HDF5 cache: {config.hdf5_cache_mb:.1f} MB")

    # RNG
    rng = np.random.default_rng(config.seed)

    # Config dict for JIT
    config_dict = {
        "num_actions": config.num_actions,
        "with_prior": config.with_prior,
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
            dataset=dataset,
            rng=rng,
            batch_size=config.batch_size,
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
                # log.info(f"Starting batch sampling for step {step}...")
                batch = dataset.sample_batch(config.batch_size, rng, include_teammate_actions=config.use_teammate_actions)
                # log.info(f"Completed batch sampling for step {step}.")

            batch_data = batch_to_tuple(batch, use_teammate_actions=config.use_teammate_actions)

            # Async device transfer (overlaps host-to-device copy with previous computation)
            if config.use_async_transfer:
                # Transfer data to device asynchronously (JAX will choose the appropriate device)
                batch_data = jax.tree_map(jax.device_put, batch_data)

            # Training step
            state, metrics = train_step(state, batch_data, config_dict)

            # Logging
            if step % config.log_every == 0:
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed if elapsed > 0 else 0

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
                eval_metrics = evaluate(state, dataset, config, eval_rng)
                log.info(
                    f"  Eval | "
                    f"Loss: {eval_metrics['eval_loss']:.4f} | "
                    f"Acc: {eval_metrics['eval_accuracy']:.4f}"
                )
                all_metrics[-1].update(eval_metrics)

                # CSV logging for evaluation
                if csv_logger is not None:
                    elapsed = time.time() - start_time
                    steps_per_sec = step / elapsed if elapsed > 0 else 0
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

    dataset.close()


def parse_args() -> TrainConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DPT model")

    # Data
    parser.add_argument("--h5_path", type=str, required=True)
    parser.add_argument("--index_path", type=str, required=True)

    # Model
    parser.add_argument("--num_actions", type=int, default=6)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=500)

    # Training
    parser.add_argument("--num_steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=None,
                       help="Warmup steps (default: 5% of num_steps)")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    # Eval
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_batch_size", type=int, default=None,
                       help="Evaluation batch size (default: same as batch_size)")
    parser.add_argument("--eval_batches", type=int, default=3)

    # Output
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="outputs/dpt_runs")
    parser.add_argument("--log_every", type=int, default=100)

    # Teammate actions (optional feature)
    parser.add_argument("--use_teammate_actions", action="store_true",
                       help="Include teammate actions in transition embedding (optional)")

    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=str, default=None,
                       help="GPU device ID(s) to use (e.g., '0', '0,1'). Use '-1' for CPU. If not specified, uses all available GPUs.")
    parser.add_argument("--csv_log", type=bool, default=True,
                       help="Enable CSV logging of metrics")
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
