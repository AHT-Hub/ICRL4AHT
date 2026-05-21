#!/usr/bin/env python3
"""Hybrid-AD Training Script for Overcooked V2.

Trains a Hybrid-AD model (AD with CNN + GRU backbone) on collected learning histories.
Uses the same data pipeline and training protocol as AD.

Usage:
    python -m benchmarks.baselines.hybrid_ad.train \
        --h5_path datasets/histories.h5 \
        --index_path datasets/histories_index.jsonl \
        --out_dir outputs/hybrid_ad
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

from benchmarks.baselines.hybrid_ad.model import HybridADConfig, HybridADModel, create_hybrid_ad_model, count_parameters
from runners.history_adapter import HistoryStore, ADBatch

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)


def get_obs_shape_from_index(index_path: str) -> Tuple[int, ...]:
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
            continue
    raise ValueError("obs_shape not found in index file")


class CSVMetricsLogger:
    def __init__(self, filepath: str, fieldnames: List[str]):
        self.filepath = filepath
        self.fieldnames = fieldnames
        self._initialized = False

    def _init_file(self):
        if self._initialized:
            return
        if os.path.isfile(self.filepath) and os.path.getsize(self.filepath) > 0:
            self._initialized = True
            return
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
        self._initialized = True

    def log(self, metrics: Dict[str, Any]):
        self._init_file()
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            row = {k: metrics.get(k, '') for k in self.fieldnames}
            writer.writerow(row)


@dataclass
class TrainConfig:
    h5_path: str = ""
    index_path: str = ""

    obs_shape: Tuple[int, ...] = field(default_factory=lambda: (9, 7, 26))
    num_actions: int = 6
    embedding_dim: int = 64
    hidden_dim: int = 256
    gru_hidden_dim: int = 256
    num_gru_layers: int = 2
    seq_len: int = 512

    embedding_dropout: float = 0.1

    reset_hidden_on_done: bool = False

    num_steps: int = 100000
    batch_size: int = 128
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0

    use_teammate_actions: bool = False

    eval_every: int = 500
    eval_batch_size: int = 128
    eval_batches: int = 1

    save_every: int = 10000
    out_dir: str = "outputs/hybrid_ad"

    seed: int = 0
    log_every: int = 100
    gpu: Optional[str] = None
    debug: bool = False

    csv_log: bool = False
    csv_interval: int = 100

    prefetch_buffer_size: int = 3
    hdf5_cache_mb: float = 512.0
    use_async_transfer: bool = True

    resume: bool = False

    def to_model_config(self) -> HybridADConfig:
        return HybridADConfig(
            obs_shape=self.obs_shape,
            num_actions=self.num_actions,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            gru_hidden_dim=self.gru_hidden_dim,
            num_gru_layers=self.num_gru_layers,
            seq_len=self.seq_len,
            embedding_dropout=self.embedding_dropout,
            use_teammate_actions=self.use_teammate_actions,
            reset_hidden_on_done=self.reset_hidden_on_done,
        )


class PrefetchDataLoader:
    def __init__(self, store, rng, batch_size, seq_len, use_teammate_actions, buffer_size=3):
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
        try:
            while not self.stop_event.is_set():
                batch = self.store.sample_ad_batch(
                    rng=self.rng,
                    batch_size=self.batch_size,
                    seq_len=self.seq_len,
                    include_teammate_actions=self.use_teammate_actions,
                )
                while not self.stop_event.is_set():
                    try:
                        self.queue.put(batch, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except Exception as e:
            self.exception = e

    def start(self):
        if self.thread is not None:
            raise RuntimeError("Prefetch thread already started")
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def get_batch(self, timeout: float = 300.0) -> ADBatch:
        if self.exception is not None:
            raise RuntimeError(f"Prefetch worker failed: {self.exception}")
        if self.thread is None or not self.thread.is_alive():
            if self.exception is not None:
                raise RuntimeError(f"Prefetch worker failed: {self.exception}")
            raise RuntimeError("Prefetch worker thread is not running")
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            if self.exception is not None:
                raise RuntimeError(f"Prefetch worker failed: {self.exception}")
            raise RuntimeError(f"Timeout waiting for batch after {timeout}s.")

    def stop(self):
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join(timeout=2.0)
            self.thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb
        self.stop()
        return False


class TrainState(train_state.TrainState):
    step: int


def create_train_state(config: TrainConfig, model: HybridADModel, params: Any) -> TrainState:
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

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
        ),
    )

    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@partial(jax.jit, static_argnames=("num_actions", "label_smoothing", "use_teammate_actions"))
def _train_step_impl(
    state: TrainState,
    batch_data: Tuple,
    dropout_rng: jax.Array,
    num_actions: int,
    label_smoothing: float,
    use_teammate_actions: bool,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    if use_teammate_actions:
        obs, prev_actions, prev_rewards, target_actions, attention_mask, prev_teammate_actions = batch_data
    else:
        obs, prev_actions, prev_rewards, target_actions, attention_mask = batch_data
        prev_teammate_actions = None

    def loss_fn(params):
        logits = state.apply_fn(
            params,
            obs,
            prev_actions,
            prev_rewards,
            attention_mask=attention_mask,
            prev_teammate_actions=prev_teammate_actions,
            train=True,
            rngs={'dropout': dropout_rng},
        )

        batch_size, seq_len = target_actions.shape
        logits_flat = logits.reshape(-1, num_actions)
        targets_flat = target_actions.reshape(-1)

        targets_onehot = jax.nn.one_hot(targets_flat, num_actions)
        if label_smoothing > 0:
            targets_onehot = targets_onehot * (1 - label_smoothing) + label_smoothing / num_actions

        log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
        per_token_loss = -jnp.sum(targets_onehot * log_probs, axis=-1)

        mask_flat = attention_mask.reshape(-1)
        masked_loss = per_token_loss * mask_flat
        loss = jnp.sum(masked_loss) / jnp.maximum(jnp.sum(mask_flat), 1.0)

        predicted = jnp.argmax(logits_flat, axis=-1)
        correct = (predicted == targets_flat) * mask_flat
        accuracy = jnp.sum(correct) / jnp.maximum(jnp.sum(mask_flat), 1.0)

        return loss, {"loss": loss, "accuracy": accuracy}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


_train_rng = jax.random.PRNGKey(0)


def train_step(state, batch_data, config_dict):
    global _train_rng
    _train_rng, dropout_rng = jax.random.split(_train_rng)
    return _train_step_impl(
        state, batch_data, dropout_rng,
        num_actions=config_dict["num_actions"],
        label_smoothing=config_dict["label_smoothing"],
        use_teammate_actions=config_dict["use_teammate_actions"],
    )


@partial(jax.jit, static_argnames=("num_actions", "use_teammate_actions"))
def _eval_step_impl(state, batch_data, num_actions, use_teammate_actions):
    if use_teammate_actions:
        obs, prev_actions, prev_rewards, target_actions, attention_mask, prev_teammate_actions = batch_data
    else:
        obs, prev_actions, prev_rewards, target_actions, attention_mask = batch_data
        prev_teammate_actions = None

    logits = state.apply_fn(
        state.params, obs, prev_actions, prev_rewards,
        attention_mask=attention_mask,
        prev_teammate_actions=prev_teammate_actions,
        train=False,
    )

    logits_flat = logits.reshape(-1, num_actions)
    targets_flat = target_actions.reshape(-1)
    targets_onehot = jax.nn.one_hot(targets_flat, num_actions)

    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    per_token_loss = -jnp.sum(targets_onehot * log_probs, axis=-1)

    mask_flat = attention_mask.reshape(-1)
    masked_loss = per_token_loss * mask_flat
    loss = jnp.sum(masked_loss) / jnp.maximum(jnp.sum(mask_flat), 1.0)

    predicted = jnp.argmax(logits_flat, axis=-1)
    correct = (predicted == targets_flat) * mask_flat
    accuracy = jnp.sum(correct) / jnp.maximum(jnp.sum(mask_flat), 1.0)

    return {"loss": loss, "accuracy": accuracy}


def eval_step(state, batch_data, config_dict):
    return _eval_step_impl(
        state, batch_data,
        num_actions=config_dict["num_actions"],
        use_teammate_actions=config_dict["use_teammate_actions"],
    )


def batch_to_tuple(batch: ADBatch, use_teammate_actions: bool = False) -> Tuple:
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


def evaluate(state, store, config, rng):
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    config_dict = {
        "num_actions": config.num_actions,
        "label_smoothing": 0.0,
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
        metrics = eval_step(state, batch_data, config_dict)
        total_loss += float(metrics["loss"])
        total_acc += float(metrics["accuracy"])
        num_batches += 1

    return {
        "eval_loss": total_loss / num_batches,
        "eval_accuracy": total_acc / num_batches,
    }


def train(config: TrainConfig):
    global _train_rng

    out_dir = Path(config.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    log.info(f"Training Hybrid-AD model")
    log.info(f"  Output: {out_dir}")
    log.info(f"  H5 path: {config.h5_path}")
    log.info(f"  Obs shape: {config.obs_shape}")
    log.info(f"  Batch size: {config.batch_size}")
    log.info(f"  Seq len: {config.seq_len}")
    log.info(f"  GRU hidden dim: {config.gru_hidden_dim}")
    log.info(f"  GRU layers: {config.num_gru_layers}")
    log.info(f"  Num steps: {config.num_steps}")
    log.info(f"  Use teammate actions: {config.use_teammate_actions}")

    _train_rng = jax.random.PRNGKey(config.seed)

    model_config = config.to_model_config()
    model, params = create_hybrid_ad_model(model_config)

    num_params = count_parameters(params)
    log.info(f"  Model parameters: {num_params:,}")

    state = create_train_state(config, model, params)

    start_step = 0
    ckpt_dir = out_dir / "checkpoints"
    if config.resume and ckpt_dir.exists():
        state = checkpoints.restore_checkpoint(ckpt_dir=str(ckpt_dir), target=state)
        start_step = int(state.step)
        if start_step > 0:
            log.info(f"Resumed from checkpoint at step {start_step}")
            _train_rng = jax.random.PRNGKey(config.seed)
            for _ in range(start_step):
                _train_rng, _ = jax.random.split(_train_rng)

    log.info("Loading dataset...")
    store = HistoryStore(config.h5_path, config.index_path, cache_size_mb=config.hdf5_cache_mb)
    log.info(f"  Histories: {len(store)}")
    log.info(f"  Tasks: {len(store.get_task_ids())}")

    if len(store) > 0:
        meta = store.get_history_meta(0)
        data_obs_shape = tuple(meta["obs_shape"])
        if data_obs_shape != config.obs_shape:
            log.warning(f"  obs_shape mismatch: config={config.obs_shape}, data={data_obs_shape}")
            config.obs_shape = data_obs_shape
            model_config = config.to_model_config()
            model, params = create_hybrid_ad_model(model_config)
            state = create_train_state(config, model, params)

    rng = np.random.default_rng(config.seed)

    config_dict = {
        "num_actions": config.num_actions,
        "label_smoothing": config.label_smoothing,
        "use_teammate_actions": config.use_teammate_actions,
    }

    csv_logger = None
    if config.csv_log:
        csv_path = out_dir / "training_metrics.csv"
        csv_fieldnames = [
            "step", "loss", "accuracy", "eval_loss", "eval_accuracy",
            "steps_per_sec", "elapsed_time"
        ]
        csv_logger = CSVMetricsLogger(str(csv_path), csv_fieldnames)

    all_metrics = []
    metrics_path = out_dir / "metrics.json"
    if config.resume and metrics_path.exists() and start_step > 0:
        try:
            with open(metrics_path, "r") as f:
                all_metrics = json.load(f)
            all_metrics = [m for m in all_metrics if m.get("step", 0) < start_step]
        except Exception:
            all_metrics = []

    log.info("Starting training...")
    start_time = time.time()

    if config.prefetch_buffer_size > 0:
        data_loader = PrefetchDataLoader(
            store=store, rng=rng, batch_size=config.batch_size,
            seq_len=config.seq_len, use_teammate_actions=config.use_teammate_actions,
            buffer_size=config.prefetch_buffer_size,
        )
        data_loader.start()
    else:
        data_loader = None

    try:
        for step in range(start_step, config.num_steps):
            if data_loader is not None:
                batch = data_loader.get_batch()
            else:
                batch = store.sample_ad_batch(
                    rng=rng, batch_size=config.batch_size,
                    seq_len=config.seq_len,
                    include_teammate_actions=config.use_teammate_actions,
                )

            batch_data = batch_to_tuple(batch, use_teammate_actions=config.use_teammate_actions)

            if config.use_async_transfer:
                batch_data = jax.tree_map(jax.device_put, batch_data)

            state, metrics = train_step(state, batch_data, config_dict)

            if jnp.isnan(metrics["loss"]):
                log.error(f"NaN loss at step {step}!")
                break

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

                if csv_logger is not None and (step % config.csv_interval == 0):
                    csv_logger.log({
                        "step": step,
                        "loss": float(metrics["loss"]),
                        "accuracy": float(metrics["accuracy"]),
                        "steps_per_sec": steps_per_sec,
                        "elapsed_time": elapsed,
                    })

            if config.eval_every > 0 and step > 0 and step % config.eval_every == 0:
                eval_rng = np.random.default_rng(config.seed + step)
                eval_metrics = evaluate(state, store, config, eval_rng)
                log.info(
                    f"  Eval | Loss: {eval_metrics['eval_loss']:.4f} | "
                    f"Acc: {eval_metrics['eval_accuracy']:.4f}"
                )
                all_metrics[-1].update(eval_metrics)

            if config.save_every > 0 and step > 0 and step % config.save_every == 0:
                ckpt_dir = out_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)
                checkpoints.save_checkpoint(ckpt_dir=str(ckpt_dir), target=state, step=step, keep=1e4)
                with open(out_dir / "metrics.json", "w") as f:
                    json.dump(all_metrics, f, indent=2)
                log.info(f"  Saved checkpoint at step {step}")

    finally:
        if data_loader is not None:
            data_loader.stop()

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    checkpoints.save_checkpoint(ckpt_dir=str(ckpt_dir), target=state, step=config.num_steps, keep=1e4)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    elapsed = time.time() - start_time
    log.info(f"Training complete! Elapsed: {elapsed:.1f}s")
    store.close()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train Hybrid-AD model")

    parser.add_argument("--h5_path", type=str, required=True)
    parser.add_argument("--index_path", type=str, required=True)

    parser.add_argument("--num_actions", type=int, default=6)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--gru_hidden_dim", type=int, default=256)
    parser.add_argument("--num_gru_layers", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=500)

    parser.add_argument("--embedding_dropout", type=float, default=0.1)

    parser.add_argument("--reset_hidden_on_done", action="store_true")

    parser.add_argument("--num_steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--eval_batches", type=int, default=3)

    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="outputs/hybrid_ad")
    parser.add_argument("--log_every", type=int, default=100)

    parser.add_argument("--use_teammate_actions", action="store_true")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-csv-log", action="store_false", dest="csv_log", default=True)
    parser.add_argument("--csv_interval", type=int, default=100)

    parser.add_argument("--prefetch_buffer_size", type=int, default=5)
    parser.add_argument("--hdf5_cache_mb", type=float, default=512.0)
    parser.add_argument("--no-async-transfer", action="store_false", dest="use_async_transfer", default=True)

    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    resolved_eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
    resolved_warmup_steps = args.warmup_steps if args.warmup_steps is not None else max(1, int(args.num_steps * 0.05))

    obs_shape = get_obs_shape_from_index(args.index_path)
    log.info(f"Index '{args.index_path}' -> obs_shape: {obs_shape}")

    return TrainConfig(
        h5_path=args.h5_path,
        index_path=args.index_path,
        obs_shape=obs_shape,
        num_actions=args.num_actions,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        gru_hidden_dim=args.gru_hidden_dim,
        num_gru_layers=args.num_gru_layers,
        seq_len=args.seq_len,
        embedding_dropout=args.embedding_dropout,
        reset_hidden_on_done=args.reset_hidden_on_done,
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
    config = parse_args()
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
