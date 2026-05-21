#!/usr/bin/env python3
"""Training script for FCP (Fictitious Co-Play) on Overcooked V2.

This script provides a standalone entry point for training FCP teammate policies
on Overcooked V2 environments, without requiring Hydra configuration.

Example Usage:
    # Train on cramped_room layout with default settings
    python scripts/train_fcp_overcooked_v2.py --layout cramped_room

    # Train with custom parameters
    python scripts/train_fcp_overcooked_v2.py \
        --layout coord_ring \
        --total_timesteps 5e6 \
        --partner_pop_size 10 \
        --num_checkpoints 5 \
        --seed 0

    # Quick test run
    python scripts/train_fcp_overcooked_v2.py \
        --layout cramped_room \
        --total_timesteps 1e5 \
        --partner_pop_size 2 \
        --num_checkpoints 2 \
        --debug

    # Save trained models
    python scripts/train_fcp_overcooked_v2.py \
        --layout cramped_room \
        --output_dir outputs/fcp_cramped_room

    # Run on specific GPU
    python scripts/train_fcp_overcooked_v2.py \
        --layout cramped_room \
        --gpu 0
"""

import argparse
import csv
import logging
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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

from agents.mlp_actor_critic_agent import MLPActorCriticPolicy
from agents.cnn_rnn_actor_critic_agent import CNNRNNActorCriticPolicy
from agents.population_interface import AgentPopulation
from envs import make_env
from envs.log_wrapper import LogWrapper
from envs.overcooked_v2 import overcooked_v2_layouts
from marl.ippo import make_train as make_ppo_train, make_train_chunked
from common.stats_utils import get_metric_names, get_stats
from common.save_load_utils import (
    save_train_run,
    load_train_run,
    save_separated_checkpoints,
    load_individual_checkpoint,
    load_all_checkpoints_from_separated,
)

# Setup logging
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)


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
        if not self._initialized:
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


def compute_checkpoint_returns(
    metrics: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Compute mean returns at each checkpoint interval from training metrics.

    Args:
        metrics: Dictionary of metric arrays with shape (num_seeds, partner_pop_size, num_updates, rollout_len, num_actors).
        config: Training configuration.

    Returns:
        Dictionary with:
            - base_returns: Mean base returns at each checkpoint, shape (num_seeds, pop_size, num_checkpoints)
            - shaped_returns: Mean shaped returns at each checkpoint, shape (num_seeds, pop_size, num_checkpoints)
    """
    num_checkpoints = config["NUM_CHECKPOINTS"]
    num_updates = config["NUM_UPDATES"]
    num_seeds = config.get("NUM_SEEDS", 1)
    pop_size = config.get("PARTNER_POP_SIZE", 1)

    # Calculate checkpoint intervals (same logic as in training)
    ckpt_interval = num_updates // max(1, num_checkpoints - 1)

    # Checkpoint update indices
    checkpoint_update_indices = []
    for ckpt_idx in range(num_checkpoints - 1):
        checkpoint_update_indices.append(ckpt_idx * ckpt_interval)  # 0-indexed
    checkpoint_update_indices.append(num_updates - 1)  # Last checkpoint at final update

    # Initialize arrays for checkpoint returns
    base_returns = np.zeros((num_seeds, pop_size, num_checkpoints))
    shaped_returns = np.zeros((num_seeds, pop_size, num_checkpoints))

    # Get mask of completed episodes
    if "returned_episode" in metrics:
        ep_done = metrics["returned_episode"]  # (num_seeds, pop_size, num_updates, rollout_len, num_actors)
    else:
        ep_done = None

    # Compute mean returns for each checkpoint
    for ckpt_idx, update_idx in enumerate(checkpoint_update_indices):
        # Use a window from previous checkpoint to current checkpoint
        if ckpt_idx == 0:
            start_idx = 0
        else:
            start_idx = checkpoint_update_indices[ckpt_idx - 1] + 1
        end_idx = update_idx + 1  # inclusive

        # Extract returns for this window
        if "returned_episode_returns" in metrics:
            window_base_returns = metrics["returned_episode_returns"][:, :, start_idx:end_idx, :, :]
            if ep_done is not None:
                window_ep_done = ep_done[:, :, start_idx:end_idx, :, :] > 0
                for seed_idx in range(num_seeds):
                    for pop_idx in range(pop_size):
                        mask = window_ep_done[seed_idx, pop_idx]
                        if np.any(mask):
                            base_returns[seed_idx, pop_idx, ckpt_idx] = np.mean(
                                window_base_returns[seed_idx, pop_idx][mask]
                            )

        if "returned_episode_shaped_returns" in metrics:
            window_shaped_returns = metrics["returned_episode_shaped_returns"][:, :, start_idx:end_idx, :, :]
            if ep_done is not None:
                window_ep_done = ep_done[:, :, start_idx:end_idx, :, :] > 0
                for seed_idx in range(num_seeds):
                    for pop_idx in range(pop_size):
                        mask = window_ep_done[seed_idx, pop_idx]
                        if np.any(mask):
                            shaped_returns[seed_idx, pop_idx, ckpt_idx] = np.mean(
                                window_shaped_returns[seed_idx, pop_idx][mask]
                            )

    return {
        "base_returns": base_returns,
        "shaped_returns": shaped_returns,
    }


def write_training_metrics_to_csv(
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: str,
    training_time: float = 0.0,
):
    """Write training metrics to CSV file.

    This function handles writing metrics from both chunked and non-chunked training.
    It writes one row per update step with aggregated metrics.

    Args:
        metrics: Dictionary of metric arrays with shape (num_seeds, partner_pop_size, num_updates, ...).
        config: Training configuration.
        output_dir: Directory to save the CSV file.
        training_time: Total training time in seconds.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "training_metrics.csv"

    partner_pop_size = config["PARTNER_POP_SIZE"]
    num_agents = 2  # OvercookedV2 always has 2 agents

    # Calculate timesteps per update
    rollout_length = config["ROLLOUT_LENGTH"]
    num_envs = config["NUM_ENVS"]
    timesteps_per_update = rollout_length * num_envs * num_agents

    # Build fieldnames (per-partner metrics)
    csv_fieldnames = ["update_step", "total_timesteps"]
    for p in range(partner_pop_size):
        csv_fieldnames.extend([
            f"p{p}_mean_base_return", f"p{p}_std_base_return",
            f"p{p}_mean_shaped_return", f"p{p}_std_shaped_return",
            f"p{p}_mean_episode_length", f"p{p}_std_episode_length",
        ])
    csv_fieldnames.append("elapsed_time")

    csv_logger = CSVMetricsLogger(str(csv_path), csv_fieldnames)

    # Get number of updates from metrics shape
    # Shape: (num_seeds, partner_pop_size, num_updates, rollout_len, num_actors)
    if "returned_episode_returns" in metrics:
        num_updates = metrics["returned_episode_returns"].shape[2]
    elif "update_steps" in metrics:
        num_updates = metrics["update_steps"].shape[2]
    else:
        log.warning("Cannot determine number of updates from metrics. Skipping CSV logging.")
        return

    # Write one row per update step
    for update_idx in range(num_updates):
        current_timesteps = (update_idx + 1) * timesteps_per_update

        csv_row = {
            "update_step": update_idx + 1,
            "total_timesteps": current_timesteps,
            "elapsed_time": training_time * (update_idx + 1) / num_updates,
        }

        # Get mask of completed episodes for this update
        # Shape: (num_seeds, partner_pop_size, rollout_len, num_actors)
        if "returned_episode" in metrics:
            ep_done_mask = metrics["returned_episode"][:, :, update_idx, :, :] > 0
        else:
            ep_done_mask = None

        # Extract per-partner metrics for this update step
        for p in range(partner_pop_size):
            # Base returns for partner p
            if "returned_episode_returns" in metrics:
                # Shape for partner p: (num_seeds, rollout_len, num_actors)
                partner_base = metrics["returned_episode_returns"][:, p, update_idx, :, :]
                if ep_done_mask is not None:
                    partner_ep_done = ep_done_mask[:, p, :, :]
                    if np.any(partner_ep_done):
                        valid_returns = partner_base[partner_ep_done]
                        csv_row[f"p{p}_mean_base_return"] = float(np.mean(valid_returns))
                        csv_row[f"p{p}_std_base_return"] = float(np.std(valid_returns))
                    else:
                        csv_row[f"p{p}_mean_base_return"] = 0.0
                        csv_row[f"p{p}_std_base_return"] = 0.0
                else:
                    csv_row[f"p{p}_mean_base_return"] = float(np.mean(partner_base))
                    csv_row[f"p{p}_std_base_return"] = float(np.std(partner_base))
            else:
                csv_row[f"p{p}_mean_base_return"] = 0.0
                csv_row[f"p{p}_std_base_return"] = 0.0

            # Shaped returns for partner p
            if "returned_episode_shaped_returns" in metrics:
                partner_shaped = metrics["returned_episode_shaped_returns"][:, p, update_idx, :, :]
                if ep_done_mask is not None:
                    partner_ep_done = ep_done_mask[:, p, :, :]
                    if np.any(partner_ep_done):
                        valid_returns = partner_shaped[partner_ep_done]
                        csv_row[f"p{p}_mean_shaped_return"] = float(np.mean(valid_returns))
                        csv_row[f"p{p}_std_shaped_return"] = float(np.std(valid_returns))
                    else:
                        csv_row[f"p{p}_mean_shaped_return"] = 0.0
                        csv_row[f"p{p}_std_shaped_return"] = 0.0
                else:
                    csv_row[f"p{p}_mean_shaped_return"] = float(np.mean(partner_shaped))
                    csv_row[f"p{p}_std_shaped_return"] = float(np.std(partner_shaped))
            else:
                # Fall back to base returns
                csv_row[f"p{p}_mean_shaped_return"] = csv_row[f"p{p}_mean_base_return"]
                csv_row[f"p{p}_std_shaped_return"] = csv_row[f"p{p}_std_base_return"]

            # Episode lengths for partner p
            if "returned_episode_lengths" in metrics:
                partner_lengths = metrics["returned_episode_lengths"][:, p, update_idx, :, :]
                if ep_done_mask is not None:
                    partner_ep_done = ep_done_mask[:, p, :, :]
                    if np.any(partner_ep_done):
                        valid_lengths = partner_lengths[partner_ep_done]
                        csv_row[f"p{p}_mean_episode_length"] = float(np.mean(valid_lengths))
                        csv_row[f"p{p}_std_episode_length"] = float(np.std(valid_lengths))
                    else:
                        csv_row[f"p{p}_mean_episode_length"] = 0.0
                        csv_row[f"p{p}_std_episode_length"] = 0.0
                else:
                    csv_row[f"p{p}_mean_episode_length"] = float(np.mean(partner_lengths))
                    csv_row[f"p{p}_std_episode_length"] = float(np.std(partner_lengths))
            else:
                csv_row[f"p{p}_mean_episode_length"] = 0.0
                csv_row[f"p{p}_std_episode_length"] = 0.0

        csv_logger.log(csv_row)

    log.info(f"Wrote training metrics to: {csv_path}")


def get_default_config(layout: str) -> Dict[str, Any]:
    """Get default FCP training configuration for a given layout.

    Uses hyperparameters from the working JaxMARL IPPO recipe for OvercookedV2.

    Args:
        layout: Name of the Overcooked V2 layout.

    Returns:
        Dictionary with training configuration.
    """
    return {
        # Environment settings
        "ENV_NAME": "overcooked-v2",
        "ENV_KWARGS": {
            "layout": layout,
            "flatten_obs": False,  # Use CNN-based observation (not flattened)
            "max_steps": 400,  # Maximum steps per episode
        },
        "ROLLOUT_LENGTH": 256,  # NUM_STEPS in JaxMARL recipe

        # FCP-specific settings
        "ALG": "fcp",
        "ACTOR_TYPE": "cnn_rnn",  # Use CNN+RNN architecture like JaxMARL
        "TOTAL_TIMESTEPS": 3e7,  # From JaxMARL recipe
        "NUM_CHECKPOINTS": 5,
        "PARTNER_POP_SIZE": 10,  # True pop size = PARTNER_POP_SIZE * NUM_CHECKPOINTS

        # PPO hyperparameters (from working JaxMARL recipe)
        "NUM_ENVS": 256,  # Critical: much larger than before
        "LR": 0.00025,  # From JaxMARL recipe
        "ANNEAL_LR": True,
        "LR_WARMUP": 0.05,  # 5% of updates for warmup
        "UPDATE_EPOCHS": 4,  # From JaxMARL recipe (was 15)
        "NUM_MINIBATCHES": 64,  # 400 actors / 50 = 8 actors per minibatch (must divide evenly)
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.25,  # From JaxMARL recipe (was 1.0)
        "ACTIVATION": "relu",  # From JaxMARL recipe (was tanh)

        # Network architecture (CNN+RNN)
        "FC_DIM_SIZE": 128,  # From JaxMARL recipe
        "GRU_HIDDEN_DIM": 128,  # From JaxMARL recipe

        # Reward shaping (from JaxMARL recipe)
        "REW_SHAPING_HORIZON": 1.5e7,  # Anneal shaped rewards over this many timesteps

        # Training settings
        "NUM_SEEDS": 1,
        "TRAIN_SEED": 0,
        "NUM_EVAL_EPISODES": 20,
        "NUM_CHUNKS": 1,  # Number of chunks for memory-efficient training
    }


def train_fcp_partners(rng, env, config: Dict[str, Any]):
    """Train a single seed of FCP partners.

    Args:
        rng: JAX random key.
        env: Environment instance.
        config: Training configuration.

    Returns:
        Training output dictionary containing checkpoints and metrics.
    """
    rngs = jax.random.split(rng, config["PARTNER_POP_SIZE"])
    train_jit = jax.jit(jax.vmap(make_ppo_train(config, env)))
    out = train_jit(rngs)
    return out


def get_fcp_population(config: Dict[str, Any], out: Dict[str, Any], env):
    """Extract flattened partner population from training output.

    Args:
        config: Training configuration.
        out: Training output dictionary.
        env: Environment instance.

    Returns:
        Tuple of (flattened_partner_params, partner_population).
    """
    num_seeds = config["NUM_SEEDS"]
    fcp_pop_size = config["PARTNER_POP_SIZE"] * config["NUM_CHECKPOINTS"]

    partner_params = out['checkpoints']
    flattened_partner_params = jax.tree.map(
        lambda x: x.reshape(num_seeds, fcp_pop_size, *x.shape[3:]),
        partner_params
    )

    # Create the appropriate policy based on actor type
    actor_type = config.get("ACTOR_TYPE", "mlp")
    if actor_type == "cnn_rnn":
        # Use CNN+RNN policy for grid-based observations
        partner_policy = CNNRNNActorCriticPolicy(
            action_dim=env.action_space(env.agents[1]).n,
            obs_shape=env.observation_space(env.agents[1]).shape,
            activation=config.get("ACTIVATION", "relu"),
            fc_dim_size=config.get("FC_DIM_SIZE", 128),
            gru_hidden_dim=config.get("GRU_HIDDEN_DIM", 128),
            use_avail_actions=True,
        )
    else:
        # Use MLP policy for flattened observations
        partner_policy = MLPActorCriticPolicy(
            action_dim=env.action_space(env.agents[1]).n,
            obs_dim=env.observation_space(env.agents[1]).shape[0],
            activation=config.get("ACTIVATION", "tanh")
        )

    partner_population = AgentPopulation(
        pop_size=fcp_pop_size,
        policy_cls=partner_policy
    )

    return flattened_partner_params, partner_population


def run_fcp_training(config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """Run FCP training.

    Args:
        config: Training configuration dictionary.
        verbose: Whether to print progress information.

    Returns:
        Dictionary containing training results:
            - partner_params: Flattened partner parameters
            - partner_population: AgentPopulation instance
            - metrics: Training metrics
            - checkpoints: Model checkpoints
    """
    rng = jax.random.PRNGKey(config["TRAIN_SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    # Create environment
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)

    if verbose:
        log.info(f"Environment: {config['ENV_NAME']}")
        log.info(f"Layout: {config['ENV_KWARGS']['layout']}")
        log.info(f"Observation space: {env.observation_space(env.agents[0]).shape}")
        log.info(f"Action space: {env.action_space(env.agents[0]).n}")
        log.info(f"Actor type: {config.get('ACTOR_TYPE', 'mlp')}")
        log.info(f"Partner population size: {config['PARTNER_POP_SIZE']}")
        log.info(f"Total checkpoints: {config['PARTNER_POP_SIZE'] * config['NUM_CHECKPOINTS']}")
        log.info(f"Total timesteps per partner: {config['TOTAL_TIMESTEPS']}")
        log.info(f"Learning rate: {config['LR']}")
        log.info(f"NUM_ENVS: {config['NUM_ENVS']}")
        if config.get('REW_SHAPING_HORIZON', 0) > 0:
            log.info(f"Reward shaping horizon: {config['REW_SHAPING_HORIZON']}")

    # Run training
    start_time = time.time()

    num_chunks = config.get("NUM_CHUNKS", 1)

    if num_chunks > 1:
        # Use chunked training to reduce GPU memory usage
        if verbose:
            log.info(f"Using chunked training with {num_chunks} chunks")

        out = _run_chunked_training(config, env, rngs, verbose)
    else:
        # Original non-chunked training
        with jax.disable_jit(False):
            vmapped_train_fn = jax.jit(
                jax.vmap(
                    partial(train_fcp_partners, env=env, config=config)
                )
            )
            out = vmapped_train_fn(rngs)

    end_time = time.time()
    training_time = end_time - start_time

    if verbose:
        log.info(f"Training completed in {training_time:.2f} seconds")

    # Extract population
    flattened_partner_params, partner_population = get_fcp_population(config, out, env)

    # Compute statistics
    metric_names = get_metric_names(config["ENV_NAME"])
    partner_metrics = out["metrics"]
    num_partner_updates = partner_metrics["returned_episode_returns"].shape[2]
    num_controlled_actors = config["NUM_ENVS"]
    partner_metrics = jax.tree.map(lambda x: x[..., :num_controlled_actors], partner_metrics)
    partner_stats = get_stats(partner_metrics, metric_names)

    # Merge updated_config (with NUM_UPDATES) back into config if available
    final_config = config.copy()
    if "updated_config" in out:
        final_config.update(out["updated_config"])

    results = {
        "partner_params": flattened_partner_params,
        "partner_population": partner_population,
        "checkpoints": out["checkpoints"],
        "metrics": out["metrics"],
        "stats": partner_stats,
        "training_time": training_time,
        "config": final_config,
    }

    return results


def _run_chunked_training(
    config: Dict[str, Any],
    env,
    rngs: jnp.ndarray,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run training in chunks to reduce GPU memory usage.

    This function trains in chunks, moving metrics to CPU after each chunk
    to free GPU memory. This allows training with larger total_timesteps
    without running out of GPU memory.

    Args:
        config: Training configuration dictionary.
        env: Environment instance.
        rngs: Random keys for each seed, shape (NUM_SEEDS,).
        verbose: Whether to print progress information.

    Returns:
        Dictionary with same structure as non-chunked training output.
    """
    num_chunks = config["NUM_CHUNKS"]
    partner_pop_size = config["PARTNER_POP_SIZE"]
    save_interval = config.get("SAVE_INTERVAL", 0)
    output_dir = config.get("OUTPUT_DIR", None)

    # Calculate NUM_UPDATES to validate divisibility
    num_updates = int(
        config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
    )

    # Validate that NUM_UPDATES is divisible by NUM_CHUNKS
    if num_updates % num_chunks != 0:
        suggested_chunks = []
        for c in [num_chunks - 1, num_chunks + 1, num_chunks - 2, num_chunks + 2]:
            if c > 0 and num_updates % c == 0:
                suggested_chunks.append(c)
        suggestion = f" Try --num_chunks {suggested_chunks[0]}" if suggested_chunks else ""
        raise ValueError(
            f"NUM_UPDATES ({num_updates}) must be divisible by NUM_CHUNKS ({num_chunks}).{suggestion}"
        )

    updates_per_chunk = num_updates // num_chunks
    if verbose:
        log.info(f"NUM_UPDATES={num_updates}, updates_per_chunk={updates_per_chunk}")
        if save_interval > 0 and output_dir:
            log.info(f"Will save checkpoints every {save_interval} chunks to {output_dir}")

    # Get the chunked training functions
    init_fn, train_chunk_fn, updated_config = make_train_chunked(config, env)

    # Create vmapped and jitted functions
    # Structure: vmap over seeds -> vmap over partners
    def init_all_partners(seed_rng):
        partner_rngs = jax.random.split(seed_rng, partner_pop_size)
        return jax.vmap(init_fn)(partner_rngs)

    def train_all_partners_one_chunk(states):
        return jax.vmap(train_chunk_fn)(states)

    init_jit = jax.jit(jax.vmap(init_all_partners))
    chunk_jit = jax.jit(jax.vmap(train_all_partners_one_chunk))

    # Initialize all training states
    if verbose:
        log.info("Initializing training states...")
    states = init_jit(rngs)

    # Train in chunks, moving metrics to CPU after each chunk
    all_chunk_metrics = []

    # Setup output directory for interval saving
    if save_interval > 0 and output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Setup CSV logging
    csv_logger = None
    csv_log_enabled = config.get("CSV_LOG", False)
    csv_interval = config.get("CSV_INTERVAL", 1)
    if csv_log_enabled and output_dir:
        csv_path = Path(output_dir) / "metrics.csv"
        # Build partner-wise fieldnames: p0_mean_base_return, p0_std_base_return, etc.
        csv_fieldnames = ["chunk", "total_chunks", "update_step", "total_timesteps"]
        for p in range(partner_pop_size):
            csv_fieldnames.extend([
                f"p{p}_mean_base_return", f"p{p}_std_base_return",
                f"p{p}_mean_shaped_return", f"p{p}_std_shaped_return",
            ])
        csv_fieldnames.append("elapsed_time")
        csv_logger = CSVMetricsLogger(str(csv_path), csv_fieldnames)
        if verbose:
            log.info(f"CSV logging enabled: {csv_path}")

    chunk_start_time = time.time()

    for chunk_idx in range(num_chunks):
        if verbose:
            log.info(f"Training chunk {chunk_idx + 1}/{num_chunks}...")

        # Run one chunk of training
        states, chunk_metrics = chunk_jit(states)

        # Move metrics to CPU to free GPU memory
        chunk_metrics_cpu = jax.device_get(chunk_metrics)
        all_chunk_metrics.append(chunk_metrics_cpu)

        # CSV logging at specified intervals
        if csv_logger is not None and ((chunk_idx + 1) % csv_interval == 0):
            # Compute current metrics from chunk data
            # chunk_metrics has shape (num_seeds, partner_pop_size, updates_per_chunk, rollout_len, num_actors)
            # IMPORTANT: Use returned_episode flag to filter for actual completed episodes
            # This works correctly even when ROLLOUT_LENGTH != max_steps
            base_returns_key = "returned_episode_returns"
            shaped_returns_key = "returned_episode_shaped_returns"
            ep_done_key = "returned_episode"

            current_update = (chunk_idx + 1) * updates_per_chunk
            current_timesteps = current_update * config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
            elapsed = time.time() - chunk_start_time

            # Build per-partner metrics
            csv_row = {
                "chunk": chunk_idx + 1,
                "total_chunks": num_chunks,
                "update_step": current_update,
                "total_timesteps": current_timesteps,
                "elapsed_time": elapsed,
            }

            # Get mask of completed episodes
            # Shape: (num_seeds, partner_pop_size, updates_per_chunk, rollout_len, num_actors)
            if ep_done_key in chunk_metrics_cpu:
                ep_done_mask = chunk_metrics_cpu[ep_done_key] > 0
            else:
                ep_done_mask = None

            # Compute per-partner statistics
            for p in range(partner_pop_size):
                # Base returns for partner p
                if base_returns_key in chunk_metrics_cpu:
                    # Get partner p's data: (num_seeds, updates_per_chunk, rollout_len, num_actors)
                    partner_base = chunk_metrics_cpu[base_returns_key][:, p, ...]
                    if ep_done_mask is not None:
                        partner_ep_done = ep_done_mask[:, p, ...]
                        if np.any(partner_ep_done):
                            valid_returns = partner_base[partner_ep_done]
                            csv_row[f"p{p}_mean_base_return"] = float(np.mean(valid_returns))
                            csv_row[f"p{p}_std_base_return"] = float(np.std(valid_returns))
                        else:
                            csv_row[f"p{p}_mean_base_return"] = 0.0
                            csv_row[f"p{p}_std_base_return"] = 0.0
                    else:
                        csv_row[f"p{p}_mean_base_return"] = float(np.mean(partner_base))
                        csv_row[f"p{p}_std_base_return"] = float(np.std(partner_base))
                else:
                    csv_row[f"p{p}_mean_base_return"] = 0.0
                    csv_row[f"p{p}_std_base_return"] = 0.0

                # Shaped returns for partner p
                if shaped_returns_key in chunk_metrics_cpu:
                    partner_shaped = chunk_metrics_cpu[shaped_returns_key][:, p, ...]
                    if ep_done_mask is not None:
                        partner_ep_done = ep_done_mask[:, p, ...]
                        if np.any(partner_ep_done):
                            valid_returns = partner_shaped[partner_ep_done]
                            csv_row[f"p{p}_mean_shaped_return"] = float(np.mean(valid_returns))
                            csv_row[f"p{p}_std_shaped_return"] = float(np.std(valid_returns))
                        else:
                            csv_row[f"p{p}_mean_shaped_return"] = 0.0
                            csv_row[f"p{p}_std_shaped_return"] = 0.0
                    else:
                        csv_row[f"p{p}_mean_shaped_return"] = float(np.mean(partner_shaped))
                        csv_row[f"p{p}_std_shaped_return"] = float(np.std(partner_shaped))
                else:
                    # Fall back to base returns
                    csv_row[f"p{p}_mean_shaped_return"] = csv_row[f"p{p}_mean_base_return"]
                    csv_row[f"p{p}_std_shaped_return"] = csv_row[f"p{p}_std_base_return"]

            csv_logger.log(csv_row)

        # Save at intervals if configured
        if save_interval > 0 and output_dir and ((chunk_idx + 1) % save_interval == 0):
            if verbose:
                log.info(f"Saving intermediate checkpoint at chunk {chunk_idx + 1}...")

            # Get current checkpoints and metrics
            current_checkpoints = jax.device_get(states["checkpoint_array"])
            current_metrics = jax.tree.map(
                lambda *chunks: np.concatenate(chunks, axis=2),
                *all_chunk_metrics
            )

            save_data = {
                "checkpoints": current_checkpoints,
                "metrics": current_metrics,
            }
            # Compute checkpoint returns from metrics
            checkpoint_returns = compute_checkpoint_returns(current_metrics, updated_config)
            # Save intermediate checkpoints separated by partner and checkpoint index
            save_path = save_separated_checkpoints(
                save_data,
                config,
                str(output_path),
                savename=f"fcp_checkpoint_chunk_{chunk_idx + 1}",
                checkpoint_returns=checkpoint_returns,
            )
            if verbose:
                log.info(f"Saved intermediate separated checkpoint to: {save_path}")

        # Force garbage collection to free GPU memory
        if chunk_idx < num_chunks - 1:
            import gc
            gc.collect()

    # Concatenate metrics from all chunks along the update dimension (axis 2)
    # Metrics shape: (num_seeds, partner_pop_size, updates_per_chunk, ...)
    def concat_metrics(*chunks):
        return np.concatenate(chunks, axis=2)

    metrics = jax.tree.map(concat_metrics, *all_chunk_metrics)

    # Get final checkpoints from state
    checkpoints = jax.device_get(states["checkpoint_array"])

    # Get final params
    final_params = jax.device_get(states["train_state"].params)

    return {
        "final_params": final_params,
        "metrics": metrics,
        "checkpoints": checkpoints,
        "final_ckpt_idx": jax.device_get(states["ckpt_idx"]),
        "updated_config": updated_config,
    }


def print_training_summary(results: Dict[str, Any]):
    """Print a summary of training results.

    Args:
        results: Training results dictionary.
    """
    config = results["config"]
    stats = results["stats"]

    print("\n" + "=" * 60)
    print("FCP Training Summary")
    print("=" * 60)
    print(f"\nLayout: {config['ENV_KWARGS']['layout']}")
    print(f"Partner population size: {config['PARTNER_POP_SIZE'] * config['NUM_CHECKPOINTS']}")
    print(f"Training time: {results['training_time']:.2f} seconds")

    # Print final statistics using proper episode filtering
    # Use returned_episode mask to only include timesteps where episodes actually ended
    if "returned_episode_returns" in stats:
        # stats shape: (num_seeds, partner_pop_size, num_updates, rollout_len, num_actors)
        # Get last update's data
        final_returns_data = stats["returned_episode_returns"][..., -1, :, :]  # Last update
        if "returned_episode" in stats:
            ep_done_mask = stats["returned_episode"][..., -1, :, :] > 0
            if np.any(ep_done_mask):
                valid_returns = final_returns_data[ep_done_mask]
                mean_return = np.mean(valid_returns)
            else:
                mean_return = 0.0
        else:
            # Fallback if no mask available
            mean_return = np.mean(final_returns_data)
        print(f"\nFinal mean episode return: {mean_return:.2f}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train FCP teammates on Overcooked V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--layout",
        type=str,
        required=True,
        choices=list(overcooked_v2_layouts.keys()),
        help="Overcooked V2 layout name",
    )

    # Training parameters (defaults from JaxMARL recipe)
    parser.add_argument(
        "--total_timesteps",
        type=float,
        default=3e7,
        help="Total timesteps per partner (default: 3e7, from JaxMARL recipe)",
    )
    parser.add_argument(
        "--partner_pop_size",
        type=int,
        default=10,
        help="Number of partners to train (default: 10)",
    )
    parser.add_argument(
        "--num_checkpoints",
        type=int,
        default=5,
        help="Number of checkpoints to save per partner (default: 5)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=256,
        help="Number of parallel environments (default: 256, from JaxMARL recipe)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.00025,
        help="Learning rate (default: 0.00025, from JaxMARL recipe)",
    )
    parser.add_argument(
        "--actor_type",
        type=str,
        default="cnn_rnn",
        choices=["mlp", "rnn", "cnn_rnn"],
        help="Actor network type (default: cnn_rnn, from JaxMARL recipe)",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=1,
        help="Number of training chunks for memory efficiency (default: 1). "
             "Use >1 to reduce GPU memory usage for large total_timesteps. "
             "Example: --num_chunks 25 for 5e7 timesteps.",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="Save checkpoints and metrics every N chunks during training (0: only save at end). "
             "Requires --num_chunks > 1 and --output_dir to be set. "
             "Example: --num_chunks 10 --save_interval 2 saves every 2 chunks.",
    )

    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save outputs (default: None, no saving)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with reduced settings",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--list_layouts",
        action="store_true",
        help="List available layouts and exit",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU device ID(s) to use (e.g., '0', '0,1'). Use '-1' for CPU. If not specified, uses all available GPUs.",
    )
    parser.add_argument(
        "--csv_log",
        type=bool,
        default=True,
        help="Enable CSV logging of metrics",
    )
    parser.add_argument(
        "--csv_interval",
        type=int,
        default=1,
        help="Log metrics to CSV every N chunks (default: 1, every chunk)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=400,
        help="Maximum steps per episode (default: 400)",
    )

    args = parser.parse_args()

    # Handle list layouts
    if args.list_layouts:
        print("\nAvailable Overcooked V2 Layouts:")
        print("-" * 40)
        for layout_name in sorted(overcooked_v2_layouts.keys()):
            print(f"  {layout_name}")
        print()
        return

    # Build configuration
    config = get_default_config(args.layout)
    config["TOTAL_TIMESTEPS"] = args.total_timesteps
    config["PARTNER_POP_SIZE"] = args.partner_pop_size
    config["NUM_CHECKPOINTS"] = args.num_checkpoints
    config["NUM_ENVS"] = args.num_envs
    config["TRAIN_SEED"] = args.seed
    config["LR"] = args.lr
    config["NUM_CHUNKS"] = args.num_chunks
    config["SAVE_INTERVAL"] = args.save_interval
    config["OUTPUT_DIR"] = args.output_dir
    config["ACTOR_TYPE"] = args.actor_type
    config["CSV_LOG"] = args.csv_log
    config["CSV_INTERVAL"] = args.csv_interval
    config["ENV_KWARGS"]["max_steps"] = args.max_steps

    # Validate save_interval settings
    if args.save_interval > 0:
        if args.num_chunks <= 1:
            log.warning("--save_interval requires --num_chunks > 1. Interval saving disabled.")
            config["SAVE_INTERVAL"] = 0
        elif args.output_dir is None:
            log.warning("--save_interval requires --output_dir to be set. Interval saving disabled.")
            config["SAVE_INTERVAL"] = 0

    # Debug mode overrides
    if args.debug:
        config["TOTAL_TIMESTEPS"] = 1e5
        config["PARTNER_POP_SIZE"] = 2
        config["NUM_CHECKPOINTS"] = 2
        config["NUM_ENVS"] = 4
        config["NUM_MINIBATCHES"] = 2  # Must be <= NUM_ACTORS (NUM_ENVS * 2)
        config["UPDATE_EPOCHS"] = 2
        log.info("Running in DEBUG mode with reduced settings")

    # Run training
    if not args.quiet:
        log.info(f"Starting FCP training on layout: {args.layout}")
        if args.gpu is not None:
            if args.gpu == "-1":
                log.info("Using CPU")
            else:
                log.info(f"Using GPU: {args.gpu}")
        else:
            log.info(f"Using all available devices: {jax.devices()}")

    try:
        results = run_fcp_training(config, verbose=not args.quiet)

        if not args.quiet:
            print_training_summary(results)

        # Save results if output directory specified
        if args.output_dir:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            save_data = {
                "checkpoints": results["checkpoints"],
                "metrics": results["metrics"],
            }
            # Compute checkpoint returns from metrics
            # Use results["config"] which contains NUM_UPDATES computed during training
            checkpoint_returns = compute_checkpoint_returns(results["metrics"], results["config"])
            # Save checkpoints separated by partner and checkpoint index
            # Structure: fcp_train_run/pi_0/ckpt_0, pi_0/ckpt_1, ..., pi_1/ckpt_0, ...
            save_path = save_separated_checkpoints(
                save_data, results["config"], str(output_path), savename="fcp_train_run",
                checkpoint_returns=checkpoint_returns,
            )
            log.info(f"Saved separated training results to: {save_path}")

            # Write detailed training metrics to CSV (for non-chunked training)
            # Chunked training writes metrics progressively during training
            if results["config"].get("CSV_LOG", False) and results["config"].get("NUM_CHUNKS", 1) <= 1:
                write_training_metrics_to_csv(
                    metrics=results["metrics"],
                    config=results["config"],
                    output_dir=args.output_dir,
                    training_time=results.get("training_time", 0.0),
                )

        return results

    except Exception as e:
        log.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
