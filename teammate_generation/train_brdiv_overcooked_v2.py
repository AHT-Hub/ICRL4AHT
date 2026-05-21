#!/usr/bin/env python3
"""Training script for BRDiv (Best Response Diversity) on Overcooked V2.

This script provides a standalone entry point for training BRDiv teammate policies
on Overcooked V2 environments, without requiring Hydra configuration.

BRDiv trains a population of diverse teammates by optimizing a diversity objective
that encourages confederate policies to achieve high self-play returns while
achieving low cross-play returns with other confederates' best responses.

Example Usage:
    # Train on cramped_room layout with default settings
    python scripts/train_brdiv_overcooked_v2.py --layout cramped_room

    # Train with custom parameters
    python scripts/train_brdiv_overcooked_v2.py \
        --layout coord_ring \
        --total_timesteps 4.5e7 \
        --partner_pop_size 4 \
        --xp_loss_weight 0.5 \
        --seed 0

    # Quick test run
    python scripts/train_brdiv_overcooked_v2.py \
        --layout cramped_room \
        --total_timesteps 1e5 \
        --partner_pop_size 2 \
        --num_checkpoints 2 \
        --debug

    # Save trained models
    python scripts/train_brdiv_overcooked_v2.py \
        --layout cramped_room \
        --output_dir outputs/brdiv_cramped_room

    # Run on specific GPU
    python scripts/train_brdiv_overcooked_v2.py \
        --layout cramped_room \
        --gpu 0

References:
    Rahman et al., "Generating Diverse Cooperative Agents by Learning
    Incompatible Policies", TMLR 2023. https://arxiv.org/abs/2207.14138
"""

import argparse
import csv
import logging
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, Any, Optional, NamedTuple, List

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
import optax
from flax.training.train_state import TrainState

from agents.mlp_actor_critic_agent import ActorWithConditionalCriticPolicy
from agents.cnn_rnn_actor_critic_agent import CNNRNNActorCriticWithConditionalCriticPolicy
from agents.population_interface import AgentPopulation
from envs import make_env
from envs.log_wrapper import LogWrapper
from envs.overcooked_v2 import overcooked_v2_layouts
from common.stats_utils import get_metric_names
from common.run_episodes import run_episodes
from common.save_load_utils import save_train_run, save_separated_checkpoints_multi
from marl.ppo_utils import unbatchify, _create_minibatches

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
        metrics: Dictionary of metric arrays with shape (num_seeds, num_updates, rollout_len, num_actors).
        config: Training configuration.

    Returns:
        Dictionary with:
            - base_returns: Mean base returns at each checkpoint, shape (num_seeds, num_checkpoints)
            - shaped_returns: Mean shaped returns at each checkpoint, shape (num_seeds, num_checkpoints)
    """
    num_checkpoints = config["NUM_CHECKPOINTS"]
    num_updates = config["NUM_UPDATES"]
    num_seeds = config.get("NUM_SEEDS", 1)

    # Calculate checkpoint intervals (same logic as in training)
    ckpt_interval = num_updates // max(1, num_checkpoints - 1)

    # Checkpoint update indices
    checkpoint_update_indices = []
    for ckpt_idx in range(num_checkpoints - 1):
        checkpoint_update_indices.append(ckpt_idx * ckpt_interval)  # 0-indexed
    checkpoint_update_indices.append(num_updates - 1)  # Last checkpoint at final update

    # Initialize arrays for checkpoint returns
    base_returns = np.zeros((num_seeds, num_checkpoints))
    shaped_returns = np.zeros((num_seeds, num_checkpoints))

    # Get mask of completed episodes
    if "returned_episode" in metrics:
        ep_done = metrics["returned_episode"]  # (num_seeds, num_updates, rollout_len, num_actors)
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
            window_base_returns = metrics["returned_episode_returns"][:, start_idx:end_idx, :, :]
            if ep_done is not None:
                window_ep_done = ep_done[:, start_idx:end_idx, :, :] > 0
                for seed_idx in range(num_seeds):
                    mask = window_ep_done[seed_idx]
                    if np.any(mask):
                        base_returns[seed_idx, ckpt_idx] = np.mean(window_base_returns[seed_idx][mask])

        if "returned_episode_shaped_returns" in metrics:
            window_shaped_returns = metrics["returned_episode_shaped_returns"][:, start_idx:end_idx, :, :]
            if ep_done is not None:
                window_ep_done = ep_done[:, start_idx:end_idx, :, :] > 0
                for seed_idx in range(num_seeds):
                    mask = window_ep_done[seed_idx]
                    if np.any(mask):
                        shaped_returns[seed_idx, ckpt_idx] = np.mean(window_shaped_returns[seed_idx][mask])

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
        metrics: Dictionary of metric arrays with shape (num_seeds, num_updates, ...).
        config: Training configuration.
        output_dir: Directory to save the CSV file.
        training_time: Total training time in seconds.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "training_metrics.csv"

    partner_pop_size = config["PARTNER_POP_SIZE"]
    num_agents = 2  # BRDiv always has 2 agents

    # Calculate timesteps per update
    rollout_length = config["ROLLOUT_LENGTH"]
    num_envs = config["NUM_ENVS"]
    timesteps_per_update = rollout_length * num_envs * num_agents

    # Build fieldnames
    csv_fieldnames = [
        "update_step", "total_timesteps",
        # Confederate agent metrics
        "mean_conf_return", "std_conf_return",
        "mean_conf_value_loss", "mean_conf_pg_loss", "mean_conf_entropy",
        # BR agent metrics
        "mean_br_return", "std_br_return",
        "mean_br_value_loss", "mean_br_pg_loss", "mean_br_entropy",
        "elapsed_time",
    ]
    # Add per-population-member metrics
    for p in range(partner_pop_size):
        csv_fieldnames.extend([
            f"conf_{p}_mean_return", f"br_{p}_mean_return",
        ])

    csv_logger = CSVMetricsLogger(str(csv_path), csv_fieldnames)

    # Get number of updates from metrics shape
    # Shape: (num_seeds, num_updates, rollout_len, num_actors)
    if "returned_episode_returns" in metrics:
        num_updates = metrics["returned_episode_returns"].shape[1]
    elif "update_steps" in metrics:
        num_updates = metrics["update_steps"].shape[1]
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
        # IMPORTANT: Use returned_episode flag to filter for actual completed episodes
        # This works correctly even when ROLLOUT_LENGTH != max_steps
        # Shape: (num_seeds, rollout_len, num_actors)
        if "returned_episode" in metrics:
            ep_done_mask = metrics["returned_episode"][:, update_idx, :, :] > 0
        else:
            ep_done_mask = None

        # Extract metrics for this update step
        # Confederate agent returns
        if "returned_episode_returns" in metrics:
            conf_returns = metrics["returned_episode_returns"][:, update_idx, :, :]
            if ep_done_mask is not None and np.any(ep_done_mask):
                valid_returns = conf_returns[ep_done_mask]
                csv_row["mean_conf_return"] = float(np.mean(valid_returns))
                csv_row["std_conf_return"] = float(np.std(valid_returns))
            else:
                csv_row["mean_conf_return"] = 0.0
                csv_row["std_conf_return"] = 0.0
        else:
            csv_row["mean_conf_return"] = 0.0
            csv_row["std_conf_return"] = 0.0

        # BR agent returns (use conf returns as proxy since they share the game)
        csv_row["mean_br_return"] = csv_row["mean_conf_return"]
        csv_row["std_br_return"] = csv_row["std_conf_return"]

        # Loss metrics (these don't need episode filtering - they're per-minibatch)
        if "value_loss_conf_agent" in metrics:
            csv_row["mean_conf_value_loss"] = float(np.mean(metrics["value_loss_conf_agent"][:, update_idx, ...]))
        else:
            csv_row["mean_conf_value_loss"] = 0.0

        if "pg_loss_conf_agent" in metrics:
            csv_row["mean_conf_pg_loss"] = float(np.mean(metrics["pg_loss_conf_agent"][:, update_idx, ...]))
        else:
            csv_row["mean_conf_pg_loss"] = 0.0

        if "entropy_conf" in metrics:
            csv_row["mean_conf_entropy"] = float(np.mean(metrics["entropy_conf"][:, update_idx, ...]))
        else:
            csv_row["mean_conf_entropy"] = 0.0

        if "value_loss_br_agent" in metrics:
            csv_row["mean_br_value_loss"] = float(np.mean(metrics["value_loss_br_agent"][:, update_idx, ...]))
        else:
            csv_row["mean_br_value_loss"] = 0.0

        if "pg_loss_br_agent" in metrics:
            csv_row["mean_br_pg_loss"] = float(np.mean(metrics["pg_loss_br_agent"][:, update_idx, ...]))
        else:
            csv_row["mean_br_pg_loss"] = 0.0

        if "entropy_br" in metrics:
            csv_row["mean_br_entropy"] = float(np.mean(metrics["entropy_br"][:, update_idx, ...]))
        else:
            csv_row["mean_br_entropy"] = 0.0

        # Per-population member metrics
        for p in range(partner_pop_size):
            csv_row[f"conf_{p}_mean_return"] = csv_row["mean_conf_return"]
            csv_row[f"br_{p}_mean_return"] = csv_row["mean_br_return"]

        csv_logger.log(csv_row)

    log.info(f"Wrote training metrics to: {csv_path}")


class XPTransition(NamedTuple):
    """Transition data for cross-play training."""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    self_onehot_id: jnp.ndarray
    oppo_onehot_id: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


def get_default_config(layout: str) -> Dict[str, Any]:
    """Get default BRDiv training configuration for a given layout.

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
            "flatten_obs": False,  # Use grid-based observations for CNN+RNN (like FCP)
            "max_steps": 400,  # Maximum steps per episode
        },
        "ROLLOUT_LENGTH": 256,  # NUM_STEPS in JaxMARL recipe

        # BRDiv-specific settings
        "ALG": "brdiv",
        "ACTOR_TYPE": "cnn_rnn",  # Use CNN+RNN architecture like FCP
        "TOTAL_TIMESTEPS": 3e7,  # From JaxMARL recipe
        "NUM_CHECKPOINTS": 5,
        "PARTNER_POP_SIZE": 10,

        # XP loss weight:
        # SP weight = 1 + 2*XP weight
        # As XP weight -> 0, SP/(SP+XP) -> 1
        # If XP weight -> infinity, XP/(SP+XP) -> 1/3, SP/(SP+XP) -> 2/3
        "XP_LOSS_WEIGHTS": 0.5,

        # PPO hyperparameters (from working JaxMARL recipe)
        "NUM_ENVS": 256,  # Critical: much larger than before
        "LR": 0.00025,  # From JaxMARL recipe
        "ANNEAL_LR": True,
        "LR_WARMUP": 0.05,  # 5% of updates for warmup
        "UPDATE_EPOCHS": 4,  # From JaxMARL recipe (was 15)
        "NUM_MINIBATCHES": 64,  # 100 actors / 50 = 2 actors per minibatch (must divide evenly)
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,  # From JaxMARL recipe (was 0.95)
        "CLIP_EPS": 0.2,  # From JaxMARL recipe (was 0.05)
        "ENT_COEF": 0.01,  # From JaxMARL recipe (was 0.01)
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.25,  # From JaxMARL recipe (was 1.0)
        "ACTIVATION": "relu",  # From JaxMARL recipe (was tanh)

        # Network architecture (CNN+RNN, from JaxMARL recipe)
        "FC_DIM_SIZE": 128,  # From JaxMARL recipe
        "GRU_HIDDEN_DIM": 128,  # From JaxMARL recipe

        # Reward shaping (from JaxMARL recipe)
        "REW_SHAPING_HORIZON": 1.5e7,  # Anneal shaped rewards over this many timesteps

        # Training settings
        "NUM_SEEDS": 1,
        "TRAIN_SEED": 0,
        "NUM_EVAL_EPISODES": 20,
        "NUM_CHUNKS": 1,  # Number of chunks for memory-efficient training

        # Evaluation settings
        # EVAL_MAX_STEPS: Max steps per episode during evaluation
        # Should match environment's actual max_steps (400 for OvercookedV2)
        # This ensures evaluation episodes run to completion, not truncated at ROLLOUT_LENGTH
        "EVAL_MAX_STEPS": 400,
    }


def _get_all_ids(pop_size: int):
    """Generate all confederate-BR ID pairs for cross-play evaluation.

    Args:
        pop_size: Population size.

    Returns:
        Tuple of (all_conf_ids, all_br_ids) arrays.
    """
    cross_product = np.meshgrid(
        np.arange(pop_size),
        np.arange(pop_size)
    )
    agent_id_cartesian_product = np.stack([g.ravel() for g in cross_product], axis=-1)
    all_conf_ids = agent_id_cartesian_product[:, 1]
    all_br_ids = agent_id_cartesian_product[:, 0]
    return all_conf_ids, all_br_ids


def gather_params(partner_params_pytree, idx_vec):
    """Gather parameters for specific indices from a parameter pytree.

    Args:
        partner_params_pytree: Pytree with all partner params.
        idx_vec: Vector of indices with shape (num_envs,).

    Returns:
        New pytree where each leaf has shape (num_envs, ...).
    """
    def gather_leaf(leaf):
        def slice_one(idx):
            return leaf[idx]
        return jax.vmap(slice_one)(idx_vec)

    return jax.tree.map(gather_leaf, partner_params_pytree)


def train_brdiv_partners(train_rng, env, config: Dict[str, Any], conf_policy, br_policy):
    """Train BRDiv confederate and best response policies.

    This implements the core BRDiv training loop, training confederate policies
    to achieve high self-play returns while achieving low cross-play returns
    with other confederates' best responses.

    Args:
        train_rng: JAX random key.
        env: Environment instance.
        config: Training configuration.
        conf_policy: Policy class for confederate agents.
        br_policy: Policy class for best response agents.

    Returns:
        Dictionary with training results.
    """
    num_agents = env.num_agents
    assert num_agents == 2, "BRDiv requires exactly 2 agents"

    # Compute derived config values
    config["NUM_GAME_AGENTS"] = num_agents
    config["NUM_CONF_ACTORS"] = config["NUM_ENVS"]
    config["NUM_BR_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // (num_agents * config["ROLLOUT_LENGTH"] * config["NUM_ENVS"])
    )

    def make_brdiv_agents(config):
        # Linear schedule (original)
        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        # Cosine decay with warmup schedule (from JaxMARL)
        def create_warmup_cosine_schedule():
            base_learning_rate = config["LR"]
            lr_warmup = config.get("LR_WARMUP", 0.0)
            update_steps = config["NUM_UPDATES"]
            warmup_steps = int(lr_warmup * update_steps)

            steps_per_epoch = config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]

            warmup_fn = optax.linear_schedule(
                init_value=0.0,
                end_value=base_learning_rate,
                transition_steps=warmup_steps * steps_per_epoch,
            )
            cosine_epochs = max(update_steps - warmup_steps, 1)

            cosine_fn = optax.cosine_decay_schedule(
                init_value=base_learning_rate,
                decay_steps=cosine_epochs * steps_per_epoch
            )
            schedule_fn = optax.join_schedules(
                schedules=[warmup_fn, cosine_fn],
                boundaries=[warmup_steps * steps_per_epoch],
            )
            return schedule_fn

        # Reward shaping annealing schedule (from JaxMARL)
        rew_shaping_horizon = config.get("REW_SHAPING_HORIZON", 0)
        if rew_shaping_horizon > 0:
            rew_shaping_anneal = optax.linear_schedule(
                init_value=1.0,
                end_value=0.0,
                transition_steps=int(rew_shaping_horizon)
            )
        else:
            rew_shaping_anneal = None

        def train(rng):
            rng, init_conf_rng, init_br_rng = jax.random.split(rng, 3)
            all_conf_init_rngs = jax.random.split(init_conf_rng, config["PARTNER_POP_SIZE"])
            all_br_init_rngs = jax.random.split(init_br_rng, config["PARTNER_POP_SIZE"])
            identity_matrix = jnp.eye(config["PARTNER_POP_SIZE"])

            init_conf_hstate = conf_policy.init_hstate(config["NUM_CONF_ACTORS"])
            init_br_hstate = br_policy.init_hstate(config["NUM_BR_ACTORS"])

            def init_train_states(rng_agents, rng_brs):
                def init_single_pair_optimizers(rng_agent, rng_br):
                    init_params_conf = conf_policy.init_params(rng_agent)
                    init_params_br = br_policy.init_params(rng_br)
                    return init_params_conf, init_params_br

                init_all_networks_and_optimizers = jax.vmap(init_single_pair_optimizers)
                all_conf_params, all_br_params = init_all_networks_and_optimizers(rng_agents, rng_brs)

                # Select learning rate schedule
                if config["ANNEAL_LR"]:
                    if config.get("LR_WARMUP", 0.0) > 0:
                        lr_schedule = create_warmup_cosine_schedule()
                    else:
                        lr_schedule = linear_schedule
                else:
                    lr_schedule = config["LR"]

                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(
                        learning_rate=lr_schedule,
                        eps=1e-5
                    ),
                )
                tx_br = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(
                        learning_rate=lr_schedule,
                        eps=1e-5
                    ),
                )

                train_state_conf = TrainState.create(
                    apply_fn=conf_policy.network.apply,
                    params=all_conf_params,
                    tx=tx,
                )
                train_state_br = TrainState.create(
                    apply_fn=br_policy.network.apply,
                    params=all_br_params,
                    tx=tx_br,
                )
                return train_state_conf, train_state_br

            all_conf_optims, all_br_optims = init_train_states(
                all_conf_init_rngs, all_br_init_rngs
            )

            def forward_pass_conf(params, obs, id, done, avail_actions, hstate, rng):
                act, val, pi, new_hstate = conf_policy.get_action_value_policy(
                    params=params,
                    obs=obs[jnp.newaxis, ...],
                    done=done[jnp.newaxis, ...],
                    avail_actions=avail_actions,
                    hstate=hstate,
                    rng=rng,
                    aux_obs=id[jnp.newaxis, ...]
                )
                return act, val, pi, new_hstate

            def forward_pass_br(params, obs, id, done, avail_actions, hstate, rng):
                act, val, pi, new_hstate = br_policy.get_action_value_policy(
                    params=params,
                    obs=obs[jnp.newaxis, ...],
                    done=done[jnp.newaxis, ...],
                    avail_actions=avail_actions,
                    hstate=hstate,
                    rng=rng,
                    aux_obs=id[jnp.newaxis, ...]
                )
                return act, val, pi, new_hstate

            def _env_step(runner_state, unused):
                (
                    all_train_state_conf, all_train_state_br, last_conf_ids, last_br_ids,
                    env_state, last_obs, last_done, last_conf_h, last_br_h, rng, anneal_factor
                ) = runner_state
                rng, act0_rng, act1_rng, step_rng, conf_sampling_rng, br_sampling_rng = jax.random.split(rng, 6)

                needs_resample = last_done["__all__"]
                resampled_conf_ids = jax.random.randint(conf_sampling_rng, (config["NUM_CONF_ACTORS"],), 0, config["PARTNER_POP_SIZE"])
                resampled_br_ids = jax.random.randint(br_sampling_rng, (config["NUM_BR_ACTORS"],), 0, config["PARTNER_POP_SIZE"])

                updated_conf_ids = jnp.where(needs_resample, resampled_conf_ids, last_conf_ids)
                updated_br_ids = jnp.where(needs_resample, resampled_br_ids, last_br_ids)

                if last_conf_h is not None:
                    updated_conf_h = jnp.where(needs_resample, init_conf_hstate, last_conf_h)
                else:
                    updated_conf_h = last_conf_h

                if last_br_h is not None:
                    updated_br_h = jnp.where(needs_resample, init_br_hstate, last_br_h)
                else:
                    updated_br_h = last_br_h

                updated_conf_params = gather_params(all_train_state_conf.params, updated_conf_ids)
                updated_br_params = gather_params(all_train_state_br.params, updated_br_ids)

                updated_conf_onehot_ids = identity_matrix[updated_conf_ids]
                updated_br_onehot_ids = identity_matrix[updated_br_ids]

                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                act0_rng = jax.random.split(act0_rng, config["NUM_ENVS"])
                act_0, val_0, pi_0, new_conf_h = jax.vmap(forward_pass_conf)(
                    updated_conf_params,
                    last_obs["agent_0"], updated_br_onehot_ids, last_done["agent_0"], avail_actions_0,
                    updated_conf_h, act0_rng
                )
                logp_0 = pi_0.log_prob(act_0)
                act_0, val_0, logp_0 = act_0.squeeze(), val_0.squeeze(), logp_0.squeeze()

                act1_rng = jax.random.split(act1_rng, config["NUM_ENVS"])
                act_1, val_1, pi_1, new_br_h = jax.vmap(forward_pass_br)(
                    updated_br_params,
                    last_obs["agent_1"], updated_conf_onehot_ids, last_done["agent_1"], avail_actions_1,
                    updated_br_h, act1_rng
                )
                logp_1 = pi_1.log_prob(act_1)
                act_1, val_1, logp_1 = act_1.squeeze(), val_1.squeeze(), logp_1.squeeze()

                combined_actions = jnp.concatenate([act_0, act_1], axis=0)
                env_act = unbatchify(combined_actions, env.agents, config["NUM_ENVS"], num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                obs_next, env_state_next, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    step_rngs, env_state, env_act
                )
                info_0 = jax.tree.map(lambda x: x[:, 0], info)
                info_1 = jax.tree.map(lambda x: x[:, 1], info)

                # Apply reward shaping with annealing (from JaxMARL recipe)
                # Base rewards come from the environment
                base_reward_0 = reward["agent_0"]
                base_reward_1 = reward["agent_1"]

                # Add shaped rewards with annealing factor
                shaped_reward_0 = info_0.get("shaped_reward", jnp.zeros_like(base_reward_0))
                shaped_reward_1 = info_1.get("shaped_reward", jnp.zeros_like(base_reward_1))
                total_reward_0 = base_reward_0 + anneal_factor * shaped_reward_0
                total_reward_1 = base_reward_1 + anneal_factor * shaped_reward_1

                def _compute_rewards(conf_id, br_id, agent_rew):
                    return jax.lax.cond(
                        jnp.equal(jnp.argmax(conf_id, axis=-1), jnp.argmax(br_id, axis=-1)),
                        lambda x: x,
                        lambda x: -x,
                        agent_rew
                    )

                agent_0_rews = jax.vmap(_compute_rewards)(updated_conf_onehot_ids, updated_br_onehot_ids, total_reward_1)
                agent_1_rews = jax.vmap(_compute_rewards)(updated_conf_onehot_ids, updated_br_onehot_ids, total_reward_0)

                transition_0 = XPTransition(
                    done=done["agent_0"],
                    action=act_0,
                    value=val_0,
                    self_onehot_id=updated_conf_onehot_ids,
                    oppo_onehot_id=updated_br_onehot_ids,
                    reward=agent_0_rews,
                    log_prob=logp_0,
                    obs=last_obs["agent_0"],
                    info=info_0,
                    avail_actions=avail_actions_0
                )

                transition_1 = XPTransition(
                    done=done["agent_1"],
                    action=act_1,
                    value=val_1,
                    self_onehot_id=updated_br_onehot_ids,
                    oppo_onehot_id=updated_conf_onehot_ids,
                    reward=agent_1_rews,
                    log_prob=logp_1,
                    obs=last_obs["agent_1"],
                    info=info_1,
                    avail_actions=avail_actions_1
                )

                new_runner_state = (
                    all_train_state_conf, all_train_state_br, updated_conf_ids, updated_br_ids,
                    env_state_next, obs_next, done, new_conf_h, new_br_h, rng, anneal_factor
                )
                return new_runner_state, (transition_0, transition_1)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            def run_all_episodes(rng, train_state_conf, train_state_br):
                conf_ids, br_ids = _get_all_ids(config["PARTNER_POP_SIZE"])
                gathered_conf_model_params = gather_params(train_state_conf.params, conf_ids)
                gathered_br_model_params = gather_params(train_state_br.params, br_ids)

                rng, eval_rng = jax.random.split(rng)

                def run_episodes_fixed_rng(conf_param, br_param):
                    return run_episodes(
                        eval_rng, env,
                        conf_param, conf_policy,
                        br_param, br_policy,
                        config["EVAL_MAX_STEPS"], config["NUM_EVAL_EPISODES"],
                    )

                ep_infos = jax.vmap(run_episodes_fixed_rng)(
                    gathered_conf_model_params, gathered_br_model_params,
                )
                return ep_infos

            def _update_epoch(update_state, unused):
                def _update_minbatch(all_train_states, all_data):
                    train_state_conf, train_state_br = all_train_states
                    minbatch_conf, minbatch_br = all_data

                    def _loss_fn(param, agent_policy, minbatch, agent_id):
                        init_hstate, _init_done, traj_batch, gae, target_v = minbatch
                        squeezed_param = jax.tree.map(lambda x: jnp.squeeze(x, 0), param)
                        _, value, pi, _ = agent_policy.get_action_value_policy(
                            params=squeezed_param,
                            obs=traj_batch.obs,
                            done=traj_batch.done,
                            avail_actions=traj_batch.avail_actions,
                            hstate=init_hstate,
                            rng=jax.random.PRNGKey(0),
                            aux_obs=traj_batch.oppo_onehot_id
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        is_relevant = jnp.equal(
                            jnp.argmax(traj_batch.self_onehot_id, axis=-1),
                            agent_id
                        )
                        loss_weights = jnp.where(is_relevant, 1, 0).astype(jnp.float32)

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - target_v)
                        value_losses_clipped = jnp.square(value_pred_clipped - target_v)
                        value_loss = jax.lax.cond(
                            loss_weights.sum() == 0,
                            lambda x: jnp.zeros_like(x).astype(jnp.float32),
                            lambda x: x,
                            (loss_weights * jnp.maximum(value_losses, value_losses_clipped)).sum() / (loss_weights.sum() + 1e-8)
                        )

                        n = config["PARTNER_POP_SIZE"]
                        is_sp = jnp.equal(
                            jnp.argmax(traj_batch.self_onehot_id, axis=-1),
                            jnp.argmax(traj_batch.oppo_onehot_id, axis=-1)
                        )
                        sp_weight = (1 + 2 * config["XP_LOSS_WEIGHTS"]) * (n / 2)
                        xp_weight = config["XP_LOSS_WEIGHTS"] * (n / (2 * (n - 1)))
                        actor_weights = jnp.where(is_sp, sp_weight, xp_weight)

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                        pg_loss_1 = ratio * gae_norm * actor_weights
                        pg_loss_2 = jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"]
                        ) * gae_norm * actor_weights
                        pg_loss = jax.lax.cond(
                            loss_weights.sum() == 0,
                            lambda x: jnp.zeros_like(x).astype(jnp.float32),
                            lambda x: x,
                            -(loss_weights * jnp.minimum(pg_loss_1, pg_loss_2)).sum() / (loss_weights.sum() + 1e-8)
                        )

                        entropy = jax.lax.cond(
                            loss_weights.sum() == 0,
                            lambda x: jnp.zeros_like(x).astype(jnp.float32),
                            lambda x: x,
                            (loss_weights * pi.entropy()).sum() / (loss_weights.sum() + 1e-8)
                        )

                        total_loss = pg_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, pg_loss, entropy)

                    possible_agent_ids = jnp.expand_dims(jnp.arange(config["PARTNER_POP_SIZE"]), 1)
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

                    def gather_conf_params_and_return_grads(agent_id):
                        param_vector = gather_params(train_state_conf.params, agent_id)
                        (loss_val_conf, aux_vals_conf), grads_conf = grad_fn(
                            param_vector, conf_policy, minbatch_conf, agent_id
                        )
                        return (loss_val_conf, aux_vals_conf), grads_conf

                    def gather_br_params_and_return_grads(agent_id):
                        param_vector = gather_params(train_state_br.params, agent_id)
                        (loss_val_br, aux_vals_br), grads_br = grad_fn(
                            param_vector, br_policy, minbatch_br, agent_id
                        )
                        return (loss_val_br, aux_vals_br), grads_br

                    (loss_val_conf, aux_vals_conf), grads_conf = jax.vmap(gather_conf_params_and_return_grads)(possible_agent_ids)
                    (loss_val_br, aux_vals_br), grads_br = jax.vmap(gather_br_params_and_return_grads)(possible_agent_ids)

                    grads_conf_new = jax.tree.map(lambda x: jnp.squeeze(x, 1), grads_conf)
                    grads_br_new = jax.tree.map(lambda x: jnp.squeeze(x, 1), grads_br)
                    train_state_conf = train_state_conf.apply_gradients(grads=grads_conf_new)
                    train_state_br = train_state_br.apply_gradients(grads=grads_br_new)
                    return (train_state_conf, train_state_br), ((loss_val_conf, aux_vals_conf), (loss_val_br, aux_vals_br))

                (
                    train_state_conf, train_state_br,
                    traj_batch_conf, traj_batch_br,
                    advantages_conf, advantages_br,
                    targets_conf, targets_br,
                    init_conf_h_for_update, init_br_h_for_update,
                    rng
                ) = update_state
                rng, perm_rng_conf, perm_rng_br = jax.random.split(rng, 3)

                # Use the ACTUAL initial hidden state from the start of the rollout
                # (not fresh zeros) - this is critical when ROLLOUT_LENGTH != max_steps
                minibatches_conf = _create_minibatches(
                    traj_batch_conf, advantages_conf, targets_conf, init_conf_h_for_update,
                    config["NUM_CONF_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_conf
                )
                minibatches_br = _create_minibatches(
                    traj_batch_br, advantages_br, targets_br, init_br_h_for_update,
                    config["NUM_BR_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_br
                )

                (train_state_conf, train_state_br), all_losses = jax.lax.scan(
                    _update_minbatch, (train_state_conf, train_state_br), (minibatches_conf, minibatches_br)
                )

                update_state = (
                    train_state_conf, train_state_br,
                    traj_batch_conf, traj_batch_br,
                    advantages_conf, advantages_br,
                    targets_conf, targets_br,
                    init_conf_h_for_update, init_br_h_for_update,
                    rng
                )
                return update_state, all_losses

            def _update_step(update_runner_state, unused):
                (
                    all_train_state_conf, all_train_state_br,
                    last_env_state, last_obs, last_done, last_conf_h, last_br_h,
                    rng, update_steps
                ) = update_runner_state

                # Save the initial hidden state BEFORE the rollout for PPO update epochs
                # This is critical when ROLLOUT_LENGTH != max_steps (episodes span multiple rollouts)
                initial_conf_hstate_for_update = last_conf_h
                initial_br_hstate_for_update = last_br_h

                # Compute reward shaping annealing factor for this update step
                if rew_shaping_anneal is not None:
                    current_timestep = update_steps * config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
                    anneal_factor = rew_shaping_anneal(current_timestep)
                else:
                    anneal_factor = 0.0

                rng, conf_sampling_rng, br_sampling_rng = jax.random.split(rng, 3)

                conf_ids = jax.random.randint(conf_sampling_rng, (config["NUM_ENVS"],), 0, config["PARTNER_POP_SIZE"])
                br_ids = jax.random.randint(br_sampling_rng, (config["NUM_ENVS"],), 0, config["PARTNER_POP_SIZE"])

                runner_state = (
                    all_train_state_conf, all_train_state_br, conf_ids, br_ids,
                    last_env_state, last_obs, last_done, last_conf_h, last_br_h, rng, anneal_factor
                )
                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
                )
                (
                    all_train_state_conf, all_train_state_br, last_conf_ids, last_br_ids,
                    last_env_state, last_obs, last_done, last_conf_h, last_br_h, rng, _
                ) = runner_state

                last_conf_params = gather_params(all_train_state_conf.params, last_conf_ids)
                last_br_params = gather_params(all_train_state_br.params, last_br_ids)

                last_conf_one_hots = identity_matrix[last_conf_ids]
                last_br_one_hots = identity_matrix[last_br_ids]

                traj_batch_conf, traj_batch_br = traj_batch

                avail_actions_0 = jax.vmap(env.get_avail_actions)(last_env_state.env_state)["agent_0"].astype(jnp.float32)
                _, last_val_conf, _, _ = jax.vmap(forward_pass_conf)(
                    params=last_conf_params,
                    obs=last_obs["agent_0"],
                    id=last_br_one_hots,
                    done=last_done["agent_0"],
                    avail_actions=avail_actions_0,
                    hstate=last_conf_h,
                    rng=jax.random.split(jax.random.PRNGKey(0), config["NUM_ENVS"])
                )
                last_val_conf = last_val_conf.squeeze()
                advantages_conf, targets_conf = _calculate_gae(traj_batch_conf, last_val_conf)

                avail_actions_1 = jax.vmap(env.get_avail_actions)(last_env_state.env_state)["agent_1"].astype(jnp.float32)
                _, last_val_br, _, _ = jax.vmap(forward_pass_br)(
                    params=last_br_params,
                    obs=last_obs["agent_1"],
                    id=last_conf_one_hots,
                    done=last_done["agent_1"],
                    avail_actions=avail_actions_1,
                    hstate=last_br_h,
                    rng=jax.random.split(jax.random.PRNGKey(0), config["NUM_ENVS"])
                )
                last_val_br = last_val_br.squeeze()
                advantages_br, targets_br = _calculate_gae(traj_batch_br, last_val_br)

                rng, update_rng = jax.random.split(rng, 2)
                update_state = (
                    all_train_state_conf, all_train_state_br,
                    traj_batch_conf, traj_batch_br,
                    advantages_conf, advantages_br,
                    targets_conf, targets_br,
                    initial_conf_hstate_for_update, initial_br_hstate_for_update,
                    update_rng
                )

                update_state, all_losses = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
                )
                all_train_state_conf, all_train_state_br = update_state[:2]
                (_, (value_loss_conf, pg_loss_conf, entropy_conf)), (_, (value_loss_br, pg_loss_br, entropy_br)) = all_losses

                metric = traj_batch_conf.info
                metric["update_steps"] = update_steps
                metric["value_loss_conf_agent"] = value_loss_conf
                metric["value_loss_br_agent"] = value_loss_br
                metric["pg_loss_conf_agent"] = pg_loss_conf
                metric["pg_loss_br_agent"] = pg_loss_br
                metric["entropy_conf"] = entropy_conf
                metric["entropy_br"] = entropy_br

                new_runner_state = (
                    all_train_state_conf, all_train_state_br,
                    last_env_state, last_obs, last_done, last_conf_h, last_br_h,
                    rng, update_steps + 1
                )
                return (new_runner_state, metric)

            # Checkpoint saving setup
            ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1)
            num_ckpts = config["NUM_CHECKPOINTS"]

            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                    params_pytree
                )

            def _update_step_with_ckpt(state_with_ckpt, unused):
                (
                    update_runner_state, checkpoint_array_conf, checkpoint_array_br, ckpt_idx,
                    eval_info
                ) = state_with_ckpt

                new_runner_state, metric = _update_step(update_runner_state, None)

                train_state_conf, train_state_br, last_env_state, last_obs, last_done, last_conf_h, last_br_h, rng, update_steps = new_runner_state

                to_store = jnp.logical_or(
                    jnp.equal(jnp.mod(update_steps - 1, ckpt_and_eval_interval), 0),
                    jnp.equal(update_steps, config["NUM_UPDATES"])
                )

                def store_and_eval_ckpt(args):
                    ckpt_arr_and_ep_infos, rng, cidx = args
                    ckpt_arr_conf, ckpt_arr_br, _ = ckpt_arr_and_ep_infos
                    new_ckpt_arr_conf = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr_conf, train_state_conf.params
                    )
                    new_ckpt_arr_br = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr_br, train_state_br.params
                    )

                    rng, eval_rng = jax.random.split(rng)
                    ep_last_info = run_all_episodes(eval_rng, train_state_conf, train_state_br)

                    return ((new_ckpt_arr_conf, new_ckpt_arr_br, ep_last_info), rng, cidx + 1)

                def skip_ckpt(args):
                    return args

                (checkpoint_array_and_infos, rng, ckpt_idx) = jax.lax.cond(
                    to_store,
                    store_and_eval_ckpt,
                    skip_ckpt,
                    ((checkpoint_array_conf, checkpoint_array_br, eval_info), rng, ckpt_idx)
                )
                checkpoint_array_conf, checkpoint_array_br, eval_ep_last_info = checkpoint_array_and_infos

                metric["eval_ep_last_info"] = eval_ep_last_info

                return ((
                    train_state_conf, train_state_br,
                    last_env_state, last_obs, last_done, last_conf_h, last_br_h, rng, update_steps
                ), checkpoint_array_conf, checkpoint_array_br, ckpt_idx, eval_ep_last_info), metric

            # Initialize training
            checkpoint_array_conf = init_ckpt_array(all_conf_optims.params)
            checkpoint_array_br = init_ckpt_array(all_br_optims.params)
            ckpt_idx = 0
            update_steps = 0

            rng, rng_eval = jax.random.split(rng, 2)
            eval_ep_last_info = run_all_episodes(rng_eval, all_conf_optims, all_br_optims)

            rng, reset_rng = jax.random.split(rng)
            reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
            init_obs, init_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
            init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

            init_conf_h = conf_policy.init_hstate(config["NUM_CONF_ACTORS"])
            init_br_h = br_policy.init_hstate(config["NUM_BR_ACTORS"])

            update_runner_state = (
                all_conf_optims, all_br_optims,
                init_env_state, init_obs, init_done, init_conf_h, init_br_h,
                rng, update_steps
            )

            state_with_ckpt = (
                update_runner_state, checkpoint_array_conf,
                checkpoint_array_br, ckpt_idx, eval_ep_last_info
            )

            # Run training
            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt,
                state_with_ckpt,
                xs=None,
                length=config["NUM_UPDATES"]
            )

            (
                final_runner_state, checkpoint_array_conf, checkpoint_array_br,
                final_ckpt_idx, all_ep_infos
            ) = state_with_ckpt

            out = {
                "final_params_conf": final_runner_state[0].params,
                "final_params_br": final_runner_state[1].params,
                "checkpoints_conf": checkpoint_array_conf,
                "checkpoints_br": checkpoint_array_br,
                "metrics": metrics,
                "all_pair_returns": all_ep_infos
            }
            return out

        return train

    train_fn = make_brdiv_agents(config)
    out = train_fn(train_rng)
    return out


def make_train_brdiv_chunked(config: Dict[str, Any], env, conf_policy, br_policy):
    """Create training functions for chunked BRDiv training to reduce GPU memory usage.

    Args:
        config: Training configuration dict. Must include NUM_CHUNKS.
        env: Environment instance.
        conf_policy: Confederate policy.
        br_policy: Best response policy.

    Returns:
        Tuple of (init_fn, train_one_chunk_fn, updated_config):
            - init_fn(rng) -> initial_state: Initialize training state
            - train_one_chunk_fn(state) -> (new_state, metrics): Run one chunk of updates
            - updated_config: Config with computed values like NUM_UPDATES
    """
    config = config.copy()
    num_agents = env.num_agents
    assert num_agents == 2, "BRDiv requires exactly 2 agents"

    config["NUM_GAME_AGENTS"] = num_agents
    config["NUM_CONF_ACTORS"] = config["NUM_ENVS"]
    config["NUM_BR_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // (num_agents * config["ROLLOUT_LENGTH"] * config["NUM_ENVS"])
    )

    num_chunks = config.get("NUM_CHUNKS", 1)
    updates_per_chunk = config["NUM_UPDATES"] // num_chunks
    # Actual total updates may be less than NUM_UPDATES due to integer division
    actual_total_updates = updates_per_chunk * num_chunks

    ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1)
    num_ckpts = config["NUM_CHECKPOINTS"]

    # Linear schedule (original)
    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    # Cosine decay with warmup schedule (from JaxMARL)
    def create_warmup_cosine_schedule():
        base_learning_rate = config["LR"]
        lr_warmup = config.get("LR_WARMUP", 0.0)
        update_steps = config["NUM_UPDATES"]
        warmup_steps = int(lr_warmup * update_steps)

        steps_per_epoch = config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]

        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=warmup_steps * steps_per_epoch,
        )
        cosine_epochs = max(update_steps - warmup_steps, 1)

        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate,
            decay_steps=cosine_epochs * steps_per_epoch
        )
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps * steps_per_epoch],
        )
        return schedule_fn

    # Reward shaping annealing schedule (from JaxMARL)
    rew_shaping_horizon = config.get("REW_SHAPING_HORIZON", 0)
    if rew_shaping_horizon > 0:
        rew_shaping_anneal = optax.linear_schedule(
            init_value=1.0,
            end_value=0.0,
            transition_steps=int(rew_shaping_horizon)
        )
    else:
        rew_shaping_anneal = None

    identity_matrix = jnp.eye(config["PARTNER_POP_SIZE"])
    init_conf_hstate = conf_policy.init_hstate(config["NUM_CONF_ACTORS"])
    init_br_hstate = br_policy.init_hstate(config["NUM_BR_ACTORS"])

    def init_train_state(rng):
        """Initialize BRDiv training state."""
        rng, init_conf_rng, init_br_rng = jax.random.split(rng, 3)
        all_conf_init_rngs = jax.random.split(init_conf_rng, config["PARTNER_POP_SIZE"])
        all_br_init_rngs = jax.random.split(init_br_rng, config["PARTNER_POP_SIZE"])

        def init_single_pair_optimizers(rng_agent, rng_br):
            init_params_conf = conf_policy.init_params(rng_agent)
            init_params_br = br_policy.init_params(rng_br)
            return init_params_conf, init_params_br

        init_all_networks_and_optimizers = jax.vmap(init_single_pair_optimizers)
        all_conf_params, all_br_params = init_all_networks_and_optimizers(all_conf_init_rngs, all_br_init_rngs)

        # Select learning rate schedule
        if config["ANNEAL_LR"]:
            if config.get("LR_WARMUP", 0.0) > 0:
                lr_schedule = create_warmup_cosine_schedule()
            else:
                lr_schedule = linear_schedule
        else:
            lr_schedule = config["LR"]

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(
                learning_rate=lr_schedule,
                eps=1e-5
            ),
        )
        tx_br = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(
                learning_rate=lr_schedule,
                eps=1e-5
            ),
        )

        train_state_conf = TrainState.create(
            apply_fn=conf_policy.network.apply,
            params=all_conf_params,
            tx=tx,
        )
        train_state_br = TrainState.create(
            apply_fn=br_policy.network.apply,
            params=all_br_params,
            tx=tx_br,
        )

        # Initialize checkpoint arrays
        def init_ckpt_array(params_pytree):
            return jax.tree.map(
                lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                params_pytree
            )

        checkpoint_array_conf = init_ckpt_array(train_state_conf.params)
        checkpoint_array_br = init_ckpt_array(train_state_br.params)

        # Initialize environment
        rng, reset_rng, eval_rng = jax.random.split(rng, 3)
        reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
        init_obs, init_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
        init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

        init_conf_h = conf_policy.init_hstate(config["NUM_CONF_ACTORS"])
        init_br_h = br_policy.init_hstate(config["NUM_BR_ACTORS"])

        # Run initial evaluation
        def run_all_episodes_for_init(rng, train_state_conf, train_state_br):
            conf_ids, br_ids = _get_all_ids(config["PARTNER_POP_SIZE"])
            gathered_conf_model_params = gather_params(train_state_conf.params, conf_ids)
            gathered_br_model_params = gather_params(train_state_br.params, br_ids)

            rng, eval_rng = jax.random.split(rng)

            def run_episodes_fixed_rng(conf_param, br_param):
                return run_episodes(
                    eval_rng, env,
                    conf_param, conf_policy,
                    br_param, br_policy,
                    config["EVAL_MAX_STEPS"], config["NUM_EVAL_EPISODES"],
                )

            ep_infos = jax.vmap(run_episodes_fixed_rng)(
                gathered_conf_model_params, gathered_br_model_params,
            )
            return ep_infos

        eval_ep_last_info = run_all_episodes_for_init(eval_rng, train_state_conf, train_state_br)

        return {
            "train_state_conf": train_state_conf,
            "train_state_br": train_state_br,
            "env_state": init_env_state,
            "obs": init_obs,
            "done": init_done,
            "conf_hstate": init_conf_h,
            "br_hstate": init_br_h,
            "rng": rng,
            "update_steps": jnp.array(0, dtype=jnp.int32),
            "checkpoint_array_conf": checkpoint_array_conf,
            "checkpoint_array_br": checkpoint_array_br,
            "ckpt_idx": jnp.array(0, dtype=jnp.int32),
            "eval_info": eval_ep_last_info,
        }

    def train_one_chunk(state):
        """Run one chunk of BRDiv training updates."""
        # Unpack state
        train_state_conf = state["train_state_conf"]
        train_state_br = state["train_state_br"]
        env_state = state["env_state"]
        obs = state["obs"]
        done = state["done"]
        conf_hstate = state["conf_hstate"]
        br_hstate = state["br_hstate"]
        rng = state["rng"]
        update_steps = state["update_steps"]
        checkpoint_array_conf = state["checkpoint_array_conf"]
        checkpoint_array_br = state["checkpoint_array_br"]
        ckpt_idx = state["ckpt_idx"]
        eval_info = state["eval_info"]

        def forward_pass_conf(params, obs, id, done, avail_actions, hstate, rng):
            act, val, pi, new_hstate = conf_policy.get_action_value_policy(
                params=params,
                obs=obs[jnp.newaxis, ...],
                done=done[jnp.newaxis, ...],
                avail_actions=avail_actions,
                hstate=hstate,
                rng=rng,
                aux_obs=id[jnp.newaxis, ...]
            )
            return act, val, pi, new_hstate

        def forward_pass_br(params, obs, id, done, avail_actions, hstate, rng):
            act, val, pi, new_hstate = br_policy.get_action_value_policy(
                params=params,
                obs=obs[jnp.newaxis, ...],
                done=done[jnp.newaxis, ...],
                avail_actions=avail_actions,
                hstate=hstate,
                rng=rng,
                aux_obs=id[jnp.newaxis, ...]
            )
            return act, val, pi, new_hstate

        def _env_step(runner_state, unused):
            (
                all_train_state_conf, all_train_state_br, last_conf_ids, last_br_ids,
                env_state, last_obs, last_done, last_conf_h, last_br_h, rng, anneal_factor
            ) = runner_state
            rng, act0_rng, act1_rng, step_rng, conf_sampling_rng, br_sampling_rng = jax.random.split(rng, 6)

            needs_resample = last_done["__all__"]
            resampled_conf_ids = jax.random.randint(conf_sampling_rng, (config["NUM_CONF_ACTORS"],), 0, config["PARTNER_POP_SIZE"])
            resampled_br_ids = jax.random.randint(br_sampling_rng, (config["NUM_BR_ACTORS"],), 0, config["PARTNER_POP_SIZE"])

            updated_conf_ids = jnp.where(needs_resample, resampled_conf_ids, last_conf_ids)
            updated_br_ids = jnp.where(needs_resample, resampled_br_ids, last_br_ids)

            if last_conf_h is not None:
                updated_conf_h = jnp.where(needs_resample, init_conf_hstate, last_conf_h)
            else:
                updated_conf_h = last_conf_h

            if last_br_h is not None:
                updated_br_h = jnp.where(needs_resample, init_br_hstate, last_br_h)
            else:
                updated_br_h = last_br_h

            updated_conf_params = gather_params(all_train_state_conf.params, updated_conf_ids)
            updated_br_params = gather_params(all_train_state_br.params, updated_br_ids)

            updated_conf_onehot_ids = identity_matrix[updated_conf_ids]
            updated_br_onehot_ids = identity_matrix[updated_br_ids]

            avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
            avail_actions = jax.lax.stop_gradient(avail_actions)
            avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
            avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

            act0_rng = jax.random.split(act0_rng, config["NUM_ENVS"])
            act_0, val_0, pi_0, new_conf_h = jax.vmap(forward_pass_conf)(
                updated_conf_params,
                last_obs["agent_0"], updated_br_onehot_ids, last_done["agent_0"], avail_actions_0,
                updated_conf_h, act0_rng
            )
            logp_0 = pi_0.log_prob(act_0)
            act_0, val_0, logp_0 = act_0.squeeze(), val_0.squeeze(), logp_0.squeeze()

            act1_rng = jax.random.split(act1_rng, config["NUM_ENVS"])
            act_1, val_1, pi_1, new_br_h = jax.vmap(forward_pass_br)(
                updated_br_params,
                last_obs["agent_1"], updated_conf_onehot_ids, last_done["agent_1"], avail_actions_1,
                updated_br_h, act1_rng
            )
            logp_1 = pi_1.log_prob(act_1)
            act_1, val_1, logp_1 = act_1.squeeze(), val_1.squeeze(), logp_1.squeeze()

            combined_actions = jnp.concatenate([act_0, act_1], axis=0)
            env_act = unbatchify(combined_actions, env.agents, config["NUM_ENVS"], num_agents)
            env_act = {k: v.flatten() for k, v in env_act.items()}

            step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
            obs_next, env_state_next, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                step_rngs, env_state, env_act
            )
            info_0 = jax.tree.map(lambda x: x[:, 0], info)
            info_1 = jax.tree.map(lambda x: x[:, 1], info)

            # Apply reward shaping with annealing (from JaxMARL recipe)
            # Base rewards come from the environment
            base_reward_0 = reward["agent_0"]
            base_reward_1 = reward["agent_1"]

            # Add shaped rewards with annealing factor
            shaped_reward_0 = info_0.get("shaped_reward", jnp.zeros_like(base_reward_0))
            shaped_reward_1 = info_1.get("shaped_reward", jnp.zeros_like(base_reward_1))
            total_reward_0 = base_reward_0 + anneal_factor * shaped_reward_0
            total_reward_1 = base_reward_1 + anneal_factor * shaped_reward_1

            def _compute_rewards(conf_id, br_id, agent_rew):
                return jax.lax.cond(
                    jnp.equal(jnp.argmax(conf_id, axis=-1), jnp.argmax(br_id, axis=-1)),
                    lambda x: x,
                    lambda x: -x,
                    agent_rew
                )

            agent_0_rews = jax.vmap(_compute_rewards)(updated_conf_onehot_ids, updated_br_onehot_ids, total_reward_1)
            agent_1_rews = jax.vmap(_compute_rewards)(updated_conf_onehot_ids, updated_br_onehot_ids, total_reward_0)

            transition_0 = XPTransition(
                done=done["agent_0"],
                action=act_0,
                value=val_0,
                self_onehot_id=updated_conf_onehot_ids,
                oppo_onehot_id=updated_br_onehot_ids,
                reward=agent_0_rews,
                log_prob=logp_0,
                obs=last_obs["agent_0"],
                info=info_0,
                avail_actions=avail_actions_0
            )

            transition_1 = XPTransition(
                done=done["agent_1"],
                action=act_1,
                value=val_1,
                self_onehot_id=updated_br_onehot_ids,
                oppo_onehot_id=updated_conf_onehot_ids,
                reward=agent_1_rews,
                log_prob=logp_1,
                obs=last_obs["agent_1"],
                info=info_1,
                avail_actions=avail_actions_1
            )

            new_runner_state = (
                all_train_state_conf, all_train_state_br, updated_conf_ids, updated_br_ids,
                env_state_next, obs_next, done, new_conf_h, new_br_h, rng, anneal_factor
            )
            return new_runner_state, (transition_0, transition_1)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = transition.done, transition.value, transition.reward
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        def run_all_episodes(rng, train_state_conf, train_state_br):
            conf_ids, br_ids = _get_all_ids(config["PARTNER_POP_SIZE"])
            gathered_conf_model_params = gather_params(train_state_conf.params, conf_ids)
            gathered_br_model_params = gather_params(train_state_br.params, br_ids)

            rng, eval_rng = jax.random.split(rng)

            def run_episodes_fixed_rng(conf_param, br_param):
                return run_episodes(
                    eval_rng, env,
                    conf_param, conf_policy,
                    br_param, br_policy,
                    config["EVAL_MAX_STEPS"], config["NUM_EVAL_EPISODES"],
                )

            ep_infos = jax.vmap(run_episodes_fixed_rng)(
                gathered_conf_model_params, gathered_br_model_params,
            )
            return ep_infos

        def _update_epoch(update_state, unused):
            def _update_minbatch(all_train_states, all_data):
                train_state_conf, train_state_br = all_train_states
                minbatch_conf, minbatch_br = all_data

                def _loss_fn(param, agent_policy, minbatch, agent_id):
                    init_hstate, _init_done, traj_batch, gae, target_v = minbatch
                    squeezed_param = jax.tree.map(lambda x: jnp.squeeze(x, 0), param)
                    _, value, pi, _ = agent_policy.get_action_value_policy(
                        params=squeezed_param,
                        obs=traj_batch.obs,
                        done=traj_batch.done,
                        avail_actions=traj_batch.avail_actions,
                        hstate=init_hstate,
                        rng=jax.random.PRNGKey(0),
                        aux_obs=traj_batch.oppo_onehot_id
                    )
                    log_prob = pi.log_prob(traj_batch.action)

                    is_relevant = jnp.equal(
                        jnp.argmax(traj_batch.self_onehot_id, axis=-1),
                        agent_id
                    )
                    loss_weights = jnp.where(is_relevant, 1, 0).astype(jnp.float32)

                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - target_v)
                    value_losses_clipped = jnp.square(value_pred_clipped - target_v)
                    value_loss = jax.lax.cond(
                        loss_weights.sum() == 0,
                        lambda x: jnp.zeros_like(x).astype(jnp.float32),
                        lambda x: x,
                        (loss_weights * jnp.maximum(value_losses, value_losses_clipped)).sum() / (loss_weights.sum() + 1e-8)
                    )

                    n = config["PARTNER_POP_SIZE"]
                    is_sp = jnp.equal(
                        jnp.argmax(traj_batch.self_onehot_id, axis=-1),
                        jnp.argmax(traj_batch.oppo_onehot_id, axis=-1)
                    )
                    sp_weight = (1 + 2 * config["XP_LOSS_WEIGHTS"]) * (n / 2)
                    xp_weight = config["XP_LOSS_WEIGHTS"] * (n / (2 * (n - 1)))
                    actor_weights = jnp.where(is_sp, sp_weight, xp_weight)

                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                    pg_loss_1 = ratio * gae_norm * actor_weights
                    pg_loss_2 = jnp.clip(
                        ratio,
                        1.0 - config["CLIP_EPS"],
                        1.0 + config["CLIP_EPS"]
                    ) * gae_norm * actor_weights
                    pg_loss = jax.lax.cond(
                        loss_weights.sum() == 0,
                        lambda x: jnp.zeros_like(x).astype(jnp.float32),
                        lambda x: x,
                        -(loss_weights * jnp.minimum(pg_loss_1, pg_loss_2)).sum() / (loss_weights.sum() + 1e-8)
                    )

                    entropy = jax.lax.cond(
                        loss_weights.sum() == 0,
                        lambda x: jnp.zeros_like(x).astype(jnp.float32),
                        lambda x: x,
                        (loss_weights * pi.entropy()).sum() / (loss_weights.sum() + 1e-8)
                    )

                    total_loss = pg_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                    return total_loss, (value_loss, pg_loss, entropy)

                possible_agent_ids = jnp.expand_dims(jnp.arange(config["PARTNER_POP_SIZE"]), 1)
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

                def gather_conf_params_and_return_grads(agent_id):
                    param_vector = gather_params(train_state_conf.params, agent_id)
                    (loss_val_conf, aux_vals_conf), grads_conf = grad_fn(
                        param_vector, conf_policy, minbatch_conf, agent_id
                    )
                    return (loss_val_conf, aux_vals_conf), grads_conf

                def gather_br_params_and_return_grads(agent_id):
                    param_vector = gather_params(train_state_br.params, agent_id)
                    (loss_val_br, aux_vals_br), grads_br = grad_fn(
                        param_vector, br_policy, minbatch_br, agent_id
                    )
                    return (loss_val_br, aux_vals_br), grads_br

                (loss_val_conf, aux_vals_conf), grads_conf = jax.vmap(gather_conf_params_and_return_grads)(possible_agent_ids)
                (loss_val_br, aux_vals_br), grads_br = jax.vmap(gather_br_params_and_return_grads)(possible_agent_ids)

                grads_conf_new = jax.tree.map(lambda x: jnp.squeeze(x, 1), grads_conf)
                grads_br_new = jax.tree.map(lambda x: jnp.squeeze(x, 1), grads_br)
                train_state_conf = train_state_conf.apply_gradients(grads=grads_conf_new)
                train_state_br = train_state_br.apply_gradients(grads=grads_br_new)
                return (train_state_conf, train_state_br), ((loss_val_conf, aux_vals_conf), (loss_val_br, aux_vals_br))

            (
                train_state_conf, train_state_br,
                traj_batch_conf, traj_batch_br,
                advantages_conf, advantages_br,
                targets_conf, targets_br,
                init_conf_h_for_update, init_br_h_for_update,
                rng
            ) = update_state
            rng, perm_rng_conf, perm_rng_br = jax.random.split(rng, 3)

            # Use the ACTUAL initial hidden state from the start of the rollout
            # (not fresh zeros) - this is critical when ROLLOUT_LENGTH != max_steps
            minibatches_conf = _create_minibatches(
                traj_batch_conf, advantages_conf, targets_conf, init_conf_h_for_update,
                config["NUM_CONF_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_conf
            )
            minibatches_br = _create_minibatches(
                traj_batch_br, advantages_br, targets_br, init_br_h_for_update,
                config["NUM_BR_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_br
            )

            (train_state_conf, train_state_br), all_losses = jax.lax.scan(
                _update_minbatch, (train_state_conf, train_state_br), (minibatches_conf, minibatches_br)
            )

            update_state = (
                train_state_conf, train_state_br,
                traj_batch_conf, traj_batch_br,
                advantages_conf, advantages_br,
                targets_conf, targets_br,
                init_conf_h_for_update, init_br_h_for_update,
                rng
            )
            return update_state, all_losses

        def _update_step(update_runner_state, unused):
            (
                all_train_state_conf, all_train_state_br,
                last_env_state, last_obs, last_done, last_conf_h, last_br_h,
                rng, update_steps
            ) = update_runner_state

            # Save the initial hidden state BEFORE the rollout for PPO update epochs
            # This is critical when ROLLOUT_LENGTH != max_steps (episodes span multiple rollouts)
            initial_conf_hstate_for_update = last_conf_h
            initial_br_hstate_for_update = last_br_h

            # Compute reward shaping annealing factor for this update step
            if rew_shaping_anneal is not None:
                current_timestep = update_steps * config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
                anneal_factor = rew_shaping_anneal(current_timestep)
            else:
                anneal_factor = 0.0

            rng, conf_sampling_rng, br_sampling_rng = jax.random.split(rng, 3)

            conf_ids = jax.random.randint(conf_sampling_rng, (config["NUM_ENVS"],), 0, config["PARTNER_POP_SIZE"])
            br_ids = jax.random.randint(br_sampling_rng, (config["NUM_ENVS"],), 0, config["PARTNER_POP_SIZE"])

            runner_state = (
                all_train_state_conf, all_train_state_br, conf_ids, br_ids,
                last_env_state, last_obs, last_done, last_conf_h, last_br_h, rng, anneal_factor
            )
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
            )
            (
                all_train_state_conf, all_train_state_br, last_conf_ids, last_br_ids,
                last_env_state, last_obs, last_done, last_conf_h, last_br_h, rng, _
            ) = runner_state

            last_conf_params = gather_params(all_train_state_conf.params, last_conf_ids)
            last_br_params = gather_params(all_train_state_br.params, last_br_ids)

            last_conf_one_hots = identity_matrix[last_conf_ids]
            last_br_one_hots = identity_matrix[last_br_ids]

            traj_batch_conf, traj_batch_br = traj_batch

            avail_actions_0 = jax.vmap(env.get_avail_actions)(last_env_state.env_state)["agent_0"].astype(jnp.float32)
            _, last_val_conf, _, _ = jax.vmap(forward_pass_conf)(
                params=last_conf_params,
                obs=last_obs["agent_0"],
                id=last_br_one_hots,
                done=last_done["agent_0"],
                avail_actions=avail_actions_0,
                hstate=last_conf_h,
                rng=jax.random.split(jax.random.PRNGKey(0), config["NUM_ENVS"])
            )
            last_val_conf = last_val_conf.squeeze()
            advantages_conf, targets_conf = _calculate_gae(traj_batch_conf, last_val_conf)

            avail_actions_1 = jax.vmap(env.get_avail_actions)(last_env_state.env_state)["agent_1"].astype(jnp.float32)
            _, last_val_br, _, _ = jax.vmap(forward_pass_br)(
                params=last_br_params,
                obs=last_obs["agent_1"],
                id=last_conf_one_hots,
                done=last_done["agent_1"],
                avail_actions=avail_actions_1,
                hstate=last_br_h,
                rng=jax.random.split(jax.random.PRNGKey(0), config["NUM_ENVS"])
            )
            last_val_br = last_val_br.squeeze()
            advantages_br, targets_br = _calculate_gae(traj_batch_br, last_val_br)

            rng, update_rng = jax.random.split(rng, 2)
            update_state = (
                all_train_state_conf, all_train_state_br,
                traj_batch_conf, traj_batch_br,
                advantages_conf, advantages_br,
                targets_conf, targets_br,
                initial_conf_hstate_for_update, initial_br_hstate_for_update,
                update_rng
            )

            update_state, all_losses = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            all_train_state_conf, all_train_state_br = update_state[:2]
            (_, (value_loss_conf, pg_loss_conf, entropy_conf)), (_, (value_loss_br, pg_loss_br, entropy_br)) = all_losses

            metric = traj_batch_conf.info
            metric["update_steps"] = update_steps
            metric["value_loss_conf_agent"] = value_loss_conf
            metric["value_loss_br_agent"] = value_loss_br
            metric["pg_loss_conf_agent"] = pg_loss_conf
            metric["pg_loss_br_agent"] = pg_loss_br
            metric["entropy_conf"] = entropy_conf
            metric["entropy_br"] = entropy_br

            new_runner_state = (
                all_train_state_conf, all_train_state_br,
                last_env_state, last_obs, last_done, last_conf_h, last_br_h,
                rng, update_steps + 1
            )
            return (new_runner_state, metric)

        def _update_step_with_ckpt(state_with_ckpt, unused):
            (
                update_runner_state, checkpoint_array_conf, checkpoint_array_br, ckpt_idx,
                eval_info
            ) = state_with_ckpt

            new_runner_state, metric = _update_step(update_runner_state, None)

            train_state_conf, train_state_br, last_env_state, last_obs, last_done, last_conf_h, last_br_h, rng, update_steps = new_runner_state

            to_store = jnp.logical_or(
                jnp.equal(jnp.mod(update_steps - 1, ckpt_and_eval_interval), 0),
                jnp.equal(update_steps, actual_total_updates)  # Use actual total, not config value
            )

            def store_and_eval_ckpt(args):
                ckpt_arr_and_ep_infos, rng, cidx = args
                ckpt_arr_conf, ckpt_arr_br, _ = ckpt_arr_and_ep_infos
                new_ckpt_arr_conf = jax.tree.map(
                    lambda c_arr, p: c_arr.at[cidx].set(p),
                    ckpt_arr_conf, train_state_conf.params
                )
                new_ckpt_arr_br = jax.tree.map(
                    lambda c_arr, p: c_arr.at[cidx].set(p),
                    ckpt_arr_br, train_state_br.params
                )

                rng, eval_rng = jax.random.split(rng)
                ep_last_info = run_all_episodes(eval_rng, train_state_conf, train_state_br)

                return ((new_ckpt_arr_conf, new_ckpt_arr_br, ep_last_info), rng, cidx + 1)

            def skip_ckpt(args):
                return args

            (checkpoint_array_and_infos, rng, ckpt_idx) = jax.lax.cond(
                to_store,
                store_and_eval_ckpt,
                skip_ckpt,
                ((checkpoint_array_conf, checkpoint_array_br, eval_info), rng, ckpt_idx)
            )
            checkpoint_array_conf, checkpoint_array_br, eval_ep_last_info = checkpoint_array_and_infos

            metric["eval_ep_last_info"] = eval_ep_last_info

            return ((
                train_state_conf, train_state_br,
                last_env_state, last_obs, last_done, last_conf_h, last_br_h, rng, update_steps
            ), checkpoint_array_conf, checkpoint_array_br, ckpt_idx, eval_ep_last_info), metric

        # Run the chunk
        update_runner_state = (
            train_state_conf, train_state_br,
            env_state, obs, done, conf_hstate, br_hstate,
            rng, update_steps
        )

        state_with_ckpt = (
            update_runner_state, checkpoint_array_conf,
            checkpoint_array_br, ckpt_idx, eval_info
        )

        # Run updates_per_chunk updates
        state_with_ckpt, metrics = jax.lax.scan(
            _update_step_with_ckpt,
            state_with_ckpt,
            xs=None,
            length=updates_per_chunk
        )

        (
            final_runner_state, checkpoint_array_conf, checkpoint_array_br,
            final_ckpt_idx, final_eval_info
        ) = state_with_ckpt
        (
            train_state_conf, train_state_br,
            env_state, obs, done, conf_h, br_h, rng, update_steps
        ) = final_runner_state

        new_state = {
            "train_state_conf": train_state_conf,
            "train_state_br": train_state_br,
            "env_state": env_state,
            "obs": obs,
            "done": done,
            "conf_hstate": conf_h,
            "br_hstate": br_h,
            "rng": rng,
            "update_steps": update_steps,
            "checkpoint_array_conf": checkpoint_array_conf,
            "checkpoint_array_br": checkpoint_array_br,
            "ckpt_idx": final_ckpt_idx,
            "eval_info": final_eval_info,
        }

        return new_state, metrics

    return init_train_state, train_one_chunk, config


def get_brdiv_population(config: Dict[str, Any], out: Dict[str, Any], env):
    """Extract partner population from BRDiv training output.

    Args:
        config: Training configuration.
        out: Training output dictionary.
        env: Environment instance.

    Returns:
        Tuple of (partner_params, partner_population).
    """
    brdiv_pop_size = config["PARTNER_POP_SIZE"]
    partner_params = out['final_params_conf']

    # Compute observation dimension (handle both flattened and grid-based)
    obs_shape = env.observation_space(env.agents[1]).shape
    if len(obs_shape) == 1:
        obs_dim = obs_shape[0]
    else:
        obs_dim = int(np.prod(obs_shape))

    # Create policy based on actor type
    actor_type = config.get("ACTOR_TYPE", "mlp")
    activation = config.get("ACTIVATION", "relu")

    if actor_type == "cnn_rnn":
        partner_policy = CNNRNNActorCriticWithConditionalCriticPolicy(
            action_dim=env.action_space(env.agents[1]).n,
            obs_shape=obs_shape,
            pop_size=brdiv_pop_size,
            activation=activation,
            fc_dim_size=config.get("FC_DIM_SIZE", 128),
            gru_hidden_dim=config.get("GRU_HIDDEN_DIM", 128),
        )
    else:
        partner_policy = ActorWithConditionalCriticPolicy(
            action_dim=env.action_space(env.agents[1]).n,
            obs_dim=obs_dim,
            pop_size=brdiv_pop_size,
            activation=activation,
        )

    partner_population = AgentPopulation(
        pop_size=brdiv_pop_size,
        policy_cls=partner_policy
    )

    return partner_params, partner_population


def run_brdiv_training(config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """Run BRDiv training.

    Args:
        config: Training configuration dictionary.
        verbose: Whether to print progress information.

    Returns:
        Dictionary containing training results.
    """
    # Create environment
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)

    # Compute observation dimension (handle both flattened and grid-based)
    obs_shape = env.observation_space(env.agents[0]).shape
    if len(obs_shape) == 1:
        obs_dim = obs_shape[0]
    else:
        obs_dim = int(np.prod(obs_shape))

    if verbose:
        log.info(f"Environment: {config['ENV_NAME']}")
        log.info(f"Layout: {config['ENV_KWARGS']['layout']}")
        log.info(f"Observation space: {obs_shape}")
        log.info(f"Observation dim (flattened): {obs_dim}")
        log.info(f"Action space: {env.action_space(env.agents[0]).n}")
        log.info(f"Actor type: {config.get('ACTOR_TYPE', 'mlp')}")
        log.info(f"Partner population size: {config['PARTNER_POP_SIZE']}")
        log.info(f"XP loss weight: {config['XP_LOSS_WEIGHTS']}")
        log.info(f"Total timesteps: {config['TOTAL_TIMESTEPS']}")
        log.info(f"Learning rate: {config['LR']}")
        log.info(f"NUM_ENVS: {config['NUM_ENVS']}")
        if config.get('REW_SHAPING_HORIZON', 0) > 0:
            log.info(f"Reward shaping horizon: {config['REW_SHAPING_HORIZON']}")

    # Initialize policies based on actor type
    activation = config.get("ACTIVATION", "relu")
    actor_type = config.get("ACTOR_TYPE", "mlp")

    if actor_type == "cnn_rnn":
        # Use CNN+RNN policy with conditional critic (like FCP)
        conf_policy = CNNRNNActorCriticWithConditionalCriticPolicy(
            action_dim=env.action_space(env.agents[0]).n,
            obs_shape=obs_shape,
            pop_size=config["PARTNER_POP_SIZE"],
            activation=activation,
            fc_dim_size=config.get("FC_DIM_SIZE", 128),
            gru_hidden_dim=config.get("GRU_HIDDEN_DIM", 128),
        )
        br_policy = CNNRNNActorCriticWithConditionalCriticPolicy(
            action_dim=env.action_space(env.agents[0]).n,
            obs_shape=obs_shape,
            pop_size=config["PARTNER_POP_SIZE"],
            activation=activation,
            fc_dim_size=config.get("FC_DIM_SIZE", 128),
            gru_hidden_dim=config.get("GRU_HIDDEN_DIM", 128),
        )
    else:
        # Use MLP policy with conditional critic (original)
        conf_policy = ActorWithConditionalCriticPolicy(
            action_dim=env.action_space(env.agents[0]).n,
            obs_dim=obs_dim,
            pop_size=config["PARTNER_POP_SIZE"],
            activation=activation,
        )
        br_policy = ActorWithConditionalCriticPolicy(
            action_dim=env.action_space(env.agents[0]).n,
            obs_dim=obs_dim,
            pop_size=config["PARTNER_POP_SIZE"],
            activation=activation,
        )

    # Generate random seeds
    rng = jax.random.PRNGKey(config["TRAIN_SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    # Run training
    start_time = time.time()

    num_chunks = config.get("NUM_CHUNKS", 1)

    if num_chunks > 1:
        # Use chunked training to reduce GPU memory usage
        if verbose:
            log.info(f"Using chunked training with {num_chunks} chunks")

        out = _run_brdiv_chunked_training(config, env, rngs, conf_policy, br_policy, verbose)
    else:
        # Original non-chunked training
        with jax.disable_jit(False):
            vmapped_train_fn = jax.jit(
                jax.vmap(
                    partial(
                        train_brdiv_partners,
                        env=env,
                        config=config,
                        conf_policy=conf_policy,
                        br_policy=br_policy
                    )
                )
            )
            out = vmapped_train_fn(rngs)

    end_time = time.time()
    training_time = end_time - start_time

    if verbose:
        log.info(f"Training completed in {training_time:.2f} seconds")

    # Extract population
    partner_params, partner_population = get_brdiv_population(config, out, env)

    # Merge updated_config (with NUM_UPDATES) back into config if available
    final_config = config.copy()
    if "updated_config" in out:
        final_config.update(out["updated_config"])

    results = {
        "partner_params": partner_params,
        "partner_population": partner_population,
        "final_params_conf": out["final_params_conf"],
        "final_params_br": out["final_params_br"],
        "checkpoints_conf": out["checkpoints_conf"],
        "checkpoints_br": out["checkpoints_br"],
        "metrics": out["metrics"],
        "all_pair_returns": out["all_pair_returns"],
        "training_time": training_time,
        "config": final_config,
    }

    return results


def _run_brdiv_chunked_training(
    config: Dict[str, Any],
    env,
    rngs: jnp.ndarray,
    conf_policy,
    br_policy,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run BRDiv training in chunks to reduce GPU memory usage.

    Args:
        config: Training configuration dictionary.
        env: Environment instance.
        rngs: Random keys for each seed, shape (NUM_SEEDS,).
        conf_policy: Confederate policy.
        br_policy: Best response policy.
        verbose: Whether to print progress information.

    Returns:
        Dictionary with same structure as non-chunked training output.
    """
    num_chunks = config["NUM_CHUNKS"]
    save_interval = config.get("SAVE_INTERVAL", 0)
    output_dir = config.get("OUTPUT_DIR", None)

    # Calculate NUM_UPDATES to validate divisibility
    num_agents = env.num_agents
    num_updates = int(
        config["TOTAL_TIMESTEPS"] // (num_agents * config["ROLLOUT_LENGTH"] * config["NUM_ENVS"])
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
    init_fn, train_chunk_fn, updated_config = make_train_brdiv_chunked(config, env, conf_policy, br_policy)

    # Create vmapped and jitted functions
    init_jit = jax.jit(jax.vmap(init_fn))
    chunk_jit = jax.jit(jax.vmap(train_chunk_fn))

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
    partner_pop_size = config["PARTNER_POP_SIZE"]

    if csv_log_enabled and output_dir:
        csv_path = Path(output_dir) / "metrics.csv"
        # Build fieldnames for BRDiv-specific metrics
        csv_fieldnames = [
            "chunk", "total_chunks", "update_step", "total_timesteps",
            # Confederate agent metrics
            "mean_conf_return", "std_conf_return",
            "mean_conf_value_loss", "mean_conf_pg_loss", "mean_conf_entropy",
            # BR agent metrics
            "mean_br_return", "std_br_return",
            "mean_br_value_loss", "mean_br_pg_loss", "mean_br_entropy",
            # Elapsed time
            "elapsed_time",
        ]
        # Add per-population-member metrics
        for p in range(partner_pop_size):
            csv_fieldnames.extend([
                f"conf_{p}_mean_return", f"br_{p}_mean_return",
            ])
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
            # IMPORTANT: Use returned_episode flag to filter for actual completed episodes
            # This works correctly even when ROLLOUT_LENGTH != max_steps
            current_update = (chunk_idx + 1) * updates_per_chunk
            current_timesteps = current_update * config["ROLLOUT_LENGTH"] * config["NUM_ENVS"] * num_agents
            elapsed = time.time() - chunk_start_time

            # Build CSV row with basic info
            csv_row = {
                "chunk": chunk_idx + 1,
                "total_chunks": num_chunks,
                "update_step": current_update,
                "total_timesteps": current_timesteps,
                "elapsed_time": elapsed,
            }

            # Get mask of completed episodes
            # Shape: (num_seeds, updates_per_chunk, rollout_len, num_actors)
            ep_done_key = "returned_episode"
            if ep_done_key in chunk_metrics_cpu:
                ep_done_mask = chunk_metrics_cpu[ep_done_key] > 0
            else:
                ep_done_mask = None

            # Extract metrics from chunk data
            # Confederate agent returns
            conf_returns_key = "returned_episode_returns"
            if conf_returns_key in chunk_metrics_cpu:
                conf_returns = chunk_metrics_cpu[conf_returns_key]
                if ep_done_mask is not None and np.any(ep_done_mask):
                    valid_returns = conf_returns[ep_done_mask]
                    csv_row["mean_conf_return"] = float(np.mean(valid_returns))
                    csv_row["std_conf_return"] = float(np.std(valid_returns))
                else:
                    csv_row["mean_conf_return"] = 0.0
                    csv_row["std_conf_return"] = 0.0
            else:
                csv_row["mean_conf_return"] = 0.0
                csv_row["std_conf_return"] = 0.0

            # BR agent returns (use conf returns as proxy since they share rewards)
            csv_row["mean_br_return"] = csv_row["mean_conf_return"]
            csv_row["std_br_return"] = csv_row["std_conf_return"]

            # Loss metrics (these don't need episode filtering - they're per-minibatch)
            if "value_loss_conf_agent" in chunk_metrics_cpu:
                csv_row["mean_conf_value_loss"] = float(np.mean(chunk_metrics_cpu["value_loss_conf_agent"]))
            else:
                csv_row["mean_conf_value_loss"] = 0.0

            if "pg_loss_conf_agent" in chunk_metrics_cpu:
                csv_row["mean_conf_pg_loss"] = float(np.mean(chunk_metrics_cpu["pg_loss_conf_agent"]))
            else:
                csv_row["mean_conf_pg_loss"] = 0.0

            if "entropy_conf" in chunk_metrics_cpu:
                csv_row["mean_conf_entropy"] = float(np.mean(chunk_metrics_cpu["entropy_conf"]))
            else:
                csv_row["mean_conf_entropy"] = 0.0

            if "value_loss_br_agent" in chunk_metrics_cpu:
                csv_row["mean_br_value_loss"] = float(np.mean(chunk_metrics_cpu["value_loss_br_agent"]))
            else:
                csv_row["mean_br_value_loss"] = 0.0

            if "pg_loss_br_agent" in chunk_metrics_cpu:
                csv_row["mean_br_pg_loss"] = float(np.mean(chunk_metrics_cpu["pg_loss_br_agent"]))
            else:
                csv_row["mean_br_pg_loss"] = 0.0

            if "entropy_br" in chunk_metrics_cpu:
                csv_row["mean_br_entropy"] = float(np.mean(chunk_metrics_cpu["entropy_br"]))
            else:
                csv_row["mean_br_entropy"] = 0.0

            # Per-population member metrics (if available from eval_info)
            for p in range(partner_pop_size):
                csv_row[f"conf_{p}_mean_return"] = csv_row["mean_conf_return"]
                csv_row[f"br_{p}_mean_return"] = csv_row["mean_br_return"]

            csv_logger.log(csv_row)

        # Save at intervals if configured
        if save_interval > 0 and output_dir and ((chunk_idx + 1) % save_interval == 0):
            if verbose:
                log.info(f"Saving intermediate checkpoint at chunk {chunk_idx + 1}...")

            # Get current checkpoints and metrics
            current_checkpoints_conf = jax.device_get(states["checkpoint_array_conf"])
            current_checkpoints_br = jax.device_get(states["checkpoint_array_br"])
            current_params_conf = jax.device_get(states["train_state_conf"].params)
            current_params_br = jax.device_get(states["train_state_br"].params)
            current_metrics = jax.tree.map(
                lambda *chunks: np.concatenate(chunks, axis=1),
                *all_chunk_metrics
            )

            save_data = {
                "final_params_conf": current_params_conf,
                "final_params_br": current_params_br,
                "checkpoints_conf": current_checkpoints_conf,
                "checkpoints_br": current_checkpoints_br,
                "metrics": current_metrics,
            }
            # Compute checkpoint returns from metrics
            checkpoint_returns = compute_checkpoint_returns(current_metrics, updated_config)
            # Save intermediate checkpoints separated by agent type and checkpoint index
            save_path = save_separated_checkpoints_multi(
                save_data,
                config,
                str(output_path),
                savename=f"brdiv_checkpoint_chunk_{chunk_idx + 1}",
                checkpoint_keys={"conf": "checkpoints_conf", "br": "checkpoints_br"},
                checkpoint_returns=checkpoint_returns,
            )
            if verbose:
                log.info(f"Saved intermediate separated checkpoint to: {save_path}")

        # Force garbage collection to free GPU memory
        if chunk_idx < num_chunks - 1:
            import gc
            gc.collect()

    # Concatenate metrics from all chunks along the update dimension (axis 1 for single seed)
    # Metrics shape: (num_seeds, updates_per_chunk, ...)
    def concat_metrics(*chunks):
        return np.concatenate(chunks, axis=1)

    metrics = jax.tree.map(concat_metrics, *all_chunk_metrics)

    # Get final checkpoints and params from state
    checkpoints_conf = jax.device_get(states["checkpoint_array_conf"])
    checkpoints_br = jax.device_get(states["checkpoint_array_br"])
    final_params_conf = jax.device_get(states["train_state_conf"].params)
    final_params_br = jax.device_get(states["train_state_br"].params)
    all_pair_returns = jax.device_get(states["eval_info"])

    return {
        "final_params_conf": final_params_conf,
        "final_params_br": final_params_br,
        "checkpoints_conf": checkpoints_conf,
        "checkpoints_br": checkpoints_br,
        "metrics": metrics,
        "all_pair_returns": all_pair_returns,
        "updated_config": updated_config,
    }


def print_training_summary(results: Dict[str, Any]):
    """Print a summary of training results.

    Args:
        results: Training results dictionary.
    """
    config = results["config"]
    metrics = results["metrics"]
    pop_size = config["PARTNER_POP_SIZE"]

    print("\n" + "=" * 60)
    print("BRDiv Training Summary")
    print("=" * 60)
    print(f"\nLayout: {config['ENV_KWARGS']['layout']}")
    print(f"Partner population size: {pop_size}")
    print(f"XP loss weight: {config['XP_LOSS_WEIGHTS']}")
    print(f"Training time: {results['training_time']:.2f} seconds")

    # Compute SP and XP returns
    all_returns = np.asarray(metrics["eval_ep_last_info"]["returned_episode_returns"])

    all_conf_ids, all_br_ids = _get_all_ids(pop_size)
    sp_mask = (all_conf_ids == all_br_ids)

    sp_returns = all_returns[:, -1, sp_mask].mean()
    xp_returns = all_returns[:, -1, ~sp_mask].mean()

    print(f"\nFinal Self-Play (SP) mean return: {sp_returns:.2f}")
    print(f"Final Cross-Play (XP) mean return: {xp_returns:.2f}")
    print(f"SP - XP gap: {sp_returns - xp_returns:.2f}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train BRDiv teammates on Overcooked V2",
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

    # Training parameters
    parser.add_argument(
        "--total_timesteps",
        type=float,
        default=3e7,
        help="Total timesteps (default: 3e7, from JaxMARL recipe)",
    )
    parser.add_argument(
        "--partner_pop_size",
        type=int,
        default=10,
        help="Population size (default: 10)",
    )
    parser.add_argument(
        "--num_checkpoints",
        type=int,
        default=5,
        help="Number of checkpoints (default: 5)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=256,
        help="Number of parallel environments (default: 256, from JaxMARL recipe)",
    )
    parser.add_argument(
        "--xp_loss_weight",
        type=float,
        default=0.5,
        help="Cross-play loss weight (default: 0.5)",
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
        choices=["mlp", "cnn_rnn"],
        help="Actor network type (default: cnn_rnn, same as FCP)",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=1,
        help="Number of training chunks for memory efficiency (default: 1). "
             "Use >1 to reduce GPU memory usage for large total_timesteps.",
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
        help="Enable CSV logging of metrics (default: True)",
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
    config["XP_LOSS_WEIGHTS"] = args.xp_loss_weight
    config["TRAIN_SEED"] = args.seed
    config["LR"] = args.lr
    config["ACTOR_TYPE"] = args.actor_type
    config["NUM_CHUNKS"] = args.num_chunks
    config["SAVE_INTERVAL"] = args.save_interval
    config["OUTPUT_DIR"] = args.output_dir
    config["CSV_LOG"] = args.csv_log
    config["CSV_INTERVAL"] = args.csv_interval
    config["ENV_KWARGS"]["max_steps"] = args.max_steps

    # Set flatten_obs based on actor type
    if args.actor_type == "cnn_rnn":
        config["ENV_KWARGS"]["flatten_obs"] = False  # CNN+RNN uses grid observations
    else:
        config["ENV_KWARGS"]["flatten_obs"] = True   # MLP uses flattened observations

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
        config["NUM_ENVS"] = 8
        config["NUM_MINIBATCHES"] = 4  # Must be <= NUM_ACTORS (NUM_ENVS * 2)
        config["UPDATE_EPOCHS"] = 2
        config["NUM_EVAL_EPISODES"] = 2
        log.info("Running in DEBUG mode with reduced settings")

    # Run training
    if not args.quiet:
        log.info(f"Starting BRDiv training on layout: {args.layout}")
        if args.gpu is not None:
            if args.gpu == "-1":
                log.info("Using CPU")
            else:
                log.info(f"Using GPU: {args.gpu}")
        else:
            log.info(f"Using all available devices: {jax.devices()}")

    try:
        results = run_brdiv_training(config, verbose=not args.quiet)

        if not args.quiet:
            print_training_summary(results)

        # Save results if output directory specified
        if args.output_dir:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            save_data = {
                "final_params_conf": results["final_params_conf"],
                "final_params_br": results["final_params_br"],
                "checkpoints_conf": results["checkpoints_conf"],
                "checkpoints_br": results["checkpoints_br"],
                "metrics": results["metrics"],
            }
            # Compute checkpoint returns from metrics
            # Use results["config"] which contains NUM_UPDATES computed during training
            checkpoint_returns = compute_checkpoint_returns(results["metrics"], results["config"])
            # Save checkpoints separated by agent type and checkpoint index
            # Structure: brdiv_train_run/conf/ckpt_0, conf/ckpt_1, ..., br/ckpt_0, ...
            save_path = save_separated_checkpoints_multi(
                save_data, results["config"], str(output_path),
                savename="brdiv_train_run",
                checkpoint_keys={"conf": "checkpoints_conf", "br": "checkpoints_br"},
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
