#!/usr/bin/env python3
"""Training script for LBRDiv (Lagrangian Best Response Diversity) on Overcooked V2.

This script provides a standalone entry point for training LBRDiv teammate policies
on Overcooked V2 environments, without requiring Hydra configuration.

LBRDiv extends BRDiv by using Lagrangian relaxation to enforce diversity constraints.
Instead of directly negating cross-play rewards, it maintains two matrices of Lagrange
multipliers that adaptively weight the self-play and cross-play objectives.

Example Usage:
    # Train on cramped_room layout with default settings
    python -m teammate_generation.train_lbrdiv_overcooked_v2 --layout cramped_room

    # Train with custom parameters
    python -m teammate_generation.train_lbrdiv_overcooked_v2 \\
        --layout coord_ring \\
        --total_timesteps 4.5e7 \\
        --partner_pop_size 4 \\
        --tolerance_factor 1.0 \\
        --lagrange_lr 0.01 \\
        --seed 0

    # Quick test run
    python -m teammate_generation.train_lbrdiv_overcooked_v2 \\
        --layout cramped_room \\
        --total_timesteps 1e5 \\
        --partner_pop_size 2 \\
        --num_checkpoints 2 \\
        --debug

References:
    Rahman et al., "Minimum Coverage Sets for Training Robust Ad Hoc Teamplay Agents",
    AAAI 2024. https://ojs.aaai.org/index.php/AAAI/article/view/29702
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

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)


class CSVMetricsLogger:
    """Helper class to write metrics to CSV file."""

    def __init__(self, filepath: str, fieldnames: List[str]):
        self.filepath = filepath
        self.fieldnames = fieldnames
        self._initialized = False

    def _init_file(self):
        if not self._initialized:
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


def compute_checkpoint_returns(
    metrics: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Compute mean returns at each checkpoint interval from training metrics."""
    num_checkpoints = config["NUM_CHECKPOINTS"]
    num_updates = config["NUM_UPDATES"]
    num_seeds = config.get("NUM_SEEDS", 1)

    ckpt_interval = num_updates // max(1, num_checkpoints - 1)

    checkpoint_update_indices = []
    for ckpt_idx in range(num_checkpoints - 1):
        checkpoint_update_indices.append(ckpt_idx * ckpt_interval)
    checkpoint_update_indices.append(num_updates - 1)

    base_returns = np.zeros((num_seeds, num_checkpoints))
    shaped_returns = np.zeros((num_seeds, num_checkpoints))

    if "returned_episode" in metrics:
        ep_done = metrics["returned_episode"]
    else:
        ep_done = None

    for ckpt_idx, update_idx in enumerate(checkpoint_update_indices):
        if ckpt_idx == 0:
            start_idx = 0
        else:
            start_idx = checkpoint_update_indices[ckpt_idx - 1] + 1
        end_idx = update_idx + 1

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
    """Write training metrics to CSV file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "training_metrics.csv"

    partner_pop_size = config["PARTNER_POP_SIZE"]
    num_agents = 2

    rollout_length = config["ROLLOUT_LENGTH"]
    num_envs = config["NUM_ENVS"]
    timesteps_per_update = rollout_length * num_envs * num_agents

    csv_fieldnames = [
        "update_step", "total_timesteps",
        "mean_conf_return", "std_conf_return",
        "mean_conf_value_loss", "mean_conf_pg_loss", "mean_conf_entropy",
        "mean_br_return", "std_br_return",
        "mean_br_value_loss", "mean_br_pg_loss", "mean_br_entropy",
        "elapsed_time",
    ]
    for p in range(partner_pop_size):
        csv_fieldnames.extend([
            f"conf_{p}_mean_return", f"br_{p}_mean_return",
        ])

    csv_logger = CSVMetricsLogger(str(csv_path), csv_fieldnames)

    if "returned_episode_returns" in metrics:
        num_updates = metrics["returned_episode_returns"].shape[1]
    elif "update_steps" in metrics:
        num_updates = metrics["update_steps"].shape[1]
    else:
        log.warning("Cannot determine number of updates from metrics. Skipping CSV logging.")
        return

    for update_idx in range(num_updates):
        current_timesteps = (update_idx + 1) * timesteps_per_update

        csv_row = {
            "update_step": update_idx + 1,
            "total_timesteps": current_timesteps,
            "elapsed_time": training_time * (update_idx + 1) / num_updates,
        }

        if "returned_episode" in metrics:
            ep_done_mask = metrics["returned_episode"][:, update_idx, :, :] > 0
        else:
            ep_done_mask = None

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

        csv_row["mean_br_return"] = csv_row["mean_conf_return"]
        csv_row["std_br_return"] = csv_row["std_conf_return"]

        for key, csv_key in [
            ("value_loss_conf_agent", "mean_conf_value_loss"),
            ("pg_loss_conf_agent", "mean_conf_pg_loss"),
            ("entropy_conf", "mean_conf_entropy"),
            ("value_loss_br_agent", "mean_br_value_loss"),
            ("pg_loss_br_agent", "mean_br_pg_loss"),
            ("entropy_br", "mean_br_entropy"),
        ]:
            if key in metrics:
                csv_row[csv_key] = float(np.mean(metrics[key][:, update_idx, ...]))
            else:
                csv_row[csv_key] = 0.0

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
    """Get default LBRDiv training configuration for a given layout."""
    return {
        "ENV_NAME": "overcooked-v2",
        "ENV_KWARGS": {
            "layout": layout,
            "flatten_obs": False,
            "max_steps": 400,
        },
        "ROLLOUT_LENGTH": 256,

        "ALG": "lbrdiv",
        "ACTOR_TYPE": "cnn_rnn",
        "TOTAL_TIMESTEPS": 3e7,
        "NUM_CHECKPOINTS": 5,
        "PARTNER_POP_SIZE": 10,

        # LBRDiv-specific: Lagrangian relaxation parameters
        "TOLERANCE_FACTOR": 0.1,
        "LAGRANGE_LR": 0.01,

        # PPO hyperparameters (from working JaxMARL recipe)
        "NUM_ENVS": 256,
        "LR": 0.00025,
        "ANNEAL_LR": True,
        "LR_WARMUP": 0.05,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 64,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.25,
        "ACTIVATION": "relu",

        "FC_DIM_SIZE": 128,
        "GRU_HIDDEN_DIM": 128,

        "REW_SHAPING_HORIZON": 1.5e7,

        "NUM_SEEDS": 1,
        "TRAIN_SEED": 0,
        "NUM_EVAL_EPISODES": 20,
        "NUM_CHUNKS": 1,

        "EVAL_MAX_STEPS": 400,
    }


def _get_all_ids(pop_size: int):
    """Generate all confederate-BR ID pairs for cross-play evaluation."""
    cross_product = np.meshgrid(
        np.arange(pop_size),
        np.arange(pop_size)
    )
    agent_id_cartesian_product = np.stack([g.ravel() for g in cross_product], axis=-1)
    all_conf_ids = agent_id_cartesian_product[:, 1]
    all_br_ids = agent_id_cartesian_product[:, 0]
    return all_conf_ids, all_br_ids


def gather_params(partner_params_pytree, idx_vec):
    """Gather parameters for specific indices from a parameter pytree."""
    def gather_leaf(leaf):
        def slice_one(idx):
            return leaf[idx]
        return jax.vmap(slice_one)(idx_vec)

    return jax.tree.map(gather_leaf, partner_params_pytree)


def train_lbrdiv_partners(train_rng, env, config: Dict[str, Any], conf_policy, br_policy):
    """Train LBRDiv confederate and best response policies.

    LBRDiv uses Lagrangian relaxation to enforce diversity constraints instead of
    directly negating cross-play rewards. Two Lagrange multiplier matrices control
    how strongly the diversity constraints are enforced.
    """
    num_agents = env.num_agents
    assert num_agents == 2, "LBRDiv requires exactly 2 agents"

    config["NUM_GAME_AGENTS"] = num_agents
    config["NUM_CONF_ACTORS"] = config["NUM_ENVS"]
    config["NUM_BR_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // (num_agents * config["ROLLOUT_LENGTH"] * config["NUM_ENVS"])
    )

    def make_lbrdiv_agents(config):
        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

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

                if config["ANNEAL_LR"]:
                    if config.get("LR_WARMUP", 0.0) > 0:
                        lr_schedule = create_warmup_cosine_schedule()
                    else:
                        lr_schedule = linear_schedule
                else:
                    lr_schedule = config["LR"]

                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=lr_schedule, eps=1e-5),
                )
                tx_br = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=lr_schedule, eps=1e-5),
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

                # Apply reward shaping with annealing
                base_reward_1 = reward["agent_1"]
                shaped_reward_1 = info_1.get("shaped_reward", jnp.zeros_like(base_reward_1))
                total_reward_1 = base_reward_1 + anneal_factor * shaped_reward_1

                # LBRDiv key difference from BRDiv: rewards are NOT negated for XP.
                # Diversity is enforced through Lagrange multipliers instead.
                transition_0 = XPTransition(
                    done=done["agent_0"],
                    action=act_0,
                    value=val_0,
                    self_onehot_id=updated_conf_onehot_ids,
                    oppo_onehot_id=updated_br_onehot_ids,
                    reward=total_reward_1,
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
                    reward=total_reward_1,
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
                    minbatch_conf, minbatch_br, lms_vertical, lms_horizontal = all_data

                    def _loss_fn(param, agent_policy, minbatch, agent_id, lms_vertical, lms_horizontal):
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
                        int_self_id = jnp.argmax(traj_batch.self_onehot_id, axis=-1)
                        int_oppo_id = jnp.argmax(traj_batch.oppo_onehot_id, axis=-1)

                        # Compute Lagrange-multiplier-based actor weights
                        def _gather_sp_weights(ids):
                            s_id, _ = ids
                            return jnp.sum(lms_vertical, axis=-1)[s_id], jnp.sum(lms_horizontal, axis=-1)[s_id]

                        def _gather_xp_weights(ids):
                            s_id, o_id = ids
                            return -lms_vertical[s_id][o_id], -lms_horizontal[o_id][s_id]

                        def _get_weights(s_id, o_id):
                            return jax.lax.cond(
                                jnp.equal(s_id, o_id),
                                _gather_sp_weights,
                                _gather_xp_weights,
                                (s_id, o_id)
                            )

                        # Value loss
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

                        # Compute actor weights from Lagrange multipliers with importance reweighting
                        n = config["PARTNER_POP_SIZE"]
                        is_sp = jnp.equal(
                            jnp.argmax(traj_batch.self_onehot_id, axis=-1),
                            jnp.argmax(traj_batch.oppo_onehot_id, axis=-1)
                        )
                        weights1, weights2 = jax.vmap(jax.vmap(_get_weights))(int_self_id, int_oppo_id)
                        actor_weights_sp = (weights1 + weights2) * (n / 2)
                        actor_weights_xp = (weights1 + weights2) * (n / (2 * (n - 1)))
                        actor_weights = jnp.where(is_sp, actor_weights_sp, actor_weights_xp)

                        # Policy gradient loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                        pg_loss_1 = ratio * actor_weights * gae_norm
                        pg_loss_2 = jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"]
                        ) * actor_weights * gae_norm
                        pg_loss = jax.lax.cond(
                            loss_weights.sum() == 0,
                            lambda x: jnp.zeros_like(x).astype(jnp.float32),
                            lambda x: x,
                            -(loss_weights * jnp.minimum(pg_loss_1, pg_loss_2)).sum() / (loss_weights.sum() + 1e-8)
                        )

                        # Weight entropy based on actor weights
                        all_sp_weights1, all_sp_weights2 = jax.vmap(_gather_sp_weights)((int_self_id, int_self_id))
                        entropy_scaler = jnp.maximum(all_sp_weights1, all_sp_weights2)

                        entropy = jax.lax.cond(
                            loss_weights.sum() == 0,
                            lambda x: jnp.zeros_like(x).astype(jnp.float32),
                            lambda x: x,
                            (loss_weights * entropy_scaler * pi.entropy()).sum() / (loss_weights.sum() + 1e-8)
                        )

                        total_loss = pg_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, pg_loss, entropy)

                    possible_agent_ids = jnp.expand_dims(jnp.arange(config["PARTNER_POP_SIZE"]), 1)
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

                    def gather_conf_params_and_return_grads(agent_id):
                        # Transpose LM matrices for confederate side to ensure symmetry
                        param_vector = gather_params(train_state_conf.params, agent_id)
                        (loss_val_conf, aux_vals_conf), grads_conf = grad_fn(
                            param_vector, conf_policy, minbatch_conf, agent_id,
                            jnp.transpose(lms_vertical), jnp.transpose(lms_horizontal)
                        )
                        return (loss_val_conf, aux_vals_conf), grads_conf

                    def gather_br_params_and_return_grads(agent_id):
                        param_vector = gather_params(train_state_br.params, agent_id)
                        (loss_val_br, aux_vals_br), grads_br = grad_fn(
                            param_vector, br_policy, minbatch_br, agent_id,
                            lms_vertical, lms_horizontal
                        )
                        return (loss_val_br, aux_vals_br), grads_br

                    (loss_val_conf, aux_vals_conf), grads_conf = jax.vmap(gather_conf_params_and_return_grads)(possible_agent_ids)
                    (loss_val_br, aux_vals_br), grads_br = jax.vmap(gather_br_params_and_return_grads)(possible_agent_ids)

                    grads_conf_new = jax.tree.map(lambda x: jnp.squeeze(x, 1), grads_conf)
                    grads_br_new = jax.tree.map(lambda x: jnp.squeeze(x, 1), grads_br)
                    train_state_conf = train_state_conf.apply_gradients(grads=grads_conf_new)
                    train_state_br = train_state_br.apply_gradients(grads=grads_br_new)
                    return (train_state_conf, train_state_br), ((loss_val_conf, aux_vals_conf), (loss_val_br, aux_vals_br))

                # --- Lagrange multiplier gradient computation ---

                def compute_lagrange_grads_same(params_br, batch, target_value, ids):
                    """Compute Lagrange gradients for self-play pairs (conf_id == br_id)."""
                    conf_id, br_id = ids

                    all_target_value = jnp.reshape(target_value, (-1, 1))
                    repeated_value_sp = jnp.repeat(
                        jnp.reshape(all_target_value, (1, -1)),
                        config["PARTNER_POP_SIZE"],
                        axis=0
                    )

                    # Compute grad_sp_vary_conf
                    relevant_conf_params = gather_params(params_br, jnp.reshape(conf_id, (1,)))
                    relevant_conf_params = jax.tree.map(lambda x: jnp.squeeze(x, 0), relevant_conf_params)

                    def _get_value_xp_vary_conf(param, agent_onehot_id):
                        ts, bs = batch.obs.shape[:2]
                        agent_onehot_id = agent_onehot_id[jnp.newaxis, jnp.newaxis, ...].repeat(ts, axis=0).repeat(bs, axis=1)
                        _, value_xp_vary_conf, _, _ = br_policy.get_action_value_policy(
                            params=param,
                            obs=batch.obs,
                            done=batch.done,
                            avail_actions=batch.avail_actions,
                            hstate=init_br_hstate,
                            rng=jax.random.PRNGKey(0),
                            aux_obs=agent_onehot_id
                        )
                        return value_xp_vary_conf.reshape(ts * bs)

                    all_possible_value_xp_vary_conf = jax.vmap(
                        lambda agent_id: _get_value_xp_vary_conf(relevant_conf_params, agent_id)
                    )(jnp.eye(config["PARTNER_POP_SIZE"]))

                    all_possible_value_xp_vary_conf = all_possible_value_xp_vary_conf.at[conf_id].set(
                        repeated_value_sp[conf_id]
                    )

                    offsetting_thresholds = jnp.zeros_like(repeated_value_sp)
                    offsetting_thresholds = offsetting_thresholds.at[conf_id].set(
                        config["TOLERANCE_FACTOR"] * jnp.ones_like(offsetting_thresholds[conf_id])
                    )
                    grad_sp_vary_conf = repeated_value_sp + offsetting_thresholds - (
                        all_possible_value_xp_vary_conf + config["TOLERANCE_FACTOR"] * jnp.ones_like(offsetting_thresholds)
                    )

                    # Compute grad_sp_vary_br
                    relevant_params = gather_params(params_br, jnp.arange(config["PARTNER_POP_SIZE"]))

                    def _get_value_xp_vary_br(param):
                        ts, bs = batch.obs.shape[:2]
                        conf_one_hot = jnp.eye(config["PARTNER_POP_SIZE"])[conf_id]
                        conf_one_hot = conf_one_hot[jnp.newaxis, jnp.newaxis, ...].repeat(ts, axis=0).repeat(bs, axis=1)
                        _, value_xp_vary_br, _, _ = br_policy.get_action_value_policy(
                            params=param,
                            obs=batch.obs,
                            done=batch.done,
                            avail_actions=batch.avail_actions,
                            hstate=init_br_hstate,
                            rng=jax.random.PRNGKey(0),
                            aux_obs=conf_one_hot
                        )
                        return value_xp_vary_br.reshape(ts * bs)

                    all_possible_value_xp_vary_br = jax.vmap(
                        lambda param: _get_value_xp_vary_br(param)
                    )(relevant_params)

                    all_possible_value_xp_vary_br = jnp.reshape(
                        all_possible_value_xp_vary_br, (config["PARTNER_POP_SIZE"], -1)
                    )
                    all_possible_value_xp_vary_br = all_possible_value_xp_vary_br.at[conf_id].set(
                        repeated_value_sp[conf_id]
                    )

                    grad_sp_vary_br = repeated_value_sp + offsetting_thresholds - (
                        all_possible_value_xp_vary_br + config["TOLERANCE_FACTOR"] * jnp.ones_like(offsetting_thresholds)
                    )

                    # Compute loss weights
                    all_self_id_int = jnp.reshape(
                        batch.self_onehot_id, (-1, jnp.shape(batch.self_onehot_id)[-1])
                    ).argmax(axis=-1)

                    all_oppo_id_int = jnp.reshape(
                        batch.oppo_onehot_id, (-1, jnp.shape(batch.oppo_onehot_id)[-1])
                    ).argmax(axis=-1)

                    self_is_conf = jnp.equal(all_self_id_int, conf_id).astype(jnp.float32)
                    oppo_is_conf = jnp.equal(all_oppo_id_int, conf_id).astype(jnp.float32)
                    loss_weights = self_is_conf * oppo_is_conf
                    repeated_loss_weights = jnp.repeat(
                        jnp.expand_dims(loss_weights, axis=0),
                        config["PARTNER_POP_SIZE"],
                        axis=0
                    )

                    vertical_grads = jnp.sum(grad_sp_vary_conf * repeated_loss_weights, axis=-1) / (jnp.sum(loss_weights) + 1e-8)
                    horizontal_grads = jnp.sum(grad_sp_vary_br * repeated_loss_weights, axis=-1) / (jnp.sum(loss_weights) + 1e-8)

                    output_grad_matrix_vertical = jnp.zeros((config["PARTNER_POP_SIZE"], config["PARTNER_POP_SIZE"]))
                    output_grad_matrix_horizontal = jnp.zeros((config["PARTNER_POP_SIZE"], config["PARTNER_POP_SIZE"]))

                    output_grad_matrix_vertical = output_grad_matrix_vertical.at[conf_id].set(vertical_grads)
                    output_grad_matrix_horizontal = output_grad_matrix_horizontal.at[conf_id].set(horizontal_grads)

                    return output_grad_matrix_vertical, output_grad_matrix_horizontal

                def compute_lagrange_grads_diff(params_br, batch, target_returns, ids):
                    """Compute Lagrange gradients for cross-play pairs (conf_id != br_id)."""
                    conf_id, br_id = ids
                    param_conf_id = gather_params(params_br, jnp.reshape(conf_id, (1,)))
                    param_br_id = gather_params(params_br, jnp.reshape(br_id, (1,)))

                    param_br_id = jax.tree.map(lambda x: jnp.squeeze(x, 0), param_br_id)
                    param_conf_id = jax.tree.map(lambda x: jnp.squeeze(x, 0), param_conf_id)

                    all_self_id_int = jnp.reshape(
                        batch.self_onehot_id, (-1, jnp.shape(batch.self_onehot_id)[-1])
                    ).argmax(axis=-1)

                    all_oppo_id_int = jnp.reshape(
                        batch.oppo_onehot_id, (-1, jnp.shape(batch.oppo_onehot_id)[-1])
                    ).argmax(axis=-1)

                    all_target_returns = jnp.reshape(target_returns, (-1))

                    oppo_is_conf = jnp.equal(all_oppo_id_int, conf_id).astype(jnp.float32)
                    self_is_br = jnp.equal(all_self_id_int, br_id).astype(jnp.float32)
                    loss_weights = oppo_is_conf * self_is_br

                    ts, bs = batch.obs.shape[:2]

                    conf_one_hot = jnp.eye(config["PARTNER_POP_SIZE"])[conf_id]
                    conf_one_hot = conf_one_hot[jnp.newaxis, jnp.newaxis, ...].repeat(ts, axis=0).repeat(bs, axis=1)
                    br_one_hot = jnp.eye(config["PARTNER_POP_SIZE"])[br_id]
                    br_one_hot = br_one_hot[jnp.newaxis, jnp.newaxis, ...].repeat(ts, axis=0).repeat(bs, axis=1)

                    _, value_sp_pop_is_br, _, _ = br_policy.get_action_value_policy(
                        params=param_br_id,
                        obs=batch.obs,
                        done=batch.done,
                        avail_actions=batch.avail_actions,
                        hstate=init_br_hstate,
                        rng=jax.random.PRNGKey(0),
                        aux_obs=br_one_hot
                    )
                    value_sp_pop_is_br = value_sp_pop_is_br.reshape(bs * ts)

                    _, value_sp_pop_is_not_br, _, _ = br_policy.get_action_value_policy(
                        params=param_conf_id,
                        obs=batch.obs,
                        done=batch.done,
                        avail_actions=batch.avail_actions,
                        hstate=init_br_hstate,
                        rng=jax.random.PRNGKey(0),
                        aux_obs=conf_one_hot
                    )
                    value_sp_pop_is_not_br = value_sp_pop_is_not_br.reshape(bs * ts)

                    vertical_diff = value_sp_pop_is_br - all_target_returns - config["TOLERANCE_FACTOR"]
                    horizontal_diff = value_sp_pop_is_not_br - all_target_returns - config["TOLERANCE_FACTOR"]

                    total_grad_vertical = (loss_weights * vertical_diff).sum() / (loss_weights.sum() + 1e-8)
                    total_grad_horizontal = (loss_weights * horizontal_diff).sum() / (loss_weights.sum() + 1e-8)

                    output_grad_matrix_vertical = jnp.zeros((config["PARTNER_POP_SIZE"], config["PARTNER_POP_SIZE"]))
                    output_grad_matrix_horizontal = jnp.zeros((config["PARTNER_POP_SIZE"], config["PARTNER_POP_SIZE"]))

                    output_grad_matrix_vertical = output_grad_matrix_vertical.at[br_id, conf_id].set(total_grad_vertical)
                    output_grad_matrix_horizontal = output_grad_matrix_horizontal.at[conf_id, br_id].set(total_grad_horizontal)

                    return output_grad_matrix_vertical, output_grad_matrix_horizontal

                def _compute_indiv_lagrange_grads(conf_id, br_id):
                    return jax.lax.cond(
                        conf_id == br_id,
                        lambda ids: compute_lagrange_grads_same(train_state_br.params, traj_batch_br, targets_br, ids),
                        lambda ids: compute_lagrange_grads_diff(train_state_br.params, traj_batch_br, targets_br, ids),
                        (conf_id, br_id)
                    )

                # --- End Lagrange gradient computation ---

                (
                    train_state_conf, train_state_br,
                    traj_batch_conf, traj_batch_br,
                    advantages_conf, advantages_br,
                    targets_conf, targets_br,
                    init_conf_h_for_update, init_br_h_for_update,
                    rng, lms_vertical, lms_horizontal
                ) = update_state
                rng, perm_rng_conf, perm_rng_br = jax.random.split(rng, 3)

                minibatches_conf = _create_minibatches(
                    traj_batch_conf, advantages_conf, targets_conf, init_conf_h_for_update,
                    config["NUM_CONF_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_conf
                )
                minibatches_br = _create_minibatches(
                    traj_batch_br, advantages_br, targets_br, init_br_h_for_update,
                    config["NUM_BR_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_br
                )

                num_minibatches = minibatches_br[1].obs.shape[0]
                repeated_lms_vertical = lms_vertical[jnp.newaxis, ...].repeat(num_minibatches, axis=0)
                repeated_lms_horizontal = lms_horizontal[jnp.newaxis, ...].repeat(num_minibatches, axis=0)

                (train_state_conf, train_state_br), all_losses = jax.lax.scan(
                    _update_minbatch, (train_state_conf, train_state_br),
                    (minibatches_conf, minibatches_br, repeated_lms_vertical, repeated_lms_horizontal)
                )

                # Update Lagrange multipliers
                all_conf_ids, all_br_ids = _get_all_ids(config["PARTNER_POP_SIZE"])
                all_lagrange_grads = jax.vmap(_compute_indiv_lagrange_grads)(all_conf_ids, all_br_ids)
                averaged_grad_vertical = jnp.sum(all_lagrange_grads[0], axis=0)
                averaged_grad_horizontal = jnp.sum(all_lagrange_grads[1], axis=0)

                lms_vertical_new = jnp.maximum(
                    lms_vertical - config["LAGRANGE_LR"] * averaged_grad_vertical,
                    0.5 * jnp.eye(config["PARTNER_POP_SIZE"])
                )
                lms_vertical_new = jnp.fill_diagonal(
                    lms_vertical_new, 0.5 * jnp.ones((config["PARTNER_POP_SIZE"]), dtype=jnp.float32),
                    inplace=False
                )

                lms_horizontal_new = jnp.maximum(
                    lms_horizontal - config["LAGRANGE_LR"] * averaged_grad_horizontal,
                    0.5 * jnp.eye(config["PARTNER_POP_SIZE"]),
                )
                lms_horizontal_new = jnp.fill_diagonal(
                    lms_horizontal_new, 0.5 * jnp.ones((config["PARTNER_POP_SIZE"]), dtype=jnp.float32),
                    inplace=False
                )

                update_state = (
                    train_state_conf, train_state_br,
                    traj_batch_conf, traj_batch_br,
                    advantages_conf, advantages_br,
                    targets_conf, targets_br,
                    init_conf_h_for_update, init_br_h_for_update,
                    rng, lms_vertical_new, lms_horizontal_new
                )
                return update_state, all_losses

            def _update_step(update_runner_state, unused):
                (
                    all_train_state_conf, all_train_state_br,
                    last_env_state, last_obs, last_done, last_conf_h, last_br_h,
                    rng, update_steps, lms_vertical, lms_horizontal
                ) = update_runner_state

                initial_conf_hstate_for_update = last_conf_h
                initial_br_hstate_for_update = last_br_h

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
                    update_rng, lms_vertical, lms_horizontal
                )

                update_state, all_losses = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
                )
                all_train_state_conf, all_train_state_br = update_state[:2]
                lms_vertical, lms_horizontal = update_state[-2:]
                (_, (value_loss_conf, pg_loss_conf, entropy_conf)), (_, (value_loss_br, pg_loss_br, entropy_br)) = all_losses

                metric = traj_batch_conf.info
                metric["lms_vertical"] = lms_vertical
                metric["lms_horizontal"] = lms_horizontal
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
                    rng, update_steps + 1,
                    lms_vertical, lms_horizontal
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

                (
                    train_state_conf, train_state_br,
                    last_env_state, last_obs, last_done, last_conf_h, last_br_h,
                    rng, update_steps, lms_vertical, lms_horizontal
                ) = new_runner_state

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
                    last_env_state, last_obs, last_done, last_conf_h, last_br_h,
                    rng, update_steps, lms_vertical, lms_horizontal
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

            # Initialize Lagrange multipliers
            lagrange_multipliers_vertical = 0.5 * jnp.eye(config["PARTNER_POP_SIZE"])
            lagrange_multipliers_horizontal = 0.5 * jnp.eye(config["PARTNER_POP_SIZE"])

            update_runner_state = (
                all_conf_optims, all_br_optims,
                init_env_state, init_obs, init_done, init_conf_h, init_br_h,
                rng, update_steps,
                lagrange_multipliers_vertical, lagrange_multipliers_horizontal
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

    train_fn = make_lbrdiv_agents(config)
    out = train_fn(train_rng)
    return out


def get_lbrdiv_population(config: Dict[str, Any], out: Dict[str, Any], env):
    """Extract partner population from LBRDiv training output."""
    lbrdiv_pop_size = config["PARTNER_POP_SIZE"]
    partner_params = out['final_params_conf']

    obs_shape = env.observation_space(env.agents[1]).shape
    if len(obs_shape) == 1:
        obs_dim = obs_shape[0]
    else:
        obs_dim = int(np.prod(obs_shape))

    actor_type = config.get("ACTOR_TYPE", "mlp")
    activation = config.get("ACTIVATION", "relu")

    if actor_type == "cnn_rnn":
        partner_policy = CNNRNNActorCriticWithConditionalCriticPolicy(
            action_dim=env.action_space(env.agents[1]).n,
            obs_shape=obs_shape,
            pop_size=lbrdiv_pop_size,
            activation=activation,
            fc_dim_size=config.get("FC_DIM_SIZE", 128),
            gru_hidden_dim=config.get("GRU_HIDDEN_DIM", 128),
        )
    else:
        partner_policy = ActorWithConditionalCriticPolicy(
            action_dim=env.action_space(env.agents[1]).n,
            obs_dim=obs_dim,
            pop_size=lbrdiv_pop_size,
            activation=activation,
        )

    partner_population = AgentPopulation(
        pop_size=lbrdiv_pop_size,
        policy_cls=partner_policy
    )

    return partner_params, partner_population


def run_lbrdiv_training(config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """Run LBRDiv training."""
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)

    obs_shape = env.observation_space(env.agents[0]).shape
    if len(obs_shape) == 1:
        obs_dim = obs_shape[0]
    else:
        obs_dim = int(np.prod(obs_shape))

    if verbose:
        log.info(f"Environment: {config['ENV_NAME']}")
        log.info(f"Layout: {config['ENV_KWARGS']['layout']}")
        log.info(f"Observation space: {obs_shape}")
        log.info(f"Action space: {env.action_space(env.agents[0]).n}")
        log.info(f"Actor type: {config.get('ACTOR_TYPE', 'mlp')}")
        log.info(f"Partner population size: {config['PARTNER_POP_SIZE']}")
        log.info(f"Tolerance factor: {config['TOLERANCE_FACTOR']}")
        log.info(f"Lagrange LR: {config['LAGRANGE_LR']}")
        log.info(f"Total timesteps: {config['TOTAL_TIMESTEPS']}")
        log.info(f"Learning rate: {config['LR']}")
        log.info(f"NUM_ENVS: {config['NUM_ENVS']}")
        if config.get('REW_SHAPING_HORIZON', 0) > 0:
            log.info(f"Reward shaping horizon: {config['REW_SHAPING_HORIZON']}")

    activation = config.get("ACTIVATION", "relu")
    actor_type = config.get("ACTOR_TYPE", "mlp")

    if actor_type == "cnn_rnn":
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

    rng = jax.random.PRNGKey(config["TRAIN_SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    start_time = time.time()

    with jax.disable_jit(False):
        vmapped_train_fn = jax.jit(
            jax.vmap(
                partial(
                    train_lbrdiv_partners,
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

    partner_params, partner_population = get_lbrdiv_population(config, out, env)

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
        "config": config,
    }

    return results


def print_training_summary(results: Dict[str, Any]):
    """Print a summary of training results."""
    config = results["config"]
    print("\n" + "=" * 60)
    print("LBRDiv Training Summary")
    print("=" * 60)
    print(f"  Layout: {config['ENV_KWARGS']['layout']}")
    print(f"  Population size: {config['PARTNER_POP_SIZE']}")
    print(f"  Tolerance factor: {config['TOLERANCE_FACTOR']}")
    print(f"  Lagrange LR: {config['LAGRANGE_LR']}")
    print(f"  Total timesteps: {config['TOTAL_TIMESTEPS']:.0f}")
    print(f"  Training time: {results['training_time']:.2f}s")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Train LBRDiv teammate policies on Overcooked V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--layout", type=str, default="cramped_room",
        help="Overcooked V2 layout name",
    )
    parser.add_argument(
        "--total_timesteps", type=float, default=3e7,
        help="Total training timesteps (default: 3e7)",
    )
    parser.add_argument(
        "--partner_pop_size", type=int, default=10,
        help="Number of partner policies in the population (default: 10)",
    )
    parser.add_argument(
        "--tolerance_factor", type=float, default=0.1,
        help="Tolerance factor for Lagrangian diversity constraints (default: 0.1)",
    )
    parser.add_argument(
        "--lagrange_lr", type=float, default=0.01,
        help="Learning rate for Lagrange multiplier updates (default: 0.01)",
    )
    parser.add_argument(
        "--num_checkpoints", type=int, default=5,
        help="Number of checkpoints to save (default: 5)",
    )
    parser.add_argument(
        "--num_envs", type=int, default=256,
        help="Number of parallel environments (default: 256)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.00025,
        help="Learning rate (default: 0.00025)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save results",
    )
    parser.add_argument(
        "--gpu", type=str, default=None,
        help="GPU device ID (use -1 for CPU)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run with reduced settings for debugging",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress most log output",
    )
    parser.add_argument(
        "--list_layouts", action="store_true",
        help="List available layouts and exit",
    )
    parser.add_argument(
        "--actor_type", type=str, default="cnn_rnn",
        choices=["mlp", "cnn_rnn"],
        help="Policy architecture (default: cnn_rnn)",
    )
    parser.add_argument(
        "--csv_log", action="store_true",
        help="Enable CSV metric logging",
    )
    parser.add_argument(
        "--max_steps", type=int, default=400,
        help="Maximum steps per episode (default: 400)",
    )

    args = parser.parse_args()

    if args.list_layouts:
        print("\nAvailable Overcooked V2 Layouts:")
        print("-" * 40)
        for layout_name in sorted(overcooked_v2_layouts.keys()):
            print(f"  {layout_name}")
        print()
        return

    config = get_default_config(args.layout)
    config["TOTAL_TIMESTEPS"] = args.total_timesteps
    config["PARTNER_POP_SIZE"] = args.partner_pop_size
    config["TOLERANCE_FACTOR"] = args.tolerance_factor
    config["LAGRANGE_LR"] = args.lagrange_lr
    config["NUM_CHECKPOINTS"] = args.num_checkpoints
    config["NUM_ENVS"] = args.num_envs
    config["TRAIN_SEED"] = args.seed
    config["LR"] = args.lr
    config["OUTPUT_DIR"] = args.output_dir
    config["ACTOR_TYPE"] = args.actor_type
    config["CSV_LOG"] = args.csv_log
    config["ENV_KWARGS"]["max_steps"] = args.max_steps

    if args.debug:
        config["TOTAL_TIMESTEPS"] = 1e5
        config["PARTNER_POP_SIZE"] = 2
        config["NUM_CHECKPOINTS"] = 2
        config["NUM_ENVS"] = 4
        config["NUM_MINIBATCHES"] = 2
        config["UPDATE_EPOCHS"] = 2
        log.info("Running in DEBUG mode with reduced settings")

    if not args.quiet:
        log.info(f"Starting LBRDiv training on layout: {args.layout}")
        if args.gpu is not None:
            if args.gpu == "-1":
                log.info("Using CPU")
            else:
                log.info(f"Using GPU: {args.gpu}")
        else:
            log.info(f"Using all available devices: {jax.devices()}")

    try:
        results = run_lbrdiv_training(config, verbose=not args.quiet)

        if not args.quiet:
            print_training_summary(results)

        if args.output_dir:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            save_data = {
                "checkpoints_conf": results["checkpoints_conf"],
                "checkpoints_br": results["checkpoints_br"],
                "metrics": results["metrics"],
            }
            checkpoint_returns = compute_checkpoint_returns(results["metrics"], results["config"])
            save_path = save_separated_checkpoints_multi(
                save_data, results["config"], str(output_path), savename="lbrdiv_train_run",
                checkpoint_returns=checkpoint_returns,
            )
            log.info(f"Saved training results to: {save_path}")

            if results["config"].get("CSV_LOG", False):
                write_training_metrics_to_csv(
                    metrics=results["metrics"],
                    config=results["config"],
                    output_dir=args.output_dir,
                    training_time=results.get("training_time", 0.0),
                )

    except Exception as e:
        log.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
