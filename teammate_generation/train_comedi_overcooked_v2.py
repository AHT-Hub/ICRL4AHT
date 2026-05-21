#!/usr/bin/env python3
"""Training script for CoMeDi (Cooperative Meta-Diversity) on Overcooked V2.

This script provides a standalone entry point for training CoMeDi teammate policies
on Overcooked V2 environments, without requiring Hydra configuration.

CoMeDi trains a population of diverse teammates sequentially. The first policy is
trained via standard IPPO self-play. Each subsequent policy is trained using four
types of rollouts (XP, SP, MP, SMP) with a combined loss that encourages diversity
from existing population members while maintaining cooperative ability.

Example Usage:
    # Train on cramped_room layout with default settings
    python -m teammate_generation.train_comedi_overcooked_v2 --layout cramped_room

    # Train with custom parameters
    python -m teammate_generation.train_comedi_overcooked_v2 \\
        --layout coord_ring \\
        --total_timesteps 3e7 \\
        --partner_pop_size 5 \\
        --comedi_alpha 1.0 \\
        --comedi_beta 0.5 \\
        --seed 0

    # Quick test run
    python -m teammate_generation.train_comedi_overcooked_v2 \\
        --layout cramped_room \\
        --total_timesteps 5e5 \\
        --partner_pop_size 2 \\
        --num_checkpoints 2 \\
        --debug

References:
    Sarkar et al., "Diverse Conventions for Human-AI Collaboration",
    NeurIPS 2023. https://openreview.net/forum?id=MljeRycu9s
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
from agents.population_interface import AgentPopulation
from envs import make_env
from envs.log_wrapper import LogWrapper, LogEnvState
from envs.overcooked_v2 import overcooked_v2_layouts
from common.stats_utils import get_metric_names
from common.run_episodes import run_episodes
from common.save_load_utils import save_train_run, save_separated_checkpoints_multi
from marl.ippo import make_train as make_ppo_train
from marl.ppo_utils import Transition, unbatchify, _create_minibatches

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
    """Compute mean returns at each checkpoint interval from training metrics.

    CoMeDi metrics have shape (num_seeds, pop_size-1, num_updates, rollout_len, num_envs).
    We average across the pop dimension before computing per-checkpoint returns.
    """
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

    def _collapse_pop_dim(arr):
        if arr.ndim == 5:
            return np.mean(arr, axis=1)
        return arr

    if "returned_episode" in metrics:
        ep_done = _collapse_pop_dim(np.array(metrics["returned_episode"]))
    else:
        ep_done = None

    for ckpt_idx, update_idx in enumerate(checkpoint_update_indices):
        if ckpt_idx == 0:
            start_idx = 0
        else:
            start_idx = checkpoint_update_indices[ckpt_idx - 1] + 1
        end_idx = update_idx + 1

        if "returned_episode_returns" in metrics:
            all_base = _collapse_pop_dim(np.array(metrics["returned_episode_returns"]))
            window_base_returns = all_base[:, start_idx:end_idx, :, :]
            if ep_done is not None:
                window_ep_done = ep_done[:, start_idx:end_idx, :, :] > 0
                for seed_idx in range(num_seeds):
                    mask = window_ep_done[seed_idx]
                    if np.any(mask):
                        base_returns[seed_idx, ckpt_idx] = np.mean(window_base_returns[seed_idx][mask])

        if "returned_episode_shaped_returns" in metrics:
            all_shaped = _collapse_pop_dim(np.array(metrics["returned_episode_shaped_returns"]))
            window_shaped_returns = all_shaped[:, start_idx:end_idx, :, :]
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
    timesteps_per_update = 4 * rollout_length * num_envs * num_agents

    csv_fieldnames = [
        "update_step", "total_timesteps",
        "mean_conf_return", "std_conf_return",
        "mean_conf_value_loss_xp", "mean_conf_value_loss_sp", "mean_conf_value_loss_mp",
        "mean_conf_pg_loss_xp", "mean_conf_pg_loss_sp", "mean_conf_pg_loss_mp",
        "mean_conf_entropy_xp", "mean_conf_entropy_sp", "mean_conf_entropy_mp",
        "mean_reward_ego", "mean_reward_br_sp", "mean_reward_br_mp",
        "elapsed_time",
    ]

    csv_logger = CSVMetricsLogger(str(csv_path), csv_fieldnames)

    if "average_rewards_ego" in metrics:
        # metrics shape: (num_seeds, pop_size-1, num_updates, ...)
        num_pop_iterations = metrics["average_rewards_ego"].shape[1]
        num_updates = metrics["average_rewards_ego"].shape[2]
    elif "update_steps" in metrics:
        num_pop_iterations = metrics["update_steps"].shape[1]
        num_updates = metrics["update_steps"].shape[2]
    else:
        log.warning("Cannot determine number of updates from metrics. Skipping CSV logging.")
        return

    for pop_iter in range(num_pop_iterations):
        for update_idx in range(num_updates):
            current_timesteps = (update_idx + 1) * timesteps_per_update

            csv_row = {
                "update_step": update_idx + 1,
                "total_timesteps": current_timesteps,
                "elapsed_time": training_time * (update_idx + 1) / num_updates,
            }

            for key, csv_key in [
                ("value_loss_conf_xp", "mean_conf_value_loss_xp"),
                ("value_loss_conf_sp", "mean_conf_value_loss_sp"),
                ("value_loss_conf_mp", "mean_conf_value_loss_mp"),
                ("pg_loss_conf_xp", "mean_conf_pg_loss_xp"),
                ("pg_loss_conf_sp", "mean_conf_pg_loss_sp"),
                ("pg_loss_conf_mp", "mean_conf_pg_loss_mp"),
                ("entropy_conf_xp", "mean_conf_entropy_xp"),
                ("entropy_conf_sp", "mean_conf_entropy_sp"),
                ("entropy_conf_mp", "mean_conf_entropy_mp"),
                ("average_rewards_ego", "mean_reward_ego"),
                ("average_rewards_br_sp", "mean_reward_br_sp"),
                ("average_rewards_br_mp2", "mean_reward_br_mp"),
            ]:
                if key in metrics:
                    csv_row[csv_key] = float(np.mean(metrics[key][:, pop_iter, update_idx, ...]))
                else:
                    csv_row[csv_key] = 0.0

            csv_logger.log(csv_row)

    log.info(f"Wrote training metrics to: {csv_path}")


class ResetTransition(NamedTuple):
    """Stores extra information for resetting agents to a point in some trajectory."""
    env_state: LogEnvState
    conf_obs: jnp.ndarray
    partner_obs: jnp.ndarray
    conf_done: jnp.ndarray
    partner_done: jnp.ndarray


def get_default_config(layout: str) -> Dict[str, Any]:
    """Get default CoMeDi training configuration for a given layout."""
    return {
        "ENV_NAME": "overcooked-v2",
        "ENV_KWARGS": {
            "layout": layout,
            "flatten_obs": True,
            "max_steps": 400,
        },
        "ROLLOUT_LENGTH": 256,

        "ALG": "comedi",
        "ACTOR_TYPE": "pseudo_actor_with_conditional_critic",
        "TOTAL_TIMESTEPS": 3e7,
        "NUM_CHECKPOINTS": 5,
        "PARTNER_POP_SIZE": 5,

        # CoMeDi-specific parameters
        "COMEDI_ALPHA": 1.0,
        "COMEDI_BETA": 0.5,
        "NUM_ARGMAX_ROLLOUT_EPS": 20,

        # PPO hyperparameters
        "NUM_ENVS": 64,
        "LR": 0.00025,
        "ANNEAL_LR": True,
        "LR_WARMUP": 0.05,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.25,
        "ACTIVATION": "tanh",

        "REW_SHAPING_HORIZON": 1.5e7,

        "NUM_SEEDS": 1,
        "TRAIN_SEED": 0,
        "NUM_EVAL_EPISODES": 20,

        "EVAL_MAX_STEPS": 400,
    }


def gather_params(partner_params_pytree, idx_vec):
    """Gather parameters for specific indices from a parameter pytree."""
    def gather_leaf(leaf):
        def slice_one(idx):
            return leaf[idx]
        return jax.vmap(slice_one)(idx_vec)

    return jax.tree.map(gather_leaf, partner_params_pytree)


def train_comedi_partners(train_rng, env, config: Dict[str, Any]):
    """Train CoMeDi confederate policies.

    CoMeDi trains a population of diverse teammates sequentially. The first
    policy is trained via IPPO self-play. Each subsequent policy is trained
    using four types of rollouts (XP, SP, MP, SMP) with a combined loss.
    """
    num_agents = env.num_agents
    assert num_agents == 2, "CoMeDi requires exactly 2 agents"

    config["NUM_GAME_AGENTS"] = num_agents
    config["NUM_ACTORS"] = num_agents * config["NUM_ENVS"]
    config["NUM_CONTROLLED_ACTORS"] = config["NUM_ACTORS"]
    config["POP_SIZE"] = config["PARTNER_POP_SIZE"]

    total_timesteps_per_iter = int(config["TOTAL_TIMESTEPS"]) // config["PARTNER_POP_SIZE"]
    config["TOTAL_TIMESTEPS_PER_ITERATION"] = total_timesteps_per_iter

    config["NUM_UPDATES"] = total_timesteps_per_iter // (
        4 * num_agents * config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
    )

    def make_comedi_agents(config):
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

        def train_init_ippo_partners(config, partner_rng, env):
            ippo_config = config.copy()
            ippo_config["TOTAL_TIMESTEPS"] = config["TOTAL_TIMESTEPS_PER_ITERATION"]
            ippo_config["ACTOR_TYPE"] = "pseudo_actor_with_conditional_critic"
            ippo_config["POP_SIZE"] = config["PARTNER_POP_SIZE"]
            out = make_ppo_train(ippo_config, env)(partner_rng)
            return out

        def train(rng):
            rng, init_ppo_rng, init_conf_rng = jax.random.split(rng, 3)

            init_ppo_partner = train_init_ippo_partners(config, init_ppo_rng, env)

            obs_dim = env.observation_space(env.agents[0]).shape[0]
            policy = ActorWithConditionalCriticPolicy(
                action_dim=env.action_space(env.agents[0]).n,
                obs_dim=obs_dim,
                pop_size=config["POP_SIZE"],
                activation=config.get("ACTIVATION", "tanh"),
            )

            dummy_init_params = policy.init_params(init_conf_rng)

            # Inline population buffer: pre-allocate params buffer
            population_buffer = jax.tree.map(
                lambda x: jnp.zeros((config["PARTNER_POP_SIZE"],) + x.shape, x.dtype),
                dummy_init_params
            )
            # Add initial IPPO params at index 0
            population_buffer = jax.tree.map(
                lambda buf, p: buf.at[0].set(p),
                population_buffer, init_ppo_partner["final_params"]
            )

            def add_conf_policy(pop_buffer, func_input):
                num_existing_agents, rng = func_input
                rng, init_conf_rng = jax.random.split(rng)

                init_params = policy.init_params(init_conf_rng)

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

                train_state = TrainState.create(
                    apply_fn=policy.network.apply,
                    params=init_params,
                    tx=tx,
                )

                rng, reset_rng_sp, reset_rng_xp, reset_rng_mp, reset_rng_mp2 = jax.random.split(rng, 5)

                reset_rngs_sp = jax.random.split(reset_rng_sp, config["NUM_ENVS"])
                reset_rngs_xp = jax.random.split(reset_rng_xp, config["NUM_ENVS"])
                reset_rngs_mp = jax.random.split(reset_rng_mp, config["NUM_ENVS"])
                reset_rngs_mp2 = jax.random.split(reset_rng_mp2, config["NUM_ENVS"])

                obsv_xp, env_state_xp = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_xp)
                obsv_sp, env_state_sp = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_sp)
                obsv_mp, env_state_mp = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_mp)
                obsv_mp2, env_state_mp2 = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_mp2)

                ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1)
                num_ckpts = config["NUM_CHECKPOINTS"]

                def init_ckpt_array(params_pytree):
                    return jax.tree.map(
                        lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                        params_pytree
                    )

                rng, eval_rng = jax.random.split(rng, 2)

                def per_id_run_episode_fixed_rng(agent0_param, agent1_id):
                    agent1_param = gather_params(
                        pop_buffer,
                        agent1_id * jnp.ones((1,), dtype=jnp.int32)
                    )
                    agent1_param = jax.tree.map(lambda y: jnp.squeeze(y, 0), agent1_param)
                    all_outs = run_episodes(
                        rng=eval_rng, env=env,
                        agent_0_param=agent0_param, agent_0_policy=policy,
                        agent_1_param=agent1_param, agent_1_policy=policy,
                        max_episode_steps=config["EVAL_MAX_STEPS"],
                        num_eps=config["NUM_ARGMAX_ROLLOUT_EPS"]
                    )
                    return all_outs

                def _env_step_conf_ego(runner_state, unused):
                    """XP rollout: agent_0 = confederate (training), agent_1 = ego (from buffer)."""
                    train_state, xp_param, xp_id, env_state, last_obs, last_dones, rng = runner_state
                    rng, act_rng, partner_rng, step_rng = jax.random.split(rng, 4)

                    obs_0 = last_obs["agent_0"]
                    obs_1 = last_obs["agent_1"]

                    avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                    avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                    avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                    xp_one_hot_id = jnp.eye(config["POP_SIZE"])[xp_id]
                    xp_one_hot_id = jnp.expand_dims(jnp.expand_dims(xp_one_hot_id, 0), 0)
                    aux_obs = jnp.repeat(xp_one_hot_id, config["NUM_ENVS"], axis=1)

                    act_0, val_0, pi_0, _ = policy.get_action_value_policy(
                        params=train_state.params,
                        obs=obs_0.reshape(1, config["NUM_ENVS"], -1),
                        done=last_dones["agent_0"].reshape(1, config["NUM_ENVS"]),
                        avail_actions=jax.lax.stop_gradient(avail_actions_0),
                        hstate=None,
                        rng=act_rng,
                        aux_obs=aux_obs
                    )
                    logp_0 = pi_0.log_prob(act_0)
                    act_0 = act_0.squeeze()
                    logp_0 = logp_0.squeeze()
                    val_0 = val_0.squeeze()

                    act_1, _, _, _ = policy.get_action_value_policy(
                        params=xp_param,
                        obs=obs_1.reshape(1, config["NUM_ENVS"], -1),
                        done=last_dones["agent_1"].reshape(1, config["NUM_ENVS"]),
                        avail_actions=jax.lax.stop_gradient(avail_actions_1),
                        hstate=None,
                        rng=partner_rng,
                        aux_obs=aux_obs
                    )
                    act_1 = act_1.squeeze()

                    combined_actions = jnp.concatenate([act_0, act_1], axis=0)
                    env_act = unbatchify(combined_actions, env.agents, config["NUM_ENVS"], num_agents)
                    env_act = {k: v.flatten() for k, v in env_act.items()}

                    step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                    obs_next, env_state_next, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                        step_rngs, env_state, env_act
                    )
                    info_0 = jax.tree.map(lambda x: x[:, 0], info)

                    base_reward_1 = reward["agent_1"]
                    shaped_reward_1 = info_0.get("shaped_reward", jnp.zeros_like(base_reward_1))
                    if rew_shaping_anneal is not None:
                        current_ts = train_state.step * config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
                        anneal = rew_shaping_anneal(current_ts)
                    else:
                        anneal = 0.0
                    total_reward_1 = base_reward_1 + anneal * shaped_reward_1

                    transition = Transition(
                        done=done["agent_0"],
                        action=act_0,
                        value=val_0,
                        reward=total_reward_1,
                        log_prob=logp_0,
                        obs=obs_0,
                        info=info_0,
                        avail_actions=avail_actions_0
                    )
                    new_runner_state = (train_state, xp_param, xp_id, env_state_next, obs_next, done, rng)
                    return new_runner_state, transition

                def _env_step_conf_br(runner_state, unused):
                    """SP/SMP rollout: agent_0 = confederate, agent_1 = best response (both self-play)."""
                    train_state, env_state, last_obs, last_dones, rng, current_trained_pop_id, reset_traj_batch = runner_state
                    rng, conf_rng, br_rng, step_rng = jax.random.split(rng, 4)

                    def gather_sampled(data_pytree, flat_indices, first_nonbatch_dim: int):
                        batch_size = config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
                        flat_data = jax.tree.map(lambda x: x.reshape(batch_size, *x.shape[first_nonbatch_dim:]), data_pytree)
                        sampled_data = jax.tree.map(lambda x: x[flat_indices], flat_data)
                        return sampled_data

                    if reset_traj_batch is not None:
                        rng, sample_rng = jax.random.split(rng)
                        needs_resample = last_dones["__all__"]

                        total_reset_states = config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
                        sampled_indices = jax.random.randint(
                            sample_rng, shape=(config["NUM_ENVS"],), minval=0, maxval=total_reset_states
                        )

                        sampled_env_state = gather_sampled(reset_traj_batch.env_state, sampled_indices, first_nonbatch_dim=2)
                        sampled_conf_obs = gather_sampled(reset_traj_batch.conf_obs, sampled_indices, first_nonbatch_dim=2)
                        sampled_br_obs = gather_sampled(reset_traj_batch.partner_obs, sampled_indices, first_nonbatch_dim=2)
                        sampled_conf_done = gather_sampled(reset_traj_batch.conf_done, sampled_indices, first_nonbatch_dim=2)
                        sampled_br_done = gather_sampled(reset_traj_batch.partner_done, sampled_indices, first_nonbatch_dim=2)

                        env_state = jax.tree.map(
                            lambda sampled, original: jnp.where(
                                needs_resample.reshape((-1,) + (1,) * (original.ndim - 1)),
                                sampled, original
                            ),
                            sampled_env_state,
                            env_state
                        )
                        obs_0 = jnp.where(needs_resample[:, jnp.newaxis], sampled_conf_obs, last_obs["agent_0"])
                        obs_1 = jnp.where(needs_resample[:, jnp.newaxis], sampled_br_obs, last_obs["agent_1"])
                        dones_0 = jnp.where(needs_resample, sampled_conf_done, last_dones["agent_0"])
                        dones_1 = jnp.where(needs_resample, sampled_br_done, last_dones["agent_1"])
                    else:
                        obs_0, obs_1 = last_obs["agent_0"], last_obs["agent_1"]
                        dones_0, dones_1 = last_dones["agent_0"], last_dones["agent_1"]

                    avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                    avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                    avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                    sp_one_hot_id = jnp.eye(config["POP_SIZE"])[current_trained_pop_id]
                    sp_one_hot_id = jnp.expand_dims(jnp.expand_dims(sp_one_hot_id, 0), 0)
                    aux_obs = jnp.repeat(sp_one_hot_id, config["NUM_ENVS"], 1)

                    act_0, val_0, pi_0, _ = policy.get_action_value_policy(
                        params=train_state.params,
                        obs=obs_0.reshape(1, config["NUM_ENVS"], -1),
                        done=dones_0.reshape(1, config["NUM_ENVS"]),
                        avail_actions=jax.lax.stop_gradient(avail_actions_0),
                        hstate=None,
                        rng=conf_rng,
                        aux_obs=aux_obs
                    )
                    logp_0 = pi_0.log_prob(act_0)
                    act_0 = act_0.squeeze()
                    logp_0 = logp_0.squeeze()
                    val_0 = val_0.squeeze()

                    act_1, val_1, pi_1, _ = policy.get_action_value_policy(
                        params=train_state.params,
                        obs=obs_1.reshape(1, config["NUM_ENVS"], -1),
                        done=dones_1.reshape(1, config["NUM_ENVS"]),
                        avail_actions=jax.lax.stop_gradient(avail_actions_1),
                        hstate=None,
                        rng=br_rng,
                        aux_obs=aux_obs
                    )
                    logp_1 = pi_1.log_prob(act_1)
                    act_1 = act_1.squeeze()
                    logp_1 = logp_1.squeeze()
                    val_1 = val_1.squeeze()

                    combined_actions = jnp.concatenate([act_0, act_1], axis=0)
                    env_act = unbatchify(combined_actions, env.agents, config["NUM_ENVS"], num_agents)
                    env_act = {k: v.flatten() for k, v in env_act.items()}

                    step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                    obs_next, env_state_next, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                        step_rngs, env_state, env_act
                    )
                    info_0 = jax.tree.map(lambda x: x[:, 0], info)
                    info_1 = jax.tree.map(lambda x: x[:, 1], info)

                    base_reward_0 = reward["agent_0"]
                    base_reward_1 = reward["agent_1"]
                    shaped_reward_0 = info_0.get("shaped_reward", jnp.zeros_like(base_reward_0))
                    shaped_reward_1 = info_1.get("shaped_reward", jnp.zeros_like(base_reward_1))
                    if rew_shaping_anneal is not None:
                        current_ts = train_state.step * config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
                        anneal = rew_shaping_anneal(current_ts)
                    else:
                        anneal = 0.0
                    total_reward_0 = base_reward_0 + anneal * shaped_reward_0
                    total_reward_1 = base_reward_1 + anneal * shaped_reward_1

                    transition_0 = Transition(
                        done=done["agent_0"],
                        action=act_0,
                        value=val_0,
                        reward=total_reward_0,
                        log_prob=logp_0,
                        obs=obs_0,
                        info=info_0,
                        avail_actions=avail_actions_0
                    )
                    transition_1 = Transition(
                        done=done["agent_1"],
                        action=act_1,
                        value=val_1,
                        reward=total_reward_1,
                        log_prob=logp_1,
                        obs=obs_1,
                        info=info_1,
                        avail_actions=avail_actions_1
                    )
                    new_runner_state = (train_state, env_state_next, obs_next, done, rng, current_trained_pop_id, reset_traj_batch)
                    return new_runner_state, (transition_0, transition_1)

                def _env_step_mixed(runner_state, unused):
                    """MP rollout: agent_0 = confederate, agent_1 = ego OR best response (random)."""
                    train_state_conf, ego_param, env_state, last_obs, last_dones, rng, current_trained_pop_id = runner_state
                    rng, act_rng, ego_act_rng, br_act_rng, partner_choice_rng, step_rng = jax.random.split(rng, 6)

                    obs_0 = last_obs["agent_0"]
                    obs_1 = last_obs["agent_1"]

                    avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                    avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                    avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                    xp_one_hot_id = jnp.eye(config["POP_SIZE"])[current_trained_pop_id]
                    xp_one_hot_id = jnp.expand_dims(jnp.expand_dims(xp_one_hot_id, 0), 0)
                    aux_obs = jnp.repeat(xp_one_hot_id, config["NUM_ENVS"], axis=1)

                    act_0, val_0, pi_0, _ = policy.get_action_value_policy(
                        params=train_state_conf.params,
                        obs=obs_0.reshape(1, config["NUM_ENVS"], -1),
                        done=last_dones["agent_0"].reshape(1, config["NUM_ENVS"]),
                        avail_actions=jax.lax.stop_gradient(avail_actions_0),
                        hstate=None,
                        rng=act_rng,
                        aux_obs=aux_obs
                    )
                    act_0 = act_0.squeeze()

                    act_ego, _, _, _ = policy.get_action_value_policy(
                        params=ego_param,
                        obs=obs_1.reshape(1, config["NUM_ENVS"], -1),
                        done=last_dones["agent_1"].reshape(1, config["NUM_ENVS"]),
                        avail_actions=jax.lax.stop_gradient(avail_actions_1),
                        hstate=None,
                        rng=ego_act_rng,
                        aux_obs=aux_obs
                    )

                    act_br, _, _, _ = policy.get_action_value_policy(
                        params=train_state.params,
                        obs=obs_1.reshape(1, config["NUM_ENVS"], -1),
                        done=last_dones["agent_1"].reshape(1, config["NUM_ENVS"]),
                        avail_actions=jax.lax.stop_gradient(avail_actions_1),
                        hstate=None,
                        rng=br_act_rng,
                        aux_obs=aux_obs
                    )

                    act_ego = act_ego.squeeze()
                    act_br = act_br.squeeze()
                    partner_choice = jax.random.randint(partner_choice_rng, shape=(config["NUM_ENVS"],), minval=0, maxval=2)
                    act_1 = jnp.where(partner_choice == 0, act_ego, act_br)

                    combined_actions = jnp.concatenate([act_0, act_1], axis=0)
                    env_act = unbatchify(combined_actions, env.agents, config["NUM_ENVS"], num_agents)
                    env_act = {k: v.flatten() for k, v in env_act.items()}

                    step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                    obs_next, env_state_next, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                        step_rngs, env_state, env_act
                    )

                    reset_transition = ResetTransition(
                        env_state=env_state,
                        conf_obs=obs_0,
                        partner_obs=obs_1,
                        conf_done=last_dones["agent_0"],
                        partner_done=last_dones["agent_1"],
                    )
                    new_runner_state = (train_state_conf, ego_param, env_state_next, obs_next, done, rng, current_trained_pop_id)
                    return new_runner_state, reset_transition

                def _update_step(update_with_ckpt_runner_state, unused):
                    update_runner_state, checkpoint_array, ckpt_idx = update_with_ckpt_runner_state
                    (
                        train_state, pop_buffer,
                        env_state_sp, obsv_sp,
                        env_state_xp, obsv_xp,
                        env_state_mp, obsv_mp,
                        env_state_mp2, obsv_mp2,
                        last_dones_xp,
                        last_dones_sp,
                        last_dones_mp,
                        last_dones_mp2,
                        rng, update_steps,
                        num_prev_trained_conf
                    ) = update_runner_state

                    # Step 1: Argmax ego partner selection
                    valid_sampling_indices = jnp.arange(config["POP_SIZE"])
                    run_all_rollouts = jax.vmap(per_id_run_episode_fixed_rng, in_axes=(None, 0))(
                        train_state.params, valid_sampling_indices)

                    all_mean_returns = run_all_rollouts["returned_episode_returns"][:, :, 0].mean(axis=-1)
                    masked_mean_returns = jnp.where(
                        valid_sampling_indices >= num_prev_trained_conf, -jnp.inf, all_mean_returns
                    )
                    max_means_id = masked_mean_returns.argmax()
                    xp_param = jax.tree.map(
                        lambda x: x[max_means_id],
                        pop_buffer
                    )

                    rng, rng_xp, rng_sp, rng_mp, rng_mp2 = jax.random.split(rng, 5)

                    # Step 2: XP rollout (conf vs ego from buffer)
                    runner_state_xp = (train_state, xp_param, max_means_id, env_state_xp, obsv_xp, last_dones_xp, rng_xp)
                    runner_state_xp, traj_batch_xp = jax.lax.scan(
                        _env_step_conf_ego, runner_state_xp, None, config["ROLLOUT_LENGTH"])
                    (train_state, xp_param, max_means_id, env_state_xp, last_obs_xp, last_dones_xp, rng_xp) = runner_state_xp

                    # Step 3: SP rollout (conf vs self)
                    runner_state_sp = (train_state, env_state_sp, obsv_sp, last_dones_sp, rng_sp, num_prev_trained_conf, None)
                    runner_state_sp, (traj_batch_sp_agent0, traj_batch_sp_agent1) = jax.lax.scan(
                        _env_step_conf_br, runner_state_sp, None, config["ROLLOUT_LENGTH"])
                    (train_state, env_state_sp, last_obs_sp, last_dones_sp, rng_sp, num_prev_trained_conf, _) = runner_state_sp

                    # Step 4: MP rollout (conf vs mixed ego/br) -> produces reset states
                    runner_state_mp = (train_state, xp_param, env_state_mp, obsv_mp, last_dones_mp, rng_mp, num_prev_trained_conf)
                    runner_state_mp, traj_batch_mp = jax.lax.scan(
                        _env_step_mixed, runner_state_mp, None, config["ROLLOUT_LENGTH"])
                    (train_state, xp_param, env_state_mp, last_obs_mp, last_dones_mp, rng_mp, num_prev_trained_conf) = runner_state_mp

                    # Step 5: SMP rollout (conf vs self, but reset to MP states)
                    runner_state_smp = (train_state, env_state_mp2, obsv_mp2, last_dones_mp2, rng_mp2, num_prev_trained_conf, traj_batch_mp)
                    runner_state_smp, (traj_batch_smp0, traj_batch_smp1) = jax.lax.scan(
                        _env_step_conf_br, runner_state_smp, None, config["ROLLOUT_LENGTH"])
                    (train_state, env_state_mp2, last_obs_mp2, last_dones_mp2, rng_mp2, num_prev_trained_conf, _) = runner_state_smp

                    # Step 6: Compute advantages and targets for all rollout types
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

                    def _compute_advantages_and_targets(env_state, policy_params,
                                                        last_obs, last_dones, traj_batch, agent_name, value_idx):
                        avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)[agent_name].astype(jnp.float32)

                        one_hot_id = jnp.eye(config["POP_SIZE"])[value_idx]
                        one_hot_id = jnp.expand_dims(jnp.expand_dims(one_hot_id, 0), 0)
                        aux_obs = jnp.repeat(one_hot_id, last_obs[agent_name].shape[0], axis=1)

                        _, vals, _, _ = policy.get_action_value_policy(
                            params=policy_params,
                            obs=last_obs[agent_name].reshape(1, last_obs[agent_name].shape[0], -1),
                            done=last_dones[agent_name].reshape(1, last_obs[agent_name].shape[0]),
                            avail_actions=jax.lax.stop_gradient(avail_actions),
                            hstate=None,
                            rng=jax.random.PRNGKey(0),
                            aux_obs=aux_obs
                        )
                        last_val = vals.squeeze()
                        advantages, targets = _calculate_gae(traj_batch, last_val)
                        return advantages, targets

                    advantages_xp_conf, targets_xp_conf = _compute_advantages_and_targets(
                        env_state_xp, train_state.params,
                        last_obs_xp, last_dones_xp, traj_batch_xp, "agent_0", value_idx=max_means_id)

                    advantages_sp_conf, targets_sp_conf = _compute_advantages_and_targets(
                        env_state_sp, train_state.params,
                        last_obs_sp, last_dones_sp, traj_batch_sp_agent0, "agent_0", value_idx=num_prev_trained_conf)

                    advantages_sp_br, targets_sp_br = _compute_advantages_and_targets(
                        env_state_sp, train_state.params,
                        last_obs_sp, last_dones_sp, traj_batch_sp_agent1, "agent_1", value_idx=num_prev_trained_conf)

                    advantages_mp_conf, targets_mp_conf = _compute_advantages_and_targets(
                        env_state_mp2, train_state.params,
                        last_obs_mp2, last_dones_mp2, traj_batch_smp0, "agent_0", value_idx=num_prev_trained_conf)

                    advantages_mp_br, targets_mp_br = _compute_advantages_and_targets(
                        env_state_mp2, train_state.params,
                        last_obs_mp2, last_dones_mp2, traj_batch_smp1, "agent_1", value_idx=num_prev_trained_conf)

                    # Step 7: PPO update with combined loss
                    def _update_epoch(update_state, unused):
                        def _compute_ppo_value_loss(pred_value, traj_batch, target_v):
                            value_pred_clipped = traj_batch.value + (
                                pred_value - traj_batch.value
                            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                            value_losses = jnp.square(pred_value - target_v)
                            value_losses_clipped = jnp.square(value_pred_clipped - target_v)
                            value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()
                            return value_loss

                        def _compute_ppo_pg_loss(log_prob, traj_batch, gae):
                            ratio = jnp.exp(log_prob - traj_batch.log_prob)
                            gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                            pg_loss_1 = ratio * gae_norm
                            pg_loss_2 = jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"]) * gae_norm
                            pg_loss = -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))
                            return pg_loss

                        def _update_minbatch_conf(train_state_conf, batch_infos):
                            minbatch_xp, minbatch_sp1, minbatch_sp2, minbatch_mp1, minbatch_mp2, xp_id, sp_id = batch_infos
                            # Unpack 5-tuple from _create_minibatches: (init_hstate, init_done, traj_batch, advantages, targets)
                            traj_batch_xp = minbatch_xp[2]
                            advantages_xp = minbatch_xp[3]
                            returns_xp = minbatch_xp[4]

                            traj_batch_sp1 = minbatch_sp1[2]
                            advantages_sp1 = minbatch_sp1[3]
                            returns_sp1 = minbatch_sp1[4]

                            traj_batch_sp2 = minbatch_sp2[2]
                            advantages_sp2 = minbatch_sp2[3]
                            returns_sp2 = minbatch_sp2[4]

                            traj_batch_mp1 = minbatch_mp1[2]
                            advantages_mp1 = minbatch_mp1[3]
                            returns_mp1 = minbatch_mp1[4]

                            traj_batch_mp2 = minbatch_mp2[2]
                            advantages_mp2 = minbatch_mp2[3]
                            returns_mp2 = minbatch_mp2[4]

                            def _loss_fn_conf(params, traj_batch_xp, gae_xp, target_v_xp,
                                            traj_batch_sp, gae_sp, target_v_sp,
                                            traj_batch_sp2, gae_sp2, target_v_sp2,
                                            traj_batch_mp, gae_mp, target_v_mp,
                                            traj_batch_mp2, gae_mp2, target_v_mp2):
                                xp_one_hot_id = jnp.eye(config["POP_SIZE"])[xp_id]
                                xp_one_hot_id = jnp.expand_dims(jnp.expand_dims(xp_one_hot_id, 0), 0)

                                sp_one_hot_id = jnp.eye(config["POP_SIZE"])[sp_id]
                                sp_one_hot_id = jnp.expand_dims(jnp.expand_dims(sp_one_hot_id, 0), 0)

                                aux_obs_xp = jnp.repeat(xp_one_hot_id, traj_batch_xp.obs.shape[1], axis=1)
                                aux_obs_xp = jnp.repeat(aux_obs_xp, traj_batch_xp.obs.shape[0], axis=0)

                                aux_obs_sp = jnp.repeat(xp_one_hot_id, traj_batch_sp.obs.shape[1], axis=1)
                                aux_obs_sp = jnp.repeat(aux_obs_sp, traj_batch_sp.obs.shape[0], axis=0)

                                _, value_xp, pi_xp, _ = policy.get_action_value_policy(
                                    params=params,
                                    obs=traj_batch_xp.obs,
                                    done=traj_batch_xp.done,
                                    avail_actions=traj_batch_xp.avail_actions,
                                    hstate=None,
                                    rng=jax.random.PRNGKey(0),
                                    aux_obs=aux_obs_xp
                                )

                                _, value_sp, pi_sp, _ = policy.get_action_value_policy(
                                    params=params,
                                    obs=traj_batch_sp.obs,
                                    done=traj_batch_sp.done,
                                    avail_actions=traj_batch_sp.avail_actions,
                                    hstate=None,
                                    rng=jax.random.PRNGKey(0),
                                    aux_obs=aux_obs_sp
                                )

                                _, value_sp2, pi_sp2, _ = policy.get_action_value_policy(
                                    params=params,
                                    obs=traj_batch_sp2.obs,
                                    done=traj_batch_sp2.done,
                                    avail_actions=traj_batch_sp2.avail_actions,
                                    hstate=None,
                                    rng=jax.random.PRNGKey(0),
                                    aux_obs=aux_obs_sp
                                )

                                _, value_mp, pi_mp, _ = policy.get_action_value_policy(
                                    params=params,
                                    obs=traj_batch_mp.obs,
                                    done=traj_batch_mp.done,
                                    avail_actions=traj_batch_mp.avail_actions,
                                    hstate=None,
                                    rng=jax.random.PRNGKey(0),
                                    aux_obs=aux_obs_sp
                                )

                                _, value_mp2, pi_mp2, _ = policy.get_action_value_policy(
                                    params=params,
                                    obs=traj_batch_mp2.obs,
                                    done=traj_batch_mp2.done,
                                    avail_actions=traj_batch_mp2.avail_actions,
                                    hstate=None,
                                    rng=jax.random.PRNGKey(0),
                                    aux_obs=aux_obs_sp
                                )

                                log_prob_xp = pi_xp.log_prob(traj_batch_xp.action)
                                log_prob_sp = pi_sp.log_prob(traj_batch_sp.action)
                                log_prob_sp2 = pi_sp2.log_prob(traj_batch_sp2.action)
                                log_prob_mp = pi_mp.log_prob(traj_batch_mp.action)
                                log_prob_mp2 = pi_mp2.log_prob(traj_batch_mp2.action)

                                value_loss_xp = _compute_ppo_value_loss(value_xp, traj_batch_xp, target_v_xp)
                                value_loss_sp = _compute_ppo_value_loss(value_sp, traj_batch_sp, target_v_sp)
                                value_loss_sp2 = _compute_ppo_value_loss(value_sp2, traj_batch_sp2, target_v_sp2)
                                value_loss_mp = _compute_ppo_value_loss(value_mp, traj_batch_mp, target_v_mp)
                                value_loss_mp2 = _compute_ppo_value_loss(value_mp2, traj_batch_mp2, target_v_mp2)

                                pg_loss_xp = _compute_ppo_pg_loss(log_prob_xp, traj_batch_xp, gae_xp)
                                pg_loss_sp = _compute_ppo_pg_loss(log_prob_sp, traj_batch_sp, gae_sp)
                                pg_loss_sp2 = _compute_ppo_pg_loss(log_prob_sp2, traj_batch_sp2, gae_sp2)
                                pg_loss_mp = _compute_ppo_pg_loss(log_prob_mp, traj_batch_mp, gae_mp)
                                pg_loss_mp2 = _compute_ppo_pg_loss(log_prob_mp2, traj_batch_mp2, gae_mp2)

                                entropy_xp = jnp.mean(pi_xp.entropy())
                                entropy_sp = jnp.mean(pi_sp.entropy())
                                entropy_sp2 = jnp.mean(pi_sp2.entropy())
                                entropy_mp = jnp.mean(pi_mp.entropy())
                                entropy_mp2 = jnp.mean(pi_mp2.entropy())

                                xp_pg_weight = -config["COMEDI_ALPHA"]
                                sp_pg_weight = 1.0
                                mp2_pg_weight = config["COMEDI_BETA"]

                                xp_loss = xp_pg_weight * pg_loss_xp + config["VF_COEF"] * value_loss_xp - config["ENT_COEF"] * entropy_xp
                                sp_loss = sp_pg_weight * pg_loss_sp + config["VF_COEF"] * value_loss_sp - config["ENT_COEF"] * entropy_sp
                                sp2_loss = sp_pg_weight * pg_loss_sp2 + config["VF_COEF"] * value_loss_sp2 - config["ENT_COEF"] * entropy_sp2
                                mp_loss = mp2_pg_weight * pg_loss_mp + config["VF_COEF"] * value_loss_mp - config["ENT_COEF"] * entropy_mp
                                mp2_loss = mp2_pg_weight * pg_loss_mp2 + config["VF_COEF"] * value_loss_mp2 - config["ENT_COEF"] * entropy_mp2

                                total_loss = sp_loss + sp2_loss + xp_loss + mp2_loss + mp_loss
                                return total_loss, (value_loss_xp, value_loss_sp + value_loss_sp2, value_loss_mp + value_loss_mp2,
                                                    pg_loss_xp, pg_loss_sp + pg_loss_sp2, pg_loss_mp + pg_loss_mp2,
                                                    entropy_xp, entropy_sp + entropy_sp2, entropy_mp + entropy_mp2)

                            grad_fn = jax.value_and_grad(_loss_fn_conf, has_aux=True)
                            (loss_val, aux_vals), grads = grad_fn(
                                train_state_conf.params,
                                traj_batch_xp, advantages_xp, returns_xp,
                                traj_batch_sp1, advantages_sp1, returns_sp1,
                                traj_batch_sp2, advantages_sp2, returns_sp2,
                                traj_batch_mp1, advantages_mp1, returns_mp1,
                                traj_batch_mp2, advantages_mp2, returns_mp2)
                            train_state_conf = train_state_conf.apply_gradients(grads=grads)
                            return train_state_conf, (loss_val, aux_vals)

                        (
                            train_state_conf, traj_batch_xp,
                            traj_batch_sp_conf, traj_batch_sp_br,
                            traj_batch_mp_conf, traj_batch_mp_br,
                            advantages_xp_conf, advantages_sp_conf,
                            advantages_sp_br, advantages_mp_conf,
                            advantages_mp_br, targets_xp_conf,
                            targets_sp_conf, targets_sp_br,
                            targets_mp_conf, targets_mp_br,
                            rng, xp_id, sp_id
                        ) = update_state

                        rng, perm_rng_xp, perm_rng_sp_conf, perm_rng_sp_br, perm_rng_mp2_conf, perm_rng_mp2_br = jax.random.split(rng, 6)

                        minibatches_xp = _create_minibatches(
                            traj_batch_xp, advantages_xp_conf, targets_xp_conf, None,
                            config["NUM_ENVS"], config["NUM_MINIBATCHES"], perm_rng_xp
                        )
                        minibatches_sp_conf = _create_minibatches(
                            traj_batch_sp_conf, advantages_sp_conf, targets_sp_conf, None,
                            config["NUM_ENVS"], config["NUM_MINIBATCHES"], perm_rng_sp_conf
                        )
                        minibatches_sp_br = _create_minibatches(
                            traj_batch_sp_br, advantages_sp_br, targets_sp_br, None,
                            config["NUM_ENVS"], config["NUM_MINIBATCHES"], perm_rng_sp_br
                        )
                        minibatches_mp_conf = _create_minibatches(
                            traj_batch_mp_conf, advantages_mp_conf, targets_mp_conf, None,
                            config["NUM_ENVS"], config["NUM_MINIBATCHES"], perm_rng_mp2_conf
                        )
                        minibatches_mp_br = _create_minibatches(
                            traj_batch_mp_br, advantages_mp_br, targets_mp_br, None,
                            config["NUM_ENVS"], config["NUM_MINIBATCHES"], perm_rng_mp2_br
                        )

                        # Get number of minibatches from the traj_batch component (index 2 in the 5-tuple)
                        num_mb = minibatches_xp[2].obs.shape[0]
                        repeated_xp_id = jnp.repeat(xp_id, num_mb, axis=0)
                        repeated_sp_id = jnp.repeat(sp_id, num_mb, axis=0)

                        train_state_conf, total_loss_conf = jax.lax.scan(
                            _update_minbatch_conf, train_state_conf, (
                                minibatches_xp, minibatches_sp_conf, minibatches_sp_br,
                                minibatches_mp_conf, minibatches_mp_br, repeated_xp_id, repeated_sp_id
                            )
                        )

                        update_state = (train_state_conf,
                            traj_batch_xp, traj_batch_sp_conf, traj_batch_sp_br, traj_batch_mp_conf, traj_batch_mp_br,
                            advantages_xp_conf, advantages_sp_conf, advantages_sp_br, advantages_mp_conf, advantages_mp_br,
                            targets_xp_conf, targets_sp_conf, targets_sp_br, targets_mp_conf, targets_mp_br,
                            rng, xp_id, sp_id
                        )
                        return update_state, total_loss_conf

                    rng, sub_rng = jax.random.split(rng, 2)
                    update_state = (
                        train_state,
                        traj_batch_xp, traj_batch_sp_agent0,
                        traj_batch_sp_agent1,
                        traj_batch_smp0, traj_batch_smp1,
                        advantages_xp_conf,
                        advantages_sp_conf, advantages_sp_br,
                        advantages_mp_conf, advantages_mp_br,
                        targets_xp_conf, targets_sp_conf,
                        targets_sp_br, targets_mp_conf,
                        targets_mp_br, sub_rng,
                        max_means_id, num_prev_trained_conf
                    )
                    update_state, conf_losses = jax.lax.scan(
                        _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                    train_state = update_state[0]

                    (
                        conf_value_loss_xp, conf_value_loss_sp, conf_value_loss_mp,
                        conf_pg_loss_xp, conf_pg_loss_sp, conf_pg_loss_mp,
                        conf_entropy_xp, conf_entropy_sp, conf_entropy_mp
                    ) = conf_losses[1]

                    new_update_runner_state = (
                        train_state, pop_buffer,
                        env_state_sp, last_obs_sp,
                        env_state_xp, last_obs_xp,
                        env_state_mp, last_obs_mp,
                        env_state_mp2, last_obs_mp2,
                        last_dones_xp, last_dones_sp,
                        last_dones_mp, last_dones_mp2,
                        rng, update_steps + 1, num_prev_trained_conf
                    )

                    metric = traj_batch_xp.info
                    metric["update_steps"] = update_steps
                    metric["value_loss_conf_xp"] = conf_value_loss_xp
                    metric["value_loss_conf_sp"] = conf_value_loss_sp
                    metric["value_loss_conf_mp"] = conf_value_loss_mp
                    metric["pg_loss_conf_xp"] = conf_pg_loss_xp
                    metric["pg_loss_conf_sp"] = conf_pg_loss_sp
                    metric["pg_loss_conf_mp"] = conf_pg_loss_mp
                    metric["entropy_conf_xp"] = conf_entropy_xp
                    metric["entropy_conf_sp"] = conf_entropy_sp
                    metric["entropy_conf_mp"] = conf_entropy_mp
                    metric["average_rewards_ego"] = jnp.mean(traj_batch_xp.reward)
                    metric["average_rewards_br_sp"] = jnp.mean(traj_batch_sp_agent1.reward)
                    metric["average_rewards_br_mp2"] = jnp.mean(traj_batch_smp1.reward)

                    return (new_update_runner_state, checkpoint_array, ckpt_idx + 1), metric

                # Eval at start
                xp_eval_returns = jax.vmap(per_id_run_episode_fixed_rng, in_axes=(None, 0))(
                    train_state.params, jnp.arange(config["POP_SIZE"]))
                sp_eval_returns = run_episodes(
                    eval_rng, env,
                    agent_0_param=train_state.params, agent_0_policy=policy,
                    agent_1_param=train_state.params, agent_1_policy=policy,
                    max_episode_steps=config["EVAL_MAX_STEPS"],
                    num_eps=config["NUM_EVAL_EPISODES"]
                )

                update_steps = 0
                init_done_xp = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
                init_done_sp = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
                init_done_mp = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
                init_done_mp2 = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

                update_runner_state = (
                    train_state, pop_buffer,
                    env_state_sp, obsv_sp,
                    env_state_xp, obsv_xp,
                    env_state_mp, obsv_mp,
                    env_state_mp2, obsv_mp2,
                    init_done_xp, init_done_sp,
                    init_done_mp, init_done_mp2,
                    rng, update_steps,
                    num_existing_agents
                )

                checkpoint_array = init_ckpt_array(train_state.params)
                ckpt_idx = 0
                update_with_ckpt_runner_state = (update_runner_state, checkpoint_array, ckpt_idx, xp_eval_returns, sp_eval_returns)

                def _update_step_with_ckpt(state_with_ckpt, unused):
                    (update_runner_state, checkpoint_array, ckpt_idx, xp_eval_returns, sp_eval_returns) = state_with_ckpt

                    new_state_with_ckpt, metric = _update_step(
                        (update_runner_state, checkpoint_array, ckpt_idx),
                        None
                    )
                    new_update_runner_state = new_state_with_ckpt[0]
                    train_state = new_update_runner_state[0]
                    rng, update_steps = new_update_runner_state[-3], new_update_runner_state[-2]

                    to_store = jnp.logical_or(
                        jnp.equal(jnp.mod(update_steps - 1, ckpt_and_eval_interval), 0),
                        jnp.equal(update_steps, config["NUM_UPDATES"])
                    )

                    def store_and_eval_ckpt(args):
                        ckpt_arr_conf, rng, cidx, _, _ = args
                        new_ckpt_arr_conf = jax.tree.map(
                            lambda c_arr, p: c_arr.at[cidx].set(p),
                            ckpt_arr_conf, train_state.params
                        )

                        xp_eval_returns = jax.vmap(per_id_run_episode_fixed_rng, in_axes=(None, 0))(
                            train_state.params, jnp.arange(config["POP_SIZE"]))
                        sp_eval_returns = run_episodes(
                            eval_rng, env,
                            agent_0_param=train_state.params, agent_0_policy=policy,
                            agent_1_param=train_state.params, agent_1_policy=policy,
                            max_episode_steps=config["EVAL_MAX_STEPS"],
                            num_eps=config["NUM_EVAL_EPISODES"]
                        )

                        return (new_ckpt_arr_conf, rng, cidx + 1, xp_eval_returns, sp_eval_returns)

                    def skip_ckpt(args):
                        return args

                    rng, store_and_eval_rng = jax.random.split(rng, 2)
                    (checkpoint_array, store_and_eval_rng, ckpt_idx, xp_eval_returns, sp_eval_returns) = jax.lax.cond(
                        to_store,
                        store_and_eval_ckpt,
                        skip_ckpt,
                        (checkpoint_array, store_and_eval_rng, ckpt_idx, xp_eval_returns, sp_eval_returns)
                    )

                    return (new_update_runner_state, checkpoint_array,
                            ckpt_idx, xp_eval_returns, sp_eval_returns), (metric, xp_eval_returns, sp_eval_returns)

                new_update_with_ckpt_runner_state, (metric, xp_eval_returns, sp_eval_returns) = jax.lax.scan(
                    _update_step_with_ckpt,
                    update_with_ckpt_runner_state,
                    xs=None,
                    length=config["NUM_UPDATES"],
                )
                new_update_runner_state, new_checkpoint_array, _, _, _ = new_update_with_ckpt_runner_state
                final_train_state = new_update_runner_state[0]

                # Add trained policy to population buffer
                updated_pop_buffer = jax.tree.map(
                    lambda buf, p: buf.at[num_existing_agents].set(p),
                    pop_buffer, final_train_state.params
                )
                conf_checkpoints = new_checkpoint_array
                return updated_pop_buffer, (conf_checkpoints, metric, xp_eval_returns, sp_eval_returns)

            rngs = jax.random.split(rng, config["PARTNER_POP_SIZE"])
            rng, add_conf_iter_rngs = rngs[0], rngs[1:]

            iter_ids = jnp.arange(1, config["PARTNER_POP_SIZE"])
            final_population_buffer, (conf_checkpoints, metric, xp_eval_returns, sp_eval_returns) = jax.lax.scan(
                add_conf_policy, population_buffer, (iter_ids, add_conf_iter_rngs)
            )

            out = {
                "final_params_conf": final_population_buffer,
                "checkpoints_conf": conf_checkpoints,
                "metrics": metric,
                "last_ep_infos_xp": xp_eval_returns,
                "last_ep_infos_sp": sp_eval_returns
            }

            return out
        return train

    train_fn = make_comedi_agents(config)
    out = train_fn(train_rng)
    return out


def get_comedi_population(config: Dict[str, Any], out: Dict[str, Any], env):
    """Extract partner population from CoMeDi training output."""
    comedi_pop_size = config["PARTNER_POP_SIZE"]
    partner_params = out['final_params_conf']

    obs_dim = env.observation_space(env.agents[1]).shape[0]

    partner_policy = ActorWithConditionalCriticPolicy(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=obs_dim,
        pop_size=comedi_pop_size,
        activation=config.get("ACTIVATION", "tanh")
    )

    partner_population = AgentPopulation(
        pop_size=comedi_pop_size,
        policy_cls=partner_policy
    )

    return partner_params, partner_population


def run_comedi_training(config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """Run CoMeDi training."""
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
        log.info(f"Partner population size: {config['PARTNER_POP_SIZE']}")
        log.info(f"CoMeDi alpha: {config['COMEDI_ALPHA']}")
        log.info(f"CoMeDi beta: {config['COMEDI_BETA']}")
        log.info(f"Total timesteps: {config['TOTAL_TIMESTEPS']}")
        log.info(f"Learning rate: {config['LR']}")
        log.info(f"NUM_ENVS: {config['NUM_ENVS']}")
        if config.get('REW_SHAPING_HORIZON', 0) > 0:
            log.info(f"Reward shaping horizon: {config['REW_SHAPING_HORIZON']}")

    rng = jax.random.PRNGKey(config["TRAIN_SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    start_time = time.time()

    with jax.disable_jit(False):
        vmapped_train_fn = jax.jit(
            jax.vmap(
                partial(
                    train_comedi_partners,
                    env=env,
                    config=config,
                )
            )
        )
        out = vmapped_train_fn(rngs)

    end_time = time.time()
    training_time = end_time - start_time

    if verbose:
        log.info(f"Training completed in {training_time:.2f} seconds")

    partner_params, partner_population = get_comedi_population(config, out, env)

    results = {
        "partner_params": partner_params,
        "partner_population": partner_population,
        "final_params_conf": out["final_params_conf"],
        "checkpoints_conf": out["checkpoints_conf"],
        "metrics": out["metrics"],
        "last_ep_infos_xp": out["last_ep_infos_xp"],
        "last_ep_infos_sp": out["last_ep_infos_sp"],
        "training_time": training_time,
        "config": config,
    }

    return results


def print_training_summary(results: Dict[str, Any]):
    """Print a summary of training results."""
    config = results["config"]
    print("\n" + "=" * 60)
    print("CoMeDi Training Summary")
    print("=" * 60)
    print(f"  Layout: {config['ENV_KWARGS']['layout']}")
    print(f"  Population size: {config['PARTNER_POP_SIZE']}")
    print(f"  Alpha (XP weight): {config['COMEDI_ALPHA']}")
    print(f"  Beta (MP weight): {config['COMEDI_BETA']}")
    print(f"  Total timesteps: {config['TOTAL_TIMESTEPS']:.0f}")
    print(f"  Training time: {results['training_time']:.2f}s")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Train CoMeDi teammate policies on Overcooked V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--layout", type=str, default="cramped_room",
        help="Overcooked V2 layout name",
    )
    parser.add_argument(
        "--total_timesteps", type=float, default=3e7,
        help="Total training timesteps across all population members (default: 3e7)",
    )
    parser.add_argument(
        "--partner_pop_size", type=int, default=5,
        help="Number of partner policies in the population (default: 5)",
    )
    parser.add_argument(
        "--comedi_alpha", type=float, default=1.0,
        help="Weight for XP loss (negated for diversity) (default: 1.0)",
    )
    parser.add_argument(
        "--comedi_beta", type=float, default=0.5,
        help="Weight for mixed-play loss (default: 0.5)",
    )
    parser.add_argument(
        "--num_checkpoints", type=int, default=5,
        help="Number of checkpoints to save per population member (default: 5)",
    )
    parser.add_argument(
        "--num_envs", type=int, default=64,
        help="Number of parallel environments (default: 64)",
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
        "--csv_log", action="store_true",
        help="Enable CSV metric logging",
    )
    parser.add_argument(
        "--max_steps", type=int, default=400,
        help="Maximum steps per episode (default: 400)",
    )
    parser.add_argument(
        "--num_minibatches", type=int, default=8,
        help="Number of minibatches for PPO updates (default: 8, must divide NUM_ENVS)",
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
    config["COMEDI_ALPHA"] = args.comedi_alpha
    config["COMEDI_BETA"] = args.comedi_beta
    config["NUM_CHECKPOINTS"] = args.num_checkpoints
    config["NUM_ENVS"] = args.num_envs
    config["TRAIN_SEED"] = args.seed
    config["LR"] = args.lr
    config["OUTPUT_DIR"] = args.output_dir
    config["CSV_LOG"] = args.csv_log
    config["ENV_KWARGS"]["max_steps"] = args.max_steps
    config["NUM_MINIBATCHES"] = args.num_minibatches

    if args.debug:
        config["TOTAL_TIMESTEPS"] = 5e5
        config["PARTNER_POP_SIZE"] = 2
        config["NUM_CHECKPOINTS"] = 2
        config["NUM_ENVS"] = 4
        config["NUM_MINIBATCHES"] = 2
        config["UPDATE_EPOCHS"] = 2
        log.info("Running in DEBUG mode with reduced settings")

    if not args.quiet:
        log.info(f"Starting CoMeDi training on layout: {args.layout}")
        if args.gpu is not None:
            if args.gpu == "-1":
                log.info("Using CPU")
            else:
                log.info(f"Using GPU: {args.gpu}")
        else:
            log.info(f"Using all available devices: {jax.devices()}")

    try:
        results = run_comedi_training(config, verbose=not args.quiet)

        if not args.quiet:
            print_training_summary(results)

        if args.output_dir:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            save_data = {
                "checkpoints_conf": results["checkpoints_conf"],
                "metrics": results["metrics"],
            }
            checkpoint_returns = compute_checkpoint_returns(results["metrics"], results["config"])
            save_path = save_separated_checkpoints_multi(
                save_data, results["config"], str(output_path), savename="comedi_train_run",
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
