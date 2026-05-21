"""TaskRunner: Train ego PPO policy against fixed teammates.

This module implements a training runner that:
1. Loads a task specification from a JSONL manifest
2. Instantiates the fixed teammate policy from the spec
3. Trains an ego PPO agent from scratch against the teammate
4. Records full interaction history for later ICRL baseline training

Usage:
    python -m runners.task_runner --manifest <path> --out_dir <path> --task_idx <int>
    python -m runners.task_runner --manifest <path> --out_dir <path> --task_id <string>

Example:
    python -m runners.task_runner \\
        --manifest benchmarks/overcooked_icrl/teammate_train.jsonl \\
        --out_dir outputs/task_runs \\
        --task_idx 0 \\
        --total_steps 1e6 \\
        --num_envs 8 \\
        --record_envs 4

    # Run on specific GPU
    python -m runners.task_runner \\
        --manifest benchmarks/overcooked_icrl/teammate_train.jsonl \\
        --out_dir outputs/task_runs \\
        --task_idx 0 \\
        --gpu 0
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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

from benchmarks.manifest_schema import TaskEntry, TeammateSpec, load_manifest
from envs import make_env
from envs.log_wrapper import LogWrapper
from teammate_wrapper.registry import make_teammate, TeammatePolicy
from teammate_wrapper.specs import HeuristicTeammateSpec, RLTeammateSpec
from teammate_wrapper.theta_sampling import ThetaSpec, sample_theta, theta_from_json
from agents.initialize_agents import initialize_ego_agent
from agents.population_interface import AgentPopulation, HeuristicPolicyPopulation, DummyPolicyPopulation
from marl.ppo_utils import Transition, unbatchify, _create_minibatches

from runners.history_recorder import HistoryRecorder

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
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


@dataclass
class TaskRunnerConfig:
    """Configuration for TaskRunner."""
    # Task specification
    manifest_path: str
    task_idx: Optional[int] = None
    task_id: Optional[str] = None

    # Output
    out_dir: str = "outputs/task_runs"

    # PPO hyperparameters (updated with JaxMARL recipe defaults)
    total_steps: int = 15_000_000
    rollout_length: int = 256
    num_envs: int = 256
    num_minibatches: int = 64
    update_epochs: int = 4
    lr: float = 0.00025
    lr_warmup: float = 0.1  # Fraction of updates for LR warmup
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.25
    anneal_lr: bool = True
    actor_type: str = "cnn_rnn"  # cnn_rnn from JaxMARL recipe
    activation: str = "relu"

    # Network architecture (CNN+RNN)
    fc_dim_size: int = 128
    gru_hidden_dim: int = 128

    # Reward shaping
    rew_shaping_horizon: int = 15_000_000  # Anneal shaped rewards over this many timesteps (0 = disabled)

    # Recording options
    record_envs: int = 256  # Number of envs to record (subset of num_envs)
    record_teammate_actions: bool = True
    record_first_steps: int = 100  # Only record first N steps of each episode (0 = record all)

    # Training options
    seed: Optional[int] = None  # Override task seed if set
    log_every: int = 10  # Log every N updates
    save_ckpt: bool = True
    save_interval: int = 10  # Save intermediate checkpoints every N updates (0 = disabled)
    gpu: Optional[str] = None  # GPU device ID(s) to use (e.g., '0', '0,1')

    # Debug
    debug: bool = False

    # CSV logging
    csv_log: bool = False  # Enable CSV logging
    csv_interval: int = 10  # Log to CSV every N updates

    # Environment settings
    max_steps: int = 400  # Maximum steps per episode

    def to_ppo_config(self) -> Dict[str, Any]:
        """Convert to PPO config dict format."""
        return {
            "TOTAL_TIMESTEPS": self.total_steps,
            "ROLLOUT_LENGTH": self.rollout_length,
            "NUM_ENVS": self.num_envs,
            "NUM_MINIBATCHES": self.num_minibatches,
            "UPDATE_EPOCHS": self.update_epochs,
            "LR": self.lr,
            "LR_WARMUP": self.lr_warmup,
            "GAMMA": self.gamma,
            "GAE_LAMBDA": self.gae_lambda,
            "CLIP_EPS": self.clip_eps,
            "ENT_COEF": self.ent_coef,
            "VF_COEF": self.vf_coef,
            "MAX_GRAD_NORM": self.max_grad_norm,
            "ANNEAL_LR": self.anneal_lr,
            "EGO_ACTOR_TYPE": self.actor_type,
            "ACTOR_TYPE": self.actor_type,  # Also add this for compatibility with IPPO
            "NUM_CHECKPOINTS": 1,
            "ACTIVATION": self.activation,
            "FC_DIM_SIZE": self.fc_dim_size,
            "GRU_HIDDEN_DIM": self.gru_hidden_dim,
            "REW_SHAPING_HORIZON": self.rew_shaping_horizon,
        }


class TaskRunner:
    """Runner for training ego PPO against a fixed teammate with history recording.

    This class encapsulates the full training pipeline:
    1. Load task from manifest
    2. Create environment and teammate
    3. Train PPO ego agent
    4. Record interaction history
    5. Save results
    """

    def __init__(self, config: TaskRunnerConfig):
        """Initialize TaskRunner.

        Args:
            config: TaskRunnerConfig with all settings
        """
        self.config = config
        self.task_entry: Optional[TaskEntry] = None
        self.env = None
        self.teammate_policy: Optional[TeammatePolicy] = None
        self.recorder: Optional[HistoryRecorder] = None
        self.teammate_actor_type: str = "cnn_rnn"  # Default, updated in _create_teammate

        # Load task
        self._load_task()

    def _load_task(self):
        """Load task specification from manifest."""
        manifest_path = Path(self.config.manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        entries = load_manifest(str(manifest_path))

        if self.config.task_idx is not None:
            if self.config.task_idx >= len(entries):
                raise ValueError(
                    f"task_idx {self.config.task_idx} out of range. "
                    f"Manifest has {len(entries)} entries."
                )
            self.task_entry = entries[self.config.task_idx]

        elif self.config.task_id is not None:
            matching = [e for e in entries if e.task_id == self.config.task_id]
            if not matching:
                raise ValueError(
                    f"task_id '{self.config.task_id}' not found in manifest.\n"
                    f"Available task_ids: {[e.task_id for e in entries[:10]]}..."
                )
            self.task_entry = matching[0]

        else:
            raise ValueError("Must specify either --task_idx or --task_id")

        log.info(f"Loaded task: {self.task_entry.task_id}")
        log.info(f"  Layout: {self.task_entry.layout_name}")
        log.info(f"  Track: {self.task_entry.track}/{self.task_entry.split}")
        log.info(f"  Teammate: {self.task_entry.teammate.get_display_name()}")

    def _create_env(self):
        """Create the Overcooked V2 environment.

        For CNN+RNN actor type, uses grid-based observations (not flattened).
        For MLP/RNN actor types, uses flattened observations.
        """
        # Set flatten_obs based on actor type
        # CNN+RNN requires grid observations (not flattened)
        # MLP/RNN require flattened observations
        flatten_obs = self.config.actor_type != "cnn_rnn"

        env_kwargs = {
            "layout": self.task_entry.layout_name,
            "flatten_obs": flatten_obs,
            "max_steps": self.config.max_steps,
        }

        self.env = make_env("overcooked-v2", env_kwargs)
        self.env = LogWrapper(self.env)

        log.info(f"Created environment: overcooked-v2/{self.task_entry.layout_name}")

        # Log observation shape
        obs_shape = self.env.observation_space(self.env.agents[0]).shape
        log.info(f"  Obs shape: {obs_shape}")
        log.info(f"  Action dim: {self.env.action_space(self.env.agents[0]).n}")

    def _create_teammate(self):
        """Create the fixed teammate policy from task spec."""
        teammate_spec = self.task_entry.teammate

        if teammate_spec.kind == "heuristic":
            # Heuristic teammates don't use neural network observations
            self.teammate_actor_type = "heuristic"

            # Resolve theta from spec
            if teammate_spec.theta is not None:
                # Use expanded theta dict
                theta = theta_from_json(teammate_spec.family, teammate_spec.theta)
            else:
                # Sample theta from theta_id
                theta_spec = ThetaSpec(
                    family=teammate_spec.family,
                    theta_id=teammate_spec.theta_id,
                    split=self.task_entry.split,
                    base_seed=teammate_spec.base_seed,
                )
                theta = sample_theta(theta_spec)

            heuristic_spec = HeuristicTeammateSpec(
                family=teammate_spec.family,
                theta=theta,
                use_log_wrapper=True,
                start_cooking_interaction=False,
            )
            self.teammate_policy = make_teammate(heuristic_spec, self.env, "agent_1")

        elif teammate_spec.kind == "rl":
            # Check checkpoint exists
            if teammate_spec.ckpt is None:
                raise ValueError(
                    f"RL teammate requires checkpoint path.\n"
                    f"Task: {self.task_entry.task_id}\n"
                    f"Teammate: {teammate_spec.family}"
                )

            ckpt_path = teammate_spec.ckpt
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"Teammate checkpoint not found: {ckpt_path}\n"
                    f"Task: {self.task_entry.task_id}"
                )

            extra = teammate_spec.extra or {}

            # Store teammate's actor_type for proper observation handling
            # CNN+RNN requires spatial observations, others use flattened
            self.teammate_actor_type = extra.get("actor_type", "cnn_rnn")

            rl_spec = RLTeammateSpec(
                algo=teammate_spec.family,
                ckpt_path=ckpt_path,
                use_log_wrapper=True,
                extra=extra,
            )
            self.teammate_policy = make_teammate(rl_spec, self.env, "agent_1")

        else:
            raise ValueError(f"Unknown teammate kind: {teammate_spec.kind}")

        log.info(f"Created teammate: {self.teammate_policy.name}")
        log.info(f"  Teammate actor type: {self.teammate_actor_type}")

    def _create_teammate_population(self):
        """Wrap teammate policy in AgentPopulation interface for PPO training."""
        if self.task_entry.teammate.kind == "heuristic":
            # Create HeuristicPolicyPopulation wrapper
            population = HeuristicPolicyPopulation(policy_cls=self.teammate_policy._policy)
            # Params don't matter for heuristic, but need something for vmap
            dummy_params = jax.tree.map(
                lambda x: x[jnp.newaxis, ...],
                {"dummy": jnp.zeros((1,))}
            )
            return population, dummy_params

        else:
            # RL teammate
            population = DummyPolicyPopulation(
                policy_cls=self.teammate_policy._policy,
                test_mode=True,
            )
            # Add batch dim to params
            params = jax.tree.map(
                lambda x: x[jnp.newaxis, ...],
                self.teammate_policy._params
            )
            return population, params

    def run(self) -> Dict[str, Any]:
        """Run the full training pipeline.

        Returns:
            Dict with:
                - final_params: Trained ego policy parameters
                - metrics: Training metrics
                - task_dir: Output directory path
        """
        # Setup
        self._create_env()
        self._create_teammate()

        # Determine seed
        seed = self.config.seed if self.config.seed is not None else self.task_entry.seed
        rng = jax.random.PRNGKey(seed)

        # Initialize ego agent
        ppo_config = self.config.to_ppo_config()
        rng, init_rng = jax.random.split(rng)
        ego_policy, init_ego_params = initialize_ego_agent(ppo_config, self.env, init_rng)

        log.info(f"Initialized ego agent: {self.config.actor_type}")

        # Create teammate population wrapper
        teammate_population, teammate_params = self._create_teammate_population()

        # Initialize recorder
        self.recorder = HistoryRecorder(
            out_dir=self.config.out_dir,
            task_uid=self.task_entry.task_id,
            task_spec=self.task_entry.to_json(),
            ppo_config=ppo_config,
            record_envs=min(self.config.record_envs, self.config.num_envs),
            total_envs=self.config.num_envs,
            seed=seed,
            extra_metadata={
                "actor_type": self.config.actor_type,
                "manifest_path": self.config.manifest_path,
                "record_first_steps": self.config.record_first_steps,
            },
            record_first_steps=self.config.record_first_steps,
        )

        if self.config.record_first_steps > 0:
            log.info(f"Recording first {self.config.record_first_steps} steps of each episode")
        else:
            log.info("Recording all steps of each episode")

        # Run training
        log.info("Starting PPO training...")
        start_time = time.time()

        rng, train_rng = jax.random.split(rng)
        result = self._train_ppo(
            ppo_config=ppo_config,
            train_rng=train_rng,
            ego_policy=ego_policy,
            init_ego_params=init_ego_params,
            teammate_population=teammate_population,
            teammate_params=teammate_params,
        )

        elapsed = time.time() - start_time
        log.info(f"Training completed in {elapsed:.1f}s")

        # Save results
        task_dir = self.recorder.save(
            final_params=result["final_params"],
            save_checkpoint=self.config.save_ckpt,
        )

        log.info(f"Results saved to: {task_dir}")

        return {
            "final_params": result["final_params"],
            "metrics": result["metrics"],
            "task_dir": task_dir,
        }

    def _train_ppo(
        self,
        ppo_config: Dict[str, Any],
        train_rng: jax.Array,
        ego_policy,
        init_ego_params,
        teammate_population: AgentPopulation,
        teammate_params,
    ) -> Dict[str, Any]:
        """Run PPO training with history recording.

        This is a modified version of train_ppo_ego_agent that records
        transitions for later ICRL baseline training.
        """
        config = ppo_config
        env = self.env

        # Compute training schedule
        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        config["NUM_CONTROLLED_ACTORS"] = config["NUM_ENVS"]
        config["NUM_UNCONTROLLED_ACTORS"] = config["NUM_ENVS"]
        config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"]) // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
        config["NUM_ACTIONS"] = env.action_space(env.agents[0]).n

        # Validate that NUM_CONTROLLED_ACTORS is divisible by NUM_MINIBATCHES
        if config["NUM_CONTROLLED_ACTORS"] % config["NUM_MINIBATCHES"] != 0:
            raise ValueError(
                f"NUM_CONTROLLED_ACTORS ({config['NUM_CONTROLLED_ACTORS']} = {config['NUM_ENVS']} envs) "
                f"must be divisible by NUM_MINIBATCHES ({config['NUM_MINIBATCHES']}). "
                f"Suggested NUM_MINIBATCHES values that divide {config['NUM_CONTROLLED_ACTORS']}: "
                f"{[i for i in [4, 8, 16, 32, 64, 128, 256] if config['NUM_CONTROLLED_ACTORS'] % i == 0]}"
            )

        num_agents = env.num_agents
        assert num_agents == 2, "Expected exactly 2 agents"

        # Learning rate schedule
        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        # Cosine decay with warmup schedule (from JaxMARL recipe)
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

        # Reward shaping annealing schedule (from JaxMARL recipe)
        rew_shaping_horizon = config.get("REW_SHAPING_HORIZON", 0)
        if rew_shaping_horizon > 0:
            rew_shaping_anneal = optax.linear_schedule(
                init_value=1.0,
                end_value=0.0,
                transition_steps=int(rew_shaping_horizon)
            )
        else:
            rew_shaping_anneal = None

        # Optimizer
        if config["ANNEAL_LR"]:
            # Use warmup + cosine decay if LR_WARMUP is specified, otherwise use linear
            if config.get("LR_WARMUP", 0.0) > 0:
                lr_schedule = create_warmup_cosine_schedule()
            else:
                lr_schedule = linear_schedule
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=lr_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=ego_policy.network.apply,
            params=init_ego_params,
            tx=tx,
        )

        # Initialize hidden states
        init_ego_hstate = ego_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
        init_partner_hstate = teammate_population.init_hstate(config["NUM_UNCONTROLLED_ACTORS"])

        # Tracking
        all_metrics = []
        total_steps = 0
        record_envs = self.config.record_envs

        # Setup CSV logging
        csv_logger = None
        if self.config.csv_log:
            csv_path = Path(self.config.out_dir) / self.task_entry.task_id / "metrics.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_fieldnames = [
                "update", "total_steps", "episodes",
                "mean_base_return", "mean_shaped_return",
                "elapsed_time"
            ]
            csv_logger = CSVMetricsLogger(str(csv_path), csv_fieldnames)
            log.info(f"CSV logging enabled: {csv_path}")

        training_start_time = time.time()

        # Track episode returns:
        # - base_returns: from LogWrapper's returned_episode_returns (base reward only)
        # - shaped_returns: cumulative training reward (base + annealed*shaped), tracked manually
        episode_base_returns = []
        episode_shaped_returns = []
        # Per-env accumulators for shaped (training) reward
        current_shaped_return = np.zeros(config["NUM_ENVS"], dtype=np.float32)

        # =====================
        # Initialize environment state ONCE (continue across updates)
        # This is critical for proper episode tracking when ROLLOUT_LENGTH != max_steps
        # =====================
        rng, reset_rng = jax.random.split(train_rng)
        train_rng = rng
        reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
        obs, env_state = jax.vmap(env.reset)(reset_rngs)
        done = {k: jnp.zeros((config["NUM_ENVS"],), dtype=bool) for k in env.agents + ["__all__"]}

        # Initialize hidden states (will be continued across updates)
        ego_hstate = init_ego_hstate
        partner_hstate = init_partner_hstate

        # Sample partner indices (single partner)
        partner_indices = jnp.zeros(config["NUM_UNCONTROLLED_ACTORS"], dtype=jnp.int32)

        # =====================
        # Define rollout step function for jax.lax.scan (MUCH faster than Python loop)
        # Note: params passed in carry to avoid closure capture issues with JAX tracing
        # =====================
        def _rollout_step(carry, unused):
            """Single rollout step - designed for jax.lax.scan."""
            (obs, env_state, done, ego_hstate, partner_hstate, rng, anneal_factor, params) = carry

            rng, step_rng, actor_rng, partner_rng = jax.random.split(rng, 4)

            # Get available actions
            avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
            avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
            avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

            # Handle observation shape based on actor type
            if config.get("EGO_ACTOR_TYPE", "mlp") == "cnn_rnn":
                obs_input_0 = obs["agent_0"][jnp.newaxis, :]  # (1, NUM_ENVS, H, W, C)
            else:
                obs_input_0 = obs["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)

            # Ego agent action - use params from carry
            act_0, val_0, pi_0, new_ego_hstate = ego_policy.get_action_value_policy(
                params=params,
                obs=obs_input_0,
                done=done["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                avail_actions=avail_actions_0,
                hstate=ego_hstate,
                rng=actor_rng,
            )
            logp_0 = pi_0.log_prob(act_0)
            act_0 = act_0.squeeze()
            logp_0 = logp_0.squeeze()
            val_0 = val_0.squeeze()

            # Teammate action
            if self.teammate_actor_type == "cnn_rnn":
                teammate_obs = obs["agent_1"][:, jnp.newaxis, jnp.newaxis, ...]
                teammate_done = done["agent_1"][:, jnp.newaxis, jnp.newaxis]
                teammate_avail = avail_actions_1[:, jnp.newaxis, jnp.newaxis, :]
            else:
                teammate_obs = obs["agent_1"].reshape(config["NUM_CONTROLLED_ACTORS"], 1, -1)
                teammate_done = done["agent_1"].reshape(config["NUM_CONTROLLED_ACTORS"], 1, -1)
                teammate_avail = avail_actions_1

            act_1, new_partner_hstate = teammate_population.get_actions(
                teammate_params,
                partner_indices,
                teammate_obs,
                teammate_done,
                teammate_avail,
                partner_hstate,
                partner_rng,
                env_state=env_state,
                aux_obs=None,
            )
            act_1 = act_1.squeeze()

            # Step environment
            env_act = {"agent_0": act_0, "agent_1": act_1}
            step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
            obs_next, env_state_next, reward, done_next, info = jax.vmap(env.step)(
                step_rngs, env_state, env_act
            )

            # Apply reward shaping with annealing
            reward_0 = reward["agent_0"]
            shaped_reward = info.get("shaped_reward", jnp.zeros((config["NUM_ENVS"], 2)))
            if shaped_reward.ndim == 2:
                shaped_reward_0 = shaped_reward[:, 0]
            else:
                shaped_reward_0 = shaped_reward
            reward_0_shaped = reward_0 + anneal_factor * shaped_reward_0

            # Get episode info for tracking
            base_returns = info.get("returned_episode_returns", jnp.zeros((config["NUM_ENVS"], 2)))
            if base_returns.ndim == 2:
                base_returns_0 = base_returns[:, 0]
            else:
                base_returns_0 = base_returns

            # Transition data to store
            transition = {
                "obs": obs["agent_0"],
                "action": act_0,
                "value": val_0,
                "log_prob": logp_0,
                "avail_actions": avail_actions_0,
                "teammate_action": act_1,
                "reward_shaped": reward_0_shaped,
                "reward_base": reward["agent_0"],
                "done": done_next["__all__"],
                "done_agent": done_next["agent_0"],
                "base_returns": base_returns_0,
            }

            new_carry = (obs_next, env_state_next, done_next, new_ego_hstate, new_partner_hstate, rng, anneal_factor, params)
            return new_carry, transition

        # =====================
        # Define PPO update functions for jax.lax.scan (MUCH faster than Python loops)
        # =====================
        def _ppo_loss_fn(params, init_hstate, init_done, traj_batch, gae, target_v):
            """PPO loss function."""
            shifted_done = jnp.concatenate([init_done, traj_batch.done[:-1]], axis=0)

            _, value, pi, _ = ego_policy.get_action_value_policy(
                params=params,
                obs=traj_batch.obs,
                done=shifted_done,
                avail_actions=traj_batch.avail_actions,
                hstate=init_hstate,
                rng=jax.random.PRNGKey(0),
            )
            log_prob = pi.log_prob(traj_batch.action)

            # Value loss (clipped)
            value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                -config["CLIP_EPS"], config["CLIP_EPS"]
            )
            value_losses = jnp.square(value - target_v)
            value_losses_clipped = jnp.square(value_pred_clipped - target_v)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

            # Policy loss (clipped)
            ratio = jnp.exp(log_prob - traj_batch.log_prob)
            gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
            pg_loss_1 = ratio * gae_norm
            pg_loss_2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae_norm
            pg_loss = -jnp.minimum(pg_loss_1, pg_loss_2).mean()

            # Entropy bonus
            entropy = pi.entropy().mean()

            total_loss = pg_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
            return total_loss, (value_loss, pg_loss, entropy)

        def _update_minibatch(train_state, batch_info):
            """Single minibatch update - for jax.lax.scan over minibatches."""
            init_hstate, init_done, traj_batch, advantages, targets = batch_info
            grad_fn = jax.value_and_grad(_ppo_loss_fn, has_aux=True)
            (loss, aux), grads = grad_fn(train_state.params, init_hstate, init_done, traj_batch, advantages, targets)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, (loss, aux)

        def _update_epoch(update_state, unused):
            """Single epoch of PPO updates - for jax.lax.scan over epochs."""
            train_state, init_hstate, init_done, traj_batch, advantages, targets, rng = update_state
            rng, perm_rng = jax.random.split(rng)

            minibatches = _create_minibatches(
                traj_batch, advantages, targets, init_hstate,
                config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng,
                init_done=init_done
            )

            # Scan over minibatches instead of Python loop
            train_state, loss_info = jax.lax.scan(_update_minibatch, train_state, minibatches)

            update_state = (train_state, init_hstate, init_done, traj_batch, advantages, targets, rng)
            return update_state, loss_info

        # =====================
        # Training loop
        # =====================
        for update_idx in range(config["NUM_UPDATES"]):
            rng, rollout_rng = jax.random.split(train_rng)
            train_rng = rng

            # Save initial states for PPO update
            initial_ego_hstate_for_update = ego_hstate
            initial_done_for_update = done["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"])

            # Compute reward shaping annealing factor
            if rew_shaping_anneal is not None:
                current_timestep = update_idx * config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
                anneal_factor = rew_shaping_anneal(current_timestep)
            else:
                anneal_factor = 0.0

            # =====================
            # Collect rollout using jax.lax.scan (FAST!)
            # =====================
            carry = (obs, env_state, done, ego_hstate, partner_hstate, rollout_rng, anneal_factor, train_state.params)
            carry, rollout_data = jax.lax.scan(_rollout_step, carry, None, length=config["ROLLOUT_LENGTH"])
            obs, env_state, done, ego_hstate, partner_hstate, _, _, _ = carry

            # Stack rollout into Transition
            traj_batch = Transition(
                done=rollout_data["done"],
                action=rollout_data["action"],
                value=rollout_data["value"],
                reward=rollout_data["reward_shaped"],
                log_prob=rollout_data["log_prob"],
                obs=rollout_data["obs"],
                info={},
                avail_actions=rollout_data["avail_actions"],
            )

            # =====================
            # Recording and metrics (after rollout, batched)
            # =====================
            if self.config.record_envs > 0:
                # Batch record - transfer to CPU once
                obs_np = np.asarray(rollout_data["obs"])
                act_np = np.asarray(rollout_data["action"])
                reward_np = np.asarray(rollout_data["reward_base"])
                done_np = np.asarray(rollout_data["done_agent"])
                teammate_act_np = np.asarray(rollout_data["teammate_action"]) if self.config.record_teammate_actions else None

                for step_idx in range(config["ROLLOUT_LENGTH"]):
                    self.recorder.record_step(
                        obs=obs_np[step_idx],
                        action=act_np[step_idx],
                        reward=reward_np[step_idx],
                        done=done_np[step_idx],
                        teammate_action=teammate_act_np[step_idx] if teammate_act_np is not None else None,
                    )

            # Track episode completions (batch processing)
            ep_done_all = np.asarray(rollout_data["done"])
            base_returns_all = np.asarray(rollout_data["base_returns"])
            reward_shaped_all = np.asarray(rollout_data["reward_shaped"])

            for step_idx in range(config["ROLLOUT_LENGTH"]):
                current_shaped_return += reward_shaped_all[step_idx]
                ep_done = ep_done_all[step_idx]
                if np.any(ep_done):
                    completed_base = base_returns_all[step_idx][ep_done]
                    episode_base_returns.extend(completed_base.tolist())
                    completed_shaped = current_shaped_return[ep_done]
                    episode_shaped_returns.extend(completed_shaped.tolist())
                    current_shaped_return[ep_done] = 0.0

            total_steps += config["NUM_ENVS"] * config["ROLLOUT_LENGTH"]

            # =====================
            # Compute GAE
            # =====================
            avail_actions_final = jax.vmap(env.get_avail_actions)(env_state.env_state)["agent_0"].astype(jnp.float32)

            if config.get("EGO_ACTOR_TYPE", "mlp") == "cnn_rnn":
                obs_input_final = obs["agent_0"][jnp.newaxis, :]
            else:
                obs_input_final = obs["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)

            _, last_val, _, _ = ego_policy.get_action_value_policy(
                params=train_state.params,
                obs=obs_input_final,
                done=done["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                avail_actions=avail_actions_final,
                hstate=ego_hstate,
                rng=jax.random.PRNGKey(0),
            )
            last_val = last_val.squeeze()

            advantages, targets = self._calculate_gae(traj_batch, last_val, config)

            # =====================
            # PPO updates using jax.lax.scan (FAST! - replaces nested Python loops)
            # =====================
            rng, update_rng = jax.random.split(train_rng)
            train_rng = rng

            update_state = (train_state, initial_ego_hstate_for_update, initial_done_for_update,
                           traj_batch, advantages, targets, update_rng)
            update_state, _ = jax.lax.scan(_update_epoch, update_state, None, length=config["UPDATE_EPOCHS"])
            train_state = update_state[0]

            # =====================
            # Logging
            # =====================
            # Use our tracked returns for both base and shaped
            recent_base = episode_base_returns[-100:] if episode_base_returns else []
            recent_shaped = episode_shaped_returns[-100:] if episode_shaped_returns else []
            mean_base_return = np.mean(recent_base) if recent_base else 0.0
            mean_shaped_return = np.mean(recent_shaped) if recent_shaped else 0.0

            if update_idx % self.config.log_every == 0 or update_idx == config["NUM_UPDATES"] - 1:
                log.info(
                    f"Update {update_idx + 1}/{config['NUM_UPDATES']} | "
                    f"Steps: {total_steps:,} | "
                    f"Episodes: {len(episode_base_returns)} | "
                    f"Mean Base Return: {mean_base_return:.2f} | "
                    f"Mean Shaped Return: {mean_shaped_return:.2f}"
                )

            self.recorder.update_training_progress(update_idx + 1, total_steps)

            all_metrics.append({
                "update": update_idx,
                "total_steps": total_steps,
                "episodes": len(episode_base_returns),
                "mean_base_return": mean_base_return,
                "mean_shaped_return": mean_shaped_return,
            })

            # CSV logging at specified intervals
            if csv_logger is not None and (update_idx % self.config.csv_interval == 0 or update_idx == config["NUM_UPDATES"] - 1):
                elapsed = time.time() - training_start_time
                csv_logger.log({
                    "update": update_idx + 1,
                    "total_steps": total_steps,
                    "episodes": len(episode_base_returns),
                    "mean_base_return": mean_base_return,
                    "mean_shaped_return": mean_shaped_return,
                    "elapsed_time": elapsed,
                })

            # Save intermediate checkpoint if configured
            if self.config.save_interval > 0 and (update_idx + 1) % self.config.save_interval == 0:
                log.info(f"Saving incremental checkpoint at update {update_idx + 1}...")
                ckpt_path = self.recorder.save_incremental(
                    update_step=update_idx + 1,
                    current_params=train_state.params,
                    metrics=all_metrics.copy(),
                    clear_buffer=True,
                )
                if ckpt_path:
                    log.info(f"  -> Saved chunk to: {ckpt_path}")
                    log.info(f"  -> Buffer cleared to free memory")

        return {
            "final_params": train_state.params,
            "metrics": all_metrics,
        }

    def _calculate_gae(self, traj_batch: Transition, last_val: jax.Array, config: Dict[str, Any]):
        """Compute Generalized Advantage Estimation."""
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
        targets = advantages + traj_batch.value
        return advantages, targets

    def _ppo_update_step(
        self,
        train_state: TrainState,
        ego_policy,
        traj_batch: Transition,
        advantages: jax.Array,
        returns: jax.Array,
        init_hstate: Any,
        init_done: jax.Array,
        config: Dict[str, Any],
    ):
        """Single PPO update step."""
        def _loss_fn(params, init_hstate, init_done, traj_batch, gae, target_v):
            # Create shifted done signal for correct RNN hidden state reset
            # During rollout: RNN uses last_done (done from step t-1) for reset at step t
            # traj_batch.done[t] = done AFTER step t's action (i.e., for step t+1's reset)
            # So we need: shifted_done[0] = init_done, shifted_done[t] = done[t-1] for t > 0
            # This ensures RNN resets at the START of new episodes, not at the END of old ones
            shifted_done = jnp.concatenate([init_done, traj_batch.done[:-1]], axis=0)

            _, value, pi, _ = ego_policy.get_action_value_policy(
                params=params,
                obs=traj_batch.obs,
                done=shifted_done,
                avail_actions=traj_batch.avail_actions,
                hstate=init_hstate,
                rng=jax.random.PRNGKey(0),
            )
            log_prob = pi.log_prob(traj_batch.action)

            # Value loss (clipped)
            value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                -config["CLIP_EPS"], config["CLIP_EPS"]
            )
            value_losses = jnp.square(value - target_v)
            value_losses_clipped = jnp.square(value_pred_clipped - target_v)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

            # Policy loss (clipped)
            ratio = jnp.exp(log_prob - traj_batch.log_prob)
            gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
            pg_loss_1 = ratio * gae_norm
            pg_loss_2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae_norm
            pg_loss = -jnp.minimum(pg_loss_1, pg_loss_2).mean()

            # Entropy bonus
            entropy = pi.entropy().mean()

            total_loss = pg_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
            return total_loss, (value_loss, pg_loss, entropy)

        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(train_state.params, init_hstate, init_done, traj_batch, advantages, returns)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, (loss, aux)


def run_task(config: TaskRunnerConfig) -> Dict[str, Any]:
    """Run a single task with the given configuration.

    This is the main entry point for programmatic use.

    Args:
        config: TaskRunnerConfig with all settings

    Returns:
        Dict with final_params, metrics, and task_dir
    """
    runner = TaskRunner(config)
    return runner.run()


def parse_args() -> TaskRunnerConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ego PPO agent against fixed teammate from manifest."
    )

    # Required arguments
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="Path to JSONL manifest file"
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="Output directory for results"
    )

    # Task selection (one required)
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--task_idx", type=int,
        help="Index of task in manifest (0-based)"
    )
    task_group.add_argument(
        "--task_id", type=str,
        help="Task ID string from manifest"
    )

    # PPO hyperparameters (defaults from JaxMARL recipe)
    parser.add_argument("--total_steps", type=float, default=6e7, help="Total training steps (default: 6e7, from JaxMARL)")
    parser.add_argument("--rollout_length", type=int, default=256, help="Rollout length per update (default: 256)")
    parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel environments (default: 1024)")
    parser.add_argument("--num_minibatches", type=int, default=64, help="Number of minibatches (default: 64)")
    parser.add_argument("--update_epochs", type=int, default=4, help="PPO update epochs (default: 4)")
    parser.add_argument("--lr", type=float, default=0.00025, help="Learning rate (default: 0.00025)")
    parser.add_argument("--lr_warmup", type=float, default=0.1, help="Fraction of updates for LR warmup (default: 0.1)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon (default: 0.2)")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.25, help="Max gradient norm (default: 0.25)")
    parser.add_argument("--anneal_lr", action="store_true", default=True, help="Anneal learning rate")
    parser.add_argument("--no_anneal_lr", action="store_false", dest="anneal_lr", help="Don't anneal LR")
    parser.add_argument("--actor_type", type=str, default="cnn_rnn", choices=["mlp", "rnn", "s5", "cnn_rnn"],
                       help="Ego agent architecture (default: cnn_rnn)")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh"],
                       help="Activation function (default: relu)")
    parser.add_argument("--fc_dim_size", type=int, default=128, help="FC layer hidden dimension (default: 128)")
    parser.add_argument("--gru_hidden_dim", type=int, default=128, help="GRU hidden dimension (default: 128)")
    parser.add_argument("--rew_shaping_horizon", type=float, default=6e7,
                       help="Reward shaping horizon (timesteps to anneal shaped rewards, 0 = disabled)")

    # Recording options
    parser.add_argument("--record_envs", type=int, default=1024,
                       help="Number of envs to record (subset of num_envs)")
    parser.add_argument("--no_record_teammate", action="store_false", dest="record_teammate_actions",
                       help="Don't record teammate actions")
    parser.add_argument("--record_first_steps", type=int, default=100,
                       help="Only record first N steps of each episode (0 = record all, default: 100)")

    # Training options
    parser.add_argument("--seed", type=int, default=None, help="Override task seed")
    parser.add_argument("--log_every", type=int, default=10, help="Log every N updates")
    parser.add_argument("--no_save_ckpt", action="store_false", dest="save_ckpt",
                       help="Don't save final checkpoint")
    parser.add_argument("--save_interval", type=int, default=10,
                       help="Save intermediate checkpoints every N updates (0=disabled). "
                            "Saves history, metrics, and model params at regular intervals.")
    parser.add_argument("--gpu", type=str, default=None,
                       help="GPU device ID(s) to use (e.g., '0', '0,1'). Use '-1' for CPU. If not specified, uses all available GPUs.")
    parser.add_argument("--debug", action="store_true", help="Debug mode (verbose output)")
    parser.add_argument("--csv_log", type=bool, default=True,
                       help="Enable CSV logging of metrics")
    parser.add_argument("--csv_interval", type=int, default=10,
                       help="Log to CSV every N updates (default: 10)")
    parser.add_argument("--max_steps", type=int, default=400,
                       help="Maximum steps per episode (default: 400)")

    args = parser.parse_args()

    return TaskRunnerConfig(
        manifest_path=args.manifest,
        task_idx=args.task_idx,
        task_id=args.task_id,
        out_dir=args.out_dir,
        total_steps=int(args.total_steps),
        rollout_length=args.rollout_length,
        num_envs=args.num_envs,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        lr=args.lr,
        lr_warmup=args.lr_warmup,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        anneal_lr=args.anneal_lr,
        actor_type=args.actor_type,
        activation=args.activation,
        fc_dim_size=args.fc_dim_size,
        gru_hidden_dim=args.gru_hidden_dim,
        rew_shaping_horizon=int(args.rew_shaping_horizon),
        record_envs=args.record_envs,
        record_teammate_actions=args.record_teammate_actions,
        record_first_steps=args.record_first_steps,
        seed=args.seed,
        log_every=args.log_every,
        save_ckpt=args.save_ckpt,
        save_interval=args.save_interval,
        gpu=args.gpu,
        debug=args.debug,
        csv_log=args.csv_log,
        csv_interval=args.csv_interval,
        max_steps=args.max_steps,
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

    if config.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        result = run_task(config)
        print(f"\nTraining completed successfully!")
        print(f"Results saved to: {result['task_dir']}")
        return 0
    except Exception as e:
        log.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
