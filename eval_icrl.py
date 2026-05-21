#!/usr/bin/env python3
"""ICRL Benchmark Evaluation CLI.

This script evaluates ICRL algorithms (DPT, AD, AMAGO-offline, Hybrid-AD,
random baseline) on the Overcooked V2 benchmark across both tracks:
teammate and layout.

Features:
- Resume-friendly: writes per-task JSON results, skips completed tasks
- Deterministic: given seed produces identical results
- Multi-algorithm: evaluate multiple algorithms in one run
- Multi-track: evaluate all tracks or a subset
- Vectorized parallel evaluation: batch multiple tasks for efficient GPU utilization

Example commands:
    # Full evaluation on all tracks
    python eval_icrl.py --algo dpt,ad,amago_offline,hybrid_ad --tracks teammate,layout \
        --episodes 100 --seed 0 --out results/

    # Run on specific GPU
    python eval_icrl.py --algo dpt,ad --tracks teammate,layout \
        --episodes 100 --seed 0 --out results/ --gpu 0

    # Run with multiple seeds (3 runs: seed 0, 1, 2)
    # Results saved in results/seed_0/, results/seed_1/, results/seed_2/
    python eval_icrl.py --algo dpt --tracks teammate --episodes 100 \
        --seed 0 --num_seeds 3 --out results/

    # Run with vectorized parallel evaluation (batch size 4)
    python eval_icrl.py --algo dpt --tracks teammate --episodes 100 \
        --seed 0 --out results/ --run_parallel --parallel_batch_size 4

    # Smoke test (2 tasks per track, 3 episodes each)
    python eval_icrl.py --algo dpt --tracks teammate --episodes 3 \
        --max_tasks_per_track 2 --out results_smoke/

    # Resume interrupted evaluation
    python eval_icrl.py --algo dpt,ad --tracks teammate,layout \
        --episodes 100 --seed 0 --out results/

    # Force re-evaluation of all tasks
    python eval_icrl.py --algo dpt --tracks teammate --episodes 100 \
        --seed 0 --out results/ --force
"""

import argparse
import dataclasses
import json
import logging
import os
import pickle
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
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

from benchmarks.manifest_schema import TaskEntry, load_manifest
from envs import make_env
from envs.log_wrapper import LogWrapper
from teammate_wrapper.registry import make_teammate
from teammate_wrapper.specs import HeuristicTeammateSpec, RLTeammateSpec
from teammate_wrapper.theta_sampling import theta_from_json, ThetaSpec, sample_theta

log = logging.getLogger(__name__)

# ==============================================================================
# Constants and Default Paths
# ==============================================================================

BENCHMARK_DIR = Path("benchmarks/overcooked_icrl")
TRACK_DIRS = {
    "teammate": BENCHMARK_DIR / "track_teammate",
    "layout": BENCHMARK_DIR / "track_layout",
}
MANIFEST_NAMES = {
    "train": "manifest_train.jsonl",
    "test": "manifest_test.jsonl",
}

SUPPORTED_ALGOS = {"dpt", "ad", "random", "amago_offline", "hybrid_ad"}
SUPPORTED_TRACKS = {"teammate", "layout"}

# Known heuristic teammate families (for filtering)
HEURISTIC_TEAMMATES = {"assembly_line", "territory", "utility_greedy", "recipe_aware_button"}

# Result JSON schema version
RESULT_SCHEMA_VERSION = 1

# Checkpoint schema version for resume functionality
CHECKPOINT_SCHEMA_VERSION = 1


# ==============================================================================
# Batch Evaluation Checkpoint (for resume functionality)
# ==============================================================================

@dataclass
class BatchEvalCheckpoint:
    """Checkpoint for resuming batch evaluation.

    Stores all intermediate state needed to resume evaluation from a specific episode.
    Saved after each episode completes to allow resume on interruption.
    """
    # Schema version for forward compatibility
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    # Algorithm info
    algorithm: str = ""

    # Task identifiers (list of task_ids in the batch)
    task_ids: List[str] = field(default_factory=list)

    # Progress tracking
    completed_episodes: int = 0  # Number of fully completed episodes
    total_episodes: int = 0

    # Random state (as list for JSON serialization)
    rng_key: List[int] = field(default_factory=list)  # JAX PRNGKey as list

    # Results accumulated so far (per-task lists)
    all_episode_returns: List[List[float]] = field(default_factory=list)
    all_episode_lengths: List[List[int]] = field(default_factory=list)
    all_per_episode: List[List[Dict[str, Any]]] = field(default_factory=list)

    # Buffer state (algorithm-specific, stored as dict for flexibility)
    buffer_state: Optional[Dict[str, Any]] = None

    # Metadata
    created_at: str = ""
    last_updated: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BatchEvalCheckpoint":
        """Create from dictionary."""
        return cls(**d)


def get_batch_checkpoint_path(out_dir: Path, algo: str, track: str, batch_id: str) -> Path:
    """Get path to batch checkpoint file.

    Args:
        out_dir: Output directory
        algo: Algorithm name
        track: Track name
        batch_id: Unique batch identifier (e.g., hash of task_ids or first task_id)

    Returns:
        Path to checkpoint file
    """
    checkpoint_dir = out_dir / algo / track / "_checkpoints"
    return checkpoint_dir / f"batch_{batch_id}.ckpt"


def save_batch_checkpoint(checkpoint: BatchEvalCheckpoint, checkpoint_path: Path):
    """Save batch checkpoint to disk.

    Uses pickle for efficient numpy array serialization.

    Args:
        checkpoint: Checkpoint to save
        checkpoint_path: Path to save to
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.last_updated = datetime.utcnow().isoformat() + "Z"

    # Use temporary file + rename for atomic writes
    temp_path = checkpoint_path.with_suffix(".tmp")
    try:
        with open(temp_path, "wb") as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Atomic rename
        temp_path.replace(checkpoint_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def load_batch_checkpoint(checkpoint_path: Path) -> Optional[BatchEvalCheckpoint]:
    """Load batch checkpoint from disk.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Checkpoint if valid, None otherwise
    """
    if not checkpoint_path.exists():
        return None
    try:
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        # Validate schema version
        if checkpoint.schema_version != CHECKPOINT_SCHEMA_VERSION:
            log.warning(f"Checkpoint schema version mismatch: {checkpoint.schema_version} vs {CHECKPOINT_SCHEMA_VERSION}")
            return None
        return checkpoint
    except Exception as e:
        log.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return None


def delete_batch_checkpoint(checkpoint_path: Path):
    """Delete batch checkpoint file.

    Called after successful completion of batch evaluation.

    Args:
        checkpoint_path: Path to checkpoint file
    """
    try:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    except Exception as e:
        log.warning(f"Failed to delete checkpoint {checkpoint_path}: {e}")


def generate_batch_id(task_entries: List) -> str:
    """Generate a unique batch ID from task entries.

    Uses the first task_id and batch size to create a deterministic ID.

    Args:
        task_entries: List of TaskEntry objects

    Returns:
        Unique batch identifier string
    """
    if not task_entries:
        return "empty"
    # Use first task_id and count to create deterministic ID
    first_id = task_entries[0].task_id
    count = len(task_entries)
    return f"{first_id}_n{count}"


# ==============================================================================
# Configuration Dataclasses
# ==============================================================================

@dataclass
class EvalConfig:
    """Global evaluation configuration."""
    # Algorithms to evaluate
    algos: List[str] = field(default_factory=lambda: ["dpt", "ad"])

    # Tracks to evaluate
    tracks: List[str] = field(default_factory=lambda: ["teammate", "layout"])

    # Layout and teammate filtering (None = all)
    layouts: Optional[List[str]] = None  # Filter by specific layouts
    teammates: Optional[List[str]] = None  # Filter by specific teammate families

    # Evaluation parameters
    num_episodes: int = 100
    max_steps: int = 100
    seed: int = 0
    num_seeds: int = 1  # Number of seeds to run (runs seed, seed+1, ..., seed+num_seeds-1)
    greedy: bool = True

    # Model checkpoints (algo -> checkpoint path)
    checkpoints: Dict[str, str] = field(default_factory=dict)

    # Model configs (algo -> config path, auto-detected if None)
    configs: Dict[str, Optional[str]] = field(default_factory=dict)

    # Context settings
    context_len: int = 4096  # For AD
    context_episodes: int = 10  # For DPT

    # Teammate action conditioning (optional feature)
    use_teammate_actions: bool = False  # If True, include teammate actions in context

    # Output
    out_dir: str = "results"

    # Task limits
    max_tasks_per_track: Optional[int] = None

    # Resume/force
    force: bool = False  # Re-evaluate even if result exists

    # GPU
    gpu: Optional[str] = None  # GPU device ID(s) to use

    # Parallel evaluation
    run_parallel: bool = False  # Run vectorized batch evaluation within teammate_family
    parallel_batch_size: int = 4  # Batch size for vectorized evaluation when run_parallel=True

    def validate(self):
        """Validate configuration."""
        for algo in self.algos:
            if algo not in SUPPORTED_ALGOS:
                raise ValueError(f"Unsupported algorithm: {algo}. Supported: {SUPPORTED_ALGOS}")
            if algo in ("dpt", "ad", "amago_offline", "hybrid_ad") and algo not in self.checkpoints:
                raise ValueError(f"Checkpoint required for {algo}. Use --checkpoint_{algo} <path>")

        for track in self.tracks:
            if track not in SUPPORTED_TRACKS:
                raise ValueError(f"Unsupported track: {track}. Supported: {SUPPORTED_TRACKS}")


@dataclass
class TaskResult:
    """Per-task evaluation result schema."""
    task_id: str
    track: str
    layout_name: str
    teammate_family: str
    num_episodes: int

    # Metrics
    mean_return: float
    std_return: float
    stderr_return: float
    min_return: float
    max_return: float
    median_return: float
    success_rate: float
    mean_episode_length: float

    # Per-episode details
    per_episode: List[Dict[str, Any]]

    # AUC over episode index (learning curve metric)
    auc: float

    # Context config
    context_config: Dict[str, Any]

    # Metadata
    algorithm: str
    eval_seed: int
    evaluated_at: str
    schema_version: int = RESULT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskResult":
        """Create from dictionary, handling missing optional fields gracefully."""
        # Compute stderr_return if missing
        num_episodes = d["num_episodes"]
        stderr_return = d.get("stderr_return")
        if stderr_return is None:
            stderr_return = d["std_return"] / np.sqrt(num_episodes) if num_episodes > 0 else 0.0

        return cls(
            task_id=d["task_id"],
            track=d["track"],
            layout_name=d["layout_name"],
            teammate_family=d["teammate_family"],
            num_episodes=num_episodes,
            mean_return=d["mean_return"],
            std_return=d["std_return"],
            stderr_return=stderr_return,
            min_return=d.get("min_return", 0.0),
            max_return=d.get("max_return", 0.0),
            median_return=d.get("median_return", d["mean_return"]),
            success_rate=d["success_rate"],
            mean_episode_length=d["mean_episode_length"],
            per_episode=d.get("per_episode", []),
            auc=d.get("auc", 0.0),
            context_config=d.get("context_config", {}),
            algorithm=d["algorithm"],
            eval_seed=d.get("eval_seed", 0),
            evaluated_at=d.get("evaluated_at", ""),
            schema_version=d.get("schema_version", RESULT_SCHEMA_VERSION),
        )


# ==============================================================================
# Utility Functions
# ==============================================================================

def get_git_commit() -> Optional[str]:
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return None


def parse_checkpoint_step(ckpt_path: Path) -> Optional[int]:
    """Parse checkpoint step number from checkpoint path.

    Handles checkpoint names like:
    - checkpoint_100000 -> 100000
    - checkpoint_final -> None
    - model.ckpt -> None

    Args:
        ckpt_path: Path to checkpoint file or directory

    Returns:
        Step number if parseable, None otherwise
    """
    name = ckpt_path.name
    if "_" not in name:
        return None
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return None


def discover_layouts_and_teammates(track: str) -> Dict[str, List[str]]:
    """Discover all layouts and their available teammates for a track.

    Returns:
        Dict mapping layout_name -> list of teammate families
    """
    track_dir = TRACK_DIRS.get(track)
    if track_dir is None or not track_dir.exists():
        return {}

    result = {}
    for layout_dir in track_dir.iterdir():
        if not layout_dir.is_dir():
            continue
        layout_name = layout_dir.name
        teammates = []
        for teammate_dir in layout_dir.iterdir():
            if not teammate_dir.is_dir():
                continue
            # Check if manifest_test.jsonl exists
            manifest_path = teammate_dir / "manifest_test.jsonl"
            if manifest_path.exists():
                teammates.append(teammate_dir.name)
        if teammates:
            result[layout_name] = sorted(teammates)
    return result


def get_manifest_path(track: str, layout: str, teammate: str, split: str = "test") -> Path:
    """Get manifest path for a track, layout, and teammate."""
    track_dir = TRACK_DIRS.get(track)
    if track_dir is None:
        raise ValueError(f"Unknown track: {track}")
    manifest_name = MANIFEST_NAMES.get(split)
    if manifest_name is None:
        raise ValueError(f"Unknown split: {split}")
    return track_dir / layout / teammate / manifest_name


def load_test_manifests(
    track: str,
    layouts: Optional[List[str]] = None,
    teammates: Optional[List[str]] = None,
) -> Dict[str, Dict[str, List[TaskEntry]]]:
    """Load test manifests for a track, organized by layout and teammate.

    Args:
        track: Track name ("teammate" or "layout")
        layouts: Optional list of layouts to filter by (None = all)
        teammates: Optional list of teammate families to filter by (None = all)

    Returns:
        Nested dict: layout_name -> teammate_family -> list of TaskEntry
    """
    available = discover_layouts_and_teammates(track)
    if not available:
        log.warning(f"No layouts found for track '{track}'")
        return {}

    result = {}
    for layout_name, available_teammates in available.items():
        # Filter by layouts if specified
        if layouts is not None and layout_name not in layouts:
            continue

        layout_tasks = {}
        for teammate_family in available_teammates:
            # Filter by teammates if specified
            if teammates is not None and teammate_family not in teammates:
                continue

            manifest_path = get_manifest_path(track, layout_name, teammate_family, "test")
            if not manifest_path.exists():
                continue

            try:
                tasks = load_manifest(str(manifest_path))
                if tasks:
                    layout_tasks[teammate_family] = tasks
            except Exception as e:
                log.warning(f"Failed to load manifest {manifest_path}: {e}")

        if layout_tasks:
            result[layout_name] = layout_tasks

    return result


def load_test_manifest(track: str) -> List[TaskEntry]:
    """Load test manifest for a track (DEPRECATED - use load_test_manifests instead).

    This is kept for backward compatibility but now loads all manifests.
    """
    all_tasks = []
    manifests = load_test_manifests(track)
    for layout_name, teammate_tasks in manifests.items():
        for teammate_family, tasks in teammate_tasks.items():
            all_tasks.extend(tasks)
    return all_tasks


def get_result_path(out_dir: Path, algo: str, track: str, layout: str, teammate: str, task_id: str) -> Path:
    """Get path to per-task result JSON, organized by layout and teammate."""
    return out_dir / algo / track / layout / teammate / f"{task_id}.json"


def result_exists(result_path: Path) -> bool:
    """Check if a valid result file exists."""
    if not result_path.exists():
        return False
    try:
        with open(result_path, "r") as f:
            data = json.load(f)
        # Basic validation
        return (
            "task_id" in data and
            "mean_return" in data and
            "schema_version" in data
        )
    except Exception:
        return False


def save_result(result: TaskResult, result_path: Path):
    """Save task result to JSON file."""
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)


def create_teammate(task_entry: TaskEntry, env) -> Any:
    """Create teammate policy from task entry.

    Reuses logic from benchmarks/baselines/{dpt,ad}/eval.py.
    """
    teammate_spec = task_entry.teammate

    if teammate_spec.kind == "heuristic":
        if teammate_spec.theta is not None:
            theta = theta_from_json(teammate_spec.family, teammate_spec.theta)
        else:
            theta_spec = ThetaSpec(
                family=teammate_spec.family,
                theta_id=teammate_spec.theta_id,
                split=task_entry.split,
                base_seed=teammate_spec.base_seed,
            )
            theta = sample_theta(theta_spec)

        spec = HeuristicTeammateSpec(
            family=teammate_spec.family,
            theta=theta,
            use_log_wrapper=True,
            start_cooking_interaction=False,
        )
        return make_teammate(spec, env, "agent_1")

    elif teammate_spec.kind == "rl":
        if teammate_spec.ckpt is None:
            raise ValueError("RL teammate requires checkpoint")

        spec = RLTeammateSpec(
            algo=teammate_spec.family,
            ckpt_path=teammate_spec.ckpt,
            use_log_wrapper=True,
            extra=teammate_spec.extra or {},
        )
        return make_teammate(spec, env, "agent_1")

    else:
        raise ValueError(f"Unknown teammate kind: {teammate_spec.kind}")


def create_env_for_task(task_entry: TaskEntry, max_steps: int = 100):
    """Create environment for a task."""
    env_kwargs = {
        "layout": task_entry.layout_name,
        "max_steps": max_steps,
    }

    env = make_env("overcooked-v2", env_kwargs)
    env = LogWrapper(env)
    return env


# ==============================================================================
# Random Baseline Agent
# ==============================================================================

class RandomAgent:
    """Random action baseline."""

    def __init__(self, num_actions: int = 6):
        self.num_actions = num_actions

    def get_action(self, rng: jax.Array) -> int:
        """Get random action."""
        return int(jax.random.randint(rng, (), 0, self.num_actions))


# ==============================================================================
# DPT Evaluation
# ==============================================================================

def load_dpt_model(checkpoint_path: str, config_path: Optional[str] = None):
    """Load DPT model from checkpoint."""
    from benchmarks.baselines.dpt.model import DPTModel, create_dpt_model
    from benchmarks.baselines.dpt.train import TrainConfig, TrainState
    from flax.training import checkpoints
    import optax

    # Convert to absolute path (orbax requires absolute paths)
    ckpt_path = Path(checkpoint_path).resolve()

    # Find config
    if config_path is None:
        parent = ckpt_path.parent.parent
        config_file = parent / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Could not find config.json at {config_file}")
        config_path = str(config_file)

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Filter to only known TrainConfig fields (ignore extra fields like 'layout')
    filtered_config = _filter_dataclass_fields(TrainConfig, config_dict)
    # Convert obs_shape from list to tuple (JSON doesn't preserve tuples)
    if "obs_shape" in filtered_config and isinstance(filtered_config["obs_shape"], list):
        filtered_config["obs_shape"] = tuple(filtered_config["obs_shape"])
    config = TrainConfig(**filtered_config)
    model_config = config.to_model_config()
    model, dummy_params = create_dpt_model(model_config)

    tx = optax.adam(1e-4)
    dummy_state = TrainState.create(
        apply_fn=model.apply,
        params=dummy_params,
        tx=tx,
    )

    state = checkpoints.restore_checkpoint(
        ckpt_dir=str(ckpt_path.parent),
        target=dummy_state,
        step=parse_checkpoint_step(ckpt_path),
    )

    # Return model_config (DPTConfig) not config (TrainConfig)
    return model, state.params, model_config


def evaluate_dpt_task(
    model,
    params,
    task_entry: TaskEntry,
    config: EvalConfig,
    model_config,
    rng: jax.Array,
) -> TaskResult:
    """Evaluate DPT on a single task."""
    from benchmarks.baselines.dpt.buffer import EpisodeBuffer

    env = create_env_for_task(task_entry, max_steps=config.max_steps)
    teammate = create_teammate(task_entry, env)

    # Use obs_shape from model_config (DPTConfig has obs_shape, not obs_dim)
    obs_shape = model_config.obs_shape
    buffer = EpisodeBuffer(
        max_episodes=config.context_episodes,
        max_steps=config.max_steps,
        obs_shape=obs_shape,
        use_teammate_actions=config.use_teammate_actions,
    )

    episode_returns = []
    episode_lengths = []
    per_episode = []

    for ep_idx in range(config.num_episodes):
        rng, reset_rng, step_rng = jax.random.split(rng, 3)

        obs, env_state = env.reset(reset_rng)
        done = {k: False for k in env.agents + ["__all__"]}

        # Initialize teammate carry state
        rng, init_rng = jax.random.split(step_rng)
        step_rng = rng
        teammate_carry = teammate.init(init_rng)

        episode_return = 0.0
        step_count = 0

        for step in range(config.max_steps):
            ego_obs = np.array(obs["agent_0"])
            ctx_obs, ctx_actions, ctx_next_obs, ctx_rewards, ctx_teammate_actions = buffer.get_context(model_config.seq_len)

            if config.greedy:
                action = model.get_action(
                    params,
                    jnp.array(ego_obs),
                    jnp.array(ctx_obs),
                    jnp.array(ctx_actions),
                    jnp.array(ctx_next_obs),
                    jnp.array(ctx_rewards),
                    greedy=True,
                    context_teammate_actions=jnp.array(ctx_teammate_actions) if ctx_teammate_actions is not None else None,
                )
            else:
                rng, sample_rng = jax.random.split(step_rng)
                step_rng = rng
                action = model.get_action(
                    params,
                    jnp.array(ego_obs),
                    jnp.array(ctx_obs),
                    jnp.array(ctx_actions),
                    jnp.array(ctx_next_obs),
                    jnp.array(ctx_rewards),
                    rng=sample_rng,
                    greedy=False,
                    context_teammate_actions=jnp.array(ctx_teammate_actions) if ctx_teammate_actions is not None else None,
                )

            action = int(action)

            # Teammate action
            avail_actions_1 = env.get_avail_actions(env_state.env_state)["agent_1"]
            rng, teammate_rng = jax.random.split(step_rng)
            step_rng = rng
            teammate_carry, teammate_action = teammate.act(
                teammate_carry,
                obs["agent_1"],
                jnp.array(done["agent_1"]),
                teammate_rng,
                env_state=env_state,  # Pass full LogEnvState, teammate code unwraps it
                avail_actions=avail_actions_1,
            )
            teammate_action = int(jnp.asarray(teammate_action).squeeze())

            # Step environment
            rng, step_env_rng = jax.random.split(step_rng)
            step_rng = rng
            env_actions = {"agent_0": action, "agent_1": teammate_action}
            next_obs, env_state, rewards, done_next, info = env.step(step_env_rng, env_state, env_actions)

            next_ego_obs = np.array(next_obs["agent_0"])
            reward = float(rewards["agent_0"])
            buffer.add_transition(
                ego_obs,
                action,
                next_ego_obs,
                reward,
                teammate_action=teammate_action if config.use_teammate_actions else None,
            )

            episode_return += reward
            step_count += 1

            obs = next_obs
            done = done_next

            if done["__all__"]:
                break

        buffer.finish_episode()

        episode_returns.append(episode_return)
        episode_lengths.append(step_count)
        per_episode.append({
            "episode_id": ep_idx,
            "return": episode_return,
            "length": step_count,
            "success": episode_return > 0,
        })

    returns = np.array(episode_returns)

    return TaskResult(
        task_id=task_entry.task_id,
        track=task_entry.track,
        layout_name=task_entry.layout_name,
        teammate_family=task_entry.teammate.family,
        num_episodes=config.num_episodes,
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        stderr_return=float(np.std(returns) / np.sqrt(len(returns))),
        min_return=float(np.min(returns)),
        max_return=float(np.max(returns)),
        median_return=float(np.median(returns)),
        success_rate=float(np.mean([e["success"] for e in per_episode])),
        mean_episode_length=float(np.mean(episode_lengths)),
        per_episode=per_episode,
        auc=float(np.sum(returns)),
        context_config={
            "algorithm": "dpt",
            "context_episodes": config.context_episodes,
            "context_update": "episode-level",
            "use_teammate_actions": config.use_teammate_actions,
        },
        algorithm="dpt",
        eval_seed=config.seed,
        evaluated_at=datetime.utcnow().isoformat() + "Z",
    )


# ==============================================================================
# AD Evaluation
# ==============================================================================

def _filter_dataclass_fields(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Filter config_dict to only include fields that exist in the dataclass.

    This allows loading checkpoints even when config.json has extra fields
    that were added during training but aren't part of the TrainConfig.
    """
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
    # Log any ignored fields for debugging
    ignored = set(config_dict.keys()) - valid_fields
    if ignored:
        log.debug(f"Ignoring unknown config fields: {ignored}")
    return filtered


def load_ad_model(checkpoint_path: str, config_path: Optional[str] = None):
    """Load AD model from checkpoint."""
    from benchmarks.baselines.ad.model import ADModel, create_ad_model
    from benchmarks.baselines.ad.train import TrainConfig, TrainState
    from flax.training import checkpoints
    import optax

    # Convert to absolute path (orbax requires absolute paths)
    ckpt_path = Path(checkpoint_path).resolve()

    if config_path is None:
        parent = ckpt_path.parent.parent
        config_file = parent / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Could not find config.json at {config_file}")
        config_path = str(config_file)

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Filter to only known TrainConfig fields (ignore extra fields like 'layout')
    filtered_config = _filter_dataclass_fields(TrainConfig, config_dict)
    # Convert obs_shape from list to tuple (JSON doesn't preserve tuples)
    if "obs_shape" in filtered_config and isinstance(filtered_config["obs_shape"], list):
        filtered_config["obs_shape"] = tuple(filtered_config["obs_shape"])
    config = TrainConfig(**filtered_config)
    model_config = config.to_model_config()
    model, dummy_params = create_ad_model(model_config)

    tx = optax.adam(1e-4)
    dummy_state = TrainState.create(
        apply_fn=model.apply,
        params=dummy_params,
        tx=tx,
    )

    state = checkpoints.restore_checkpoint(
        ckpt_dir=str(ckpt_path.parent),
        target=dummy_state,
        step=parse_checkpoint_step(ckpt_path),
    )

    # Return model_config (ADConfig) not config (TrainConfig)
    return model, state.params, model_config


def evaluate_ad_task(
    model,
    params,
    task_entry: TaskEntry,
    config: EvalConfig,
    model_config,
    rng: jax.Array,
) -> TaskResult:
    """Evaluate AD on a single task with step-level context updates."""
    from benchmarks.baselines.ad.buffer import OnlineBuffer

    env = create_env_for_task(task_entry, max_steps=config.max_steps)
    teammate = create_teammate(task_entry, env)

    # Use obs_shape from model_config (ADConfig has obs_shape, not obs_dim)
    obs_shape = model_config.obs_shape
    buffer = OnlineBuffer(
        max_len=config.context_len,
        obs_shape=obs_shape,
        use_teammate_actions=config.use_teammate_actions,
    )

    episode_returns = []
    episode_lengths = []
    per_episode = []

    for ep_idx in range(config.num_episodes):
        rng, reset_rng, step_rng = jax.random.split(rng, 3)

        obs, env_state = env.reset(reset_rng)
        done = {k: False for k in env.agents + ["__all__"]}

        # Initialize teammate carry state
        rng, init_rng = jax.random.split(step_rng)
        step_rng = rng
        teammate_carry = teammate.init(init_rng)

        episode_return = 0.0
        step_count = 0
        prev_action = 0
        prev_reward = 0.0
        prev_teammate_action = 0

        for step in range(config.max_steps):
            ego_obs = np.array(obs["agent_0"])

            # Add current step to buffer (AD step-level updates)
            buffer.add_step(
                obs=ego_obs,
                prev_action=prev_action,
                prev_reward=prev_reward,
                prev_teammate_action=prev_teammate_action if config.use_teammate_actions else None,
            )

            # Get context
            ctx_obs, ctx_prev_actions, ctx_prev_rewards, ctx_prev_teammate_actions = buffer.get_padded_context(
                target_len=min(buffer.current_len, config.context_len)
            )

            if len(ctx_obs) > model_config.seq_len:
                ctx_obs = ctx_obs[-model_config.seq_len:]
                ctx_prev_actions = ctx_prev_actions[-model_config.seq_len:]
                ctx_prev_rewards = ctx_prev_rewards[-model_config.seq_len:]
                if ctx_prev_teammate_actions is not None:
                    ctx_prev_teammate_actions = ctx_prev_teammate_actions[-model_config.seq_len:]

            if len(ctx_obs) == 0:
                ctx_obs = ego_obs[None, :]
                ctx_prev_actions = np.array([prev_action], dtype=np.int32)
                ctx_prev_rewards = np.array([prev_reward], dtype=np.float32)
                if config.use_teammate_actions:
                    ctx_prev_teammate_actions = np.array([prev_teammate_action], dtype=np.int32)

            # Get action
            if config.greedy:
                logits = model.apply(
                    params,
                    jnp.array(ctx_obs)[None, :, :],
                    jnp.array(ctx_prev_actions)[None, :],
                    jnp.array(ctx_prev_rewards)[None, :],
                    attention_mask=None,
                    prev_teammate_actions=jnp.array(ctx_prev_teammate_actions)[None, :] if ctx_prev_teammate_actions is not None else None,
                    train=False,
                )
                action = int(jnp.argmax(logits[0, -1, :]))
            else:
                rng, sample_rng = jax.random.split(step_rng)
                step_rng = rng
                logits = model.apply(
                    params,
                    jnp.array(ctx_obs)[None, :, :],
                    jnp.array(ctx_prev_actions)[None, :],
                    jnp.array(ctx_prev_rewards)[None, :],
                    attention_mask=None,
                    prev_teammate_actions=jnp.array(ctx_prev_teammate_actions)[None, :] if ctx_prev_teammate_actions is not None else None,
                    train=False,
                )
                action = int(jax.random.categorical(sample_rng, logits[0, -1, :]))

            # Teammate action
            avail_actions_1 = env.get_avail_actions(env_state.env_state)["agent_1"]
            rng, teammate_rng = jax.random.split(step_rng)
            step_rng = rng
            teammate_carry, teammate_action = teammate.act(
                teammate_carry,
                obs["agent_1"],
                jnp.array(done["agent_1"]),
                teammate_rng,
                env_state=env_state,  # Pass full LogEnvState, teammate code unwraps it
                avail_actions=avail_actions_1,
            )
            teammate_action = int(jnp.asarray(teammate_action).squeeze())

            # Step environment
            rng, step_env_rng = jax.random.split(step_rng)
            step_rng = rng
            env_actions = {"agent_0": action, "agent_1": teammate_action}
            next_obs, env_state, rewards, done_next, info = env.step(step_env_rng, env_state, env_actions)

            reward = float(rewards["agent_0"])
            episode_return += reward
            step_count += 1

            prev_action = action
            prev_reward = reward
            prev_teammate_action = teammate_action

            obs = next_obs
            done = done_next

            if done["__all__"]:
                break

        episode_returns.append(episode_return)
        episode_lengths.append(step_count)
        per_episode.append({
            "episode_id": ep_idx,
            "return": episode_return,
            "length": step_count,
            "success": episode_return > 0,
        })

    returns = np.array(episode_returns)

    return TaskResult(
        task_id=task_entry.task_id,
        track=task_entry.track,
        layout_name=task_entry.layout_name,
        teammate_family=task_entry.teammate.family,
        num_episodes=config.num_episodes,
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        stderr_return=float(np.std(returns) / np.sqrt(len(returns))),
        min_return=float(np.min(returns)),
        max_return=float(np.max(returns)),
        median_return=float(np.median(returns)),
        success_rate=float(np.mean([e["success"] for e in per_episode])),
        mean_episode_length=float(np.mean(episode_lengths)),
        per_episode=per_episode,
        auc=float(np.sum(returns)),
        context_config={
            "algorithm": "ad",
            "context_len": config.context_len,
            "context_update": "step-level",
            "use_teammate_actions": config.use_teammate_actions,
        },
        algorithm="ad",
        eval_seed=config.seed,
        evaluated_at=datetime.utcnow().isoformat() + "Z",
    )


# ==============================================================================
# AMAGO-offline Evaluation
# ==============================================================================

def load_amago_offline_model(checkpoint_path: str, config_path: Optional[str] = None):
    """Load AMAGO-offline model from checkpoint."""
    from benchmarks.baselines.amago_offline.model import AMAGOOfflineModel, create_amago_offline_model
    from benchmarks.baselines.amago_offline.train import TrainConfig, TrainState
    from flax.training import checkpoints
    import optax

    ckpt_path = Path(checkpoint_path).resolve()

    if config_path is None:
        parent = ckpt_path.parent.parent
        config_file = parent / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Could not find config.json at {config_file}")
        config_path = str(config_file)

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    filtered_config = _filter_dataclass_fields(TrainConfig, config_dict)
    if "obs_shape" in filtered_config and isinstance(filtered_config["obs_shape"], list):
        filtered_config["obs_shape"] = tuple(filtered_config["obs_shape"])
    config = TrainConfig(**filtered_config)
    model_config = config.to_model_config()
    model, dummy_params = create_amago_offline_model(model_config)

    tx = optax.adam(1e-4)
    dummy_state = TrainState.create(
        apply_fn=model.apply,
        params=dummy_params,
        tx=tx,
    )

    state = checkpoints.restore_checkpoint(
        ckpt_dir=str(ckpt_path.parent),
        target=dummy_state,
        step=parse_checkpoint_step(ckpt_path),
    )

    return model, state.params, model_config


def evaluate_amago_offline_task(
    model,
    params,
    task_entry: TaskEntry,
    config: EvalConfig,
    model_config,
    rng: jax.Array,
) -> TaskResult:
    """Evaluate AMAGO-offline on a single task with step-level context updates."""
    from benchmarks.baselines.ad.buffer import OnlineBuffer

    env = create_env_for_task(task_entry, max_steps=config.max_steps)
    teammate = create_teammate(task_entry, env)

    obs_shape = model_config.obs_shape
    buffer = OnlineBuffer(
        max_len=config.context_len,
        obs_shape=obs_shape,
        use_teammate_actions=config.use_teammate_actions,
    )

    episode_returns = []
    episode_lengths = []
    per_episode = []

    global_step = 0

    for ep_idx in range(config.num_episodes):
        rng, reset_rng, step_rng = jax.random.split(rng, 3)

        obs, env_state = env.reset(reset_rng)
        done = {k: False for k in env.agents + ["__all__"]}

        rng, init_rng = jax.random.split(step_rng)
        step_rng = rng
        teammate_carry = teammate.init(init_rng)

        episode_return = 0.0
        step_count = 0
        prev_action = 0
        prev_reward = 0.0
        prev_teammate_action = 0

        for step in range(config.max_steps):
            ego_obs = np.array(obs["agent_0"])

            buffer.add_step(
                obs=ego_obs,
                prev_action=prev_action,
                prev_reward=prev_reward,
                prev_teammate_action=prev_teammate_action if config.use_teammate_actions else None,
            )

            ctx_obs, ctx_prev_actions, ctx_prev_rewards, ctx_prev_teammate_actions = buffer.get_padded_context(
                target_len=min(buffer.current_len, config.context_len)
            )

            if len(ctx_obs) > model_config.seq_len:
                ctx_obs = ctx_obs[-model_config.seq_len:]
                ctx_prev_actions = ctx_prev_actions[-model_config.seq_len:]
                ctx_prev_rewards = ctx_prev_rewards[-model_config.seq_len:]
                if ctx_prev_teammate_actions is not None:
                    ctx_prev_teammate_actions = ctx_prev_teammate_actions[-model_config.seq_len:]

            if len(ctx_obs) == 0:
                ctx_obs = ego_obs[None, :]
                ctx_prev_actions = np.array([prev_action], dtype=np.int32)
                ctx_prev_rewards = np.array([prev_reward], dtype=np.float32)
                if config.use_teammate_actions:
                    ctx_prev_teammate_actions = np.array([prev_teammate_action], dtype=np.int32)

            ctx_len = len(ctx_obs)
            ctx_dones = np.zeros(ctx_len, dtype=np.float32)
            ctx_time_idxs = np.arange(max(0, global_step - ctx_len + 1), global_step + 1, dtype=np.int32)
            if len(ctx_time_idxs) != ctx_len:
                ctx_time_idxs = np.arange(ctx_len, dtype=np.int32)

            logits = model.apply(
                params,
                jnp.array(ctx_obs)[None, :, :],
                jnp.array(ctx_prev_actions)[None, :],
                jnp.array(ctx_prev_rewards)[None, :],
                jnp.array(ctx_dones)[None, :],
                jnp.array(ctx_time_idxs)[None, :],
                attention_mask=None,
                prev_teammate_actions=jnp.array(ctx_prev_teammate_actions)[None, :] if ctx_prev_teammate_actions is not None else None,
                train=False,
            )

            if config.greedy:
                action = int(jnp.argmax(logits[0, -1, :]))
            else:
                rng, sample_rng = jax.random.split(step_rng)
                step_rng = rng
                action = int(jax.random.categorical(sample_rng, logits[0, -1, :]))

            avail_actions_1 = env.get_avail_actions(env_state.env_state)["agent_1"]
            rng, teammate_rng = jax.random.split(step_rng)
            step_rng = rng
            teammate_carry, teammate_action = teammate.act(
                teammate_carry,
                obs["agent_1"],
                jnp.array(done["agent_1"]),
                teammate_rng,
                env_state=env_state,
                avail_actions=avail_actions_1,
            )
            teammate_action = int(jnp.asarray(teammate_action).squeeze())

            rng, step_env_rng = jax.random.split(step_rng)
            step_rng = rng
            env_actions = {"agent_0": action, "agent_1": teammate_action}
            next_obs, env_state, rewards, done_next, info = env.step(step_env_rng, env_state, env_actions)

            reward = float(rewards["agent_0"])
            episode_return += reward
            step_count += 1
            global_step += 1

            prev_action = action
            prev_reward = reward
            prev_teammate_action = teammate_action

            obs = next_obs
            done = done_next

            if done["__all__"]:
                break

        episode_returns.append(episode_return)
        episode_lengths.append(step_count)
        per_episode.append({
            "episode_id": ep_idx,
            "return": episode_return,
            "length": step_count,
            "success": episode_return > 0,
        })

    returns = np.array(episode_returns)

    return TaskResult(
        task_id=task_entry.task_id,
        track=task_entry.track,
        layout_name=task_entry.layout_name,
        teammate_family=task_entry.teammate.family,
        num_episodes=config.num_episodes,
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        stderr_return=float(np.std(returns) / np.sqrt(len(returns))),
        min_return=float(np.min(returns)),
        max_return=float(np.max(returns)),
        median_return=float(np.median(returns)),
        success_rate=float(np.mean([e["success"] for e in per_episode])),
        mean_episode_length=float(np.mean(episode_lengths)),
        per_episode=per_episode,
        auc=float(np.sum(returns)),
        context_config={
            "algorithm": "amago_offline",
            "context_update": "step-level/cross-episode",
            "seq_len": model_config.seq_len,
            "use_teammate_actions": config.use_teammate_actions,
            "implementation": "amago_style_in_project",
        },
        algorithm="amago_offline",
        eval_seed=config.seed,
        evaluated_at=datetime.utcnow().isoformat() + "Z",
    )


# ==============================================================================
# Hybrid-AD Evaluation
# ==============================================================================

def load_hybrid_ad_model(checkpoint_path: str, config_path: Optional[str] = None):
    """Load Hybrid-AD model from checkpoint."""
    from benchmarks.baselines.hybrid_ad.model import HybridADModel, create_hybrid_ad_model
    from benchmarks.baselines.hybrid_ad.train import TrainConfig, TrainState
    from flax.training import checkpoints
    import optax

    ckpt_path = Path(checkpoint_path).resolve()

    if config_path is None:
        parent = ckpt_path.parent.parent
        config_file = parent / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Could not find config.json at {config_file}")
        config_path = str(config_file)

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    filtered_config = _filter_dataclass_fields(TrainConfig, config_dict)
    if "obs_shape" in filtered_config and isinstance(filtered_config["obs_shape"], list):
        filtered_config["obs_shape"] = tuple(filtered_config["obs_shape"])
    config = TrainConfig(**filtered_config)
    model_config = config.to_model_config()
    model, dummy_params = create_hybrid_ad_model(model_config)

    tx = optax.adam(1e-4)
    dummy_state = TrainState.create(
        apply_fn=model.apply,
        params=dummy_params,
        tx=tx,
    )

    state = checkpoints.restore_checkpoint(
        ckpt_dir=str(ckpt_path.parent),
        target=dummy_state,
        step=parse_checkpoint_step(ckpt_path),
    )

    return model, state.params, model_config


def evaluate_hybrid_ad_task(
    model,
    params,
    task_entry: TaskEntry,
    config: EvalConfig,
    model_config,
    rng: jax.Array,
) -> TaskResult:
    """Evaluate Hybrid-AD on a single task with step-level context updates (same as AD)."""
    from benchmarks.baselines.ad.buffer import OnlineBuffer

    env = create_env_for_task(task_entry, max_steps=config.max_steps)
    teammate = create_teammate(task_entry, env)

    obs_shape = model_config.obs_shape
    buffer = OnlineBuffer(
        max_len=config.context_len,
        obs_shape=obs_shape,
        use_teammate_actions=config.use_teammate_actions,
    )

    episode_returns = []
    episode_lengths = []
    per_episode = []

    for ep_idx in range(config.num_episodes):
        rng, reset_rng, step_rng = jax.random.split(rng, 3)

        obs, env_state = env.reset(reset_rng)
        done = {k: False for k in env.agents + ["__all__"]}

        rng, init_rng = jax.random.split(step_rng)
        step_rng = rng
        teammate_carry = teammate.init(init_rng)

        episode_return = 0.0
        step_count = 0
        prev_action = 0
        prev_reward = 0.0
        prev_teammate_action = 0

        for step in range(config.max_steps):
            ego_obs = np.array(obs["agent_0"])

            buffer.add_step(
                obs=ego_obs,
                prev_action=prev_action,
                prev_reward=prev_reward,
                prev_teammate_action=prev_teammate_action if config.use_teammate_actions else None,
            )

            ctx_obs, ctx_prev_actions, ctx_prev_rewards, ctx_prev_teammate_actions = buffer.get_padded_context(
                target_len=min(buffer.current_len, config.context_len)
            )

            if len(ctx_obs) > model_config.seq_len:
                ctx_obs = ctx_obs[-model_config.seq_len:]
                ctx_prev_actions = ctx_prev_actions[-model_config.seq_len:]
                ctx_prev_rewards = ctx_prev_rewards[-model_config.seq_len:]
                if ctx_prev_teammate_actions is not None:
                    ctx_prev_teammate_actions = ctx_prev_teammate_actions[-model_config.seq_len:]

            if len(ctx_obs) == 0:
                ctx_obs = ego_obs[None, :]
                ctx_prev_actions = np.array([prev_action], dtype=np.int32)
                ctx_prev_rewards = np.array([prev_reward], dtype=np.float32)
                if config.use_teammate_actions:
                    ctx_prev_teammate_actions = np.array([prev_teammate_action], dtype=np.int32)

            if config.greedy:
                logits = model.apply(
                    params,
                    jnp.array(ctx_obs)[None, :, :],
                    jnp.array(ctx_prev_actions)[None, :],
                    jnp.array(ctx_prev_rewards)[None, :],
                    attention_mask=None,
                    prev_teammate_actions=jnp.array(ctx_prev_teammate_actions)[None, :] if ctx_prev_teammate_actions is not None else None,
                    train=False,
                )
                action = int(jnp.argmax(logits[0, -1, :]))
            else:
                rng, sample_rng = jax.random.split(step_rng)
                step_rng = rng
                logits = model.apply(
                    params,
                    jnp.array(ctx_obs)[None, :, :],
                    jnp.array(ctx_prev_actions)[None, :],
                    jnp.array(ctx_prev_rewards)[None, :],
                    attention_mask=None,
                    prev_teammate_actions=jnp.array(ctx_prev_teammate_actions)[None, :] if ctx_prev_teammate_actions is not None else None,
                    train=False,
                )
                action = int(jax.random.categorical(sample_rng, logits[0, -1, :]))

            avail_actions_1 = env.get_avail_actions(env_state.env_state)["agent_1"]
            rng, teammate_rng = jax.random.split(step_rng)
            step_rng = rng
            teammate_carry, teammate_action = teammate.act(
                teammate_carry,
                obs["agent_1"],
                jnp.array(done["agent_1"]),
                teammate_rng,
                env_state=env_state,
                avail_actions=avail_actions_1,
            )
            teammate_action = int(jnp.asarray(teammate_action).squeeze())

            rng, step_env_rng = jax.random.split(step_rng)
            step_rng = rng
            env_actions = {"agent_0": action, "agent_1": teammate_action}
            next_obs, env_state, rewards, done_next, info = env.step(step_env_rng, env_state, env_actions)

            reward = float(rewards["agent_0"])
            episode_return += reward
            step_count += 1

            prev_action = action
            prev_reward = reward
            prev_teammate_action = teammate_action

            obs = next_obs
            done = done_next

            if done["__all__"]:
                break

        episode_returns.append(episode_return)
        episode_lengths.append(step_count)
        per_episode.append({
            "episode_id": ep_idx,
            "return": episode_return,
            "length": step_count,
            "success": episode_return > 0,
        })

    returns = np.array(episode_returns)

    return TaskResult(
        task_id=task_entry.task_id,
        track=task_entry.track,
        layout_name=task_entry.layout_name,
        teammate_family=task_entry.teammate.family,
        num_episodes=config.num_episodes,
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        stderr_return=float(np.std(returns) / np.sqrt(len(returns))),
        min_return=float(np.min(returns)),
        max_return=float(np.max(returns)),
        median_return=float(np.median(returns)),
        success_rate=float(np.mean([e["success"] for e in per_episode])),
        mean_episode_length=float(np.mean(episode_lengths)),
        per_episode=per_episode,
        auc=float(np.sum(returns)),
        context_config={
            "algorithm": "hybrid_ad",
            "context_update": "step-level rolling context",
            "temporal_backend": "cnn_gru",
            "supervision": "ad_action_prediction",
            "use_teammate_actions": config.use_teammate_actions,
        },
        algorithm="hybrid_ad",
        eval_seed=config.seed,
        evaluated_at=datetime.utcnow().isoformat() + "Z",
    )


# ==============================================================================
# Vectorized Batch Evaluation (DPT)
# ==============================================================================

def evaluate_dpt_batch(
    model,
    params,
    task_entries: List[TaskEntry],
    config: EvalConfig,
    model_config,
    rng: jax.Array,
    out_dir: Optional[Path] = None,
    track: Optional[str] = None,
) -> List[TaskResult]:
    """Evaluate DPT on multiple tasks in parallel using vectorized buffers.

    This function evaluates multiple tasks simultaneously by:
    1. Creating separate environments and teammates for each task
    2. Running episodes in a synchronized manner
    3. Batching model forward passes across all tasks for GPU efficiency

    Supports resume from checkpoint:
    - Saves checkpoint after each episode
    - On restart, loads checkpoint and resumes from where it left off
    - Checkpoint is deleted after successful completion

    Args:
        model: DPT model
        params: Model parameters
        task_entries: List of task entries to evaluate
        config: Evaluation configuration
        model_config: DPT model configuration
        rng: Random key
        out_dir: Output directory for checkpoints (optional, enables resume)
        track: Track name (optional, needed for checkpoint path)

    Returns:
        List of TaskResult, one per task
    """
    from benchmarks.baselines.dpt.buffer import VectorizedEpisodeBuffer

    if not task_entries:
        return []

    batch_size = len(task_entries)
    obs_shape = model_config.obs_shape

    # Create environments and teammates for each task
    envs = []
    teammates = []
    for task_entry in task_entries:
        env = create_env_for_task(task_entry, max_steps=config.max_steps)
        teammate = create_teammate(task_entry, env)
        envs.append(env)
        teammates.append(teammate)

    # Create vectorized buffer
    buffer = VectorizedEpisodeBuffer(
        batch_size=batch_size,
        max_episodes=config.context_episodes,
        max_steps=config.max_steps,
        obs_shape=obs_shape,
        use_teammate_actions=config.use_teammate_actions,
    )

    # Results storage per task
    all_episode_returns = [[] for _ in range(batch_size)]
    all_episode_lengths = [[] for _ in range(batch_size)]
    all_per_episode = [[] for _ in range(batch_size)]

    # Check for checkpoint (resume support)
    start_episode = 0
    checkpoint_path = None
    if out_dir is not None and track is not None:
        batch_id = generate_batch_id(task_entries)
        checkpoint_path = get_batch_checkpoint_path(out_dir, "dpt", track, batch_id)
        checkpoint = load_batch_checkpoint(checkpoint_path)

        if checkpoint is not None and not config.force:
            # Validate checkpoint matches current batch
            checkpoint_task_ids = set(checkpoint.task_ids)
            current_task_ids = set(t.task_id for t in task_entries)
            if checkpoint_task_ids == current_task_ids:
                # Resume from checkpoint
                start_episode = checkpoint.completed_episodes
                rng = jax.random.PRNGKey(checkpoint.rng_key[0])
                if len(checkpoint.rng_key) > 1:
                    # Handle split key format
                    rng = jnp.array(checkpoint.rng_key, dtype=jnp.uint32)

                # Restore results
                all_episode_returns = [list(r) for r in checkpoint.all_episode_returns]
                all_episode_lengths = [list(l) for l in checkpoint.all_episode_lengths]
                all_per_episode = [list(p) for p in checkpoint.all_per_episode]

                # Restore buffer state
                if checkpoint.buffer_state is not None:
                    buffer.set_state(checkpoint.buffer_state)

                log.info(f"        Resuming from episode {start_episode}/{config.num_episodes}")
            else:
                log.warning(f"        Checkpoint task mismatch, starting fresh")

    for ep_idx in range(start_episode, config.num_episodes):
        # Reset all environments
        rng, *reset_rngs = jax.random.split(rng, batch_size + 1)

        obs_list = []
        env_states = []
        done_list = []
        teammate_carries = []

        for i in range(batch_size):
            obs, env_state = envs[i].reset(reset_rngs[i])
            obs_list.append(obs)
            env_states.append(env_state)
            done_list.append({k: False for k in envs[i].agents + ["__all__"]})

            # Initialize teammate
            rng, init_rng = jax.random.split(rng)
            teammate_carry = teammates[i].init(init_rng)
            teammate_carries.append(teammate_carry)

        episode_returns = [0.0] * batch_size
        step_counts = [0] * batch_size
        task_done = [False] * batch_size

        for step in range(config.max_steps):
            # Prepare batched observations for active tasks
            active_indices = [i for i in range(batch_size) if not task_done[i]]
            if not active_indices:
                break

            # Get batched context from buffer
            ctx_obs, ctx_actions, ctx_next_obs, ctx_rewards, ctx_teammate_actions = buffer.get_context_batch(
                model_config.seq_len
            )

            # Get current observations for all tasks (pad inactive with zeros)
            batch_ego_obs = np.zeros((batch_size,) + obs_shape, dtype=np.float32)
            for i in active_indices:
                batch_ego_obs[i] = np.array(obs_list[i]["agent_0"])

            # Batched model forward pass
            if config.greedy:
                batch_actions = model.get_action(
                    params,
                    jnp.array(batch_ego_obs),
                    jnp.array(ctx_obs),
                    jnp.array(ctx_actions),
                    jnp.array(ctx_next_obs),
                    jnp.array(ctx_rewards),
                    greedy=True,
                    context_teammate_actions=jnp.array(ctx_teammate_actions) if ctx_teammate_actions is not None else None,
                )
            else:
                rng, sample_rng = jax.random.split(rng)
                batch_actions = model.get_action(
                    params,
                    jnp.array(batch_ego_obs),
                    jnp.array(ctx_obs),
                    jnp.array(ctx_actions),
                    jnp.array(ctx_next_obs),
                    jnp.array(ctx_rewards),
                    rng=sample_rng,
                    greedy=False,
                    context_teammate_actions=jnp.array(ctx_teammate_actions) if ctx_teammate_actions is not None else None,
                )

            batch_actions = np.array(batch_actions)

            # Step each environment individually (teammates are different)
            for i in active_indices:
                action = int(batch_actions[i])
                ego_obs = np.array(obs_list[i]["agent_0"])

                # Get teammate action
                avail_actions_1 = envs[i].get_avail_actions(env_states[i].env_state)["agent_1"]
                rng, teammate_rng = jax.random.split(rng)
                teammate_carries[i], teammate_action = teammates[i].act(
                    teammate_carries[i],
                    obs_list[i]["agent_1"],
                    jnp.array(done_list[i]["agent_1"]),
                    teammate_rng,
                    env_state=env_states[i],
                    avail_actions=avail_actions_1,
                )
                teammate_action = int(jnp.asarray(teammate_action).squeeze())

                # Step environment
                rng, step_env_rng = jax.random.split(rng)
                env_actions = {"agent_0": action, "agent_1": teammate_action}
                next_obs, env_state, rewards, done_next, info = envs[i].step(
                    step_env_rng, env_states[i], env_actions
                )

                next_ego_obs = np.array(next_obs["agent_0"])
                reward = float(rewards["agent_0"])

                # Add transition to buffer
                buffer.add_transition(
                    i,
                    ego_obs,
                    action,
                    next_ego_obs,
                    reward,
                    teammate_action=teammate_action if config.use_teammate_actions else None,
                )

                episode_returns[i] += reward
                step_counts[i] += 1

                obs_list[i] = next_obs
                env_states[i] = env_state
                done_list[i] = done_next

                if done_next["__all__"]:
                    task_done[i] = True
                    buffer.finish_episode(i)

        # Finish episodes for tasks that didn't terminate early
        for i in range(batch_size):
            if not task_done[i]:
                buffer.finish_episode(i)

            all_episode_returns[i].append(episode_returns[i])
            all_episode_lengths[i].append(step_counts[i])
            all_per_episode[i].append({
                "episode_id": ep_idx,
                "return": episode_returns[i],
                "length": step_counts[i],
                "success": episode_returns[i] > 0,
            })

        # Save checkpoint after each episode
        if checkpoint_path is not None:
            # Convert JAX rng to serializable format
            rng_as_list = np.array(rng).tolist()
            checkpoint = BatchEvalCheckpoint(
                algorithm="dpt",
                task_ids=[t.task_id for t in task_entries],
                completed_episodes=ep_idx + 1,
                total_episodes=config.num_episodes,
                rng_key=rng_as_list,
                all_episode_returns=all_episode_returns,
                all_episode_lengths=all_episode_lengths,
                all_per_episode=all_per_episode,
                buffer_state=buffer.get_state(),
                created_at=datetime.utcnow().isoformat() + "Z",
            )
            save_batch_checkpoint(checkpoint, checkpoint_path)

    # Build TaskResult for each task
    results = []
    for i, task_entry in enumerate(task_entries):
        returns = np.array(all_episode_returns[i])
        results.append(TaskResult(
            task_id=task_entry.task_id,
            track=task_entry.track,
            layout_name=task_entry.layout_name,
            teammate_family=task_entry.teammate.family,
            num_episodes=config.num_episodes,
            mean_return=float(np.mean(returns)),
            std_return=float(np.std(returns)),
            stderr_return=float(np.std(returns) / np.sqrt(len(returns))),
            min_return=float(np.min(returns)),
            max_return=float(np.max(returns)),
            median_return=float(np.median(returns)),
            success_rate=float(np.mean([e["success"] for e in all_per_episode[i]])),
            mean_episode_length=float(np.mean(all_episode_lengths[i])),
            per_episode=all_per_episode[i],
            auc=float(np.sum(returns)),
            context_config={
                "algorithm": "dpt",
                "context_episodes": config.context_episodes,
                "context_update": "episode-level",
                "use_teammate_actions": config.use_teammate_actions,
                "batch_evaluation": True,
            },
            algorithm="dpt",
            eval_seed=config.seed,
            evaluated_at=datetime.utcnow().isoformat() + "Z",
        ))

    # Delete checkpoint on successful completion
    if checkpoint_path is not None:
        delete_batch_checkpoint(checkpoint_path)

    return results


# ==============================================================================
# Vectorized Batch Evaluation (AD)
# ==============================================================================

def evaluate_ad_batch(
    model,
    params,
    task_entries: List[TaskEntry],
    config: EvalConfig,
    model_config,
    rng: jax.Array,
    out_dir: Optional[Path] = None,
    track: Optional[str] = None,
) -> List[TaskResult]:
    """Evaluate AD on multiple tasks in parallel using vectorized buffers.

    This function evaluates multiple tasks simultaneously by:
    1. Creating separate environments and teammates for each task
    2. Running episodes in a synchronized manner
    3. Batching model forward passes across all tasks for GPU efficiency

    Supports resume from checkpoint:
    - Saves checkpoint after each episode
    - On restart, loads checkpoint and resumes from where it left off
    - Checkpoint is deleted after successful completion

    Args:
        model: AD model
        params: Model parameters
        task_entries: List of task entries to evaluate
        config: Evaluation configuration
        model_config: AD model configuration
        rng: Random key
        out_dir: Output directory for checkpoints (optional, enables resume)
        track: Track name (optional, needed for checkpoint path)

    Returns:
        List of TaskResult, one per task
    """
    from benchmarks.baselines.ad.buffer import VectorizedOnlineBuffer

    if not task_entries:
        return []

    batch_size = len(task_entries)
    obs_shape = model_config.obs_shape

    # Create environments and teammates for each task
    envs = []
    teammates = []
    for task_entry in task_entries:
        env = create_env_for_task(task_entry, max_steps=config.max_steps)
        teammate = create_teammate(task_entry, env)
        envs.append(env)
        teammates.append(teammate)

    # Create vectorized buffer
    buffer = VectorizedOnlineBuffer(
        batch_size=batch_size,
        max_len=config.context_len,
        obs_shape=obs_shape,
        use_teammate_actions=config.use_teammate_actions,
    )

    # Results storage per task
    all_episode_returns = [[] for _ in range(batch_size)]
    all_episode_lengths = [[] for _ in range(batch_size)]
    all_per_episode = [[] for _ in range(batch_size)]

    # Check for checkpoint (resume support)
    start_episode = 0
    checkpoint_path = None
    if out_dir is not None and track is not None:
        batch_id = generate_batch_id(task_entries)
        checkpoint_path = get_batch_checkpoint_path(out_dir, "ad", track, batch_id)
        checkpoint = load_batch_checkpoint(checkpoint_path)

        if checkpoint is not None and not config.force:
            # Validate checkpoint matches current batch
            checkpoint_task_ids = set(checkpoint.task_ids)
            current_task_ids = set(t.task_id for t in task_entries)
            if checkpoint_task_ids == current_task_ids:
                # Resume from checkpoint
                start_episode = checkpoint.completed_episodes
                rng = jax.random.PRNGKey(checkpoint.rng_key[0])
                if len(checkpoint.rng_key) > 1:
                    # Handle split key format
                    rng = jnp.array(checkpoint.rng_key, dtype=jnp.uint32)

                # Restore results
                all_episode_returns = [list(r) for r in checkpoint.all_episode_returns]
                all_episode_lengths = [list(l) for l in checkpoint.all_episode_lengths]
                all_per_episode = [list(p) for p in checkpoint.all_per_episode]

                # Restore buffer state
                if checkpoint.buffer_state is not None:
                    buffer.set_state(checkpoint.buffer_state)

                log.info(f"        Resuming from episode {start_episode}/{config.num_episodes}")
            else:
                log.warning(f"        Checkpoint task mismatch, starting fresh")

    for ep_idx in range(start_episode, config.num_episodes):
        # Reset all environments
        rng, *reset_rngs = jax.random.split(rng, batch_size + 1)

        obs_list = []
        env_states = []
        done_list = []
        teammate_carries = []

        for i in range(batch_size):
            obs, env_state = envs[i].reset(reset_rngs[i])
            obs_list.append(obs)
            env_states.append(env_state)
            done_list.append({k: False for k in envs[i].agents + ["__all__"]})

            # Initialize teammate
            rng, init_rng = jax.random.split(rng)
            teammate_carry = teammates[i].init(init_rng)
            teammate_carries.append(teammate_carry)

        episode_returns = [0.0] * batch_size
        step_counts = [0] * batch_size
        task_done = [False] * batch_size
        prev_actions = [0] * batch_size
        prev_rewards = [0.0] * batch_size
        prev_teammate_actions = [0] * batch_size

        for step in range(config.max_steps):
            # Check for active tasks
            active_indices = [i for i in range(batch_size) if not task_done[i]]
            if not active_indices:
                break

            # Add current step to buffer for all active tasks (AD step-level updates)
            for i in active_indices:
                ego_obs = np.array(obs_list[i]["agent_0"])
                buffer.add_step(
                    i,
                    obs=ego_obs,
                    prev_action=prev_actions[i],
                    prev_reward=prev_rewards[i],
                    prev_teammate_action=prev_teammate_actions[i] if config.use_teammate_actions else None,
                )

            # Get batched padded context
            target_len = min(model_config.seq_len, config.context_len)
            ctx_obs, ctx_prev_actions, ctx_prev_rewards, ctx_prev_teammate_actions = buffer.get_padded_context_batch(
                target_len
            )

            # Prepare attention mask (optional, for variable-length sequences)
            # For simplicity, we use the full context length for all

            # Batched model forward pass
            if config.greedy:
                logits = model.apply(
                    params,
                    jnp.array(ctx_obs),
                    jnp.array(ctx_prev_actions),
                    jnp.array(ctx_prev_rewards),
                    attention_mask=None,
                    prev_teammate_actions=jnp.array(ctx_prev_teammate_actions) if ctx_prev_teammate_actions is not None else None,
                    train=False,
                )
                batch_actions = jnp.argmax(logits[:, -1, :], axis=-1)
            else:
                rng, sample_rng = jax.random.split(rng)
                logits = model.apply(
                    params,
                    jnp.array(ctx_obs),
                    jnp.array(ctx_prev_actions),
                    jnp.array(ctx_prev_rewards),
                    attention_mask=None,
                    prev_teammate_actions=jnp.array(ctx_prev_teammate_actions) if ctx_prev_teammate_actions is not None else None,
                    train=False,
                )
                batch_actions = jax.random.categorical(sample_rng, logits[:, -1, :])

            batch_actions = np.array(batch_actions)

            # Step each environment individually
            for i in active_indices:
                action = int(batch_actions[i])

                # Get teammate action
                avail_actions_1 = envs[i].get_avail_actions(env_states[i].env_state)["agent_1"]
                rng, teammate_rng = jax.random.split(rng)
                teammate_carries[i], teammate_action = teammates[i].act(
                    teammate_carries[i],
                    obs_list[i]["agent_1"],
                    jnp.array(done_list[i]["agent_1"]),
                    teammate_rng,
                    env_state=env_states[i],
                    avail_actions=avail_actions_1,
                )
                teammate_action = int(jnp.asarray(teammate_action).squeeze())

                # Step environment
                rng, step_env_rng = jax.random.split(rng)
                env_actions = {"agent_0": action, "agent_1": teammate_action}
                next_obs, env_state, rewards, done_next, info = envs[i].step(
                    step_env_rng, env_states[i], env_actions
                )

                reward = float(rewards["agent_0"])
                episode_returns[i] += reward
                step_counts[i] += 1

                # Update tracking for next step
                prev_actions[i] = action
                prev_rewards[i] = reward
                prev_teammate_actions[i] = teammate_action

                obs_list[i] = next_obs
                env_states[i] = env_state
                done_list[i] = done_next

                if done_next["__all__"]:
                    task_done[i] = True

        # Record episode results
        for i in range(batch_size):
            all_episode_returns[i].append(episode_returns[i])
            all_episode_lengths[i].append(step_counts[i])
            all_per_episode[i].append({
                "episode_id": ep_idx,
                "return": episode_returns[i],
                "length": step_counts[i],
                "success": episode_returns[i] > 0,
            })

        # Save checkpoint after each episode
        if checkpoint_path is not None:
            # Convert JAX rng to serializable format
            rng_as_list = np.array(rng).tolist()
            checkpoint = BatchEvalCheckpoint(
                algorithm="ad",
                task_ids=[t.task_id for t in task_entries],
                completed_episodes=ep_idx + 1,
                total_episodes=config.num_episodes,
                rng_key=rng_as_list,
                all_episode_returns=all_episode_returns,
                all_episode_lengths=all_episode_lengths,
                all_per_episode=all_per_episode,
                buffer_state=buffer.get_state(),
                created_at=datetime.utcnow().isoformat() + "Z",
            )
            save_batch_checkpoint(checkpoint, checkpoint_path)

    # Build TaskResult for each task
    results = []
    for i, task_entry in enumerate(task_entries):
        returns = np.array(all_episode_returns[i])
        results.append(TaskResult(
            task_id=task_entry.task_id,
            track=task_entry.track,
            layout_name=task_entry.layout_name,
            teammate_family=task_entry.teammate.family,
            num_episodes=config.num_episodes,
            mean_return=float(np.mean(returns)),
            std_return=float(np.std(returns)),
            stderr_return=float(np.std(returns) / np.sqrt(len(returns))),
            min_return=float(np.min(returns)),
            max_return=float(np.max(returns)),
            median_return=float(np.median(returns)),
            success_rate=float(np.mean([e["success"] for e in all_per_episode[i]])),
            mean_episode_length=float(np.mean(all_episode_lengths[i])),
            per_episode=all_per_episode[i],
            auc=float(np.sum(returns)),
            context_config={
                "algorithm": "ad",
                "context_len": config.context_len,
                "context_update": "step-level",
                "use_teammate_actions": config.use_teammate_actions,
                "batch_evaluation": True,
            },
            algorithm="ad",
            eval_seed=config.seed,
            evaluated_at=datetime.utcnow().isoformat() + "Z",
        ))

    # Delete checkpoint on successful completion
    if checkpoint_path is not None:
        delete_batch_checkpoint(checkpoint_path)

    return results


# ==============================================================================
# Random Baseline Evaluation
# ==============================================================================

def evaluate_random_task(
    task_entry: TaskEntry,
    config: EvalConfig,
    rng: jax.Array,
) -> TaskResult:
    """Evaluate random baseline on a single task."""
    env = create_env_for_task(task_entry, max_steps=config.max_steps)
    teammate = create_teammate(task_entry, env)
    agent = RandomAgent(num_actions=6)

    episode_returns = []
    episode_lengths = []
    per_episode = []

    for ep_idx in range(config.num_episodes):
        rng, reset_rng, step_rng = jax.random.split(rng, 3)

        obs, env_state = env.reset(reset_rng)
        done = {k: False for k in env.agents + ["__all__"]}

        # Initialize teammate carry state
        rng, init_rng = jax.random.split(step_rng)
        step_rng = rng
        teammate_carry = teammate.init(init_rng)

        episode_return = 0.0
        step_count = 0

        for step in range(config.max_steps):
            # Random action
            rng, action_rng = jax.random.split(step_rng)
            step_rng = rng
            action = agent.get_action(action_rng)

            # Teammate action
            avail_actions_1 = env.get_avail_actions(env_state.env_state)["agent_1"]
            rng, teammate_rng = jax.random.split(step_rng)
            step_rng = rng
            teammate_carry, teammate_action = teammate.act(
                teammate_carry,
                obs["agent_1"],
                jnp.array(done["agent_1"]),
                teammate_rng,
                env_state=env_state,  # Pass full LogEnvState, teammate code unwraps it
                avail_actions=avail_actions_1,
            )
            teammate_action = int(jnp.asarray(teammate_action).squeeze())

            # Step environment
            rng, step_env_rng = jax.random.split(step_rng)
            step_rng = rng
            env_actions = {"agent_0": action, "agent_1": teammate_action}
            next_obs, env_state, rewards, done_next, info = env.step(step_env_rng, env_state, env_actions)

            reward = float(rewards["agent_0"])
            episode_return += reward
            step_count += 1

            obs = next_obs
            done = done_next

            if done["__all__"]:
                break

        episode_returns.append(episode_return)
        episode_lengths.append(step_count)
        per_episode.append({
            "episode_id": ep_idx,
            "return": episode_return,
            "length": step_count,
            "success": episode_return > 0,
        })

    returns = np.array(episode_returns)

    return TaskResult(
        task_id=task_entry.task_id,
        track=task_entry.track,
        layout_name=task_entry.layout_name,
        teammate_family=task_entry.teammate.family,
        num_episodes=config.num_episodes,
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        stderr_return=float(np.std(returns) / np.sqrt(len(returns))),
        min_return=float(np.min(returns)),
        max_return=float(np.max(returns)),
        median_return=float(np.median(returns)),
        success_rate=float(np.mean([e["success"] for e in per_episode])),
        mean_episode_length=float(np.mean(episode_lengths)),
        per_episode=per_episode,
        auc=float(np.sum(returns)),
        context_config={
            "algorithm": "random",
        },
        algorithm="random",
        eval_seed=config.seed,
        evaluated_at=datetime.utcnow().isoformat() + "Z",
    )


# ==============================================================================
# Vectorized Batch Evaluation (Random)
# ==============================================================================

def evaluate_random_batch(
    task_entries: List[TaskEntry],
    config: EvalConfig,
    rng: jax.Array,
    out_dir: Optional[Path] = None,
    track: Optional[str] = None,
) -> List[TaskResult]:
    """Evaluate random baseline on multiple tasks in parallel.

    This function evaluates multiple tasks simultaneously by:
    1. Creating separate environments and teammates for each task
    2. Running episodes in a synchronized manner
    3. Batching random action generation across all tasks

    Supports resume from checkpoint:
    - Saves checkpoint after each episode
    - On restart, loads checkpoint and resumes from where it left off
    - Checkpoint is deleted after successful completion

    Args:
        task_entries: List of task entries to evaluate
        config: Evaluation configuration
        rng: Random key
        out_dir: Output directory for checkpoints (optional, enables resume)
        track: Track name (optional, needed for checkpoint path)

    Returns:
        List of TaskResult, one per task
    """
    if not task_entries:
        return []

    batch_size = len(task_entries)

    # Create environments and teammates for each task
    envs = []
    teammates = []
    agents = []
    for task_entry in task_entries:
        env = create_env_for_task(task_entry, max_steps=config.max_steps)
        teammate = create_teammate(task_entry, env)
        agent = RandomAgent(num_actions=6)
        envs.append(env)
        teammates.append(teammate)
        agents.append(agent)

    # Results storage per task
    all_episode_returns = [[] for _ in range(batch_size)]
    all_episode_lengths = [[] for _ in range(batch_size)]
    all_per_episode = [[] for _ in range(batch_size)]

    # Check for checkpoint (resume support)
    start_episode = 0
    checkpoint_path = None
    if out_dir is not None and track is not None:
        batch_id = generate_batch_id(task_entries)
        checkpoint_path = get_batch_checkpoint_path(out_dir, "random", track, batch_id)
        checkpoint = load_batch_checkpoint(checkpoint_path)

        if checkpoint is not None and not config.force:
            # Validate checkpoint matches current batch
            checkpoint_task_ids = set(checkpoint.task_ids)
            current_task_ids = set(t.task_id for t in task_entries)
            if checkpoint_task_ids == current_task_ids:
                # Resume from checkpoint
                start_episode = checkpoint.completed_episodes
                rng = jax.random.PRNGKey(checkpoint.rng_key[0])
                if len(checkpoint.rng_key) > 1:
                    # Handle split key format
                    rng = jnp.array(checkpoint.rng_key, dtype=jnp.uint32)

                # Restore results
                all_episode_returns = [list(r) for r in checkpoint.all_episode_returns]
                all_episode_lengths = [list(l) for l in checkpoint.all_episode_lengths]
                all_per_episode = [list(p) for p in checkpoint.all_per_episode]

                # Note: Random baseline has no buffer state to restore

                log.info(f"        Resuming from episode {start_episode}/{config.num_episodes}")
            else:
                log.warning(f"        Checkpoint task mismatch, starting fresh")

    for ep_idx in range(start_episode, config.num_episodes):
        # Reset all environments
        rng, *reset_rngs = jax.random.split(rng, batch_size + 1)

        obs_list = []
        env_states = []
        done_list = []
        teammate_carries = []

        for i in range(batch_size):
            obs, env_state = envs[i].reset(reset_rngs[i])
            obs_list.append(obs)
            env_states.append(env_state)
            done_list.append({k: False for k in envs[i].agents + ["__all__"]})

            # Initialize teammate
            rng, init_rng = jax.random.split(rng)
            teammate_carry = teammates[i].init(init_rng)
            teammate_carries.append(teammate_carry)

        episode_returns = [0.0] * batch_size
        step_counts = [0] * batch_size
        task_done = [False] * batch_size

        for step in range(config.max_steps):
            # Check for active tasks
            active_indices = [i for i in range(batch_size) if not task_done[i]]
            if not active_indices:
                break

            # Generate random actions for all active tasks
            rng, *action_rngs = jax.random.split(rng, len(active_indices) + 1)

            # Step each environment individually
            for idx, i in enumerate(active_indices):
                # Random action
                action = agents[i].get_action(action_rngs[idx])

                # Get teammate action
                avail_actions_1 = envs[i].get_avail_actions(env_states[i].env_state)["agent_1"]
                rng, teammate_rng = jax.random.split(rng)
                teammate_carries[i], teammate_action = teammates[i].act(
                    teammate_carries[i],
                    obs_list[i]["agent_1"],
                    jnp.array(done_list[i]["agent_1"]),
                    teammate_rng,
                    env_state=env_states[i],
                    avail_actions=avail_actions_1,
                )
                teammate_action = int(jnp.asarray(teammate_action).squeeze())

                # Step environment
                rng, step_env_rng = jax.random.split(rng)
                env_actions = {"agent_0": action, "agent_1": teammate_action}
                next_obs, env_state, rewards, done_next, info = envs[i].step(
                    step_env_rng, env_states[i], env_actions
                )

                reward = float(rewards["agent_0"])
                episode_returns[i] += reward
                step_counts[i] += 1

                obs_list[i] = next_obs
                env_states[i] = env_state
                done_list[i] = done_next

                if done_next["__all__"]:
                    task_done[i] = True

        # Record episode results
        for i in range(batch_size):
            all_episode_returns[i].append(episode_returns[i])
            all_episode_lengths[i].append(step_counts[i])
            all_per_episode[i].append({
                "episode_id": ep_idx,
                "return": episode_returns[i],
                "length": step_counts[i],
                "success": episode_returns[i] > 0,
            })

        # Save checkpoint after each episode
        if checkpoint_path is not None:
            # Convert JAX rng to serializable format
            rng_as_list = np.array(rng).tolist()
            checkpoint = BatchEvalCheckpoint(
                algorithm="random",
                task_ids=[t.task_id for t in task_entries],
                completed_episodes=ep_idx + 1,
                total_episodes=config.num_episodes,
                rng_key=rng_as_list,
                all_episode_returns=all_episode_returns,
                all_episode_lengths=all_episode_lengths,
                all_per_episode=all_per_episode,
                buffer_state=None,  # Random baseline has no buffer
                created_at=datetime.utcnow().isoformat() + "Z",
            )
            save_batch_checkpoint(checkpoint, checkpoint_path)

    # Build TaskResult for each task
    results = []
    for i, task_entry in enumerate(task_entries):
        returns = np.array(all_episode_returns[i])
        results.append(TaskResult(
            task_id=task_entry.task_id,
            track=task_entry.track,
            layout_name=task_entry.layout_name,
            teammate_family=task_entry.teammate.family,
            num_episodes=config.num_episodes,
            mean_return=float(np.mean(returns)),
            std_return=float(np.std(returns)),
            stderr_return=float(np.std(returns) / np.sqrt(len(returns))),
            min_return=float(np.min(returns)),
            max_return=float(np.max(returns)),
            median_return=float(np.median(returns)),
            success_rate=float(np.mean([e["success"] for e in all_per_episode[i]])),
            mean_episode_length=float(np.mean(all_episode_lengths[i])),
            per_episode=all_per_episode[i],
            auc=float(np.sum(returns)),
            context_config={
                "algorithm": "random",
                "batch_evaluation": True,
            },
            algorithm="random",
            eval_seed=config.seed,
            evaluated_at=datetime.utcnow().isoformat() + "Z",
        ))

    # Delete checkpoint on successful completion
    if checkpoint_path is not None:
        delete_batch_checkpoint(checkpoint_path)

    return results


# ==============================================================================
# Global Parallel Evaluation (across all tasks)
# ==============================================================================

def evaluate_algo_on_track_parallel(
    algo: str,
    track: str,
    config: EvalConfig,
    rng: jax.Array,
    out_dir: Path,
    model=None,
    params=None,
    model_config=None,
) -> "TrackResults":
    """Evaluate an algorithm on a track using global parallel batching.

    This function evaluates ALL tasks across ALL layouts and teammate families
    in parallel batches, rather than iterating sequentially. This maximizes
    GPU utilization by batching model forward passes.

    Returns:
        TrackResults containing results organized by layout and teammate
    """
    log.info(f"  Loading manifests for track '{track}'...")
    manifests = load_test_manifests(track, layouts=config.layouts, teammates=config.teammates)

    if not manifests:
        log.warning(f"  No manifests found for track '{track}'")
        return TrackResults()

    # Collect all tasks with metadata
    all_tasks = []  # List of (layout_name, teammate_family, task_entry)
    for layout_name in sorted(manifests.keys()):
        layout_tasks = manifests[layout_name]
        for teammate_family in sorted(layout_tasks.keys()):
            tasks = layout_tasks[teammate_family]
            for task_entry in tasks:
                all_tasks.append((layout_name, teammate_family, task_entry))

    # Apply max_tasks_per_track limit
    if config.max_tasks_per_track is not None:
        all_tasks = all_tasks[:config.max_tasks_per_track]

    total_tasks = len(all_tasks)
    log.info(f"  Found {total_tasks} total tasks for parallel evaluation")
    log.info(f"  Batch size: {config.parallel_batch_size}")

    # Filter tasks that need evaluation
    tasks_to_eval = []
    tasks_with_results = []

    for layout_name, teammate_family, task_entry in all_tasks:
        task_id = task_entry.task_id
        result_path = get_result_path(out_dir, algo, track, layout_name, teammate_family, task_id)
        if not config.force and result_exists(result_path):
            try:
                with open(result_path, "r") as f:
                    data = json.load(f)
                result = TaskResult.from_dict(data)
                tasks_with_results.append((layout_name, teammate_family, task_entry, result))
                log.info(f"    SKIPPED: {task_id} (exists)")
            except Exception:
                tasks_to_eval.append((layout_name, teammate_family, task_entry))
        else:
            tasks_to_eval.append((layout_name, teammate_family, task_entry))

    log.info(f"  Tasks to evaluate: {len(tasks_to_eval)}, Skipped: {len(tasks_with_results)}")

    # Initialize results structure
    track_results = TrackResults()

    # Add skipped results
    for layout_name, teammate_family, task_entry, result in tasks_with_results:
        if layout_name not in track_results.by_layout_teammate:
            track_results.by_layout_teammate[layout_name] = {}
        if teammate_family not in track_results.by_layout_teammate[layout_name]:
            track_results.by_layout_teammate[layout_name][teammate_family] = []
        track_results.by_layout_teammate[layout_name][teammate_family].append(result)
        track_results.all_results.append(result)
        track_results.skipped.append(task_entry.task_id)

    # Process tasks in batches
    batch_size = config.parallel_batch_size
    task_counter = len(tasks_with_results)

    for batch_start in range(0, len(tasks_to_eval), batch_size):
        batch_end = min(batch_start + batch_size, len(tasks_to_eval))
        batch = tasks_to_eval[batch_start:batch_end]
        actual_batch_size = len(batch)

        log.info(f"\n  Processing batch {batch_start // batch_size + 1}/{(len(tasks_to_eval) + batch_size - 1) // batch_size} "
                 f"(tasks {batch_start + 1}-{batch_end} of {len(tasks_to_eval)})")

        # Extract task entries for batch evaluation
        batch_task_entries = [task_entry for _, _, task_entry in batch]

        rng, batch_rng = jax.random.split(rng)

        try:
            if algo == "dpt":
                batch_results = evaluate_dpt_batch(
                    model, params, batch_task_entries, config, model_config, batch_rng,
                    out_dir=out_dir, track=track
                )
            elif algo == "ad":
                batch_results = evaluate_ad_batch(
                    model, params, batch_task_entries, config, model_config, batch_rng,
                    out_dir=out_dir, track=track
                )
            elif algo == "random":
                batch_results = evaluate_random_batch(
                    batch_task_entries, config, batch_rng,
                    out_dir=out_dir, track=track
                )
            elif algo in ("amago_offline", "hybrid_ad"):
                batch_results = []
                for i, te in enumerate(batch_task_entries):
                    rng_i = jax.random.fold_in(batch_rng, i)
                    if algo == "amago_offline":
                        r = evaluate_amago_offline_task(model, params, te, config, model_config, rng_i)
                    else:
                        r = evaluate_hybrid_ad_task(model, params, te, config, model_config, rng_i)
                    batch_results.append(r)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

            # Save and organize results
            for (layout_name, teammate_family, task_entry), result in zip(batch, batch_results):
                task_counter += 1
                task_id = task_entry.task_id
                result_path = get_result_path(out_dir, algo, track, layout_name, teammate_family, task_id)
                save_result(result, result_path)

                # Add to results structure
                if layout_name not in track_results.by_layout_teammate:
                    track_results.by_layout_teammate[layout_name] = {}
                if teammate_family not in track_results.by_layout_teammate[layout_name]:
                    track_results.by_layout_teammate[layout_name][teammate_family] = []
                track_results.by_layout_teammate[layout_name][teammate_family].append(result)
                track_results.all_results.append(result)

                log.info(f"    [{task_counter}/{total_tasks}] {task_id} ({layout_name}/{teammate_family}) - "
                         f"mean_return={result.mean_return:.2f}, success={result.success_rate:.2%}")

        except Exception as e:
            log.error(f"  Batch evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            # Continue with next batch instead of failing completely

    return track_results


# ==============================================================================
# Main Evaluation Loop
# ==============================================================================

@dataclass
class TrackResults:
    """Results for a track, organized by layout and teammate."""
    # layout_name -> teammate_family -> list of TaskResult
    by_layout_teammate: Dict[str, Dict[str, List[TaskResult]]] = field(default_factory=dict)
    # Flat list of all results
    all_results: List[TaskResult] = field(default_factory=list)
    # Skipped task IDs
    skipped: List[str] = field(default_factory=list)


def evaluate_algo_on_track(
    algo: str,
    track: str,
    config: EvalConfig,
    rng: jax.Array,
    out_dir: Path,
    model=None,
    params=None,
    model_config=None,
) -> TrackResults:
    """Evaluate an algorithm on a track.

    Returns:
        TrackResults containing results organized by layout and teammate
    """
    log.info(f"  Loading manifests for track '{track}'...")
    manifests = load_test_manifests(track, layouts=config.layouts, teammates=config.teammates)

    if not manifests:
        log.warning(f"  No manifests found for track '{track}'")
        return TrackResults()

    # Count total tasks
    total_tasks = sum(
        len(tasks)
        for layout_tasks in manifests.values()
        for tasks in layout_tasks.values()
    )

    # Apply max_tasks_per_track limit if specified
    remaining_tasks = config.max_tasks_per_track

    log.info(f"  Found {len(manifests)} layouts with {total_tasks} total tasks")
    if config.layouts:
        log.info(f"  Filtering by layouts: {config.layouts}")
    if config.teammates:
        log.info(f"  Filtering by teammates: {config.teammates}")

    track_results = TrackResults()
    task_counter = 0

    # Iterate by layout and teammate for organized logging
    for layout_name in sorted(manifests.keys()):
        layout_tasks = manifests[layout_name]
        track_results.by_layout_teammate[layout_name] = {}

        log.info(f"\n    Layout: {layout_name}")

        for teammate_family in sorted(layout_tasks.keys()):
            tasks = layout_tasks[teammate_family]

            # Apply max_tasks limit
            if remaining_tasks is not None:
                if remaining_tasks <= 0:
                    break
                tasks = tasks[:remaining_tasks]
                remaining_tasks -= len(tasks)

            log.info(f"      Teammate: {teammate_family} ({len(tasks)} tasks)")

            teammate_results = []

            # Sequential evaluation mode
            for task_entry in tasks:
                task_counter += 1
                task_id = task_entry.task_id
                result_path = get_result_path(out_dir, algo, track, layout_name, teammate_family, task_id)

                # Check for existing result
                if not config.force and result_exists(result_path):
                    log.info(f"        [{task_counter}] {task_id} - SKIPPED (exists)")
                    track_results.skipped.append(task_id)
                    # Load existing result
                    try:
                        with open(result_path, "r") as f:
                            data = json.load(f)
                        result = TaskResult.from_dict(data)
                        teammate_results.append(result)
                        track_results.all_results.append(result)
                    except Exception as e:
                        log.warning(f"        Failed to load existing result: {e}")
                    continue

                log.info(f"        [{task_counter}] {task_id}")

                rng, eval_rng = jax.random.split(rng)

                try:
                    if algo == "dpt":
                        result = evaluate_dpt_task(
                            model, params, task_entry, config, model_config, eval_rng
                        )
                    elif algo == "ad":
                        result = evaluate_ad_task(
                            model, params, task_entry, config, model_config, eval_rng
                        )
                    elif algo == "amago_offline":
                        result = evaluate_amago_offline_task(
                            model, params, task_entry, config, model_config, eval_rng
                        )
                    elif algo == "hybrid_ad":
                        result = evaluate_hybrid_ad_task(
                            model, params, task_entry, config, model_config, eval_rng
                        )
                    elif algo == "random":
                        result = evaluate_random_task(task_entry, config, eval_rng)
                    else:
                        raise ValueError(f"Unknown algorithm: {algo}")

                    save_result(result, result_path)
                    teammate_results.append(result)
                    track_results.all_results.append(result)
                    log.info(f"          mean_return={result.mean_return:.2f}, success={result.success_rate:.2%}")

                except Exception as e:
                    log.error(f"          FAILED: {e}")
                    import traceback
                    traceback.print_exc()

            # Log teammate summary
            if teammate_results:
                mean_returns = [r.mean_return for r in teammate_results]
                log.info(f"        {teammate_family} mean: {np.mean(mean_returns):.2f} +/- {np.std(mean_returns):.2f}")
                track_results.by_layout_teammate[layout_name][teammate_family] = teammate_results

            if remaining_tasks is not None and remaining_tasks <= 0:
                break

        if remaining_tasks is not None and remaining_tasks <= 0:
            log.info(f"  Reached max_tasks_per_track limit ({config.max_tasks_per_track})")
            break

    return track_results


def run_evaluation(config: EvalConfig):
    """Run full evaluation."""
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    run_config = {
        "config": asdict(config),
        "git_commit": get_git_commit(),
        "started_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    log.info("=" * 70)
    log.info("ICRL Benchmark Evaluation")
    log.info("=" * 70)
    log.info(f"  Algorithms: {config.algos}")
    log.info(f"  Tracks: {config.tracks}")
    if config.layouts:
        log.info(f"  Layouts filter: {config.layouts}")
    if config.teammates:
        log.info(f"  Teammates filter: {config.teammates}")
    log.info(f"  Episodes per task: {config.num_episodes}")
    log.info(f"  Seed: {config.seed}")
    if config.run_parallel:
        log.info(f"  Parallel mode: ENABLED (batch_size={config.parallel_batch_size})")
    else:
        log.info(f"  Parallel mode: disabled")
    log.info(f"  Output: {config.out_dir}")
    log.info("=" * 70)

    rng = jax.random.PRNGKey(config.seed)

    all_results: Dict[str, Dict[str, TrackResults]] = {}

    for algo in config.algos:
        log.info(f"\n[{algo.upper()}]")
        all_results[algo] = {}

        # Load model if needed
        model = params = model_config = None
        if algo == "dpt":
            log.info("  Loading DPT model...")
            model, params, model_config = load_dpt_model(
                config.checkpoints["dpt"],
                config.configs.get("dpt"),
            )
            log.info(f"    Loaded from {config.checkpoints['dpt']}")
        elif algo == "ad":
            log.info("  Loading AD model...")
            model, params, model_config = load_ad_model(
                config.checkpoints["ad"],
                config.configs.get("ad"),
            )
            log.info(f"    Loaded from {config.checkpoints['ad']}")
        elif algo == "amago_offline":
            log.info("  Loading AMAGO-offline model...")
            model, params, model_config = load_amago_offline_model(
                config.checkpoints["amago_offline"],
                config.configs.get("amago_offline"),
            )
            log.info(f"    Loaded from {config.checkpoints['amago_offline']}")
        elif algo == "hybrid_ad":
            log.info("  Loading Hybrid-AD model...")
            model, params, model_config = load_hybrid_ad_model(
                config.checkpoints["hybrid_ad"],
                config.configs.get("hybrid_ad"),
            )
            log.info(f"    Loaded from {config.checkpoints['hybrid_ad']}")

        for track in config.tracks:
            log.info(f"\n  Track: {track}")

            rng, track_rng = jax.random.split(rng)

            # Use parallel evaluation if enabled and algorithm supports it
            if config.run_parallel and algo in ("dpt", "ad", "random", "amago_offline", "hybrid_ad"):
                track_results = evaluate_algo_on_track_parallel(
                    algo, track, config, track_rng, out_dir,
                    model=model, params=params, model_config=model_config,
                )
            else:
                track_results = evaluate_algo_on_track(
                    algo, track, config, track_rng, out_dir,
                    model=model, params=params, model_config=model_config,
                )

            all_results[algo][track] = track_results

            if track_results.all_results:
                mean_returns = [r.mean_return for r in track_results.all_results]
                log.info(f"\n    Track '{track}' overall mean: {np.mean(mean_returns):.2f} +/- {np.std(mean_returns):.2f}")
                log.info(f"    Evaluated: {len(track_results.all_results) - len(track_results.skipped)}, Skipped: {len(track_results.skipped)}")

    # Save summary with detailed breakdown
    run_config["completed_at"] = datetime.utcnow().isoformat() + "Z"
    run_config["summary"] = {}

    for algo in all_results:
        run_config["summary"][algo] = {}
        for track, track_results in all_results[algo].items():
            if not track_results.all_results:
                continue

            # Overall track summary
            mean_returns = [r.mean_return for r in track_results.all_results]
            track_summary = {
                "num_tasks": len(track_results.all_results),
                "mean_return": float(np.mean(mean_returns)),
                "std_return": float(np.std(mean_returns)),
                "stderr_return": float(np.std(mean_returns) / np.sqrt(len(mean_returns))),
                "by_layout": {},
                "by_teammate": {},
            }

            # Per-layout and per-teammate breakdown
            by_teammate_all = {}  # teammate -> all results across layouts

            for layout_name, layout_results in track_results.by_layout_teammate.items():
                layout_all = []
                for teammate_family, teammate_results in layout_results.items():
                    layout_all.extend(teammate_results)
                    # Accumulate per-teammate
                    if teammate_family not in by_teammate_all:
                        by_teammate_all[teammate_family] = []
                    by_teammate_all[teammate_family].extend(teammate_results)

                if layout_all:
                    layout_returns = [r.mean_return for r in layout_all]
                    track_summary["by_layout"][layout_name] = {
                        "num_tasks": len(layout_all),
                        "mean_return": float(np.mean(layout_returns)),
                        "std_return": float(np.std(layout_returns)),
                    }

            # Per-teammate summary across all layouts
            for teammate_family, teammate_results in by_teammate_all.items():
                teammate_returns = [r.mean_return for r in teammate_results]
                track_summary["by_teammate"][teammate_family] = {
                    "num_tasks": len(teammate_results),
                    "mean_return": float(np.mean(teammate_returns)),
                    "std_return": float(np.std(teammate_returns)),
                }

            run_config["summary"][algo][track] = track_summary

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    log.info("\n" + "=" * 70)
    log.info("EVALUATION COMPLETE")
    log.info("=" * 70)
    log.info(f"Results saved to: {out_dir}")
    log.info(f"Run 'python summarize.py --results_dir {out_dir}' to generate summary tables")

    return all_results


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> EvalConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate ICRL algorithms on Overcooked V2 benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all ICRL baselines on all tracks
  python eval_icrl.py --algo dpt,ad,amago_offline,hybrid_ad --tracks teammate,layout \\
      --checkpoint_dpt outputs/dpt/checkpoints/checkpoint_100000 \\
      --checkpoint_ad outputs/ad/checkpoints/checkpoint_100000 \\
      --checkpoint_amago_offline outputs/amago_offline/checkpoints/checkpoint_100000 \\
      --checkpoint_hybrid_ad outputs/hybrid_ad/checkpoints/checkpoint_100000 \\
      --episodes 100 --seed 0 --out results/

  # Evaluate on specific layout and teammate
  python eval_icrl.py --algo random --tracks teammate \\
      --layout demo_cook_simple --teammate territory \\
      --episodes 10 --out results_specific/

  # Evaluate against multiple heuristic teammates
  python eval_icrl.py --algo random --tracks teammate \\
      --layout demo_cook_simple --teammate territory,assembly_line,utility_greedy \\
      --episodes 10 --out results_multi_teammate/

  # Quick smoke test
  python eval_icrl.py --algo random --tracks teammate \\
      --episodes 3 --max_tasks_per_track 2 --out results_smoke/

  # Resume interrupted evaluation (skips completed tasks)
  python eval_icrl.py --algo dpt --tracks teammate --out results/

  # Vectorized batch evaluation (run tasks within each teammate_family in batches)
  python eval_icrl.py --algo dpt --tracks teammate \\
      --checkpoint_dpt outputs/dpt/checkpoints/checkpoint_100000 \\
      --episodes 100 --run_parallel --parallel_batch_size 4 --out results/
        """
    )

    # Algorithms
    parser.add_argument(
        "--algo", type=str, default="dpt",
        help="Comma-separated list of algorithms to evaluate (dpt,ad,amago_offline,hybrid_ad,random)"
    )
    parser.add_argument(
        "--tracks", type=str, default="teammate",
        help="Comma-separated list of tracks to evaluate"
    )

    # Layout and teammate filtering
    parser.add_argument(
        "--layout", type=str, default=None,
        help="Comma-separated list of layouts to evaluate (default: all). "
             "E.g., 'demo_cook_simple,test_time_wide'"
    )
    parser.add_argument(
        "--teammate", type=str, default=None,
        help="Comma-separated list of heuristic teammate families to evaluate (default: all). "
             "Options: assembly_line,territory,utility_greedy,recipe_aware_button"
    )

    # Model checkpoints
    parser.add_argument(
        "--checkpoint_dpt", type=str, default=None,
        help="Path to DPT checkpoint"
    )
    parser.add_argument(
        "--checkpoint_ad", type=str, default=None,
        help="Path to AD checkpoint"
    )
    parser.add_argument(
        "--config_dpt", type=str, default=None,
        help="Path to DPT config (auto-detected if None)"
    )
    parser.add_argument(
        "--config_ad", type=str, default=None,
        help="Path to AD config (auto-detected if None)"
    )

    # AMAGO-offline checkpoints
    parser.add_argument(
        "--checkpoint_amago_offline", type=str, default=None,
        help="Path to AMAGO-offline checkpoint"
    )
    parser.add_argument(
        "--config_amago_offline", type=str, default=None,
        help="Path to AMAGO-offline config (auto-detected if None)"
    )

    # Hybrid-AD checkpoints
    parser.add_argument(
        "--checkpoint_hybrid_ad", type=str, default=None,
        help="Path to Hybrid-AD checkpoint"
    )
    parser.add_argument(
        "--config_hybrid_ad", type=str, default=None,
        help="Path to Hybrid-AD config (auto-detected if None)"
    )

    # Evaluation parameters
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Number of episodes per task"
    )
    parser.add_argument(
        "--max_steps", type=int, default=100,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility (starting seed when using --num_seeds)"
    )
    parser.add_argument(
        "--num_seeds", type=int, default=1,
        help="Number of seeds to run. Runs seeds: seed, seed+1, ..., seed+num_seeds-1. "
             "Results are saved in separate subdirectories (seed_0, seed_1, etc.)"
    )
    parser.add_argument(
        "--greedy", action="store_true", default=True,
        help="Use greedy action selection"
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Sample actions instead of greedy (overrides --greedy)"
    )

    # Context settings
    parser.add_argument(
        "--context_len", type=int, default=500,
        help="Context length for AD"
    )
    parser.add_argument(
        "--context_episodes", type=int, default=5,
        help="Context episodes for DPT"
    )

    # Teammate actions (optional feature)
    parser.add_argument(
        "--use_teammate_actions", action="store_true",
        help="Include teammate actions in context (optional)"
    )

    # Output
    parser.add_argument(
        "--out", type=str, default="results",
        help="Output directory"
    )

    # Task limits
    parser.add_argument(
        "--max_tasks_per_track", type=int, default=None,
        help="Limit tasks per track (for testing)"
    )

    # Resume/force
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-evaluation of all tasks"
    )

    # GPU
    parser.add_argument(
        "--gpu", type=str, default=None,
        help="GPU device ID(s) to use (e.g., '0', '0,1'). Use '-1' for CPU. If not specified, uses all available GPUs."
    )

    # Parallel evaluation
    parser.add_argument(
        "--run_parallel", action="store_true", default=True,
        help="Run vectorized batch evaluation within each teammate_family"
    )
    parser.add_argument(
        "--parallel_batch_size", type=int, default=20,
        help="Batch size for vectorized evaluation when --run_parallel is enabled (default: 20)"
    )

    args = parser.parse_args()

    # Build config
    algos = [a.strip() for a in args.algo.split(",")]
    tracks = [t.strip() for t in args.tracks.split(",")]

    # Parse layout and teammate filters
    layouts = None
    if args.layout:
        layouts = [l.strip() for l in args.layout.split(",")]

    teammates = None
    if args.teammate:
        teammates = [t.strip() for t in args.teammate.split(",")]

    checkpoints = {}
    configs = {}
    if args.checkpoint_dpt:
        checkpoints["dpt"] = args.checkpoint_dpt
    if args.checkpoint_ad:
        checkpoints["ad"] = args.checkpoint_ad
    if args.checkpoint_amago_offline:
        checkpoints["amago_offline"] = args.checkpoint_amago_offline
    if args.checkpoint_hybrid_ad:
        checkpoints["hybrid_ad"] = args.checkpoint_hybrid_ad
    if args.config_dpt:
        configs["dpt"] = args.config_dpt
    if args.config_ad:
        configs["ad"] = args.config_ad
    if args.config_amago_offline:
        configs["amago_offline"] = args.config_amago_offline
    if args.config_hybrid_ad:
        configs["hybrid_ad"] = args.config_hybrid_ad

    return EvalConfig(
        algos=algos,
        tracks=tracks,
        layouts=layouts,
        teammates=teammates,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        num_seeds=args.num_seeds,
        greedy=not args.sample,
        checkpoints=checkpoints,
        configs=configs,
        context_len=args.context_len,
        context_episodes=args.context_episodes,
        use_teammate_actions=args.use_teammate_actions,
        out_dir=args.out,
        max_tasks_per_track=args.max_tasks_per_track,
        force=args.force,
        gpu=args.gpu,
        run_parallel=args.run_parallel,
        parallel_batch_size=args.parallel_batch_size,
    )


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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
        config.validate()

        # Run evaluation for each seed
        base_seed = config.seed
        base_out_dir = config.out_dir
        num_seeds = config.num_seeds

        log.info(f"Running evaluation with {num_seeds} seed(s): {base_seed} to {base_seed + num_seeds - 1}")

        for seed_idx in range(num_seeds):
            current_seed = base_seed + seed_idx

            # Create seed-specific output directory
            if num_seeds > 1:
                seed_out_dir = str(Path(base_out_dir) / f"seed_{current_seed}")
            else:
                seed_out_dir = base_out_dir

            log.info("\n" + "#" * 70)
            log.info(f"# SEED RUN {seed_idx + 1}/{num_seeds}: seed={current_seed}")
            log.info(f"# Output: {seed_out_dir}")
            log.info("#" * 70)

            # Update config for this seed run
            config.seed = current_seed
            config.out_dir = seed_out_dir

            run_evaluation(config)

        # Restore original config values (for consistency)
        config.seed = base_seed
        config.out_dir = base_out_dir

        if num_seeds > 1:
            log.info("\n" + "=" * 70)
            log.info(f"ALL {num_seeds} SEED RUNS COMPLETE")
            log.info(f"Results saved in: {base_out_dir}/seed_*/")
            log.info("=" * 70)

        return 0
    except Exception as e:
        log.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())