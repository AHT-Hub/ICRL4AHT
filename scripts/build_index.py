#!/usr/bin/env python3
"""Build HDF5 dataset + index from collected histories.

This script converts the per-task npz files produced by task_runner.py
into a single packed HDF5 file optimized for random access sampling during
AD/DPT/AMAGO-offline/Hybrid-AD training.

Usage:
    # Simple usage (outputs to same directory):
    python -m scripts.build_index outputs/task_runs

    # With expert action relabeling for DPT training:
    python -m scripts.build_index outputs/task_runs --relabel

    # Filter by split:
    python -m scripts.build_index outputs/task_runs --split train

    # Custom output location:
    python -m scripts.build_index outputs/task_runs --out_h5 datasets/data.h5

    # Run on specific GPU:
    python -m scripts.build_index outputs/task_runs --relabel --gpu 0

    # Run on CPU:
    python -m scripts.build_index outputs/task_runs --relabel --gpu -1

Output files (auto-generated in collected_root if not specified):
    - histories.h5: Packed HDF5 with all learning histories (+ expert_actions if --relabel)
    - histories_index.jsonl: Index for O(1) metadata lookup
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator

import h5py
import numpy as np

log = logging.getLogger(__name__)

# =============================================================================
# Relabeling Support (lazy imports to avoid JAX overhead when not needed)
# =============================================================================

_RELABEL_IMPORTS_DONE = False
_jax = None
_jnp = None
_initialize_mlp_agent = None
_initialize_rnn_agent = None
_initialize_s5_agent = None
_initialize_cnn_rnn_agent = None


def _ensure_relabel_imports():
    """Lazy import JAX and policy modules for relabeling."""
    global _RELABEL_IMPORTS_DONE, _jax, _jnp
    global _initialize_mlp_agent, _initialize_rnn_agent, _initialize_s5_agent, _initialize_cnn_rnn_agent

    if _RELABEL_IMPORTS_DONE:
        return

    import jax
    import jax.numpy as jnp
    from agents.initialize_agents import (
        initialize_mlp_agent,
        initialize_rnn_agent,
        initialize_s5_agent,
        initialize_cnn_rnn_agent,
    )

    _jax = jax
    _jnp = jnp
    _initialize_mlp_agent = initialize_mlp_agent
    _initialize_rnn_agent = initialize_rnn_agent
    _initialize_s5_agent = initialize_s5_agent
    _initialize_cnn_rnn_agent = initialize_cnn_rnn_agent
    _RELABEL_IMPORTS_DONE = True


def load_checkpoint(ckpt_dir: Path) -> Dict[str, np.ndarray]:
    """Load parameters from checkpoint directory."""
    _ensure_relabel_imports()

    params_file = ckpt_dir / "params.npz"
    if not params_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {params_file}")

    with np.load(params_file) as f:
        flat_params = {k: f[k] for k in f.files}

    # Unflatten params from dotted keys to nested dict
    result = {}
    for key, value in flat_params.items():
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = _jnp.array(value)
    return result


def get_policy_for_task(
    metadata: Dict[str, Any],
    action_dim: int,
    obs_shape: Tuple[int, ...],
) -> Tuple[Any, str]:
    """Create policy object matching the task's training config."""
    _ensure_relabel_imports()

    ppo_config = metadata.get("ppo_config", {})
    actor_type = ppo_config.get("EGO_ACTOR_TYPE", metadata.get("actor_type", "mlp"))
    obs_dim = int(np.prod(obs_shape))

    config = {
        "POLICY_INPUT_DIM": obs_dim,
        "ACTIVATION": ppo_config.get("ACTIVATION", "relu"),
        "FC_DIM_SIZE": ppo_config.get("FC_DIM_SIZE", 128),
        "GRU_HIDDEN_DIM": ppo_config.get("GRU_HIDDEN_DIM", 128),
    }

    _obs_shape = obs_shape

    class DummyEnv:
        agents = ["agent_0"]

        def action_space(self, agent):
            class ActionSpace:
                n = action_dim
            return ActionSpace()

        def observation_space(self, agent):
            class ObsSpace:
                shape = _obs_shape
            return ObsSpace()

    dummy_env = DummyEnv()
    rng = _jax.random.PRNGKey(0)

    if actor_type == "mlp":
        policy, _ = _initialize_mlp_agent(config, dummy_env, rng)
    elif actor_type == "rnn":
        policy, _ = _initialize_rnn_agent(config, dummy_env, rng)
    elif actor_type == "s5":
        policy, _ = _initialize_s5_agent(config, dummy_env, rng)
    elif actor_type == "cnn_rnn":
        policy, _ = _initialize_cnn_rnn_agent(config, dummy_env, rng)
    else:
        raise ValueError(f"Unknown actor type: {actor_type}")

    return policy, actor_type


def relabel_task_obs(
    obs: np.ndarray,
    metadata: Dict[str, Any],
    ckpt_dir: Path,
    batch_size: int = 1024,
) -> np.ndarray:
    """Relabel expert actions for observations from a single task.

    Args:
        obs: Observations array of shape (T, E, *obs_shape)
        metadata: Task metadata containing ppo_config
        ckpt_dir: Path to checkpoint directory
        batch_size: Batch size for forward passes

    Returns:
        Expert actions array of shape (T, E)
    """
    _ensure_relabel_imports()

    T, E = obs.shape[:2]
    obs_shape = obs.shape[2:]
    num_actions = 6  # Overcooked action space

    # Load checkpoint and create policy
    params = load_checkpoint(ckpt_dir)
    policy, actor_type = get_policy_for_task(metadata, num_actions, obs_shape)

    # Flatten for batch processing
    N = T * E
    obs_flat = obs.reshape(N, *obs_shape)

    avail_actions = _jnp.ones((batch_size, num_actions))
    expert_actions_flat = np.zeros(N, dtype=np.int32)
    rng = _jax.random.PRNGKey(0)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_obs = _jnp.array(obs_flat[start:end])
        actual_batch = batch_obs.shape[0]

        # Pad if needed
        if actual_batch < batch_size:
            padding = _jnp.zeros((batch_size - actual_batch,) + obs_shape)
            batch_obs = _jnp.concatenate([batch_obs, padding], axis=0)

        if actor_type == "cnn_rnn":
            batch_obs_seq = batch_obs[None, :, ...]
            done = _jnp.zeros((1, batch_size), dtype=bool)
            hstate = policy.init_hstate(batch_size)
            actions_batch, _ = policy.get_action(
                params, batch_obs_seq, done, avail_actions, hstate, rng, test_mode=True
            )
            actions_batch = actions_batch.squeeze(0)
        elif actor_type in ["rnn", "s5"]:
            batch_obs_flat = batch_obs.reshape(batch_size, -1)
            batch_obs_seq = batch_obs_flat[None, :, :]
            done = _jnp.zeros((1, batch_size), dtype=bool)
            hstate = policy.init_hstate(batch_size)
            actions_batch, _ = policy.get_action(
                params, batch_obs_seq, done, avail_actions, hstate, rng, test_mode=True
            )
            actions_batch = actions_batch.squeeze(0)
        else:
            batch_obs_flat = batch_obs.reshape(batch_size, -1)
            pi, _ = policy.network.apply(params, (batch_obs_flat, avail_actions))
            actions_batch = pi.mode()

        expert_actions_flat[start:end] = np.array(actions_batch[:actual_batch])

    return expert_actions_flat.reshape(T, E)


# =============================================================================
# History Loading Utilities (handles both regular and chunked formats)
# =============================================================================

def is_chunked_history(task_dir: Path) -> bool:
    """Check if a task directory contains chunked history data.

    Args:
        task_dir: Path to task directory

    Returns:
        True if history is stored in chunks
    """
    history_path = task_dir / "history.npz"
    if not history_path.exists():
        return False

    with np.load(history_path) as npz:
        return "_chunked" in npz.files and npz["_chunked"][0]


def load_history_arrays(task_dir: Path) -> Dict[str, np.ndarray]:
    """Load history arrays from task directory, handling both regular and chunked formats.

    Args:
        task_dir: Path to task directory

    Returns:
        Dict with arrays: obs_t, act_t, rew_t, done_t, and optionally teammate_act_t
    """
    if is_chunked_history(task_dir):
        return _load_chunked_history(task_dir)
    else:
        return _load_regular_history(task_dir)


def _load_regular_history(task_dir: Path) -> Dict[str, np.ndarray]:
    """Load regular (non-chunked) history arrays."""
    history_path = task_dir / "history.npz"
    with np.load(history_path) as npz:
        result = {
            "obs_t": npz["obs_t"],
            "act_t": npz["act_t"],
            "rew_t": npz["rew_t"],
            "done_t": npz["done_t"],
        }
        # Load teammate actions if available
        if "teammate_act_t" in npz.files:
            result["teammate_act_t"] = npz["teammate_act_t"]
        return result


def _load_chunked_history(task_dir: Path) -> Dict[str, np.ndarray]:
    """Load and concatenate chunked history arrays."""
    chunks_dir = task_dir / "chunks"

    # Load manifest if exists
    manifest_path = chunks_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        chunk_files = manifest["chunks"]
    else:
        # Fall back to scanning directory
        chunk_files = sorted([
            f.name for f in chunks_dir.glob("chunk_*.npz")
            if "_episodes" not in f.name
        ])

    if not chunk_files:
        raise ValueError(f"No chunk files found in {chunks_dir}")

    # Load and concatenate all chunks
    all_arrays: Dict[str, List[np.ndarray]] = {
        "obs_t": [], "act_t": [], "rew_t": [], "done_t": [], "teammate_act_t": []
    }

    for chunk_file in chunk_files:
        chunk_path = chunks_dir / chunk_file
        with np.load(chunk_path) as npz:
            for key in ["obs_t", "act_t", "rew_t", "done_t", "teammate_act_t"]:
                if key in npz.files and npz[key].size > 0:
                    all_arrays[key].append(npz[key])

    # Concatenate along time dimension (axis 0)
    result = {}
    for key, arrays in all_arrays.items():
        if arrays:
            result[key] = np.concatenate(arrays, axis=0)
        elif key != "teammate_act_t":  # Only include teammate_act_t if present
            result[key] = np.array([])

    return result


def get_history_shape(task_dir: Path) -> Tuple[int, int, Tuple[int, ...]]:
    """Get history shape (T, E, obs_shape) without loading full data.

    Args:
        task_dir: Path to task directory

    Returns:
        Tuple of (T, E, obs_shape) where obs_shape is a tuple like (H, W, C) or (obs_dim,)
    """
    if is_chunked_history(task_dir):
        return _get_chunked_history_shape(task_dir)
    else:
        return _get_regular_history_shape(task_dir)


def _get_regular_history_shape(task_dir: Path) -> Tuple[int, int, Tuple[int, ...]]:
    """Get shape from regular history.npz."""
    history_path = task_dir / "history.npz"
    with np.load(history_path) as npz:
        full_shape = npz["obs_t"].shape
        T = full_shape[0]
        E = full_shape[1]
        # obs_shape is everything after (T, E), e.g., (H, W, C) or (obs_dim,)
        obs_shape = full_shape[2:] if len(full_shape) > 2 else (1,)
        return T, E, obs_shape


def _get_chunked_history_shape(task_dir: Path) -> Tuple[int, int, Tuple[int, ...]]:
    """Get shape from chunked history."""
    chunks_dir = task_dir / "chunks"
    history_path = task_dir / "history.npz"

    # Get total timesteps from history.npz marker
    with np.load(history_path) as npz:
        total_timesteps = int(npz["_total_timesteps"][0])

    # Get E and obs_shape from first chunk
    manifest_path = chunks_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        first_chunk = manifest["chunks"][0]
    else:
        chunk_files = sorted([
            f.name for f in chunks_dir.glob("chunk_*.npz")
            if "_episodes" not in f.name
        ])
        first_chunk = chunk_files[0]

    with np.load(chunks_dir / first_chunk) as npz:
        full_shape = npz["obs_t"].shape
        E = full_shape[1]
        # obs_shape is everything after (T, E)
        obs_shape = full_shape[2:] if len(full_shape) > 2 else (1,)

    return total_timesteps, E, obs_shape


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BuildConfig:
    """Configuration for HDF5 dataset building."""
    collected_root: str
    out_h5: str
    out_index: str

    # Filters
    track: Optional[str] = None
    split: Optional[str] = None
    layout: Optional[str] = None
    max_tasks: Optional[int] = None
    max_histories_per_task: Optional[int] = None

    # Expert action relabeling
    relabel: bool = False  # Compute and store expert_actions using final checkpoint
    relabel_batch_size: int = 1024

    # GPU selection
    gpu: Optional[str] = None  # GPU device ID to use (e.g., '0', '1'). Use '-1' for CPU.

    # HDF5 settings
    compression: str = "gzip"
    compression_opts: int = 6
    chunk_time_size: int = 4096  # Chunk along time axis for efficient slicing

    # Debug
    debug_smoke_test: bool = False
    verbose: bool = False


# =============================================================================
# Task Discovery and Validation
# =============================================================================

@dataclass
class TaskInfo:
    """Information about a collected task."""
    task_id: str
    task_dir: Path
    metadata: Dict[str, Any]
    episodes: Dict[str, Any]

    # Extracted metadata fields
    track: str = ""
    split: str = ""
    layout: str = ""
    teammate_family: str = ""
    teammate_kind: str = ""

    # Array info
    T: int = 0  # Total timesteps
    E: int = 0  # Number of env streams (record_envs)
    obs_shape: Tuple[int, ...] = ()  # Observation shape, e.g., (H, W, C) or (obs_dim,)
    action_dim: int = 6  # Overcooked action space

    def __post_init__(self):
        """Extract fields from metadata."""
        task_spec = self.metadata.get("task_spec", {})
        self.track = task_spec.get("track", "")
        self.split = task_spec.get("split", "")
        self.layout = task_spec.get("layout_name", "")

        teammate = task_spec.get("teammate", {})
        self.teammate_family = teammate.get("family", "")
        self.teammate_kind = teammate.get("kind", "")


def compute_env_quality_scores(
    episodes: Dict[str, Any],
    num_envs: int,
    window_fraction: float = 0.1,
    min_window_size: int = 5,
) -> Dict[int, Tuple[float, float, float]]:
    """Compute quality scores for each env stream based on base_return growth.

    For each env_idx, computes:
    1. Average base_return at the end of training (last N episodes)
    2. Increment of base_return from initial to end (end avg - start avg)

    The combined score is: end_avg + increment (both metrics contribute equally)

    Args:
        episodes: Episodes dict from episodes.json with 'episodes' list
        num_envs: Number of environment streams (E)
        window_fraction: Fraction of episodes to use for start/end windows (default: 0.1)
        min_window_size: Minimum number of episodes in each window (default: 5)

    Returns:
        Dict mapping env_idx to (combined_score, end_avg, increment)
    """
    ep_list = episodes.get("episodes", [])
    if not ep_list:
        # No episodes, return zero scores for all envs
        return {i: (0.0, 0.0, 0.0) for i in range(num_envs)}

    # Group episodes by env_idx
    env_episodes: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(num_envs)}
    for ep in ep_list:
        env_idx = ep.get("env_idx", 0)
        if env_idx < num_envs:
            env_episodes[env_idx].append(ep)

    scores = {}
    for env_idx in range(num_envs):
        eps = env_episodes[env_idx]
        if not eps:
            scores[env_idx] = (0.0, 0.0, 0.0)
            continue

        # Sort by episode_id to ensure chronological order
        eps = sorted(eps, key=lambda e: e.get("episode_id", e.get("start_idx", 0)))

        # Compute window size
        n_eps = len(eps)
        window_size = max(min_window_size, int(n_eps * window_fraction))
        window_size = min(window_size, n_eps)  # Can't exceed total episodes

        # Get returns
        returns = [ep.get("total_return", 0.0) for ep in eps]

        # Compute start and end averages
        start_returns = returns[:window_size]
        end_returns = returns[-window_size:]

        start_avg = float(np.mean(start_returns)) if start_returns else 0.0
        end_avg = float(np.mean(end_returns)) if end_returns else 0.0
        increment = end_avg - start_avg

        # Combined score: prioritize both high final performance and improvement
        combined_score = end_avg + increment

        scores[env_idx] = (combined_score, end_avg, increment)

    return scores


def select_best_env_indices(
    episodes: Dict[str, Any],
    num_envs: int,
    max_histories: int,
) -> List[int]:
    """Select the best env indices based on base_return growth quality.

    Args:
        episodes: Episodes dict from episodes.json
        num_envs: Total number of environment streams
        max_histories: Maximum number of histories to select

    Returns:
        List of env indices sorted by quality (best first), limited to max_histories
    """
    scores = compute_env_quality_scores(episodes, num_envs)

    # Sort env indices by combined score (descending)
    sorted_indices = sorted(
        range(num_envs),
        key=lambda i: scores[i][0],
        reverse=True
    )

    # Return top max_histories indices
    selected = sorted_indices[:max_histories]

    # Log selection info
    if selected:
        best_idx = selected[0]
        worst_selected_idx = selected[-1]
        log.info(f"  Selected {len(selected)}/{num_envs} env streams by quality:")
        log.info(f"    Best: env_idx={best_idx}, score={scores[best_idx][0]:.2f} "
                f"(end_avg={scores[best_idx][1]:.2f}, increment={scores[best_idx][2]:.2f})")
        if len(selected) > 1:
            log.info(f"    Cutoff: env_idx={worst_selected_idx}, score={scores[worst_selected_idx][0]:.2f} "
                    f"(end_avg={scores[worst_selected_idx][1]:.2f}, increment={scores[worst_selected_idx][2]:.2f})")

    return selected


def discover_tasks(root: Path, config: BuildConfig) -> List[TaskInfo]:
    """Discover and validate all task directories.

    Args:
        root: Root directory containing task subdirectories
        config: Build configuration with filters

    Returns:
        List of TaskInfo for valid tasks matching filters
    """
    tasks = []

    if not root.exists():
        raise FileNotFoundError(f"Collected root does not exist: {root}")

    # Find all potential task directories
    candidates = sorted([
        d for d in root.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    ])

    log.info(f"Found {len(candidates)} candidate task directories")

    for task_dir in candidates:
        # Check required files exist
        history_path = task_dir / "history.npz"
        episodes_path = task_dir / "episodes.json"
        metadata_path = task_dir / "metadata.json"

        if not all(p.exists() for p in [history_path, episodes_path, metadata_path]):
            log.debug(f"Skipping incomplete task: {task_dir.name}")
            continue

        try:
            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            with open(episodes_path, "r") as f:
                episodes = json.load(f)

            task_info = TaskInfo(
                task_id=task_dir.name,
                task_dir=task_dir,
                metadata=metadata,
                episodes=episodes,
            )

            # Apply filters
            if config.track and task_info.track != config.track:
                log.debug(f"Skipping task {task_info.task_id}: track mismatch")
                continue
            if config.split and task_info.split != config.split:
                log.debug(f"Skipping task {task_info.task_id}: split mismatch")
                continue
            if config.layout and task_info.layout != config.layout:
                log.debug(f"Skipping task {task_info.task_id}: layout mismatch")
                continue

            # Load array shapes (without loading full data)
            # Supports both regular and chunked history formats
            try:
                T, E, obs_shape = get_history_shape(task_dir)
                task_info.T = T
                task_info.E = E
                task_info.obs_shape = obs_shape
            except Exception as shape_err:
                log.warning(f"Error getting shape for task {task_dir.name}: {shape_err}")
                continue

            tasks.append(task_info)

        except Exception as e:
            log.warning(f"Error loading task {task_dir.name}: {e}")
            continue

    # Apply max_tasks limit
    if config.max_tasks is not None:
        tasks = tasks[:config.max_tasks]

    log.info(f"Validated {len(tasks)} tasks for packing")
    return tasks


# =============================================================================
# HDF5 Packing
# =============================================================================

@dataclass
class HistoryEntry:
    """Index entry for a single learning history."""
    history_id: int
    task_id: str
    env_idx: int
    T: int  # sequence length
    obs_shape: Tuple[int, ...]  # Observation shape, e.g., (H, W, C) or (obs_dim,)
    action_dim: int
    track: str
    split: str
    layout: str
    teammate_family: str
    teammate_kind: str
    h5_group: str  # Path in HDF5 file
    has_teammate_actions: bool = False  # Whether teammate actions are available
    has_expert_actions: bool = False  # Whether expert actions are available

    @property
    def obs_dim(self) -> int:
        """Backward-compatible property: flattened observation dimension."""
        import numpy as np
        return int(np.prod(self.obs_shape))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert obs_shape tuple to list for JSON serialization
        d["obs_shape"] = list(self.obs_shape)
        # Also include obs_dim for backward compatibility
        d["obs_dim"] = self.obs_dim
        return d


def pack_histories(
    tasks: List[TaskInfo],
    h5_path: Path,
    config: BuildConfig,
) -> List[HistoryEntry]:
    """Pack all learning histories into a single HDF5 file.

    The main structure of the HDF5 file is:
    - One group per learning history (indexed by sequential integer)
    - Gzip compression with level 6
    - Chunk sizes optimized for time-axis slicing

    Args:
        tasks: List of validated TaskInfo objects
        h5_path: Output HDF5 file path
        config: Build configuration

    Returns:
        List of HistoryEntry for the index file
    """
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    index_entries = []
    history_id = 0

    # Use rdcc (raw data chunk cache) for better write performance
    with h5py.File(h5_path, "w", rdcc_nbytes=100 * 1024 * 1024) as h5f:
        # Store global attributes
        h5f.attrs["schema_version"] = 1
        h5f.attrs["format"] = "overcooked_learning_histories"
        h5f.attrs["compression"] = config.compression
        h5f.attrs["compression_opts"] = config.compression_opts

        for task_idx, task in enumerate(tasks):
            log.info(f"Packing task {task_idx + 1}/{len(tasks)}: {task.task_id}")

            # Load arrays (supports both regular and chunked formats)
            history_data = load_history_arrays(task.task_dir)
            obs_t = history_data["obs_t"]      # (T, E, ...) where ... is obs_shape
            act_t = history_data["act_t"]      # (T, E)
            rew_t = history_data["rew_t"]      # (T, E)
            done_t = history_data["done_t"]    # (T, E)
            teammate_act_t = history_data.get("teammate_act_t")  # (T, E) or None
            has_teammate_actions = teammate_act_t is not None

            T, E = obs_t.shape[:2]
            # obs_shape is everything after (T, E), e.g., (H, W, C) or (obs_dim,)
            obs_shape = obs_t.shape[2:] if len(obs_t.shape) > 2 else (1,)

            # Optionally compute expert actions for this task
            expert_act_t = None
            has_expert_actions = False
            if config.relabel:
                ckpt_dir = task.task_dir / "final_ckpt"
                if ckpt_dir.exists():
                    try:
                        log.info(f"  Relabeling expert actions...")
                        expert_act_t = relabel_task_obs(
                            obs_t, task.metadata, ckpt_dir,
                            batch_size=config.relabel_batch_size
                        )
                        has_expert_actions = True
                        log.info(f"  Expert actions: shape={expert_act_t.shape}")
                    except Exception as e:
                        log.warning(f"  Failed to relabel: {e}")
                else:
                    log.warning(f"  No checkpoint found for relabeling: {ckpt_dir}")

            # Determine which env streams to pack
            if config.max_histories_per_task is not None:
                # Select best env indices based on base_return growth quality
                # Quality score = end_avg + increment, where:
                #   - end_avg: average base_return at end of training
                #   - increment: base_return improvement from start to end
                selected_env_indices = select_best_env_indices(
                    task.episodes, E, config.max_histories_per_task
                )
            else:
                # Pack all env streams
                selected_env_indices = list(range(E))

            # Pack selected env streams as separate learning histories
            for env_idx in selected_env_indices:
                group_name = str(history_id)
                g = h5f.create_group(group_name)

                # Store metadata as attributes
                g.attrs["task_id"] = task.task_id
                g.attrs["env_idx"] = env_idx
                g.attrs["track"] = task.track
                g.attrs["split"] = task.split
                g.attrs["layout"] = task.layout
                g.attrs["teammate_family"] = task.teammate_family
                g.attrs["teammate_kind"] = task.teammate_kind
                g.attrs["T"] = T
                g.attrs["obs_shape"] = list(obs_shape)  # Store as list for HDF5 compatibility
                g.attrs["obs_dim"] = int(np.prod(obs_shape))  # Backward compatibility: flattened dim
                g.attrs["action_dim"] = task.action_dim
                g.attrs["has_teammate_actions"] = has_teammate_actions
                g.attrs["has_expert_actions"] = has_expert_actions

                # Calculate chunk size: chunk along time axis for efficient slicing
                # (1, chunk_time_size, ...)
                # But since we store per-history, shape is (T, ...), so chunk is (chunk_size, ...)
                time_chunk = min(config.chunk_time_size, T)

                # Create datasets with compression and chunking
                # obs: (T, *obs_shape) - e.g., (T, H, W, C) or (T, obs_dim)
                obs_data = obs_t[:, env_idx]  # Extract single env stream: (T, *obs_shape)
                obs_chunks = (time_chunk,) + obs_shape
                g.create_dataset(
                    "obs",
                    data=obs_data,
                    dtype=np.float32,
                    compression=config.compression,
                    compression_opts=config.compression_opts,
                    chunks=obs_chunks,
                )

                # actions: (T,)
                g.create_dataset(
                    "actions",
                    data=act_t[:, env_idx],
                    dtype=np.int32,
                    compression=config.compression,
                    compression_opts=config.compression_opts,
                    chunks=(time_chunk,),
                )

                # rewards: (T,)
                g.create_dataset(
                    "rewards",
                    data=rew_t[:, env_idx],
                    dtype=np.float32,
                    compression=config.compression,
                    compression_opts=config.compression_opts,
                    chunks=(time_chunk,),
                )

                # dones: (T,)
                g.create_dataset(
                    "dones",
                    data=done_t[:, env_idx],
                    dtype=np.bool_,
                    compression=config.compression,
                    compression_opts=config.compression_opts,
                    chunks=(time_chunk,),
                )

                # teammate_actions: (T,) - optional
                if has_teammate_actions:
                    g.create_dataset(
                        "teammate_actions",
                        data=teammate_act_t[:, env_idx],
                        dtype=np.int32,
                        compression=config.compression,
                        compression_opts=config.compression_opts,
                        chunks=(time_chunk,),
                    )

                # expert_actions: (T,) - optional (from relabeling)
                if has_expert_actions and expert_act_t is not None:
                    g.create_dataset(
                        "expert_actions",
                        data=expert_act_t[:, env_idx],
                        dtype=np.int32,
                        compression=config.compression,
                        compression_opts=config.compression_opts,
                        chunks=(time_chunk,),
                    )

                # Create index entry
                entry = HistoryEntry(
                    history_id=history_id,
                    task_id=task.task_id,
                    env_idx=env_idx,
                    T=T,
                    obs_shape=obs_shape,
                    action_dim=task.action_dim,
                    track=task.track,
                    split=task.split,
                    layout=task.layout,
                    teammate_family=task.teammate_family,
                    teammate_kind=task.teammate_kind,
                    h5_group=f"/{group_name}",
                    has_teammate_actions=has_teammate_actions,
                    has_expert_actions=has_expert_actions,
                )
                index_entries.append(entry)
                history_id += 1

        # Store total count
        h5f.attrs["num_histories"] = history_id

    log.info(f"Packed {history_id} learning histories into {h5_path}")
    return index_entries


def write_index(entries: List[HistoryEntry], index_path: Path) -> None:
    """Write index entries to a JSONL file.

    Args:
        entries: List of HistoryEntry objects
        index_path: Output path for index file
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)

    with open(index_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry.to_dict()) + "\n")

    log.info(f"Wrote index with {len(entries)} entries to {index_path}")


# =============================================================================
# Verification and Debug
# =============================================================================

def verify_determinism(tasks: List[TaskInfo]) -> bool:
    """Verify that task ordering is deterministic.

    Args:
        tasks: List of tasks

    Returns:
        True if ordering is deterministic (sorted by task_id)
    """
    task_ids = [t.task_id for t in tasks]
    return task_ids == sorted(task_ids)


def print_dataset_stats(h5_path: Path, index_path: Path) -> None:
    """Print statistics about the packed dataset.

    Args:
        h5_path: Path to HDF5 file
        index_path: Path to index file
    """
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    # Load index
    entries = []
    with open(index_path, "r") as f:
        for line in f:
            entries.append(json.loads(line.strip()))

    print(f"\nTotal learning histories: {len(entries)}")

    # Count by track/split/layout
    from collections import Counter
    tracks = Counter(e["track"] for e in entries)
    splits = Counter(e["split"] for e in entries)
    layouts = Counter(e["layout"] for e in entries)

    print(f"\nBy track: {dict(tracks)}")
    print(f"By split: {dict(splits)}")
    print(f"By layout: {dict(layouts)}")

    # Sequence length stats
    lengths = [e["T"] for e in entries]
    print(f"\nSequence lengths:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.1f}")

    # Obs dim stats
    obs_dims = set(e["obs_dim"] for e in entries)
    print(f"\nObservation dimensions: {obs_dims}")

    # HDF5 file info
    h5_size_mb = h5_path.stat().st_size / (1024 * 1024)
    print(f"\nHDF5 file size: {h5_size_mb:.2f} MB")

    with h5py.File(h5_path, "r") as f:
        print(f"Compression: {f.attrs.get('compression', 'unknown')}")
        print(f"Compression opts: {f.attrs.get('compression_opts', 'unknown')}")

        # Check chunking on first history
        if "0" in f:
            obs_ds = f["0/obs"]
            print(f"\nChunk info (obs dataset):")
            print(f"  Shape: {obs_ds.shape}")
            print(f"  Chunks: {obs_ds.chunks}")
            print(f"  Compression: {obs_ds.compression}")

    print("=" * 60 + "\n")


def run_smoke_test(h5_path: Path, index_path: Path, num_samples: int = 5) -> bool:
    """Run a smoke test: sample random slices and verify shapes.

    Args:
        h5_path: Path to HDF5 file
        index_path: Path to index file
        num_samples: Number of samples to draw

    Returns:
        True if all tests pass
    """
    print("\n" + "=" * 60)
    print("Smoke Test")
    print("=" * 60)

    # Import adapter (deferred to avoid circular imports during build)
    try:
        from runners.history_adapter import HistoryStore
    except ImportError:
        print("WARNING: history_adapter not found, skipping adapter smoke test")
        print("Running basic HDF5 access test instead...")
        return _run_basic_h5_test(h5_path, index_path, num_samples)

    try:
        store = HistoryStore(str(h5_path), str(index_path))
        print(f"Loaded store with {len(store)} histories")

        rng = np.random.default_rng(0)

        for i in range(num_samples):
            # Sample a random history and slice
            history_id = rng.integers(0, len(store))
            meta = store.get_history_meta(history_id)

            # Sample a random slice
            T = meta["T"]
            seq_len = min(128, T)
            start = rng.integers(0, T - seq_len + 1)

            data = store.load_slice(history_id, start, seq_len)

            print(f"\nSample {i + 1}:")
            print(f"  history_id: {history_id}")
            print(f"  task_id: {meta['task_id']}, env_idx: {meta['env_idx']}")
            print(f"  slice: [{start}:{start + seq_len}] of {T}")
            print(f"  obs shape: {data['obs'].shape}")
            print(f"  actions shape: {data['actions'].shape}")
            print(f"  rewards shape: {data['rewards'].shape}")
            print(f"  dones shape: {data['dones'].shape}")

            # Verify shapes
            # obs_shape can be (obs_dim,) for flattened or (H, W, C) for spatial observations
            expected_obs_shape = (seq_len,) + tuple(meta["obs_shape"])
            assert data["obs"].shape == expected_obs_shape, f"obs shape mismatch: {data['obs'].shape} != {expected_obs_shape}"
            assert data["actions"].shape == (seq_len,), "actions shape mismatch"
            assert data["rewards"].shape == (seq_len,), "rewards shape mismatch"
            assert data["dones"].shape == (seq_len,), "dones shape mismatch"

        print("\n[PASS] All smoke tests passed!")
        print("=" * 60 + "\n")
        return True

    except Exception as e:
        print(f"\n[FAIL] Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60 + "\n")
        return False


def _run_basic_h5_test(h5_path: Path, index_path: Path, num_samples: int) -> bool:
    """Run basic HDF5 access test without adapter."""
    try:
        # Load index
        entries = []
        with open(index_path, "r") as f:
            for line in f:
                entries.append(json.loads(line.strip()))

        rng = np.random.default_rng(0)

        with h5py.File(h5_path, "r") as f:
            for i in range(min(num_samples, len(entries))):
                idx = rng.integers(0, len(entries))
                entry = entries[idx]
                group_name = entry["h5_group"].lstrip("/")

                T = entry["T"]
                seq_len = min(128, T)
                start = rng.integers(0, T - seq_len + 1)

                obs = f[group_name]["obs"][start:start + seq_len]
                actions = f[group_name]["actions"][start:start + seq_len]
                rewards = f[group_name]["rewards"][start:start + seq_len]
                dones = f[group_name]["dones"][start:start + seq_len]

                print(f"\nSample {i + 1}:")
                print(f"  history_id: {entry['history_id']}")
                print(f"  obs shape: {obs.shape}, dtype: {obs.dtype}")
                print(f"  actions shape: {actions.shape}, dtype: {actions.dtype}")

        print("\n[PASS] Basic HDF5 access test passed!")
        return True

    except Exception as e:
        print(f"\n[FAIL] Basic test failed: {e}")
        return False


# =============================================================================
# Main
# =============================================================================

def build_dataset(config: BuildConfig) -> Tuple[Path, Path]:
    """Main entry point for building the dataset.

    Args:
        config: Build configuration

    Returns:
        Tuple of (h5_path, index_path)
    """
    collected_root = Path(config.collected_root)
    h5_path = Path(config.out_h5)
    index_path = Path(config.out_index)

    # Step 1: Discover and validate tasks
    log.info("Discovering tasks...")
    tasks = discover_tasks(collected_root, config)

    if not tasks:
        raise ValueError("No valid tasks found!")

    # Verify determinism
    if not verify_determinism(tasks):
        log.warning("Task ordering is not deterministic! Sorting by task_id...")
        tasks = sorted(tasks, key=lambda t: t.task_id)

    # Step 2: Pack into HDF5
    log.info("Packing histories into HDF5...")
    entries = pack_histories(tasks, h5_path, config)

    # Step 3: Write index
    log.info("Writing index...")
    write_index(entries, index_path)

    # Step 4: Print stats
    print_dataset_stats(h5_path, index_path)

    # Step 5: Run smoke test if requested
    if config.debug_smoke_test:
        success = run_smoke_test(h5_path, index_path)
        if not success:
            log.error("Smoke test failed!")
            sys.exit(1)

    return h5_path, index_path


def parse_args() -> BuildConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build HDF5 dataset from collected histories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Simple usage (auto-generates output paths):
    python -m scripts.build_index --collected_root outputs/task_runs

    # With filters:
    python -m scripts.build_index --collected_root outputs/task_runs --split train

    # Custom output paths:
    python -m scripts.build_index --collected_root outputs/task_runs \\
        --out_h5 datasets/my_data.h5 --out_index datasets/my_data.jsonl
""",
    )

    # Required argument
    parser.add_argument(
        "collected_root", type=str, nargs="?",
        help="Root directory containing task subdirectories"
    )
    parser.add_argument(
        "--collected_root", type=str, dest="collected_root_flag",
        help="Root directory (alternative to positional arg)"
    )

    # Optional output paths (auto-generated if not specified)
    parser.add_argument(
        "--out_h5", type=str, default=None,
        help="Output HDF5 path (default: datasets/{track}_{split}_{layout}_histories.h5)"
    )
    parser.add_argument(
        "--out_index", type=str, default=None,
        help="Output index path (default: datasets/{track}_{split}_{layout}_/histories_index.jsonl)"
    )

    # Common filters
    parser.add_argument(
        "--track", type=str, default="teammate", choices=["teammate", "layout"],
        help="Filter by track (teammate/layout)"
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "test"],
        help="Filter by split (train/test)"
    )
    parser.add_argument(
        "--layout", type=str, default=None,
        help="Filter by layout name"
    )

    # Expert action relabeling
    parser.add_argument(
        "--relabel", action="store_true", default=True,
        help="Compute and store expert_actions using final checkpoint (for DPT training)"
    )

    # Limit histories per task
    parser.add_argument(
        "--max_histories_per_task", type=int, default=128,
        help="Maximum number of histories to pack per task (select best by quality)"
    )

    # GPU selection
    parser.add_argument(
        "--gpu", type=str, default=None,
        help="GPU device ID to use (e.g., '0', '1'). Use '-1' for CPU. If not specified, uses all available GPUs."
    )

    # Advanced options (hidden from main help)
    advanced = parser.add_argument_group("advanced options")
    advanced.add_argument("--max_tasks", type=int, default=None)
    advanced.add_argument("--relabel_batch_size", type=int, default=1024)
    advanced.add_argument("--compression", type=str, default="gzip")
    advanced.add_argument("--compression_opts", type=int, default=6)
    advanced.add_argument("--chunk_time_size", type=int, default=4096)
    advanced.add_argument("--debug_smoke_test", action="store_true")
    advanced.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    # Resolve collected_root from positional or flag
    collected_root = args.collected_root or args.collected_root_flag
    if not collected_root:
        parser.error("collected_root is required (positional or --collected_root)")

    # Auto-generate output paths if not specified
    if args.out_h5 is None or args.out_index is None:
        # Build path components for default names
        track_part = args.track or "all"
        split_part = args.split or "all"
        layout_part = args.layout or "all"
        base_name = f"{track_part}_{split_part}_{layout_part}_"

        out_h5 = args.out_h5 or f"datasets/{base_name}histories.h5"
        out_index = args.out_index or f"datasets/{base_name}/histories_index.jsonl"
    else:
        out_h5 = args.out_h5
        out_index = args.out_index

    return BuildConfig(
        collected_root=collected_root,
        out_h5=out_h5,
        out_index=out_index,
        track=args.track,
        split=args.split,
        layout=args.layout,
        max_tasks=args.max_tasks,
        max_histories_per_task=args.max_histories_per_task,
        relabel=args.relabel,
        relabel_batch_size=args.relabel_batch_size,
        gpu=args.gpu,
        compression=args.compression,
        compression_opts=args.compression_opts,
        chunk_time_size=args.chunk_time_size,
        debug_smoke_test=args.debug_smoke_test,
        verbose=args.verbose,
    )


def main() -> int:
    """Main entry point."""
    config = parse_args()

    # Setup logging
    level = logging.DEBUG if config.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Setup GPU configuration (before any JAX imports)
    if config.gpu is not None:
        if config.gpu == "-1":
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
            log.info("Using CPU")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
            log.info(f"Using GPU: {config.gpu}")
    else:
        log.info("Using all available GPUs")

    try:
        build_dataset(config)
        return 0
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        return 130
    except Exception as e:
        log.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
