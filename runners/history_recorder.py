"""History Recorder for storing training interaction histories.

This module provides utilities for recording and storing learning histories
during PPO training, suitable for later ICRL baseline training.

The recorder stores:
- obs_t: Ego observations at each timestep
- act_t: Ego actions taken
- rew_t: Rewards received
- done_t: Episode termination flags
- teammate_act_t: Teammate actions (optional)
- Episode boundaries and metadata

Storage format:
    out_dir/<task_uid>/
        - history.npz (time-major arrays)
        - episodes.json (per-episode stats and indices)
        - metadata.json (full task spec, config, summary)
        - final_ckpt/ (optional ego parameters)
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""
    episode_id: int
    start_idx: int
    end_idx: int  # Exclusive
    length: int
    total_return: float
    shaped_return: Optional[float] = None
    env_idx: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "length": self.length,
            "total_return": float(self.total_return),
            "shaped_return": float(self.shaped_return) if self.shaped_return is not None else None,
            "env_idx": self.env_idx,
        }


@dataclass
class RecordingBuffer:
    """Buffer for accumulating transitions before saving.

    This class manages in-memory storage during training and provides
    methods for efficient batch recording and periodic flushing.
    """
    # Pre-allocated arrays (will be resized if needed)
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[np.ndarray] = field(default_factory=list)
    dones: List[np.ndarray] = field(default_factory=list)
    teammate_actions: List[np.ndarray] = field(default_factory=list)

    # Episode tracking
    episode_stats: List[EpisodeStats] = field(default_factory=list)
    current_episode_starts: Optional[np.ndarray] = None
    current_episode_returns: Optional[np.ndarray] = None
    current_episode_id: int = 0
    total_timesteps: int = 0

    # For incremental saving - track global state across buffer clears
    global_timestep_offset: int = 0  # Offset for global timestep indexing
    buffered_timesteps: int = 0  # Timesteps in current buffer (reset on clear)

    # For recording only first N steps of each episode
    record_first_steps: int = 0  # 0 = record all steps
    episode_step_counters: Optional[np.ndarray] = None  # Per-env step counter within episode

    def reset(self, num_envs: int, record_envs: int, record_first_steps: int = 0):
        """Reset the buffer for a new recording session.

        Args:
            num_envs: Total number of environments
            record_envs: Number of environments to record
            record_first_steps: Only record first N steps of each episode (0 = record all)
        """
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.teammate_actions = []
        self.episode_stats = []
        self.current_episode_starts = np.zeros(record_envs, dtype=np.int32)
        self.current_episode_returns = np.zeros(record_envs, dtype=np.float32)
        self.current_episode_id = 0
        self.total_timesteps = 0
        self.global_timestep_offset = 0
        self.buffered_timesteps = 0
        self.record_first_steps = record_first_steps
        self.episode_step_counters = np.zeros(record_envs, dtype=np.int32)

    def clear_buffer(self):
        """Clear transition data but preserve episode tracking state.

        This is used for incremental saving to free memory while maintaining
        continuity of episode tracking across saves.
        """
        # Update global offset before clearing
        self.global_timestep_offset = self.total_timesteps

        # Clear transition data
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.teammate_actions = []

        # Clear episode stats (they've been saved)
        self.episode_stats = []

        # Reset buffered timesteps counter
        self.buffered_timesteps = 0

        # Update episode starts to be relative to new buffer
        # (current_episode_starts tracks absolute timesteps, need to adjust)
        if self.current_episode_starts is not None:
            self.current_episode_starts = np.full_like(
                self.current_episode_starts,
                self.total_timesteps
            )

    def add_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        teammate_action: Optional[np.ndarray] = None,
    ):
        """Add a batch of transitions from vectorized environments.

        If record_first_steps > 0, only records the first record_first_steps timesteps
        of each episode. For the record_first_steps-th timestep, done is set to True
        to mark the artificial episode boundary.

        Args:
            obs: Observations (record_envs, ...)
            action: Ego actions (record_envs,)
            reward: Rewards (record_envs,)
            done: Done flags (record_envs,)
            teammate_action: Teammate actions (record_envs,), optional
        """
        record_envs = obs.shape[0]

        # Convert inputs to numpy arrays
        obs_np = np.asarray(obs)
        action_np = np.asarray(action)
        reward_np = np.asarray(reward)
        done_np = np.asarray(done)
        teammate_action_np = np.asarray(teammate_action) if teammate_action is not None else None

        # Determine recording behavior based on record_first_steps
        if self.record_first_steps > 0:
            # Per-env mask: only record if episode step < record_first_steps
            should_record = self.episode_step_counters < self.record_first_steps

            # Check if this is the last step to record (step == record_first_steps - 1)
            # For these envs, set done=True in the recorded data
            is_last_recorded_step = self.episode_step_counters == (self.record_first_steps - 1)

            # Create modified done array: True if real done OR if hitting record limit
            done_for_record = done_np.copy()
            done_for_record[is_last_recorded_step] = True

            # Check if any env should record this timestep
            # We always append data for consistent time-major arrays, but mark
            # non-recorded envs appropriately in episode tracking
            any_should_record = np.any(should_record)
        else:
            # Record all steps
            should_record = np.ones(record_envs, dtype=bool)
            done_for_record = done_np.copy()
            any_should_record = True

        # Append transition data (always append for consistent array shapes)
        if any_should_record:
            self.obs.append(obs_np)
            self.actions.append(action_np)
            self.rewards.append(reward_np)
            self.dones.append(done_for_record)

            if teammate_action_np is not None:
                self.teammate_actions.append(teammate_action_np)

            self.total_timesteps += 1
            self.buffered_timesteps += 1

        # Update episode tracking
        # Track returns using original reward (not affected by done modification)
        self.current_episode_returns += reward_np

        # Check for episode completions
        for env_idx in range(record_envs):
            # For envs still being recorded, check if episode ends
            # (either real done, or hitting record_first_steps limit)
            if should_record[env_idx]:
                episode_ends = done_for_record[env_idx]

                if episode_ends:
                    ep_length = self.total_timesteps - self.current_episode_starts[env_idx]
                    self.episode_stats.append(EpisodeStats(
                        episode_id=self.current_episode_id,
                        start_idx=int(self.current_episode_starts[env_idx]),
                        end_idx=self.total_timesteps,  # Exclusive
                        length=int(ep_length),
                        total_return=float(self.current_episode_returns[env_idx]),
                        env_idx=env_idx,
                    ))
                    self.current_episode_id += 1
                    self.current_episode_starts[env_idx] = self.total_timesteps
                    self.current_episode_returns[env_idx] = 0.0

        # When real episode ends but we're past record_first_steps, still need to
        # reset tracking state for the next episode (even though we don't create stats)
        if self.record_first_steps > 0:
            real_episode_ends = done_np.astype(bool)
            past_limit = ~should_record
            need_reset = real_episode_ends & past_limit
            if np.any(need_reset):
                self.current_episode_starts[need_reset] = self.total_timesteps
                self.current_episode_returns[need_reset] = 0.0

        # Update episode step counters
        if self.record_first_steps > 0:
            self.episode_step_counters += 1
            # Reset counters for envs where real episode ended
            self.episode_step_counters[done_np.astype(bool)] = 0

    def get_arrays(self) -> Dict[str, np.ndarray]:
        """Stack all buffered transitions into numpy arrays.

        Returns:
            Dict with time-major arrays:
                - obs_t: (T, record_envs, ...)
                - act_t: (T, record_envs)
                - rew_t: (T, record_envs)
                - done_t: (T, record_envs)
                - teammate_act_t: (T, record_envs) [if recorded]
        """
        result = {
            "obs_t": np.stack(self.obs, axis=0) if self.obs else np.array([]),
            "act_t": np.stack(self.actions, axis=0) if self.actions else np.array([]),
            "rew_t": np.stack(self.rewards, axis=0) if self.rewards else np.array([]),
            "done_t": np.stack(self.dones, axis=0) if self.dones else np.array([]),
        }

        if self.teammate_actions:
            result["teammate_act_t"] = np.stack(self.teammate_actions, axis=0)

        return result

    def has_data(self) -> bool:
        """Check if buffer has any data to save."""
        return len(self.obs) > 0


class HistoryRecorder:
    """Records and saves training histories for ICRL baseline training.

    This class manages the full lifecycle of history recording:
    1. Initialize with task metadata
    2. Record transitions during training
    3. Save to disk in npz + json format

    Example:
        recorder = HistoryRecorder(
            out_dir="outputs/tasks",
            task_uid="proto_train_cramped_t0_s42",
            task_spec=task_entry.to_json(),
            ppo_config=config,
            record_envs=4,
        )

        # During training loop:
        recorder.record_step(obs, action, reward, done, teammate_action)

        # After training:
        recorder.save()
    """

    def __init__(
        self,
        out_dir: str,
        task_uid: str,
        task_spec: Dict[str, Any],
        ppo_config: Dict[str, Any],
        record_envs: int,
        total_envs: int,
        seed: int,
        extra_metadata: Optional[Dict[str, Any]] = None,
        record_first_steps: int = 0,
    ):
        """Initialize the history recorder.

        Args:
            out_dir: Base output directory
            task_uid: Unique task identifier (used as subdirectory name)
            task_spec: Full task specification from manifest
            ppo_config: PPO hyperparameters used for training
            record_envs: Number of environments to record (subset of total)
            total_envs: Total number of vectorized environments
            seed: Random seed for this task
            extra_metadata: Additional metadata to save
            record_first_steps: Only record first N steps of each episode (0 = record all)
        """
        self.out_dir = Path(out_dir)
        self.task_uid = task_uid
        self.task_dir = self.out_dir / task_uid
        self.task_spec = task_spec
        self.ppo_config = ppo_config
        self.record_envs = record_envs
        self.total_envs = total_envs
        self.seed = seed
        self.extra_metadata = extra_metadata or {}
        self.record_first_steps = record_first_steps

        # Create output directory
        self.task_dir.mkdir(parents=True, exist_ok=True)

        # Initialize recording buffer
        self.buffer = RecordingBuffer()
        self.buffer.reset(total_envs, record_envs, record_first_steps)

        # Training progress tracking
        self.updates_completed = 0
        self.total_steps_trained = 0

        # Incremental saving state
        self.chunk_index = 0  # Current chunk number for incremental saves
        self.chunks_dir = self.task_dir / "chunks"
        self.all_episode_stats: List[EpisodeStats] = []  # Accumulated across all chunks
        self.all_episode_returns: List[float] = []  # For computing final statistics
        self.incremental_mode = False  # Whether we're using incremental saving

    def record_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        teammate_action: Optional[np.ndarray] = None,
    ):
        """Record a single timestep from the vectorized environments.

        Only the first `record_envs` environments are recorded.

        Args:
            obs: Observations from all envs (num_envs, ...)
            action: Ego actions (num_envs,)
            reward: Rewards (num_envs,)
            done: Done flags (num_envs,)
            teammate_action: Teammate actions (num_envs,), optional
        """
        # Only record first record_envs environments
        self.buffer.add_transition(
            obs=obs[:self.record_envs],
            action=action[:self.record_envs],
            reward=reward[:self.record_envs],
            done=done[:self.record_envs],
            teammate_action=teammate_action[:self.record_envs] if teammate_action is not None else None,
        )

    def record_batch(
        self,
        obs_batch: np.ndarray,
        action_batch: np.ndarray,
        reward_batch: np.ndarray,
        done_batch: np.ndarray,
        teammate_action_batch: Optional[np.ndarray] = None,
    ):
        """Record a batch of transitions (e.g., from a rollout).

        Expected shapes: (rollout_len, num_envs, ...)

        Args:
            obs_batch: (T, num_envs, ...)
            action_batch: (T, num_envs)
            reward_batch: (T, num_envs)
            done_batch: (T, num_envs)
            teammate_action_batch: (T, num_envs), optional
        """
        rollout_len = obs_batch.shape[0]
        for t in range(rollout_len):
            ta = teammate_action_batch[t] if teammate_action_batch is not None else None
            self.record_step(
                obs_batch[t],
                action_batch[t],
                reward_batch[t],
                done_batch[t],
                ta,
            )

    def update_training_progress(self, update_step: int, total_steps: int):
        """Track training progress for metadata."""
        self.updates_completed = update_step
        self.total_steps_trained = total_steps

    def get_episode_returns(self) -> List[float]:
        """Get list of episode returns recorded so far (including previous chunks)."""
        current_returns = [ep.total_return for ep in self.buffer.episode_stats]
        return self.all_episode_returns + current_returns

    def get_latest_episodes(self, n: int = 10) -> List[EpisodeStats]:
        """Get the n most recent episodes."""
        return self.buffer.episode_stats[-n:]

    def save(
        self,
        final_params: Optional[Any] = None,
        save_checkpoint: bool = True,
    ):
        """Save all recorded history to disk.

        If incremental mode was used, this saves any remaining buffer data
        and creates final consolidated files. Otherwise, saves everything
        from the buffer.

        Creates:
            - history.npz: Time-major arrays of transitions (or chunks/ directory)
            - episodes.json: Per-episode stats and indices
            - metadata.json: Full task spec, config, summary
            - final_ckpt/: Ego parameters (if save_checkpoint=True)

        Args:
            final_params: Final ego policy parameters (optional)
            save_checkpoint: Whether to save the final checkpoint
        """
        if self.incremental_mode:
            # Save any remaining data in buffer as final chunk
            if self.buffer.has_data():
                self.save_incremental(
                    update_step=self.updates_completed,
                    current_params=final_params,
                    metrics=None,
                    clear_buffer=False,  # Don't clear since we're done
                )

            # Create consolidated episodes.json with all episodes
            all_episodes = [ep.to_dict() for ep in self.all_episode_stats]
            # Also add any episodes from current buffer that weren't saved yet
            all_episodes.extend([ep.to_dict() for ep in self.buffer.episode_stats
                                if ep not in self.all_episode_stats])

            episodes_data = {
                "episodes": all_episodes,
                "total_episodes": len(all_episodes),
                "total_timesteps": self.buffer.total_timesteps,
                "num_chunks": self.chunk_index,
                "is_chunked": True,
            }
            with open(self.task_dir / "episodes.json", "w") as f:
                json.dump(episodes_data, f, indent=2)

            # Compute final summary statistics
            all_returns = self.get_episode_returns()
            summary = {
                "num_episodes": len(all_returns),
                "mean_return": float(np.mean(all_returns)) if all_returns else 0.0,
                "std_return": float(np.std(all_returns)) if all_returns else 0.0,
                "min_return": float(np.min(all_returns)) if all_returns else 0.0,
                "max_return": float(np.max(all_returns)) if all_returns else 0.0,
                "total_timesteps_recorded": self.buffer.total_timesteps,
                "record_envs": self.record_envs,
                "total_envs": self.total_envs,
                "num_chunks": self.chunk_index,
            }

            # Save final metadata
            metadata = {
                "task_spec": self.task_spec,
                "ppo_config": _serialize_config(self.ppo_config),
                "seed": self.seed,
                "summary": summary,
                "updates_completed": self.updates_completed,
                "total_steps_trained": self.total_steps_trained,
                "is_incremental": True,
                "is_complete": True,
                **self.extra_metadata,
            }
            with open(self.task_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Create a manifest file listing all chunks
            chunk_manifest = {
                "num_chunks": self.chunk_index,
                "chunks": [f"chunk_{i}.npz" for i in range(self.chunk_index)],
                "total_timesteps": self.buffer.total_timesteps,
            }
            with open(self.chunks_dir / "manifest.json", "w") as f:
                json.dump(chunk_manifest, f, indent=2)

            # Create empty history.npz as marker (actual data in chunks/)
            # This satisfies validation that expects history.npz to exist
            np.savez_compressed(
                self.task_dir / "history.npz",
                _chunked=np.array([True]),
                _num_chunks=np.array([self.chunk_index]),
                _total_timesteps=np.array([self.buffer.total_timesteps]),
            )

        else:
            # Non-incremental mode: save everything from buffer
            arrays = self.buffer.get_arrays()
            np.savez_compressed(
                self.task_dir / "history.npz",
                **arrays,
            )

            # Save episode info
            episodes_data = {
                "episodes": [ep.to_dict() for ep in self.buffer.episode_stats],
                "total_episodes": len(self.buffer.episode_stats),
                "total_timesteps": self.buffer.total_timesteps,
            }
            with open(self.task_dir / "episodes.json", "w") as f:
                json.dump(episodes_data, f, indent=2)

            # Compute summary statistics
            returns = self.get_episode_returns()
            summary = {
                "num_episodes": len(returns),
                "mean_return": float(np.mean(returns)) if returns else 0.0,
                "std_return": float(np.std(returns)) if returns else 0.0,
                "min_return": float(np.min(returns)) if returns else 0.0,
                "max_return": float(np.max(returns)) if returns else 0.0,
                "total_timesteps_recorded": self.buffer.total_timesteps,
                "record_envs": self.record_envs,
                "total_envs": self.total_envs,
            }

            # Save metadata
            metadata = {
                "task_spec": self.task_spec,
                "ppo_config": _serialize_config(self.ppo_config),
                "seed": self.seed,
                "summary": summary,
                "updates_completed": self.updates_completed,
                "total_steps_trained": self.total_steps_trained,
                **self.extra_metadata,
            }
            with open(self.task_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        # Save final checkpoint
        if save_checkpoint and final_params is not None:
            ckpt_dir = self.task_dir / "final_ckpt"
            ckpt_dir.mkdir(exist_ok=True)

            # Flatten params to numpy and save
            params_np = jax.tree.map(lambda x: np.asarray(x), final_params)

            # Save using numpy savez
            flat_params = {}
            _flatten_pytree(params_np, "", flat_params)
            np.savez(ckpt_dir / "params.npz", **flat_params)

        return str(self.task_dir)

    def save_incremental(
        self,
        update_step: int,
        current_params: Optional[Any] = None,
        metrics: Optional[List[Dict[str, Any]]] = None,
        clear_buffer: bool = True,
    ) -> str:
        """Save current buffer incrementally and optionally clear it to free memory.

        This method saves the current buffer contents as a numbered chunk,
        accumulates episode statistics, and clears the buffer to prevent
        memory overflow during long training runs.

        Output structure:
            {task_dir}/
                chunks/
                    chunk_0.npz (history data)
                    chunk_0_episodes.json
                    chunk_1.npz
                    chunk_1_episodes.json
                    ...
                params/
                    params_update_{N}.npz (latest params)
                metrics.json (cumulative metrics)

        Args:
            update_step: Current update step number
            current_params: Current ego policy parameters (optional)
            metrics: List of metrics dicts from training (optional)
            clear_buffer: Whether to clear buffer after saving (default: True)

        Returns:
            Path to the saved chunk file
        """
        self.incremental_mode = True

        # Skip if no data in buffer
        if not self.buffer.has_data():
            return ""

        # Create chunks directory
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        # Save history arrays for this chunk
        arrays = self.buffer.get_arrays()
        chunk_path = self.chunks_dir / f"chunk_{self.chunk_index}.npz"
        np.savez_compressed(chunk_path, **arrays)

        # Save episode info for this chunk
        chunk_episodes = [ep.to_dict() for ep in self.buffer.episode_stats]
        episodes_path = self.chunks_dir / f"chunk_{self.chunk_index}_episodes.json"
        episodes_data = {
            "chunk_index": self.chunk_index,
            "update_step": update_step,
            "episodes": chunk_episodes,
            "num_episodes": len(chunk_episodes),
            "timesteps_in_chunk": self.buffer.buffered_timesteps,
            "global_timestep_start": self.buffer.global_timestep_offset,
            "global_timestep_end": self.buffer.total_timesteps,
        }
        with open(episodes_path, "w") as f:
            json.dump(episodes_data, f, indent=2)

        # Accumulate episode stats for final summary
        self.all_episode_stats.extend(self.buffer.episode_stats)
        self.all_episode_returns.extend([ep.total_return for ep in self.buffer.episode_stats])

        # Save current parameters
        if current_params is not None:
            params_dir = self.task_dir / "params"
            params_dir.mkdir(parents=True, exist_ok=True)
            params_np = jax.tree.map(lambda x: np.asarray(x), current_params)
            flat_params = {}
            _flatten_pytree(params_np, "", flat_params)
            # Save with update step in name (overwrite previous to save space)
            np.savez(params_dir / f"params_update_{update_step}.npz", **flat_params)
            # Also save as "latest" for easy access
            np.savez(params_dir / "params_latest.npz", **flat_params)

        # Save cumulative metrics
        if metrics is not None:
            with open(self.task_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        # Save progress metadata
        all_returns = self.all_episode_returns
        progress_metadata = {
            "task_spec": self.task_spec,
            "ppo_config": _serialize_config(self.ppo_config),
            "seed": self.seed,
            "updates_completed": update_step,
            "total_steps_trained": self.total_steps_trained,
            "total_timesteps_recorded": self.buffer.total_timesteps,
            "num_chunks_saved": self.chunk_index + 1,
            "total_episodes": len(all_returns),
            "mean_return": float(np.mean(all_returns)) if all_returns else 0.0,
            "is_incremental": True,
            "is_complete": False,
            **self.extra_metadata,
        }
        with open(self.task_dir / "progress.json", "w") as f:
            json.dump(progress_metadata, f, indent=2)

        saved_chunk = self.chunk_index
        self.chunk_index += 1

        # Clear buffer to free memory
        if clear_buffer:
            self.buffer.clear_buffer()
            # Force garbage collection
            import gc
            gc.collect()

        return str(chunk_path)


def _serialize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert config to JSON-serializable format."""
    result = {}
    for k, v in config.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            result[k] = v
        elif isinstance(v, dict):
            result[k] = _serialize_config(v)
        elif isinstance(v, (list, tuple)):
            result[k] = list(v)
        elif hasattr(v, 'item'):
            result[k] = v.item()
        elif hasattr(v, 'tolist'):
            result[k] = v.tolist()
        else:
            result[k] = str(v)
    return result


def _flatten_pytree(tree: Any, prefix: str, flat_dict: Dict[str, np.ndarray]):
    """Flatten a pytree to a dict with dotted keys."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            _flatten_pytree(v, new_prefix, flat_dict)
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            new_prefix = f"{prefix}.{i}" if prefix else str(i)
            _flatten_pytree(v, new_prefix, flat_dict)
    elif isinstance(tree, np.ndarray):
        flat_dict[prefix] = tree
    elif hasattr(tree, '__array__'):
        flat_dict[prefix] = np.asarray(tree)


def load_history(task_dir: str, load_chunks: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Dict[str, Any]]:
    """Load a saved history from disk.

    Supports both regular and chunked (incremental) history formats.

    Args:
        task_dir: Path to the task output directory
        load_chunks: If True and history is chunked, load and concatenate all chunks.
                    If False, return empty arrays for chunked history.

    Returns:
        Tuple of:
            - history: Dict of numpy arrays (obs_t, act_t, rew_t, done_t, ...)
            - episodes: Episode info dict
            - metadata: Full metadata dict
    """
    task_dir = Path(task_dir)

    # Load history arrays
    with np.load(task_dir / "history.npz") as f:
        history = {k: f[k] for k in f.files}

    # Check if this is chunked data
    if "_chunked" in history and history["_chunked"][0]:
        if load_chunks:
            # Load and concatenate all chunks
            history = load_chunked_history(task_dir)
        else:
            # Return marker info only
            pass

    # Load episodes
    with open(task_dir / "episodes.json", "r") as f:
        episodes = json.load(f)

    # Load metadata
    with open(task_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    return history, episodes, metadata


def load_chunked_history(task_dir: str) -> Dict[str, np.ndarray]:
    """Load and concatenate all chunks from a chunked history.

    Args:
        task_dir: Path to the task output directory

    Returns:
        Dict of concatenated numpy arrays (obs_t, act_t, rew_t, done_t, ...)
    """
    task_dir = Path(task_dir)
    chunks_dir = task_dir / "chunks"

    # Load manifest
    manifest_path = chunks_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        chunk_files = manifest["chunks"]
    else:
        # Fall back to scanning directory
        chunk_files = sorted([f.name for f in chunks_dir.glob("chunk_*.npz")
                             if "_episodes" not in f.name])

    if not chunk_files:
        return {
            "obs_t": np.array([]),
            "act_t": np.array([]),
            "rew_t": np.array([]),
            "done_t": np.array([]),
        }

    # Load all chunks
    all_arrays: Dict[str, List[np.ndarray]] = {}

    for chunk_file in chunk_files:
        chunk_path = chunks_dir / chunk_file
        with np.load(chunk_path) as f:
            for key in f.files:
                if key not in all_arrays:
                    all_arrays[key] = []
                arr = f[key]
                if arr.size > 0:
                    all_arrays[key].append(arr)

    # Concatenate arrays along time dimension (axis 0)
    result = {}
    for key, arrays in all_arrays.items():
        if arrays:
            result[key] = np.concatenate(arrays, axis=0)
        else:
            result[key] = np.array([])

    return result


def verify_history_consistency(task_dir: str) -> List[str]:
    """Verify that a saved history is consistent.

    Checks:
    - Episode boundaries match done flags
    - Array shapes are consistent
    - Episode stats match actual data

    Args:
        task_dir: Path to the task output directory

    Returns:
        List of error messages (empty if consistent)
    """
    errors = []
    task_dir = Path(task_dir)

    try:
        history, episodes, metadata = load_history(str(task_dir))
    except Exception as e:
        return [f"Failed to load history: {e}"]

    # Check array shapes
    T = history["obs_t"].shape[0] if history["obs_t"].size > 0 else 0
    for key in ["act_t", "rew_t", "done_t"]:
        if key in history and history[key].shape[0] != T:
            errors.append(f"Array {key} has wrong length: {history[key].shape[0]} vs {T}")

    # Verify episode boundaries
    ep_list = episodes.get("episodes", [])
    for ep in ep_list:
        start, end = ep["start_idx"], ep["end_idx"]
        env_idx = ep["env_idx"]

        if end > T:
            errors.append(f"Episode {ep['episode_id']} end_idx {end} exceeds history length {T}")
            continue

        if start >= end:
            errors.append(f"Episode {ep['episode_id']} has invalid range [{start}, {end})")
            continue

        # Check that done flag is set at end-1
        if history["done_t"].size > 0:
            done_at_end = history["done_t"][end - 1, env_idx]
            if not done_at_end:
                errors.append(f"Episode {ep['episode_id']} done flag not set at end index {end - 1}")

    return errors
