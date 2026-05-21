"""Online buffer for AD evaluation with step-level updates.

This implements the step-by-step context buffer update strategy for Algorithm Distillation evaluation semantics.

Key difference from DPT:
- DPT updates buffer only after each episode completes
- AD updates buffer at EVERY STEP with (obs, prev_action, prev_reward)

Also provides VectorizedOnlineBuffer for batched/parallel evaluation
of multiple tasks simultaneously.

The buffer maintains a rolling window of recent steps that can be used as
context for the transformer model.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union

import numpy as np
import jax.numpy as jnp


@dataclass
class OnlineBuffer:
    """Online buffer for AD step-level context updates.

    Following AD evaluation semantics:
    - Buffer stores (obs, prev_action, prev_reward) tuples
    - Optionally stores prev_teammate_action for teammate action conditioning
    - Updated at EVERY step (not just episode boundaries)
    - Uses FIFO eviction when buffer is full (circular buffer)

    The buffer maintains the context needed for AD inference:
    - At step t, we have context from steps [t-ctx_len, t-1]
    - We use this context plus current (obs_t, prev_action_t, prev_reward_t)
      to predict action_t

    Implementation uses circular buffer indexing for O(1) insertion without
    memory allocation (avoiding np.roll which is O(n) and allocates).

    Attributes:
        max_len: Maximum number of steps to store
        obs_shape: Observation shape (can be tuple like (H, W, C) or int for flat obs)
        use_teammate_actions: Whether to track teammate actions
        obs: Observation buffer (max_len, *obs_shape)
        prev_actions: Previous action buffer (max_len,)
        prev_rewards: Previous reward buffer (max_len,)
        prev_teammate_actions: Previous teammate action buffer (max_len,), optional
        current_len: Number of valid steps in buffer (capped at max_len)
        _write_idx: Circular buffer write index (where next write will occur)
    """
    max_len: int
    obs_shape: Union[int, Tuple[int, ...]]  # Support both flat (int) and multi-dim (tuple)
    use_teammate_actions: bool = False

    # Buffers (initialized in __post_init__)
    obs: np.ndarray = field(init=False)
    prev_actions: np.ndarray = field(init=False)
    prev_rewards: np.ndarray = field(init=False)
    prev_teammate_actions: Optional[np.ndarray] = field(init=False)

    # Tracking
    current_len: int = field(default=0, init=False)
    total_steps: int = field(default=0, init=False)  # Total steps seen (for debugging)
    truncation_count: int = field(default=0, init=False)  # Number of times buffer was truncated
    _write_idx: int = field(default=0, init=False)  # Circular buffer write position

    def __post_init__(self):
        """Initialize buffers."""
        # Handle both flat obs_dim (int) and multi-dim obs_shape (tuple)
        if isinstance(self.obs_shape, int):
            obs_buffer_shape = (self.max_len, self.obs_shape)
        else:
            obs_buffer_shape = (self.max_len,) + tuple(self.obs_shape)
        self.obs = np.zeros(obs_buffer_shape, dtype=np.float32)
        self.prev_actions = np.zeros(self.max_len, dtype=np.int32)
        self.prev_rewards = np.zeros(self.max_len, dtype=np.float32)
        if self.use_teammate_actions:
            self.prev_teammate_actions = np.zeros(self.max_len, dtype=np.int32)
        else:
            self.prev_teammate_actions = None
        self.current_len = 0
        self.total_steps = 0
        self.truncation_count = 0
        self._write_idx = 0

    # Backward compatibility alias
    @property
    def obs_dim(self) -> Union[int, Tuple[int, ...]]:
        """Backward compatibility: return obs_shape."""
        return self.obs_shape

    def reset(self):
        """Reset buffer to empty state.

        Called at the start of evaluation on a new task.
        """
        self.obs.fill(0)
        self.prev_actions.fill(0)
        self.prev_rewards.fill(0)
        if self.prev_teammate_actions is not None:
            self.prev_teammate_actions.fill(0)
        self.current_len = 0
        self.total_steps = 0
        self.truncation_count = 0
        self._write_idx = 0

    def _get_ordered_indices(self, length: Optional[int] = None) -> np.ndarray:
        """Get indices to read buffer data in temporal order.

        For a circular buffer, when full, the oldest data is at _write_idx
        and newest at (_write_idx - 1) % max_len. This method returns indices
        that will read data from oldest to newest.

        Args:
            length: Number of most recent elements to get (default: current_len)

        Returns:
            Array of indices in temporal order (oldest to newest)
        """
        if length is None:
            length = self.current_len

        if self.current_len < self.max_len:
            # Buffer not full - data is in order from 0 to current_len-1
            start = max(0, self.current_len - length)
            return np.arange(start, self.current_len)
        else:
            # Buffer is full (circular)
            # Oldest is at _write_idx, newest is at (_write_idx - 1) % max_len
            # We want the last `length` elements in temporal order
            if length >= self.max_len:
                # Return all in order: from _write_idx to end, then 0 to _write_idx-1
                return np.concatenate([
                    np.arange(self._write_idx, self.max_len),
                    np.arange(0, self._write_idx)
                ])
            else:
                # Return last `length` elements
                # Newest is at (_write_idx - 1) % max_len
                # Start from (_write_idx - length) % max_len
                start = (self._write_idx - length) % self.max_len
                if start < self._write_idx:
                    # Contiguous range
                    return np.arange(start, self._write_idx)
                else:
                    # Wraps around
                    return np.concatenate([
                        np.arange(start, self.max_len),
                        np.arange(0, self._write_idx)
                    ])

    def add_step(
        self,
        obs: np.ndarray,
        prev_action: int,
        prev_reward: float,
        prev_teammate_action: Optional[int] = None,
    ):
        """Add a step to the buffer using circular indexing.

        This is called at EVERY step during AD evaluation, before selecting
        the action. It adds the current observation and the previous action/reward
        to the context buffer.

        Uses O(1) circular buffer insertion - no array copying or allocation.

        Args:
            obs: Current observation (obs_dim,)
            prev_action: Action taken at t-1 (0 for first step)
            prev_reward: Reward received at t-1 (0 for first step)
            prev_teammate_action: Teammate action at t-1 (optional)
        """
        self.total_steps += 1

        # Write at current position (circular indexing)
        self.obs[self._write_idx] = obs
        self.prev_actions[self._write_idx] = prev_action
        self.prev_rewards[self._write_idx] = prev_reward
        if self.prev_teammate_actions is not None and prev_teammate_action is not None:
            self.prev_teammate_actions[self._write_idx] = prev_teammate_action

        # Advance write index (wrap around)
        self._write_idx = (self._write_idx + 1) % self.max_len

        # Update length (capped at max_len)
        if self.current_len < self.max_len:
            self.current_len += 1
        else:
            self.truncation_count += 1

    def get_context(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Get the current context for model input.

        Returns data in temporal order (oldest to newest) from the circular buffer.

        Returns:
            Tuple of (obs, prev_actions, prev_rewards, prev_teammate_actions) each of shape (ctx_len, ...)
            where ctx_len = current_len (may be less than max_len)
            prev_teammate_actions is None if use_teammate_actions is False
        """
        if self.current_len == 0:
            # Empty buffer - return zeros with length 0
            # Handle both flat obs_dim (int) and multi-dim obs_shape (tuple)
            if isinstance(self.obs_shape, int):
                empty_obs_shape = (0, self.obs_shape)
            else:
                empty_obs_shape = (0,) + tuple(self.obs_shape)
            return (
                np.zeros(empty_obs_shape, dtype=np.float32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int32) if self.use_teammate_actions else None,
            )

        # Get indices in temporal order (handles circular buffer)
        indices = self._get_ordered_indices()

        prev_teammate = None
        if self.prev_teammate_actions is not None:
            prev_teammate = self.prev_teammate_actions[indices].copy()

        return (
            self.obs[indices].copy(),
            self.prev_actions[indices].copy(),
            self.prev_rewards[indices].copy(),
            prev_teammate,
        )

    def get_padded_context(
        self,
        target_len: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Get context padded to a fixed length.

        Useful for batched inference where all samples need same length.
        Padding is added at the BEGINNING (older positions) to maintain
        causal alignment at the end.

        Returns data in temporal order from the circular buffer.

        Args:
            target_len: Target length to pad to

        Returns:
            Tuple of (obs, prev_actions, prev_rewards, prev_teammate_actions) each of shape (target_len, ...)
            prev_teammate_actions is None if use_teammate_actions is False
        """
        if self.current_len >= target_len:
            # Return last target_len steps in temporal order
            indices = self._get_ordered_indices(target_len)
            prev_teammate = self.prev_teammate_actions[indices].copy() if self.prev_teammate_actions is not None else None
            return (
                self.obs[indices].copy(),
                self.prev_actions[indices].copy(),
                self.prev_rewards[indices].copy(),
                prev_teammate,
            )

        # Need to pad at the beginning
        pad_len = target_len - self.current_len

        # Handle both flat obs_dim (int) and multi-dim obs_shape (tuple)
        if isinstance(self.obs_shape, int):
            obs_padded_shape = (target_len, self.obs_shape)
        else:
            obs_padded_shape = (target_len,) + tuple(self.obs_shape)
        obs_padded = np.zeros(obs_padded_shape, dtype=np.float32)
        prev_actions_padded = np.zeros(target_len, dtype=np.int32)
        prev_rewards_padded = np.zeros(target_len, dtype=np.float32)
        prev_teammate_padded = np.zeros(target_len, dtype=np.int32) if self.use_teammate_actions else None

        # Copy valid data at the end (in temporal order)
        if self.current_len > 0:
            indices = self._get_ordered_indices()
            obs_padded[pad_len:] = self.obs[indices]
            prev_actions_padded[pad_len:] = self.prev_actions[indices]
            prev_rewards_padded[pad_len:] = self.prev_rewards[indices]
            if prev_teammate_padded is not None and self.prev_teammate_actions is not None:
                prev_teammate_padded[pad_len:] = self.prev_teammate_actions[indices]

        return obs_padded, prev_actions_padded, prev_rewards_padded, prev_teammate_padded

    def get_jax_context(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        """Get context as JAX arrays.

        Returns:
            Tuple of JAX arrays (obs, prev_actions, prev_rewards, prev_teammate_actions)
        """
        obs, prev_actions, prev_rewards, prev_teammate = self.get_context()
        return (
            jnp.array(obs),
            jnp.array(prev_actions),
            jnp.array(prev_rewards),
            jnp.array(prev_teammate) if prev_teammate is not None else None,
        )

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about buffer state.

        Returns:
            Dictionary with buffer stats
        """
        return {
            "current_len": self.current_len,
            "max_len": self.max_len,
            "total_steps": self.total_steps,
            "truncation_count": self.truncation_count,
            "buffer_utilization": self.current_len / self.max_len,
        }

    def print_debug(self, prefix: str = ""):
        """Print debug information."""
        info = self.get_debug_info()
        print(f"{prefix}OnlineBuffer: len={info['current_len']}/{info['max_len']}, "
              f"total_steps={info['total_steps']}, truncations={info['truncation_count']}")


@dataclass
class VectorizedOnlineBuffer:
    """Vectorized online buffer for batched AD evaluation.

    Manages N parallel OnlineBuffers for evaluating multiple tasks simultaneously.
    Provides vectorized operations for efficient batched context retrieval.

    Attributes:
        batch_size: Number of parallel tasks (N)
        max_len: Maximum context length per task
        obs_shape: Observation shape
        use_teammate_actions: Whether to track teammate actions
    """
    batch_size: int
    max_len: int
    obs_shape: Union[int, Tuple[int, ...]]
    use_teammate_actions: bool = False

    # Internal buffers (initialized in __post_init__)
    _buffers: List[OnlineBuffer] = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize N parallel buffers."""
        self._buffers = [
            OnlineBuffer(
                max_len=self.max_len,
                obs_shape=self.obs_shape,
                use_teammate_actions=self.use_teammate_actions,
            )
            for _ in range(self.batch_size)
        ]

    def reset(self):
        """Reset all buffers."""
        for buf in self._buffers:
            buf.reset()

    def reset_task(self, task_idx: int):
        """Reset buffer for a specific task."""
        self._buffers[task_idx].reset()

    def add_step(
        self,
        task_idx: int,
        obs: np.ndarray,
        prev_action: int,
        prev_reward: float,
        prev_teammate_action: Optional[int] = None,
    ):
        """Add step for a specific task."""
        self._buffers[task_idx].add_step(obs, prev_action, prev_reward, prev_teammate_action)

    def add_steps_batch(
        self,
        obs: np.ndarray,
        prev_actions: np.ndarray,
        prev_rewards: np.ndarray,
        prev_teammate_actions: Optional[np.ndarray] = None,
    ):
        """Add steps for all tasks in batch.

        Args:
            obs: Observations (batch_size, *obs_shape)
            prev_actions: Previous actions (batch_size,)
            prev_rewards: Previous rewards (batch_size,)
            prev_teammate_actions: Previous teammate actions (batch_size,), optional
        """
        for i in range(self.batch_size):
            tm_action = int(prev_teammate_actions[i]) if prev_teammate_actions is not None else None
            self._buffers[i].add_step(
                obs[i], int(prev_actions[i]), float(prev_rewards[i]), tm_action
            )

    def get_context(self, task_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Get context for a specific task."""
        return self._buffers[task_idx].get_context()

    def get_padded_context(
        self,
        task_idx: int,
        target_len: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Get padded context for a specific task."""
        return self._buffers[task_idx].get_padded_context(target_len)

    def get_padded_context_batch(
        self,
        target_len: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Get batched padded context for all tasks.

        Args:
            target_len: Target context length

        Returns:
            Tuple of batched arrays:
            - obs: (batch_size, target_len, *obs_shape)
            - prev_actions: (batch_size, target_len)
            - prev_rewards: (batch_size, target_len)
            - prev_teammate_actions: (batch_size, target_len) or None
        """
        # Get context from each buffer
        contexts = [buf.get_padded_context(target_len) for buf in self._buffers]

        # Stack into batched arrays
        batch_obs = np.stack([ctx[0] for ctx in contexts], axis=0)
        batch_prev_actions = np.stack([ctx[1] for ctx in contexts], axis=0)
        batch_prev_rewards = np.stack([ctx[2] for ctx in contexts], axis=0)

        if self.use_teammate_actions:
            batch_prev_teammate_actions = np.stack([ctx[3] for ctx in contexts], axis=0)
        else:
            batch_prev_teammate_actions = None

        return batch_obs, batch_prev_actions, batch_prev_rewards, batch_prev_teammate_actions

    @property
    def current_lens(self) -> np.ndarray:
        """Get current lengths for all tasks."""
        return np.array([buf.current_len for buf in self._buffers], dtype=np.int32)

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state for checkpointing.

        Returns:
            Dictionary containing all buffer state that can be saved to disk.
        """
        buffer_states = []
        for buf in self._buffers:
            buffer_states.append({
                "obs": buf.obs.copy(),
                "prev_actions": buf.prev_actions.copy(),
                "prev_rewards": buf.prev_rewards.copy(),
                "prev_teammate_actions": buf.prev_teammate_actions.copy() if buf.prev_teammate_actions is not None else None,
                "current_len": buf.current_len,
                "total_steps": buf.total_steps,
                "truncation_count": buf.truncation_count,
                "_write_idx": buf._write_idx,
            })
        return {
            "batch_size": self.batch_size,
            "max_len": self.max_len,
            "obs_shape": self.obs_shape,
            "use_teammate_actions": self.use_teammate_actions,
            "buffer_states": buffer_states,
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore buffer state from checkpoint.

        Args:
            state: Dictionary from get_state()
        """
        assert state["batch_size"] == self.batch_size
        assert state["max_len"] == self.max_len

        for i, buf_state in enumerate(state["buffer_states"]):
            buf = self._buffers[i]
            buf.obs[:] = buf_state["obs"]
            buf.prev_actions[:] = buf_state["prev_actions"]
            buf.prev_rewards[:] = buf_state["prev_rewards"]
            if buf.prev_teammate_actions is not None and buf_state["prev_teammate_actions"] is not None:
                buf.prev_teammate_actions[:] = buf_state["prev_teammate_actions"]
            buf.current_len = buf_state["current_len"]
            buf.total_steps = buf_state["total_steps"]
            buf.truncation_count = buf_state["truncation_count"]
            buf._write_idx = buf_state["_write_idx"]
