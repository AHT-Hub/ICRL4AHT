"""Episode buffer for DPT evaluation.

This module provides the EpisodeBuffer class for storing episode context
during DPT evaluation, following DPT evaluation semantics:
- Context buffer only updated after episode completes
- FIFO eviction when buffer is full

Also provides VectorizedEpisodeBuffer for batched/parallel evaluation
of multiple tasks simultaneously.

Performance optimized with circular buffer (O(1) insertion instead of O(n) with np.roll).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class EpisodeBuffer:
    """Buffer for storing episode context.

    Following DPT evaluation semantics:
    - Context buffer only updated after episode completes
    - FIFO eviction when buffer is full

    Uses circular buffer for O(1) insertion instead of O(n) np.roll.
    """
    max_episodes: int
    max_steps: int
    obs_shape: Union[int, Tuple[int, ...]]  # Support both flat (int) and multi-dim (tuple)
    use_teammate_actions: bool = False

    def __post_init__(self):
        """Initialize buffers."""
        # Handle both flat obs_dim (int) and multi-dim obs_shape (tuple)
        if isinstance(self.obs_shape, int):
            ep_obs_shape = (self.max_episodes, self.max_steps, self.obs_shape)
            curr_obs_shape = (self.max_steps, self.obs_shape)
        else:
            ep_obs_shape = (self.max_episodes, self.max_steps) + tuple(self.obs_shape)
            curr_obs_shape = (self.max_steps,) + tuple(self.obs_shape)

        # (max_episodes, max_steps, *obs_shape)
        self.obs = np.zeros(ep_obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.max_episodes, self.max_steps), dtype=np.int32)
        self.next_obs = np.zeros(ep_obs_shape, dtype=np.float32)
        self.rewards = np.zeros((self.max_episodes, self.max_steps), dtype=np.float32)
        if self.use_teammate_actions:
            self.teammate_actions = np.zeros((self.max_episodes, self.max_steps), dtype=np.int32)
        else:
            self.teammate_actions = None
        self.lengths = np.zeros(self.max_episodes, dtype=np.int32)

        # Current episode buffers
        self.current_obs = np.zeros(curr_obs_shape, dtype=np.float32)
        self.current_actions = np.zeros(self.max_steps, dtype=np.int32)
        self.current_next_obs = np.zeros(curr_obs_shape, dtype=np.float32)
        self.current_rewards = np.zeros(self.max_steps, dtype=np.float32)
        if self.use_teammate_actions:
            self.current_teammate_actions = np.zeros(self.max_steps, dtype=np.int32)
        else:
            self.current_teammate_actions = None
        self.current_step = 0

        # Circular buffer state
        self._head = 0  # Next position to write (oldest episode position)
        self.num_episodes = 0

    # Backward compatibility alias
    @property
    def obs_dim(self) -> Union[int, Tuple[int, ...]]:
        """Backward compatibility: return obs_shape."""
        return self.obs_shape

    def reset(self):
        """Reset all buffers."""
        self.obs.fill(0)
        self.actions.fill(0)
        self.next_obs.fill(0)
        self.rewards.fill(0)
        if self.teammate_actions is not None:
            self.teammate_actions.fill(0)
        self.lengths.fill(0)
        self.current_step = 0
        self._head = 0
        self.num_episodes = 0

    def add_transition(
        self,
        obs: np.ndarray,
        action: int,
        next_obs: np.ndarray,
        reward: float,
        teammate_action: Optional[int] = None,
    ):
        """Add transition to current episode buffer.

        Args:
            obs: Observation
            action: Action taken
            next_obs: Next observation
            reward: Reward received
            teammate_action: Teammate action taken (optional)
        """
        if self.current_step < self.max_steps:
            self.current_obs[self.current_step] = obs
            self.current_actions[self.current_step] = action
            self.current_next_obs[self.current_step] = next_obs
            self.current_rewards[self.current_step] = reward
            if self.current_teammate_actions is not None and teammate_action is not None:
                self.current_teammate_actions[self.current_step] = teammate_action
            self.current_step += 1

    def finish_episode(self):
        """Move current episode to context buffer (circular buffer, O(1) insertion)."""
        if self.current_step > 0:
            # Insert at head position (overwrites oldest episode when full)
            write_idx = self._head

            # Clear the slot before writing to avoid stale data
            self.obs[write_idx].fill(0)
            self.actions[write_idx].fill(0)
            self.next_obs[write_idx].fill(0)
            self.rewards[write_idx].fill(0)
            if self.teammate_actions is not None:
                self.teammate_actions[write_idx].fill(0)

            # Insert current episode
            self.obs[write_idx, :self.current_step] = self.current_obs[:self.current_step]
            self.actions[write_idx, :self.current_step] = self.current_actions[:self.current_step]
            self.next_obs[write_idx, :self.current_step] = self.current_next_obs[:self.current_step]
            self.rewards[write_idx, :self.current_step] = self.current_rewards[:self.current_step]
            if self.teammate_actions is not None and self.current_teammate_actions is not None:
                self.teammate_actions[write_idx, :self.current_step] = self.current_teammate_actions[:self.current_step]
            self.lengths[write_idx] = self.current_step

            # Advance head (circular)
            self._head = (self._head + 1) % self.max_episodes
            self.num_episodes = min(self.num_episodes + 1, self.max_episodes)

        # Reset current episode
        self.current_obs.fill(0)
        self.current_actions.fill(0)
        self.current_next_obs.fill(0)
        self.current_rewards.fill(0)
        if self.current_teammate_actions is not None:
            self.current_teammate_actions.fill(0)
        self.current_step = 0

    def _get_episode_order(self) -> np.ndarray:
        """Get episode indices in chronological order (oldest to newest).

        Returns:
            Array of episode indices in order from oldest to newest.
        """
        if self.num_episodes == 0:
            return np.array([], dtype=np.int32)

        if self.num_episodes < self.max_episodes:
            # Buffer not full yet: episodes are at indices 0 to num_episodes-1
            # Head points to next write position, so oldest is at 0
            return np.arange(self.num_episodes, dtype=np.int32)
        else:
            # Buffer is full: head points to oldest (will be overwritten next)
            # Order: head, head+1, ..., max-1, 0, 1, ..., head-1
            return np.concatenate([
                np.arange(self._head, self.max_episodes, dtype=np.int32),
                np.arange(0, self._head, dtype=np.int32)
            ])

    def _get_obs_buffer_shape(self, num_transitions: int) -> Tuple[int, ...]:
        """Get the shape for observation buffers with given number of transitions."""
        if isinstance(self.obs_shape, int):
            return (num_transitions, self.obs_shape)
        else:
            return (num_transitions,) + tuple(self.obs_shape)

    def get_context(self, max_transitions: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Get flattened context for model input.

        Args:
            max_transitions: Maximum number of transitions to return

        Returns:
            Tuple of (obs, actions, next_obs, rewards, teammate_actions) each of shape (K, ...)
            teammate_actions is None if use_teammate_actions is False
        """
        if self.num_episodes == 0:
            # No context yet - return zeros
            obs_shape = self._get_obs_buffer_shape(max_transitions)
            return (
                np.zeros(obs_shape, dtype=np.float32),
                np.zeros(max_transitions, dtype=np.int32),
                np.zeros(obs_shape, dtype=np.float32),
                np.zeros(max_transitions, dtype=np.float32),
                np.zeros(max_transitions, dtype=np.int32) if self.use_teammate_actions else None,
            )

        # Get episode indices in chronological order
        episode_order = self._get_episode_order()

        # Calculate total transitions to pre-allocate
        total_transitions = sum(self.lengths[idx] for idx in episode_order)

        if total_transitions == 0:
            obs_shape = self._get_obs_buffer_shape(max_transitions)
            return (
                np.zeros(obs_shape, dtype=np.float32),
                np.zeros(max_transitions, dtype=np.int32),
                np.zeros(obs_shape, dtype=np.float32),
                np.zeros(max_transitions, dtype=np.float32),
                np.zeros(max_transitions, dtype=np.int32) if self.use_teammate_actions else None,
            )

        # Pre-allocate arrays for better performance
        obs_shape = self._get_obs_buffer_shape(total_transitions)
        ctx_obs = np.zeros(obs_shape, dtype=np.float32)
        ctx_actions = np.zeros(total_transitions, dtype=np.int32)
        ctx_next_obs = np.zeros(obs_shape, dtype=np.float32)
        ctx_rewards = np.zeros(total_transitions, dtype=np.float32)
        ctx_teammate_actions = np.zeros(total_transitions, dtype=np.int32) if self.use_teammate_actions else None

        # Fill arrays in chronological order
        write_pos = 0
        for ep_idx in episode_order:
            ep_len = self.lengths[ep_idx]
            if ep_len > 0:
                ctx_obs[write_pos:write_pos + ep_len] = self.obs[ep_idx, :ep_len]
                ctx_actions[write_pos:write_pos + ep_len] = self.actions[ep_idx, :ep_len]
                ctx_next_obs[write_pos:write_pos + ep_len] = self.next_obs[ep_idx, :ep_len]
                ctx_rewards[write_pos:write_pos + ep_len] = self.rewards[ep_idx, :ep_len]
                if ctx_teammate_actions is not None and self.teammate_actions is not None:
                    ctx_teammate_actions[write_pos:write_pos + ep_len] = self.teammate_actions[ep_idx, :ep_len]
                write_pos += ep_len

        # Take last max_transitions (most recent)
        if total_transitions > max_transitions:
            ctx_obs = ctx_obs[-max_transitions:]
            ctx_actions = ctx_actions[-max_transitions:]
            ctx_next_obs = ctx_next_obs[-max_transitions:]
            ctx_rewards = ctx_rewards[-max_transitions:]
            if ctx_teammate_actions is not None:
                ctx_teammate_actions = ctx_teammate_actions[-max_transitions:]
        elif total_transitions < max_transitions:
            # Pad with zeros at the beginning
            pad_size = max_transitions - total_transitions
            pad_obs_shape = self._get_obs_buffer_shape(pad_size)
            ctx_obs = np.concatenate([np.zeros(pad_obs_shape, dtype=np.float32), ctx_obs])
            ctx_actions = np.concatenate([np.zeros(pad_size, dtype=np.int32), ctx_actions])
            ctx_next_obs = np.concatenate([np.zeros(pad_obs_shape, dtype=np.float32), ctx_next_obs])
            ctx_rewards = np.concatenate([np.zeros(pad_size, dtype=np.float32), ctx_rewards])
            if ctx_teammate_actions is not None:
                ctx_teammate_actions = np.concatenate([np.zeros(pad_size, dtype=np.int32), ctx_teammate_actions])

        return ctx_obs, ctx_actions, ctx_next_obs, ctx_rewards, ctx_teammate_actions


@dataclass
class VectorizedEpisodeBuffer:
    """Vectorized episode buffer for batched DPT evaluation.

    Manages N parallel EpisodeBuffers for evaluating multiple tasks simultaneously.
    Provides vectorized operations for efficient batched context retrieval.

    Attributes:
        batch_size: Number of parallel tasks (N)
        max_episodes: Maximum episodes to store per task
        max_steps: Maximum steps per episode
        obs_shape: Observation shape
        use_teammate_actions: Whether to track teammate actions
    """
    batch_size: int
    max_episodes: int
    max_steps: int
    obs_shape: Union[int, Tuple[int, ...]]
    use_teammate_actions: bool = False

    # Internal buffers (initialized in __post_init__)
    _buffers: List[EpisodeBuffer] = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize N parallel buffers."""
        self._buffers = [
            EpisodeBuffer(
                max_episodes=self.max_episodes,
                max_steps=self.max_steps,
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

    def add_transition(
        self,
        task_idx: int,
        obs: np.ndarray,
        action: int,
        next_obs: np.ndarray,
        reward: float,
        teammate_action: Optional[int] = None,
    ):
        """Add transition for a specific task."""
        self._buffers[task_idx].add_transition(obs, action, next_obs, reward, teammate_action)

    def add_transitions_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        next_obs: np.ndarray,
        rewards: np.ndarray,
        teammate_actions: Optional[np.ndarray] = None,
    ):
        """Add transitions for all tasks in batch.

        Args:
            obs: Observations (batch_size, *obs_shape)
            actions: Actions (batch_size,)
            next_obs: Next observations (batch_size, *obs_shape)
            rewards: Rewards (batch_size,)
            teammate_actions: Teammate actions (batch_size,), optional
        """
        for i in range(self.batch_size):
            tm_action = teammate_actions[i] if teammate_actions is not None else None
            self._buffers[i].add_transition(
                obs[i], int(actions[i]), next_obs[i], float(rewards[i]), tm_action
            )

    def finish_episode(self, task_idx: int):
        """Finish episode for a specific task."""
        self._buffers[task_idx].finish_episode()

    def finish_episodes_batch(self, mask: Optional[np.ndarray] = None):
        """Finish episodes for multiple tasks.

        Args:
            mask: Boolean mask (batch_size,) indicating which tasks to finish.
                  If None, finish all tasks.
        """
        if mask is None:
            for buf in self._buffers:
                buf.finish_episode()
        else:
            for i, buf in enumerate(self._buffers):
                if mask[i]:
                    buf.finish_episode()

    def get_context(self, task_idx: int, max_transitions: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Get context for a specific task."""
        return self._buffers[task_idx].get_context(max_transitions)

    def get_context_batch(
        self,
        max_transitions: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Get batched context for all tasks.

        Returns:
            Tuple of batched arrays:
            - obs: (batch_size, max_transitions, *obs_shape)
            - actions: (batch_size, max_transitions)
            - next_obs: (batch_size, max_transitions, *obs_shape)
            - rewards: (batch_size, max_transitions)
            - teammate_actions: (batch_size, max_transitions) or None
        """
        # Get context from each buffer
        contexts = [buf.get_context(max_transitions) for buf in self._buffers]

        # Stack into batched arrays
        batch_obs = np.stack([ctx[0] for ctx in contexts], axis=0)
        batch_actions = np.stack([ctx[1] for ctx in contexts], axis=0)
        batch_next_obs = np.stack([ctx[2] for ctx in contexts], axis=0)
        batch_rewards = np.stack([ctx[3] for ctx in contexts], axis=0)

        if self.use_teammate_actions:
            batch_teammate_actions = np.stack([ctx[4] for ctx in contexts], axis=0)
        else:
            batch_teammate_actions = None

        return batch_obs, batch_actions, batch_next_obs, batch_rewards, batch_teammate_actions

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state for checkpointing.

        Returns:
            Dictionary containing all buffer state that can be saved to disk.
        """
        buffer_states = []
        for buf in self._buffers:
            buffer_states.append({
                "obs": buf.obs.copy(),
                "actions": buf.actions.copy(),
                "next_obs": buf.next_obs.copy(),
                "rewards": buf.rewards.copy(),
                "teammate_actions": buf.teammate_actions.copy() if buf.teammate_actions is not None else None,
                "lengths": buf.lengths.copy(),
                "current_obs": buf.current_obs.copy(),
                "current_actions": buf.current_actions.copy(),
                "current_next_obs": buf.current_next_obs.copy(),
                "current_rewards": buf.current_rewards.copy(),
                "current_teammate_actions": buf.current_teammate_actions.copy() if buf.current_teammate_actions is not None else None,
                "current_step": buf.current_step,
                "_head": buf._head,
                "num_episodes": buf.num_episodes,
            })
        return {
            "batch_size": self.batch_size,
            "max_episodes": self.max_episodes,
            "max_steps": self.max_steps,
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
        assert state["max_episodes"] == self.max_episodes
        assert state["max_steps"] == self.max_steps

        for i, buf_state in enumerate(state["buffer_states"]):
            buf = self._buffers[i]
            buf.obs[:] = buf_state["obs"]
            buf.actions[:] = buf_state["actions"]
            buf.next_obs[:] = buf_state["next_obs"]
            buf.rewards[:] = buf_state["rewards"]
            if buf.teammate_actions is not None and buf_state["teammate_actions"] is not None:
                buf.teammate_actions[:] = buf_state["teammate_actions"]
            buf.lengths[:] = buf_state["lengths"]
            buf.current_obs[:] = buf_state["current_obs"]
            buf.current_actions[:] = buf_state["current_actions"]
            buf.current_next_obs[:] = buf_state["current_next_obs"]
            buf.current_rewards[:] = buf_state["current_rewards"]
            if buf.current_teammate_actions is not None and buf_state["current_teammate_actions"] is not None:
                buf.current_teammate_actions[:] = buf_state["current_teammate_actions"]
            buf.current_step = buf_state["current_step"]
            buf._head = buf_state["_head"]
            buf.num_episodes = buf_state["num_episodes"]
