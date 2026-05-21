#!/usr/bin/env python3
"""History adapter for efficient random sampling from packed HDF5 datasets.

This module provides the HistoryStore class for accessing learning histories
and high-level samplers for AD and DPT training.

Design principles (matching dataset_spec.md requirements):
- Efficient random access: slice arrays in one HDF5 call, no Python loops
- Caching: HDF5 file handle kept open, optional per-history caching
- Deterministic sampling: seeded RNG produces reproducible batches
- Output format: dictionaries matching dataset_spec.md contracts

Usage:
    from runners.history_adapter import HistoryStore

    store = HistoryStore("datasets/histories.h5", "datasets/histories_index.jsonl")

    # Low-level access
    meta = store.get_history_meta(history_id=0)
    data = store.load_slice(history_id=0, t0=100, length=256)

    # AD sampling
    rng = np.random.default_rng(0)
    batch = store.sample_ad_batch(rng, batch_size=32, seq_len=256)

    # DPT sampling
    batch = store.sample_dpt_batch(rng, batch_size=32, ctx_len=128)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import json
import numpy as np


# =============================================================================
# Type Definitions
# =============================================================================

# Batch return types following dataset_spec.md

@dataclass
class ADBatch:
    """AD (Algorithm Distillation) batch format.

    Key features:
    - Input per step: (obs_t, prev_action_{t-1}, prev_reward_{t-1})
    - Target: action_t
    - Optionally: prev_teammate_action_{t-1}

    All arrays are (batch_size, seq_len, ...) or (batch_size, seq_len).
    """
    obs: np.ndarray           # (B, L, ...) float32 - observations
    prev_actions: np.ndarray  # (B, L) int32 - previous actions (0 for t=0)
    prev_rewards: np.ndarray  # (B, L) float32 - previous rewards (0 for t=0)
    target_actions: np.ndarray  # (B, L) int32 - actions to predict
    dones: np.ndarray         # (B, L) bool - episode termination flags
    attention_mask: np.ndarray  # (B, L) bool - valid positions (1=valid, 0=padding)

    # Metadata for debugging
    history_ids: np.ndarray   # (B,) int - which history each sample came from
    start_ts: np.ndarray      # (B,) int - start timestep in history

    # Optional teammate actions
    prev_teammate_actions: Optional[np.ndarray] = None  # (B, L) int32 - previous teammate actions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "obs": self.obs,
            "prev_actions": self.prev_actions,
            "prev_rewards": self.prev_rewards,
            "target_actions": self.target_actions,
            "dones": self.dones,
            "attention_mask": self.attention_mask,
            "history_ids": self.history_ids,
            "start_ts": self.start_ts,
        }
        if self.prev_teammate_actions is not None:
            result["prev_teammate_actions"] = self.prev_teammate_actions
        return result


@dataclass
class DPTBatch:
    """DPT (Decision Pretrained Transformer) batch format.

    Key features:
    - Context: (obs, action, next_obs, reward) tuples
    - Query: single observation
    - Target: expert action for query
    - Optionally: teammate_action for each context transition

    Context is sampled from the same task as the query.
    """
    # Query
    query_obs: np.ndarray        # (B, ...) float32 - observation to predict action for
    query_target: np.ndarray     # (B,) int32 - expert action (supervision target)

    # Context
    context_obs: np.ndarray      # (B, K, ...) float32 - context observations
    context_actions: np.ndarray  # (B, K) int32 - context actions
    context_next_obs: np.ndarray # (B, K, ...) float32 - context next observations
    context_rewards: np.ndarray  # (B, K) float32 - context rewards

    # Metadata
    task_ids: List[str]          # (B,) - task IDs for each sample
    history_ids: np.ndarray      # (B,) int - history IDs for queries
    query_ts: np.ndarray         # (B,) int - query timesteps within histories

    # Optional teammate actions
    context_teammate_actions: Optional[np.ndarray] = None  # (B, K) int32 - context teammate actions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "query_obs": self.query_obs,
            "query_target": self.query_target,
            "context_obs": self.context_obs,
            "context_actions": self.context_actions,
            "context_next_obs": self.context_next_obs,
            "context_rewards": self.context_rewards,
            "task_ids": self.task_ids,
            "history_ids": self.history_ids,
            "query_ts": self.query_ts,
        }
        if self.context_teammate_actions is not None:
            result["context_teammate_actions"] = self.context_teammate_actions
        return result


# =============================================================================
# History Store
# =============================================================================

class HistoryStore:
    """Efficient random-access store for learning histories.

    Provides:
    - O(1) metadata lookup via index
    - Efficient HDF5 slicing for sequence windows
    - High-level samplers for AD and DPT training

    The store keeps the HDF5 file open for the lifetime of the object
    to avoid repeated open/close overhead.

    Example:
        store = HistoryStore("histories.h5", "histories_index.jsonl")
        meta = store.get_history_meta(0)
        data = store.load_slice(0, t0=100, length=256)
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        index_path: Union[str, Path],
        cache_size_mb: float = 100.0,
    ):
        """Initialize the history store.

        Args:
            h5_path: Path to HDF5 file
            index_path: Path to index file (JSONL)
            cache_size_mb: HDF5 chunk cache size in MB
        """
        self.h5_path = Path(h5_path)
        self.index_path = Path(index_path)

        # Load index
        self._index: List[Dict[str, Any]] = []
        self._task_to_histories: Dict[str, List[int]] = {}
        self._load_index()

        # Open HDF5 file with chunk cache
        rdcc_nbytes = int(cache_size_mb * 1024 * 1024)
        self._h5f: h5py.File = h5py.File(
            self.h5_path, "r",
            rdcc_nbytes=rdcc_nbytes,
            rdcc_nslots=10007,  # Prime number for hash table
        )

        # Precompute uniform sampling weights (can be overridden for weighted sampling)
        self._sample_weights: Optional[np.ndarray] = None

    def _load_index(self) -> None:
        """Load index from JSONL file."""
        self._index = []
        self._task_to_histories = {}

        with open(self.index_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                history_id = entry["history_id"]
                task_id = entry["task_id"]

                # Ensure obs_shape is a tuple (may be stored as list in JSON)
                if "obs_shape" in entry:
                    entry["obs_shape"] = tuple(entry["obs_shape"])
                elif "obs_dim" in entry:
                    # Backward compatibility: create obs_shape from obs_dim
                    entry["obs_shape"] = (entry["obs_dim"],)

                self._index.append(entry)

                if task_id not in self._task_to_histories:
                    self._task_to_histories[task_id] = []
                self._task_to_histories[task_id].append(history_id)

    def __len__(self) -> int:
        """Return number of learning histories."""
        return len(self._index)

    def __del__(self) -> None:
        """Close HDF5 file on deletion."""
        if hasattr(self, "_h5f") and self._h5f is not None:
            try:
                self._h5f.close()
            except Exception:
                pass

    def close(self) -> None:
        """Explicitly close the HDF5 file."""
        if self._h5f is not None:
            self._h5f.close()
            self._h5f = None

    # -------------------------------------------------------------------------
    # Low-level access
    # -------------------------------------------------------------------------

    def get_history_meta(self, history_id: int) -> Dict[str, Any]:
        """Get metadata for a history by ID.

        Args:
            history_id: History index (0-based)

        Returns:
            Dictionary with metadata fields:
            - task_id, env_idx, T, obs_dim, action_dim
            - track, split, layout, teammate_family, teammate_kind
            - h5_group
        """
        if history_id < 0 or history_id >= len(self._index):
            raise IndexError(f"history_id {history_id} out of range [0, {len(self._index)})")
        return self._index[history_id]

    def load_slice(
        self,
        history_id: int,
        t0: int,
        length: int,
        include_teammate_actions: bool = False,
        include_expert_actions: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Load a contiguous slice from a history.

        Args:
            history_id: History index (0-based)
            t0: Start timestep (inclusive)
            length: Number of timesteps to load
            include_teammate_actions: Whether to include teammate_actions if available
            include_expert_actions: Whether to include expert_actions if available

        Returns:
            Dictionary with arrays:
            - obs: (length, *obs_shape) float32 - e.g., (length, H, W, C) or (length, obs_dim)
            - actions: (length,) int32
            - rewards: (length,) float32
            - dones: (length,) bool
            - teammate_actions: (length,) int32 (only if include_teammate_actions and available)
            - expert_actions: (length,) int32 (only if include_expert_actions and available)
        """
        meta = self.get_history_meta(history_id)
        T = meta["T"]

        # Validate slice bounds
        if t0 < 0 or t0 + length > T:
            raise IndexError(f"Slice [{t0}:{t0 + length}] out of bounds for T={T}")

        group_name = meta["h5_group"].lstrip("/")
        g = self._h5f[group_name]

        # Load slices in one call each (efficient HDF5 access)
        result = {
            "obs": g["obs"][t0:t0 + length],
            "actions": g["actions"][t0:t0 + length],
            "rewards": g["rewards"][t0:t0 + length],
            "dones": g["dones"][t0:t0 + length],
        }

        # Optionally include teammate_actions if available
        if include_teammate_actions and "teammate_actions" in g:
            result["teammate_actions"] = g["teammate_actions"][t0:t0 + length]

        # Optionally include expert_actions if available
        if include_expert_actions and "expert_actions" in g:
            result["expert_actions"] = g["expert_actions"][t0:t0 + length]

        return result

    def get_task_ids(self) -> List[str]:
        """Get list of unique task IDs."""
        return list(self._task_to_histories.keys())

    def get_histories_for_task(self, task_id: str) -> List[int]:
        """Get history IDs for a given task.

        Args:
            task_id: Task identifier

        Returns:
            List of history_ids for this task
        """
        return self._task_to_histories.get(task_id, [])

    # -------------------------------------------------------------------------
    # AD Sampling
    # -------------------------------------------------------------------------

    def sample_ad_batch(
        self,
        rng: np.random.Generator,
        batch_size: int,
        seq_len: int,
        weighted: bool = False,
        include_teammate_actions: bool = False,
    ) -> ADBatch:
        """Sample a batch for AD (Algorithm Distillation) training.

        Key features:
        - Sample contiguous subsequences from learning histories
        - Do NOT cross learning history boundaries
        - May cross episode boundaries within a history

        PERFORMANCE OPTIMIZATIONS:
        1. Cross-batch HDF5 read merging: Groups all reads by history_id across
           the entire batch, so each history is read at most once
        2. Bulk random sampling: Generates all random numbers upfront
        3. Metadata caching: Caches history lengths to avoid repeated lookups

        Args:
            rng: NumPy random generator for reproducibility
            batch_size: Number of sequences to sample
            seq_len: Length of each sequence
            weighted: If True, weight histories by length (longer = more samples)
            include_teammate_actions: If True, include prev_teammate_actions in batch

        Returns:
            ADBatch with properly formatted arrays
        """
        num_histories = len(self)

        # =============================================================
        # PHASE 1: Sample history IDs and cache metadata
        # =============================================================

        if weighted:
            weights = self._get_length_weights(seq_len)
            history_ids = rng.choice(num_histories, size=batch_size, p=weights)
        else:
            history_ids = rng.integers(0, num_histories, size=batch_size)

        # Cache history metadata - T values and group names
        # Only cache for histories we'll actually use
        unique_hids = np.unique(history_ids)
        history_T_cache = {int(hid): self._index[hid]["T"] for hid in unique_hids}
        history_group_cache = {int(hid): self._index[hid]["h5_group"].lstrip("/") for hid in unique_hids}

        # Get obs_shape from first history
        first_meta = self.get_history_meta(int(history_ids[0]))
        obs_shape = first_meta["obs_shape"]

        # =============================================================
        # PHASE 2: Pre-allocate output arrays
        # =============================================================

        obs = np.zeros((batch_size, seq_len) + obs_shape, dtype=np.float32)
        prev_actions = np.zeros((batch_size, seq_len), dtype=np.int32)
        prev_rewards = np.zeros((batch_size, seq_len), dtype=np.float32)
        target_actions = np.zeros((batch_size, seq_len), dtype=np.int32)
        dones = np.zeros((batch_size, seq_len), dtype=np.bool_)
        attention_mask = np.ones((batch_size, seq_len), dtype=np.bool_)
        start_ts = np.zeros(batch_size, dtype=np.int32)

        prev_teammate_actions = None
        if include_teammate_actions:
            prev_teammate_actions = np.zeros((batch_size, seq_len), dtype=np.int32)

        # =============================================================
        # PHASE 3: Sample start positions and group reads by history
        # =============================================================

        # Data structure: hid -> [(batch_idx, start, valid_len, needs_prev_step), ...]
        hid_reads: Dict[int, List[Tuple[int, int, int, bool]]] = {}

        for i, hid in enumerate(history_ids):
            hid = int(hid)
            T = history_T_cache[hid]

            # Sample start position
            max_start = T - seq_len
            if max_start <= 0:
                start = 0
                valid_len = T
                attention_mask[i, valid_len:] = False
            else:
                start = int(rng.integers(0, max_start + 1))
                valid_len = seq_len

            start_ts[i] = start

            # Track whether we need the previous timestep for prev_actions/prev_rewards
            needs_prev = (start > 0)

            if hid not in hid_reads:
                hid_reads[hid] = []
            hid_reads[hid].append((i, start, valid_len, needs_prev))

        # =============================================================
        # PHASE 4: Execute batched reads and fill output arrays
        # =============================================================

        for hid, reads in hid_reads.items():
            group_name = history_group_cache[hid]
            g = self._h5f[group_name]

            # Determine the range we need to read for this history
            # We need [min_start - 1, max_end) where max_end = start + valid_len
            min_start = min(r[1] for r in reads)
            max_end = max(r[1] + r[2] for r in reads)

            # If any read needs prev step, extend min_start
            any_needs_prev = any(r[3] for r in reads)
            read_start = max(0, min_start - 1) if any_needs_prev else min_start

            # Single batched HDF5 read
            obs_slice = g["obs"][read_start:max_end]
            actions_slice = g["actions"][read_start:max_end]
            rewards_slice = g["rewards"][read_start:max_end]
            dones_slice = g["dones"][read_start:max_end]

            teammate_actions_slice = None
            if include_teammate_actions and "teammate_actions" in g:
                teammate_actions_slice = g["teammate_actions"][read_start:max_end]

            # Fill each batch item from the cached slice
            for batch_idx, start, valid_len, needs_prev in reads:
                # Local offset within our read slice
                local_start = start - read_start

                # Fill main data
                obs[batch_idx, :valid_len] = obs_slice[local_start:local_start + valid_len]
                target_actions[batch_idx, :valid_len] = actions_slice[local_start:local_start + valid_len]
                dones[batch_idx, :valid_len] = dones_slice[local_start:local_start + valid_len]

                # Fill prev_actions and prev_rewards
                if needs_prev:
                    # We have start > 0, so we can use start-1
                    prev_local = local_start - 1
                    prev_actions[batch_idx, :valid_len] = actions_slice[prev_local:prev_local + valid_len]
                    prev_rewards[batch_idx, :valid_len] = rewards_slice[prev_local:prev_local + valid_len]
                    if teammate_actions_slice is not None:
                        prev_teammate_actions[batch_idx, :valid_len] = teammate_actions_slice[prev_local:prev_local + valid_len]
                else:
                    # start=0: first step has no previous, rest are shifted
                    if valid_len > 1:
                        prev_actions[batch_idx, 1:valid_len] = actions_slice[local_start:local_start + valid_len - 1]
                        prev_rewards[batch_idx, 1:valid_len] = rewards_slice[local_start:local_start + valid_len - 1]
                        if teammate_actions_slice is not None:
                            prev_teammate_actions[batch_idx, 1:valid_len] = teammate_actions_slice[local_start:local_start + valid_len - 1]
                    # prev_actions[batch_idx, 0] = 0 and prev_rewards[batch_idx, 0] = 0 (already initialized)

        return ADBatch(
            obs=obs,
            prev_actions=prev_actions,
            prev_rewards=prev_rewards,
            target_actions=target_actions,
            dones=dones,
            attention_mask=attention_mask,
            history_ids=history_ids.astype(np.int32),
            start_ts=start_ts,
            prev_teammate_actions=prev_teammate_actions,
        )

    def _get_length_weights(self, min_len: int) -> np.ndarray:
        """Compute sampling weights proportional to usable length.

        Args:
            min_len: Minimum sequence length (for filtering short histories)

        Returns:
            Normalized probability weights
        """
        weights = np.array([
            max(0, entry["T"] - min_len) for entry in self._index
        ], dtype=np.float64)
        total = weights.sum()
        if total == 0:
            # Fallback to uniform
            return np.ones(len(self._index)) / len(self._index)
        return weights / total

    # -------------------------------------------------------------------------
    # DPT Sampling
    # -------------------------------------------------------------------------

    def sample_dpt_batch(
        self,
        rng: np.random.Generator,
        batch_size: int,
        ctx_len: int,
        cross_history_context: bool = True,
        include_teammate_actions: bool = False,
        use_expert_actions: bool = False,
    ) -> DPTBatch:
        """Sample a batch for DPT (Decision Pretrained Transformer) training.

        ULTRA-FAST IMPLEMENTATION using contiguous context sampling.
        Instead of randomly sampling individual transitions (slow), we sample
        contiguous chunks from histories which allows:
        1. Single HDF5 read per batch item
        2. Direct memory copy instead of element-by-element assignment
        3. Minimal Python loop overhead

        Args:
            rng: NumPy random generator for reproducibility
            batch_size: Number of samples
            ctx_len: Number of context transitions (K)
            cross_history_context: Ignored in fast mode (always samples from one history per item)
            include_teammate_actions: If True, include context_teammate_actions in batch
            use_expert_actions: If True, use expert_actions for query_target

        Returns:
            DPTBatch with properly formatted arrays
        """
        task_ids_list = self.get_task_ids()
        if not task_ids_list:
            raise ValueError("No tasks available for sampling")
        num_tasks = len(task_ids_list)

        # Get obs_shape from first history
        first_meta = self.get_history_meta(0)
        obs_shape = first_meta["obs_shape"]

        # Pre-allocate output arrays
        query_obs = np.zeros((batch_size,) + obs_shape, dtype=np.float32)
        query_target = np.zeros(batch_size, dtype=np.int32)
        query_ts = np.zeros(batch_size, dtype=np.int32)
        context_obs = np.zeros((batch_size, ctx_len) + obs_shape, dtype=np.float32)
        context_actions = np.zeros((batch_size, ctx_len), dtype=np.int32)
        context_next_obs = np.zeros((batch_size, ctx_len) + obs_shape, dtype=np.float32)
        context_rewards = np.zeros((batch_size, ctx_len), dtype=np.float32)
        task_ids_out: List[str] = []
        history_ids_out = np.zeros(batch_size, dtype=np.int32)
        context_teammate_actions = None
        if include_teammate_actions:
            context_teammate_actions = np.zeros((batch_size, ctx_len), dtype=np.int32)

        # =============================================================
        # FAST PATH: Sample contiguous context chunks
        # =============================================================

        # Pre-compute task -> histories mapping as numpy arrays for speed
        task_histories_list = [self.get_histories_for_task(task_ids_list[ti]) for ti in range(num_tasks)]

        # Sample all random values upfront (much faster than per-iteration)
        task_indices = rng.integers(0, num_tasks, size=batch_size)

        for i in range(batch_size):
            ti = task_indices[i]
            task_id = task_ids_list[ti]
            task_ids_out.append(task_id)
            histories = task_histories_list[ti]

            # Sample a history for context
            ctx_hid = histories[rng.integers(0, len(histories))]
            ctx_meta = self._index[ctx_hid]
            ctx_T = ctx_meta["T"]
            ctx_group = ctx_meta["h5_group"].lstrip("/")

            # Sample query (can be from same or different history)
            query_hid = histories[rng.integers(0, len(histories))]
            query_meta = self._index[query_hid]
            query_T = query_meta["T"]
            query_group = query_meta["h5_group"].lstrip("/")

            history_ids_out[i] = query_hid

            # Sample query timestep
            query_t = rng.integers(0, query_T)
            query_ts[i] = query_t

            # Load query data (single timestep)
            g_query = self._h5f[query_group]
            query_obs[i] = g_query["obs"][query_t]
            if use_expert_actions and "expert_actions" in g_query:
                query_target[i] = g_query["expert_actions"][query_t]
            else:
                query_target[i] = g_query["actions"][query_t]

            # Sample contiguous context chunk
            # We need ctx_len transitions, each transition needs t and t+1
            # So we need ctx_len + 1 consecutive timesteps
            max_ctx_start = ctx_T - ctx_len - 1
            if max_ctx_start < 0:
                # History too short, use what we have
                ctx_start = 0
                valid_ctx_len = max(0, ctx_T - 1)
            else:
                ctx_start = rng.integers(0, max_ctx_start + 1)
                valid_ctx_len = ctx_len

            if valid_ctx_len > 0:
                g_ctx = self._h5f[ctx_group]

                # Single contiguous read - FAST!
                obs_chunk = g_ctx["obs"][ctx_start:ctx_start + valid_ctx_len + 1]
                actions_chunk = g_ctx["actions"][ctx_start:ctx_start + valid_ctx_len]
                rewards_chunk = g_ctx["rewards"][ctx_start:ctx_start + valid_ctx_len]

                # Direct array assignment - no Python loop!
                context_obs[i, :valid_ctx_len] = obs_chunk[:-1]
                context_actions[i, :valid_ctx_len] = actions_chunk
                context_next_obs[i, :valid_ctx_len] = obs_chunk[1:]
                context_rewards[i, :valid_ctx_len] = rewards_chunk

                if include_teammate_actions and "teammate_actions" in g_ctx:
                    teammate_chunk = g_ctx["teammate_actions"][ctx_start:ctx_start + valid_ctx_len]
                    context_teammate_actions[i, :valid_ctx_len] = teammate_chunk

        return DPTBatch(
            query_obs=query_obs,
            query_target=query_target,
            context_obs=context_obs,
            context_actions=context_actions,
            context_next_obs=context_next_obs,
            context_rewards=context_rewards,
            task_ids=task_ids_out,
            history_ids=history_ids_out,
            query_ts=query_ts,
            context_teammate_actions=context_teammate_actions,
        )

    def sample_dpt_batch_random_context(
        self,
        rng: np.random.Generator,
        batch_size: int,
        ctx_len: int,
        cross_history_context: bool = True,
        include_teammate_actions: bool = False,
        use_expert_actions: bool = False,
    ) -> DPTBatch:
        """Sample DPT batch with randomly sampled context transitions (slower but more diverse).

        Use this if you need truly random context sampling. For most cases,
        sample_dpt_batch (contiguous sampling) is preferred for speed.
        """
        task_ids_list = self.get_task_ids()
        if not task_ids_list:
            raise ValueError("No tasks available for sampling")
        num_tasks = len(task_ids_list)

        first_meta = self.get_history_meta(0)
        obs_shape = first_meta["obs_shape"]

        query_obs = np.zeros((batch_size,) + obs_shape, dtype=np.float32)
        query_target = np.zeros(batch_size, dtype=np.int32)
        query_ts = np.zeros(batch_size, dtype=np.int32)
        context_obs = np.zeros((batch_size, ctx_len) + obs_shape, dtype=np.float32)
        context_actions = np.zeros((batch_size, ctx_len), dtype=np.int32)
        context_next_obs = np.zeros((batch_size, ctx_len) + obs_shape, dtype=np.float32)
        context_rewards = np.zeros((batch_size, ctx_len), dtype=np.float32)
        task_ids_out: List[str] = []
        history_ids_out = np.zeros(batch_size, dtype=np.int32)
        context_teammate_actions = None
        if include_teammate_actions:
            context_teammate_actions = np.zeros((batch_size, ctx_len), dtype=np.int32)

        task_indices = rng.integers(0, num_tasks, size=batch_size)
        task_histories_list = [self.get_histories_for_task(task_ids_list[ti]) for ti in range(num_tasks)]
        history_T_cache = {hid: self._index[hid]["T"] for hid in range(len(self._index))}
        history_group_cache = {hid: self._index[hid]["h5_group"].lstrip("/") for hid in range(len(self._index))}

        query_hids = np.zeros(batch_size, dtype=np.int32)
        for i in range(batch_size):
            ti = task_indices[i]
            task_ids_out.append(task_ids_list[ti])
            histories = task_histories_list[ti]
            query_hids[i] = histories[rng.integers(0, len(histories))]
        history_ids_out[:] = query_hids

        for i in range(batch_size):
            query_T = history_T_cache[query_hids[i]]
            query_ts[i] = rng.integers(0, query_T)

        hid_reads: Dict[int, Dict[str, List]] = {}
        for i in range(batch_size):
            hid = int(query_hids[i])
            if hid not in hid_reads:
                hid_reads[hid] = {"query": [], "context": []}
            hid_reads[hid]["query"].append((i, int(query_ts[i])))

        if cross_history_context:
            for i in range(batch_size):
                ti = task_indices[i]
                histories = task_histories_list[ti]
                num_hist = len(histories)
                ctx_hist_indices = rng.integers(0, num_hist, size=ctx_len)
                for j in range(ctx_len):
                    ctx_hid = histories[ctx_hist_indices[j]]
                    ctx_T = history_T_cache[ctx_hid]
                    if ctx_T < 2:
                        continue
                    ctx_t = int(rng.integers(0, ctx_T - 1))
                    if ctx_hid not in hid_reads:
                        hid_reads[ctx_hid] = {"query": [], "context": []}
                    hid_reads[ctx_hid]["context"].append((i, j, ctx_t))
        else:
            for i in range(batch_size):
                hid = int(query_hids[i])
                ctx_T = history_T_cache[hid]
                if ctx_T < 2:
                    continue
                ctx_ts_arr = rng.integers(0, ctx_T - 1, size=ctx_len)
                for j, ctx_t in enumerate(ctx_ts_arr):
                    hid_reads[hid]["context"].append((i, j, int(ctx_t)))

        for hid, reads in hid_reads.items():
            group_name = history_group_cache[hid]
            g = self._h5f[group_name]

            all_timesteps = set()
            for batch_idx, t in reads["query"]:
                all_timesteps.add(t)
            for batch_idx, ctx_idx, t in reads["context"]:
                all_timesteps.add(t)
                all_timesteps.add(t + 1)

            if not all_timesteps:
                continue

            min_t = min(all_timesteps)
            max_t = max(all_timesteps)

            obs_slice = g["obs"][min_t:max_t + 1]
            actions_slice = g["actions"][min_t:max_t + 1]
            rewards_slice = g["rewards"][min_t:max_t + 1]

            expert_actions_slice = None
            if use_expert_actions and "expert_actions" in g:
                expert_actions_slice = g["expert_actions"][min_t:max_t + 1]

            teammate_actions_slice = None
            if include_teammate_actions and "teammate_actions" in g:
                teammate_actions_slice = g["teammate_actions"][min_t:max_t + 1]

            for batch_idx, t in reads["query"]:
                local_t = t - min_t
                query_obs[batch_idx] = obs_slice[local_t]
                if use_expert_actions and expert_actions_slice is not None:
                    query_target[batch_idx] = expert_actions_slice[local_t]
                else:
                    query_target[batch_idx] = actions_slice[local_t]

            if reads["context"]:
                for batch_idx, ctx_idx, t in reads["context"]:
                    local_t = t - min_t
                    context_obs[batch_idx, ctx_idx] = obs_slice[local_t]
                    context_actions[batch_idx, ctx_idx] = actions_slice[local_t]
                    context_next_obs[batch_idx, ctx_idx] = obs_slice[local_t + 1]
                    context_rewards[batch_idx, ctx_idx] = rewards_slice[local_t]
                    if teammate_actions_slice is not None:
                        context_teammate_actions[batch_idx, ctx_idx] = teammate_actions_slice[local_t]

        return DPTBatch(
            query_obs=query_obs,
            query_target=query_target,
            context_obs=context_obs,
            context_actions=context_actions,
            context_next_obs=context_next_obs,
            context_rewards=context_rewards,
            task_ids=task_ids_out,
            history_ids=history_ids_out,
            query_ts=query_ts,
            context_teammate_actions=context_teammate_actions,
        )


# =============================================================================
# DPT Dataset Wrapper
# =============================================================================

class DPTDataset:
    """Simple DPT Dataset wrapper with expert actions support.

    Wraps HistoryStore to provide DPT training batches. Expert actions must be
    included in the main histories.h5 file (use build_index.py with --relabel flag).

    Usage:
        dataset = DPTDataset(
            h5_path="datasets/histories.h5",
            index_path="datasets/histories_index.jsonl",
            seq_len=512,
            use_expert_actions=True,  # Required for DPT training
        )

        batch = dataset.sample_batch(batch_size=128)
        dataset.close()
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        index_path: Union[str, Path],
        seq_len: int = 512,
        seed: int = 0,
        use_expert_actions: bool = True,
        cache_size_mb: float = 100.0,
    ):
        """Initialize DPT dataset.

        Args:
            h5_path: Path to main HDF5 history file
            index_path: Path to index JSONL file
            seq_len: Context sequence length (K)
            seed: Random seed
            use_expert_actions: Whether to use expert_actions for query_target (default True for DPT)
            cache_size_mb: HDF5 chunk cache size in MB
        """
        self.h5_path = Path(h5_path)
        self.index_path = Path(index_path)
        self.seq_len = seq_len
        self.seed = seed
        self.use_expert_actions = use_expert_actions

        # Load history store with specified cache size
        self._store = HistoryStore(h5_path, index_path, cache_size_mb=cache_size_mb)

        # Verify expert_actions are available if requested
        if use_expert_actions:
            first_meta = self._store.get_history_meta(0)
            has_expert = first_meta.get("has_expert_actions", False)
            if not has_expert:
                # Check if the h5 file actually has expert_actions in the first group
                group_name = first_meta["h5_group"].lstrip("/")
                if "expert_actions" not in self._store._h5f[group_name]:
                    raise ValueError(
                        "expert_actions not found in dataset. "
                        "Run build_index.py with --relabel flag to add expert actions."
                    )

        # RNG
        self.rng = np.random.default_rng(seed)

    @property
    def store(self) -> HistoryStore:
        """Access underlying store."""
        return self._store

    @property
    def num_tasks(self) -> int:
        """Number of tasks in the dataset."""
        return len(self._store.get_task_ids())

    def sample_batch(
        self,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
        include_teammate_actions: bool = False,
    ) -> DPTBatch:
        """Sample a batch for DPT training.

        Args:
            batch_size: Number of samples
            rng: Optional random generator (uses self.rng if None)
            include_teammate_actions: Whether to include teammate actions

        Returns:
            DPTBatch with collated samples
        """
        if rng is None:
            rng = self.rng

        # Sample batch from store with expert_actions for query_target
        batch = self._store.sample_dpt_batch(
            rng=rng,
            batch_size=batch_size,
            ctx_len=self.seq_len,
            cross_history_context=True,
            include_teammate_actions=include_teammate_actions,
            use_expert_actions=self.use_expert_actions,
        )

        return batch

    def close(self) -> None:
        """Close HDF5 files."""
        self._store.close()


# =============================================================================
# CLI for Testing
# =============================================================================

def main():
    """CLI for testing the history adapter."""
    import argparse

    parser = argparse.ArgumentParser(description="Test history adapter")
    parser.add_argument("--h5_path", type=str, required=True)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--sample_ad", action="store_true")
    parser.add_argument("--sample_dpt", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--ctx_len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    store = HistoryStore(args.h5_path, args.index_path)
    print(f"Loaded store with {len(store)} histories")
    print(f"Tasks: {len(store.get_task_ids())}")

    rng = np.random.default_rng(args.seed)

    if args.sample_ad:
        print("\nSampling AD batch...")
        batch = store.sample_ad_batch(rng, args.batch_size, args.seq_len)
        print(f"  obs: {batch.obs.shape}")
        print(f"  prev_actions: {batch.prev_actions.shape}")
        print(f"  target_actions: {batch.target_actions.shape}")
        print(f"  attention_mask: {batch.attention_mask.shape}")

    if args.sample_dpt:
        print("\nSampling DPT batch...")
        batch = store.sample_dpt_batch(rng, args.batch_size, args.ctx_len)
        print(f"  query_obs: {batch.query_obs.shape}")
        print(f"  query_target: {batch.query_target.shape}")
        print(f"  context_obs: {batch.context_obs.shape}")
        print(f"  context_actions: {batch.context_actions.shape}")

    store.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
