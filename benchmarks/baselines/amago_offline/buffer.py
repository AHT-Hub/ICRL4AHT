"""AMAGO-offline buffer: samples long sequences with timestep indices.

Uses HistoryStore's existing ADBatch sampling, adding time_idxs field.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from runners.history_adapter import HistoryStore, ADBatch


@dataclass
class AMAGOBatch:
    """AMAGO-offline training batch.

    Extends AD batch semantics with explicit time indices for positional encoding.
    """
    obs: np.ndarray              # (B, L, H, W, C) float32
    prev_actions: np.ndarray     # (B, L) int32
    prev_rewards: np.ndarray     # (B, L) float32
    target_actions: np.ndarray   # (B, L) int32
    dones: np.ndarray            # (B, L) bool
    attention_mask: np.ndarray   # (B, L) bool
    time_idxs: np.ndarray        # (B, L) int32

    history_ids: np.ndarray      # (B,) int
    start_ts: np.ndarray         # (B,) int

    prev_teammate_actions: Optional[np.ndarray] = None  # (B, L) int32


def ad_batch_to_amago_batch(ad_batch: ADBatch) -> AMAGOBatch:
    """Convert an ADBatch to AMAGOBatch by adding time_idxs."""
    batch_size, seq_len = ad_batch.prev_actions.shape

    time_idxs = np.zeros((batch_size, seq_len), dtype=np.int32)
    for i in range(batch_size):
        time_idxs[i] = np.arange(seq_len) + ad_batch.start_ts[i]

    return AMAGOBatch(
        obs=ad_batch.obs,
        prev_actions=ad_batch.prev_actions,
        prev_rewards=ad_batch.prev_rewards,
        target_actions=ad_batch.target_actions,
        dones=ad_batch.dones,
        attention_mask=ad_batch.attention_mask,
        time_idxs=time_idxs,
        history_ids=ad_batch.history_ids,
        start_ts=ad_batch.start_ts,
        prev_teammate_actions=ad_batch.prev_teammate_actions,
    )


def sample_amago_batch(
    store: HistoryStore,
    rng: np.random.Generator,
    batch_size: int,
    seq_len: int,
    include_teammate_actions: bool = False,
) -> AMAGOBatch:
    """Sample an AMAGO-style batch from HistoryStore."""
    ad_batch = store.sample_ad_batch(
        rng=rng,
        batch_size=batch_size,
        seq_len=seq_len,
        include_teammate_actions=include_teammate_actions,
    )
    return ad_batch_to_amago_batch(ad_batch)
