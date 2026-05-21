"""AMAGO-offline baseline for Overcooked V2 ICRL benchmark.

AMAGO-style offline sequence model that processes long learning histories
using timestep encoding + recurrent trajectory encoding.

Main features:
- Timestep encoder: encodes (obs, prev_action, prev_reward, done, time_idx) into tokens
- Trajectory encoder: GRU over token sequence for long-range memory
- Offline action prediction from trajectory hidden states
- Reads existing HDF5 + JSONL index learning histories

Reference: AMAGO (Grigsby et al., 2024) adapted for offline-only use in JAX/Flax.
"""

from benchmarks.baselines.amago_offline.model import (
    AMAGOOfflineConfig,
    AMAGOOfflineModel,
    create_amago_offline_model,
    count_parameters,
)

__all__ = [
    "AMAGOOfflineConfig",
    "AMAGOOfflineModel",
    "create_amago_offline_model",
    "count_parameters",
]
