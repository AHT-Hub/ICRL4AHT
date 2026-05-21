"""Algorithm Distillation (AD) baseline for Overcooked V2 ICRL benchmark.

This module implements AD approach from:
"In-context Reinforcement Learning with Algorithm Distillation" (Laskin, M., et al., ICLR 2023)

Main features:
- Step-level token format: (obs_t, prev_action_{t-1}, prev_reward_{t-1}) -> action_t
- Causal transformer with behavioral cloning loss
- Online evaluation with step-level buffer updates

Key differences from DPT:
- AD uses contiguous sequences from learning histories
- AD updates context buffer EVERY STEP during evaluation (not after episodes)
- AD predicts actions autoregressively given in-context learning history
"""

from benchmarks.baselines.ad.model import ADConfig, ADModel, create_ad_model
from benchmarks.baselines.ad.buffer import OnlineBuffer

__all__ = [
    "ADConfig",
    "ADModel",
    "create_ad_model",
    "OnlineBuffer",
]
