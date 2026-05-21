"""Hybrid-AD baseline for Overcooked V2 ICRL benchmark.

AD variant that replaces the Transformer temporal backbone with CNN + GRU
while keeping AD's step-level training protocol and action prediction target.

Main features:
- Same step-level token format as AD: (obs_t, prev_action_{t-1}, prev_reward_{t-1}) -> action_t
- CNN observation encoder + GRU recurrent backbone (no Transformer)
- Behavioral cloning loss on action prediction
- Online evaluation with step-level buffer updates (identical to AD)
"""

from benchmarks.baselines.hybrid_ad.model import HybridADConfig, HybridADModel, create_hybrid_ad_model

__all__ = [
    "HybridADConfig",
    "HybridADModel",
    "create_hybrid_ad_model",
]
