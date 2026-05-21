"""DPT (Decision Pretrained Transformer) baseline for Overcooked V2.

This module implements the DPT approach from:
"Supervised Pretraining Can Learn In-Context Reinforcement Learning" (Lee, J., et al., NeurIPS 2023)


Key components:
- model.py: Transformer model for DPT (Flax/JAX)
- train.py: Training script
- eval.py: Evaluation script
- relabel_expert_actions.py: Script to relabel actions using trained PPO policy
"""

from benchmarks.baselines.dpt.model import DPTModel, DPTConfig, create_dpt_model
from runners.history_adapter import DPTDataset

__all__ = [
    "DPTModel",
    "DPTConfig",
    "create_dpt_model",
    "DPTDataset",
]
