"""Hybrid-AD Model implementation in Flax/JAX.

Replaces AD's Transformer temporal backbone with CNN + GRU while keeping
the same token format and action prediction objective.

Architecture:
- Token format: (prev_action_emb, [prev_teammate_action_emb], prev_reward, obs_emb) concatenated
- GRU recurrent backbone (replaces Transformer blocks)
- Predicts next action at each timestep
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.baselines.ad.model import CNNObservationEncoder


@dataclass
class HybridADConfig:
    """Configuration for Hybrid-AD model."""
    obs_shape: Tuple[int, ...] = (9, 7, 26)
    num_actions: int = 6
    embedding_dim: int = 64
    hidden_dim: int = 256
    gru_hidden_dim: int = 256
    num_gru_layers: int = 2

    seq_len: int = 4096

    embedding_dropout: float = 0.1

    use_teammate_actions: bool = False

    reset_hidden_on_done: bool = False

    @property
    def obs_dim(self) -> int:
        return int(np.prod(self.obs_shape))


class GRUBackbone(nn.Module):
    """Multi-layer GRU backbone for temporal processing."""
    hidden_dim: int
    num_layers: int = 2

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        train: bool = True,
    ) -> jnp.ndarray:
        for i in range(self.num_layers):
            cell = nn.GRUCell(features=self.hidden_dim)
            batch_size, seq_len, _ = x.shape
            carry = cell.initialize_carry(jax.random.PRNGKey(0), (batch_size,))

            outputs = []
            for t in range(seq_len):
                carry, y = cell(carry, x[:, t, :])
                outputs.append(y)
            x = jnp.stack(outputs, axis=1)

        return x


class HybridADModel(nn.Module):
    """Hybrid-AD: AD with GRU backbone instead of Transformer.

    Same token format as AD:
    - Input per step: (prev_action_emb, [prev_teammate_action_emb], prev_reward, obs_emb)
    - Target: action_t

    Uses GRU for temporal processing instead of causal Transformer blocks.
    """
    config: HybridADConfig

    def setup(self):
        cfg = self.config

        self.obs_encoder = CNNObservationEncoder(embedding_dim=cfg.embedding_dim)

        self.action_embedding = nn.Embed(
            num_embeddings=cfg.num_actions,
            features=cfg.embedding_dim,
        )

        if cfg.use_teammate_actions:
            self.teammate_action_embedding = nn.Embed(
                num_embeddings=cfg.num_actions,
                features=cfg.embedding_dim,
            )

        self.embed_token = nn.Dense(cfg.hidden_dim)

        self.gru_backbone = GRUBackbone(
            hidden_dim=cfg.gru_hidden_dim,
            num_layers=cfg.num_gru_layers,
        )

        if cfg.gru_hidden_dim != cfg.hidden_dim:
            self.output_proj = nn.Dense(cfg.hidden_dim)

        self.final_norm = nn.LayerNorm()
        self.action_head = nn.Dense(cfg.num_actions)

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        prev_actions: jnp.ndarray,
        prev_rewards: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        prev_teammate_actions: Optional[jnp.ndarray] = None,
        train: bool = True,
    ) -> jnp.ndarray:
        cfg = self.config
        batch_size, seq_len = prev_rewards.shape

        obs_emb = self.obs_encoder(obs, train=train)
        action_emb = self.action_embedding(prev_actions)
        reward_emb = prev_rewards[:, :, None]

        if cfg.use_teammate_actions and prev_teammate_actions is not None:
            teammate_action_emb = self.teammate_action_embedding(prev_teammate_actions)
            sequence = jnp.concatenate([action_emb, teammate_action_emb, reward_emb, obs_emb], axis=-1)
        else:
            sequence = jnp.concatenate([action_emb, reward_emb, obs_emb], axis=-1)

        sequence = self.embed_token(sequence)

        sequence = nn.Dropout(rate=cfg.embedding_dropout, deterministic=not train)(sequence)

        x = self.gru_backbone(sequence, train=train)

        if cfg.gru_hidden_dim != cfg.hidden_dim:
            x = self.output_proj(x)

        x = self.final_norm(x)
        logits = self.action_head(x)

        return logits

    def get_action(
        self,
        params: Any,
        obs: jnp.ndarray,
        prev_action: jnp.ndarray,
        prev_reward: jnp.ndarray,
        context_obs: jnp.ndarray,
        context_prev_actions: jnp.ndarray,
        context_prev_rewards: jnp.ndarray,
        rng: Optional[jax.Array] = None,
        greedy: bool = True,
        prev_teammate_action: Optional[jnp.ndarray] = None,
        context_prev_teammate_actions: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        cfg = self.config

        single = obs.ndim == len(cfg.obs_shape)
        if single:
            obs = obs[None, ...]
            prev_action = jnp.array([prev_action])
            prev_reward = jnp.array([prev_reward])
            context_obs = context_obs[None, ...]
            context_prev_actions = context_prev_actions[None, :]
            context_prev_rewards = context_prev_rewards[None, :]
            if prev_teammate_action is not None:
                prev_teammate_action = jnp.array([prev_teammate_action])
            if context_prev_teammate_actions is not None:
                context_prev_teammate_actions = context_prev_teammate_actions[None, :]

        full_obs = jnp.concatenate([context_obs, jnp.expand_dims(obs, axis=1)], axis=1)
        full_prev_actions = jnp.concatenate([context_prev_actions, prev_action[:, None]], axis=1)
        full_prev_rewards = jnp.concatenate([context_prev_rewards, prev_reward[:, None]], axis=1)

        full_prev_teammate_actions = None
        if cfg.use_teammate_actions and prev_teammate_action is not None and context_prev_teammate_actions is not None:
            full_prev_teammate_actions = jnp.concatenate(
                [context_prev_teammate_actions, prev_teammate_action[:, None]], axis=1
            )

        logits = self.apply(
            params,
            full_obs,
            full_prev_actions,
            full_prev_rewards,
            attention_mask=None,
            prev_teammate_actions=full_prev_teammate_actions,
            train=False,
        )

        current_logits = logits[:, -1, :]

        if greedy:
            action = jnp.argmax(current_logits, axis=-1)
        else:
            assert rng is not None
            action = jax.random.categorical(rng, current_logits)

        if single:
            action = action[0]

        return action


def create_hybrid_ad_model(config: HybridADConfig) -> Tuple[HybridADModel, Any]:
    """Create and initialize a Hybrid-AD model."""
    model = HybridADModel(config)

    rng = jax.random.PRNGKey(0)
    batch_size = 2
    seq_len = 32

    dummy_obs = jnp.zeros((batch_size, seq_len) + config.obs_shape)
    dummy_prev_actions = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    dummy_prev_rewards = jnp.zeros((batch_size, seq_len))

    dummy_prev_teammate_actions = None
    if config.use_teammate_actions:
        dummy_prev_teammate_actions = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    params = model.init(
        rng,
        dummy_obs,
        dummy_prev_actions,
        dummy_prev_rewards,
        attention_mask=None,
        prev_teammate_actions=dummy_prev_teammate_actions,
        train=False,
    )

    return model, params


def count_parameters(params: Any) -> int:
    """Count total number of parameters in a pytree."""
    return sum(x.size for x in jax.tree.leaves(params))
