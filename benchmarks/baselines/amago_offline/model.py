"""AMAGO-offline Model implementation in Flax/JAX.

AMAGO-style architecture adapted for offline action prediction:
1. Timestep encoder: (obs, prev_action, prev_reward, done, time_idx) -> token
2. Trajectory encoder: GRU over token sequence
3. Action head: hidden state -> action logits

Reference: AMAGO (Grigsby et al., 2024) adapted for offline-only use.
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
class AMAGOOfflineConfig:
    """Configuration for AMAGO-offline model."""
    obs_shape: Tuple[int, ...] = (9, 7, 26)
    num_actions: int = 6
    embedding_dim: int = 64
    hidden_dim: int = 256

    # Timestep encoder
    tstep_hidden_dim: int = 256

    # Trajectory encoder
    traj_encoder: str = "gru"  # "gru" or "transformer"
    traj_hidden_dim: int = 256
    num_traj_layers: int = 2

    # Transformer-specific (only used if traj_encoder="transformer")
    num_heads: int = 4
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1

    seq_len: int = 4096

    embedding_dropout: float = 0.1

    use_teammate_actions: bool = False

    @property
    def obs_dim(self) -> int:
        return int(np.prod(self.obs_shape))


class TstepEncoder(nn.Module):
    """Timestep encoder: combines obs, prev_action, prev_reward, done, time_idx into a single token."""
    config: AMAGOOfflineConfig

    @nn.compact
    def __call__(
        self,
        obs_emb: jnp.ndarray,       # (B, L, embedding_dim)
        prev_actions: jnp.ndarray,   # (B, L) int32
        prev_rewards: jnp.ndarray,   # (B, L)
        dones: jnp.ndarray,          # (B, L)
        time_idxs: jnp.ndarray,      # (B, L) int32
        prev_teammate_actions: Optional[jnp.ndarray] = None,  # (B, L) int32
        train: bool = True,
    ) -> jnp.ndarray:
        cfg = self.config

        action_emb = nn.Embed(
            num_embeddings=cfg.num_actions,
            features=cfg.embedding_dim,
        )(prev_actions)

        reward_feat = prev_rewards[:, :, None]
        done_feat = dones.astype(jnp.float32)[:, :, None]
        time_feat = (time_idxs.astype(jnp.float32) / 1000.0)[:, :, None]

        features = [obs_emb, action_emb, reward_feat, done_feat, time_feat]

        if cfg.use_teammate_actions and prev_teammate_actions is not None:
            teammate_emb = nn.Embed(
                num_embeddings=cfg.num_actions,
                features=cfg.embedding_dim,
            )(prev_teammate_actions)
            features.append(teammate_emb)

        concat = jnp.concatenate(features, axis=-1)

        x = nn.Dense(cfg.tstep_hidden_dim)(concat)
        x = nn.relu(x)
        x = nn.Dense(cfg.hidden_dim)(x)
        x = nn.LayerNorm()(x)

        return x


class GRUTrajEncoder(nn.Module):
    """GRU-based trajectory encoder."""
    hidden_dim: int
    num_layers: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
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


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block for trajectory encoding."""
    hidden_dim: int
    num_heads: int
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1

    @nn.compact
    def __call__(self, x, mask=None, train=True):
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.attention_dropout,
            deterministic=not train,
        )(x, x, mask=mask)
        x = nn.Dropout(rate=self.residual_dropout, deterministic=not train)(x)
        x = residual + x

        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(4 * self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(rate=self.residual_dropout, deterministic=not train)(x)
        x = residual + x

        return x


class TransformerTrajEncoder(nn.Module):
    """Transformer-based trajectory encoder with causal masking."""
    config: AMAGOOfflineConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        cfg = self.config
        batch_size, seq_len, _ = x.shape

        if x.shape[-1] != cfg.hidden_dim:
            x = nn.Dense(cfg.hidden_dim)(x)

        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))[None, None, :, :]

        for _ in range(cfg.num_traj_layers):
            x = TransformerBlock(
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                attention_dropout=cfg.attention_dropout,
                residual_dropout=cfg.residual_dropout,
            )(x, mask=causal_mask, train=train)

        return x


class AMAGOOfflineModel(nn.Module):
    """AMAGO-style offline sequence model for action prediction.

    Architecture:
    1. CNN obs encoder -> obs embedding
    2. Timestep encoder: (obs_emb, prev_action, prev_reward, done, time_idx) -> token
    3. Trajectory encoder: GRU or Transformer over token sequence -> hidden states
    4. Action head: hidden states -> action logits
    """
    config: AMAGOOfflineConfig

    def setup(self):
        cfg = self.config
        self.obs_encoder = CNNObservationEncoder(embedding_dim=cfg.embedding_dim)
        self.tstep_encoder = TstepEncoder(config=cfg)

        if cfg.traj_encoder == "gru":
            self.traj_encoder = GRUTrajEncoder(
                hidden_dim=cfg.traj_hidden_dim,
                num_layers=cfg.num_traj_layers,
            )
        else:
            self.traj_encoder = TransformerTrajEncoder(config=cfg)

        if cfg.traj_hidden_dim != cfg.hidden_dim:
            self.output_proj = nn.Dense(cfg.hidden_dim)

        self.final_norm = nn.LayerNorm()
        self.action_head = nn.Dense(cfg.num_actions)

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        prev_actions: jnp.ndarray,
        prev_rewards: jnp.ndarray,
        dones: jnp.ndarray,
        time_idxs: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        prev_teammate_actions: Optional[jnp.ndarray] = None,
        train: bool = True,
    ) -> jnp.ndarray:
        cfg = self.config

        obs_emb = self.obs_encoder(obs, train=train)

        tstep_tokens = self.tstep_encoder(
            obs_emb, prev_actions, prev_rewards, dones, time_idxs,
            prev_teammate_actions=prev_teammate_actions,
            train=train,
        )

        tstep_tokens = nn.Dropout(rate=cfg.embedding_dropout, deterministic=not train)(tstep_tokens)

        x = self.traj_encoder(tstep_tokens, train=train)

        if cfg.traj_hidden_dim != cfg.hidden_dim:
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
        context_dones: jnp.ndarray,
        context_time_idxs: jnp.ndarray,
        rng: Optional[jax.Array] = None,
        greedy: bool = True,
        done: Optional[jnp.ndarray] = None,
        time_idx: Optional[jnp.ndarray] = None,
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
            context_dones = context_dones[None, :]
            context_time_idxs = context_time_idxs[None, :]
            if done is not None:
                done = jnp.array([done])
            if time_idx is not None:
                time_idx = jnp.array([time_idx])
            if prev_teammate_action is not None:
                prev_teammate_action = jnp.array([prev_teammate_action])
            if context_prev_teammate_actions is not None:
                context_prev_teammate_actions = context_prev_teammate_actions[None, :]

        full_obs = jnp.concatenate([context_obs, jnp.expand_dims(obs, axis=1)], axis=1)
        full_prev_actions = jnp.concatenate([context_prev_actions, prev_action[:, None]], axis=1)
        full_prev_rewards = jnp.concatenate([context_prev_rewards, prev_reward[:, None]], axis=1)

        if done is None:
            done = jnp.zeros_like(prev_action, dtype=jnp.float32)
        full_dones = jnp.concatenate([context_dones, done[:, None]], axis=1)

        ctx_len = context_time_idxs.shape[1]
        if time_idx is None:
            if ctx_len > 0:
                time_idx = context_time_idxs[:, -1] + 1
            else:
                time_idx = jnp.zeros_like(prev_action)
        full_time_idxs = jnp.concatenate([context_time_idxs, time_idx[:, None]], axis=1)

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
            full_dones,
            full_time_idxs,
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


def create_amago_offline_model(config: AMAGOOfflineConfig) -> Tuple[AMAGOOfflineModel, Any]:
    """Create and initialize an AMAGO-offline model."""
    model = AMAGOOfflineModel(config)

    rng = jax.random.PRNGKey(0)
    batch_size = 2
    seq_len = 32

    dummy_obs = jnp.zeros((batch_size, seq_len) + config.obs_shape)
    dummy_prev_actions = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    dummy_prev_rewards = jnp.zeros((batch_size, seq_len))
    dummy_dones = jnp.zeros((batch_size, seq_len))
    dummy_time_idxs = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    dummy_prev_teammate_actions = None
    if config.use_teammate_actions:
        dummy_prev_teammate_actions = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    params = model.init(
        rng,
        dummy_obs,
        dummy_prev_actions,
        dummy_prev_rewards,
        dummy_dones,
        dummy_time_idxs,
        attention_mask=None,
        prev_teammate_actions=dummy_prev_teammate_actions,
        train=False,
    )

    return model, params


def count_parameters(params: Any) -> int:
    """Count total number of parameters in a pytree."""
    return sum(x.size for x in jax.tree.leaves(params))
