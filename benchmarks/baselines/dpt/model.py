"""DPT Model implementation in Flax/JAX.

This implements the Decision Pretrained Transformer model adapted for Overcooked V2.
Uses CNN encoders for grid-based (H, W, C) observations following the JaxMARL IPPO recipe.

Architecture follows DPT design with the following components:
- Observation encoder: CNN for grid-based (H, W, C) observations
- Transition embedding: Linear layer combining [obs, next_obs, action_onehot, reward]
- Transformer blocks: Pre-norm with causal attention (no positional encoding by default)
- Action head: Linear layer outputting action logits
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional, Sequence, Tuple

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class DPTConfig:
    """Configuration for DPT model."""
    # Observation shape (H, W, C) for grid-based observations
    obs_shape: Tuple[int, ...] = (9, 7, 26)  # Default Overcooked padded shape
    num_actions: int = 6  # Overcooked action space
    embedding_dim: int = 64  # Observation embedding dimension
    hidden_dim: int = 256  # Transformer hidden dimension
    num_layers: int = 4  # Number of transformer layers
    num_heads: int = 4  # Number of attention heads

    # Sequence configuration
    seq_len: int = 512  # Maximum context length (transitions)

    # Dropout rates
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1

    # Architecture options
    pre_norm: bool = True  # Pre-norm vs post-norm
    with_positional_encoding: bool = False

    # Teammate action conditioning (optional)
    use_teammate_actions: bool = False  # If True, include teammate actions in transition embedding

    def __post_init__(self):
        # Validate head dimension is compatible with hidden_dim
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"

    @property
    def obs_dim(self) -> int:
        """Backward-compatible property: flattened observation dimension."""
        return int(np.prod(self.obs_shape))


class CNNObservationEncoder(nn.Module):
    """CNN encoder for grid-based observations.

    Architecture from JaxMARL IPPO recipe:
    - 3x 1x1 convs (128, 128, 8 features) for pointwise feature extraction
    - 3x 3x3 convs (16, 32, 32 features) for spatial feature extraction
    - Dense layer to output embedding size
    """
    embedding_dim: int = 64

    @nn.compact
    def __call__(self, obs: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Encode grid-based observations.

        Args:
            obs: Observations of shape (..., H, W, C)
            train: Whether in training mode

        Returns:
            Encoded observations of shape (..., embedding_dim)
        """
        # Get original shape and flatten batch dimensions
        orig_shape = obs.shape
        spatial_dims = orig_shape[-3:]  # (H, W, C)
        batch_shape = orig_shape[:-3]

        # Reshape to (batch, H, W, C)
        x = obs.reshape(-1, *spatial_dims)

        # Pointwise convolutions (1x1 kernels)
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=8,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)

        # Spatial convolutions (3x3 kernels)
        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)

        # Flatten spatial dimensions
        x = x.reshape((x.shape[0], -1))

        # Project to embedding size
        x = nn.Dense(
            features=self.embedding_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)

        # Reshape back to original batch shape
        x = x.reshape(*batch_shape, self.embedding_dim)

        return x


# Backward compatibility alias
ObservationEncoder = CNNObservationEncoder


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm or post-norm.

    Follows DPT architecture with causal self-attention.
    """
    hidden_dim: int
    num_heads: int
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    pre_norm: bool = True

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        """Apply transformer block.

        Args:
            x: Input of shape (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask
            deterministic: If True, disable dropout (inference mode)

        Returns:
            Output of shape (batch_size, seq_len, hidden_dim)
        """
        # Pre-norm architecture
        if self.pre_norm:
            # Self-attention with pre-norm
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.attention_dropout,
                deterministic=deterministic,
            )(x, x, mask=mask)
            x = nn.Dropout(rate=self.residual_dropout, deterministic=deterministic)(x)
            x = residual + x

            # MLP with pre-norm
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.Dense(4 * self.hidden_dim)(x)
            x = nn.gelu(x)
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.Dropout(rate=self.residual_dropout, deterministic=deterministic)(x)
            x = residual + x
        else:
            # Post-norm architecture
            residual = x
            x = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.attention_dropout,
                deterministic=deterministic,
            )(x, x, mask=mask)
            x = nn.Dropout(rate=self.residual_dropout, deterministic=deterministic)(x)
            x = nn.LayerNorm()(residual + x)

            residual = x
            x = nn.Dense(4 * self.hidden_dim)(x)
            x = nn.gelu(x)
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.Dropout(rate=self.residual_dropout, deterministic=deterministic)(x)
            x = nn.LayerNorm()(residual + x)

        return x


class DPTModel(nn.Module):
    """Decision Pretrained Transformer for Overcooked V2.

    Takes context transitions (obs, action, next_obs, reward) and a query observation,
    predicts the expert action for the query.

    Key features:
    - Sequence format: [query, context_1, context_2, ..., context_K]
    - Each element contains (obs_emb, action_onehot, [teammate_action_onehot], next_obs_emb, reward)
    - Query has zeros for action/next_obs/reward
    - Output action logits for all positions (use first position for query)
    """
    config: DPTConfig

    def setup(self):
        cfg = self.config

        # Observation encoder
        self.obs_encoder = ObservationEncoder(
            embedding_dim=cfg.embedding_dim,
        )

        # Transition embedding: [obs_emb, action_onehot, [teammate_action_onehot], next_obs_emb, reward] -> hidden_dim
        # Input dimension depends on use_teammate_actions:
        # - Without teammate: 2 * emb_dim + num_actions + 1
        # - With teammate: 2 * emb_dim + 2 * num_actions + 1
        # Note: nn.Dense infers input dimension automatically from first forward pass
        self.embed_transition = nn.Dense(cfg.hidden_dim)

        # Transformer blocks
        # Using a list of blocks - JAX JIT will optimize the unrolled loop efficiently
        self.blocks = [
            TransformerBlock(
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                attention_dropout=cfg.attention_dropout,
                residual_dropout=cfg.residual_dropout,
                pre_norm=cfg.pre_norm,
            )
            for _ in range(cfg.num_layers)
        ]

        # Final layer norm (if pre-norm)
        if cfg.pre_norm:
            self.final_norm = nn.LayerNorm()

        # Action head
        self.action_head = nn.Dense(cfg.num_actions)

        # Optional positional encoding
        if cfg.with_positional_encoding:
            self.pos_embedding = self.param(
                'pos_embedding',
                nn.initializers.normal(stddev=0.02),
                (1, cfg.seq_len + 1, cfg.hidden_dim)
            )

    @nn.compact
    def __call__(
        self,
        query_obs: jnp.ndarray,         # (batch_size, H, W, C)
        context_obs: jnp.ndarray,        # (batch_size, seq_len, H, W, C)
        context_actions: jnp.ndarray,    # (batch_size, seq_len) int32
        context_next_obs: jnp.ndarray,   # (batch_size, seq_len, H, W, C)
        context_rewards: jnp.ndarray,    # (batch_size, seq_len)
        context_teammate_actions: Optional[jnp.ndarray] = None,  # (batch_size, seq_len) int32
        train: bool = True,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            query_obs: Query observations (batch_size, H, W, C)
            context_obs: Context observations (batch_size, seq_len, H, W, C)
            context_actions: Context actions (batch_size, seq_len)
            context_next_obs: Context next observations (batch_size, seq_len, H, W, C)
            context_rewards: Context rewards (batch_size, seq_len)
            context_teammate_actions: Context teammate actions (batch_size, seq_len), optional
            train: Whether in training mode

        Returns:
            Action logits of shape (batch_size, seq_len + 1, num_actions) if training,
            or (batch_size, num_actions) for query only if not training.
        """
        cfg = self.config
        batch_size, seq_len = context_rewards.shape

        # Encode observations
        # Query: (batch_size, embedding_dim)
        query_obs_emb = self.obs_encoder(query_obs, train=train)
        # Context: (batch_size, seq_len, embedding_dim)
        context_obs_emb = self.obs_encoder(context_obs, train=train)
        context_next_obs_emb = self.obs_encoder(context_next_obs, train=train)

        # One-hot encode actions: (batch_size, seq_len, num_actions)
        context_actions_onehot = jax.nn.one_hot(context_actions, cfg.num_actions)

        # Build sequence: [query, context_1, ..., context_K]
        # Query has zeros for action/next_obs/reward
        zeros_obs = jnp.zeros((batch_size, 1, cfg.embedding_dim))
        zeros_action = jnp.zeros((batch_size, 1, cfg.num_actions))
        zeros_reward = jnp.zeros((batch_size, 1, 1))

        # Concatenate query embedding with zeros
        query_obs_emb_expanded = query_obs_emb[:, None, :]  # (batch, 1, emb)

        # Observation sequence: [query_obs, context_obs]
        obs_seq = jnp.concatenate([query_obs_emb_expanded, context_obs_emb], axis=1)

        # Action sequence: [zeros, context_actions]
        action_seq = jnp.concatenate([zeros_action, context_actions_onehot], axis=1)

        # Next obs sequence: [zeros, context_next_obs]
        next_obs_seq = jnp.concatenate([zeros_obs, context_next_obs_emb], axis=1)

        # Reward sequence: [zeros, context_rewards]
        reward_seq = jnp.concatenate(
            [zeros_reward, context_rewards[:, :, None]],
            axis=1
        )  # (batch, seq_len + 1, 1)

        # Combine into transition embedding
        if cfg.use_teammate_actions and context_teammate_actions is not None:
            # One-hot encode teammate actions
            context_teammate_actions_onehot = jax.nn.one_hot(context_teammate_actions, cfg.num_actions)
            teammate_action_seq = jnp.concatenate([zeros_action, context_teammate_actions_onehot], axis=1)
            # Shape: (batch, seq_len + 1, 2 * emb_dim + 2 * num_actions + 1)
            sequence = jnp.concatenate([obs_seq, action_seq, teammate_action_seq, next_obs_seq, reward_seq], axis=-1)
        else:
            # Shape: (batch, seq_len + 1, 2 * emb_dim + num_actions + 1)
            sequence = jnp.concatenate([obs_seq, action_seq, next_obs_seq, reward_seq], axis=-1)

        # Project to hidden dimension
        sequence = self.embed_transition(sequence)  # (batch, seq_len + 1, hidden_dim)

        # Add positional encoding if configured
        if cfg.with_positional_encoding:
            actual_len = sequence.shape[1]
            sequence = sequence + self.pos_embedding[:, :actual_len, :]

        # Apply dropout to embeddings
        sequence = nn.Dropout(rate=cfg.embedding_dropout, deterministic=not train)(sequence)

        # Create causal attention mask
        # DPT uses causal attention (each position can only attend to itself and previous)
        total_len = seq_len + 1
        causal_mask = jnp.tril(jnp.ones((total_len, total_len)))
        causal_mask = causal_mask[None, None, :, :]  # Add batch and head dims

        # Apply transformer blocks
        x = sequence
        for block in self.blocks:
            x = block(x, mask=causal_mask, deterministic=not train)

        # Apply final norm if pre-norm
        if cfg.pre_norm:
            x = self.final_norm(x)

        # Compute action logits
        logits = self.action_head(x)  # (batch, seq_len + 1, num_actions)

        if not train:
            # During inference, return only the query prediction (first position)
            # Note: DPT returns head[:, -1, :] but their sequence is [query, context]
            # We follow the same convention - query is at position 0
            return logits[:, 0, :]

        return logits

    def get_action(
        self,
        params: Any,
        query_obs: jnp.ndarray,
        context_obs: jnp.ndarray,
        context_actions: jnp.ndarray,
        context_next_obs: jnp.ndarray,
        context_rewards: jnp.ndarray,
        rng: Optional[jax.Array] = None,
        greedy: bool = True,
        context_teammate_actions: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Get action for query given context.

        Args:
            params: Model parameters
            query_obs: Query observation (H, W, C) or (batch, H, W, C)
            context_obs: Context observations (K, H, W, C) or (batch, K, H, W, C)
            context_actions: Context actions (K,) or (batch, K)
            context_next_obs: Context next observations (K, H, W, C) or (batch, K, H, W, C)
            context_rewards: Context rewards (K,) or (batch, K)
            rng: Random key for sampling (only used if not greedy)
            greedy: If True, return argmax action; if False, sample from distribution
            context_teammate_actions: Context teammate actions (K,) or (batch, K), optional

        Returns:
            Action (,) or (batch,)
        """
        cfg = self.config

        # Add batch dimension if needed
        # Single sample has shape (H, W, C), batched has (batch, H, W, C)
        single = query_obs.ndim == len(cfg.obs_shape)
        if single:
            query_obs = query_obs[None, ...]
            context_obs = context_obs[None, ...]
            context_actions = context_actions[None, :]
            context_next_obs = context_next_obs[None, ...]
            context_rewards = context_rewards[None, :]
            if context_teammate_actions is not None:
                context_teammate_actions = context_teammate_actions[None, :]

        # Forward pass (inference mode)
        logits = self.apply(
            params,
            query_obs,
            context_obs,
            context_actions,
            context_next_obs,
            context_rewards,
            context_teammate_actions=context_teammate_actions,
            train=False,
        )

        if greedy:
            action = jnp.argmax(logits, axis=-1)
        else:
            assert rng is not None, "Must provide rng for sampling"
            action = jax.random.categorical(rng, logits)

        if single:
            action = action[0]

        return action


def create_dpt_model(config: DPTConfig) -> Tuple[DPTModel, Any]:
    """Create and initialize a DPT model.

    Args:
        config: Model configuration

    Returns:
        Tuple of (model, init_params)
    """
    model = DPTModel(config)

    # Initialize with dummy inputs
    rng = jax.random.PRNGKey(0)
    batch_size = 2
    seq_len = 32

    # Use obs_shape for (H, W, C) observations
    dummy_query = jnp.zeros((batch_size,) + config.obs_shape)
    dummy_context_obs = jnp.zeros((batch_size, seq_len) + config.obs_shape)
    dummy_context_actions = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    dummy_context_next_obs = jnp.zeros((batch_size, seq_len) + config.obs_shape)
    dummy_context_rewards = jnp.zeros((batch_size, seq_len))

    # Include dummy teammate actions if configured
    dummy_context_teammate_actions = None
    if config.use_teammate_actions:
        dummy_context_teammate_actions = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    params = model.init(
        rng,
        dummy_query,
        dummy_context_obs,
        dummy_context_actions,
        dummy_context_next_obs,
        dummy_context_rewards,
        context_teammate_actions=dummy_context_teammate_actions,
        train=False,
    )

    return model, params


def count_parameters(params: Any) -> int:
    """Count total number of parameters in a pytree."""
    return sum(x.size for x in jax.tree.leaves(params))
