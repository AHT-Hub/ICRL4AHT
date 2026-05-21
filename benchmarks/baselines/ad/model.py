"""AD Model implementation in Flax/JAX.

This implements the Algorithm Distillation transformer model adapted for Overcooked V2.
Uses CNN encoders for grid-based (H, W, C) observations following the JaxMARL IPPO recipe.

Architecture follows AD design with these key features:
- Token format: (prev_action_emb, prev_reward, obs_emb) concatenated
- Causal transformer decoder
- Predicts next action at each timestep
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class ADConfig:
    """Configuration for AD model."""
    # Observation shape (H, W, C) for grid-based observations
    obs_shape: Tuple[int, ...] = (9, 7, 26)  # Default Overcooked padded shape
    num_actions: int = 6  # Overcooked action space
    embedding_dim: int = 64  # Embedding dimension for obs/action
    hidden_dim: int = 256  # Transformer hidden dimension
    num_layers: int = 4  # Number of transformer layers
    num_heads: int = 4  # Number of attention heads

    # Sequence configuration
    seq_len: int = 4096  # Maximum context length (steps)

    # Dropout rates
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1

    # Architecture options
    pre_norm: bool = True  # Pre-norm vs post-norm (pre-norm is standard)
    normalize_qk: bool = False  # QK normalization (from NaViT)

    # Teammate action conditioning (optional)
    use_teammate_actions: bool = False  # If True, include teammate actions in token

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
    """Transformer block with pre-norm.

    Uses causal self-attention without explicit positional encoding
    (relies on ALiBi-style implicit positional bias or learned positions).
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
        train: bool = True,
    ) -> jnp.ndarray:
        """Apply transformer block.

        Args:
            x: Input of shape (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask (batch, 1, seq, seq) or (1, 1, seq, seq)
            train: Whether in training mode

        Returns:
            Output of shape (batch_size, seq_len, hidden_dim)
        """
        if self.pre_norm:
            # Self-attention with pre-norm
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.attention_dropout,
                deterministic=not train,
            )(x, x, mask=mask)
            x = nn.Dropout(rate=self.residual_dropout, deterministic=not train)(x)
            x = residual + x

            # MLP with pre-norm
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.Dense(4 * self.hidden_dim)(x)
            x = nn.gelu(x)
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.Dropout(rate=self.residual_dropout, deterministic=not train)(x)
            x = residual + x
        else:
            # Post-norm architecture
            residual = x
            x = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.attention_dropout,
                deterministic=not train,
            )(x, x, mask=mask)
            x = nn.Dropout(rate=self.residual_dropout, deterministic=not train)(x)
            x = nn.LayerNorm()(residual + x)

            residual = x
            x = nn.Dense(4 * self.hidden_dim)(x)
            x = nn.gelu(x)
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.Dropout(rate=self.residual_dropout, deterministic=not train)(x)
            x = nn.LayerNorm()(residual + x)

        return x


class ADModel(nn.Module):
    """Algorithm Distillation Transformer for Overcooked V2.

    Takes a sequence of (obs, prev_action, prev_reward) tuples and predicts
    the action at each timestep with causal masking.

    Token format:
    - Input per step: (prev_action_emb, prev_reward, obs_emb) concatenated
    - Optionally: (prev_action_emb, prev_teammate_action_emb, prev_reward, obs_emb)
    - Target: action_t

    During training:
    - Input: full sequence of context + actions
    - Output: predicted actions at all positions
    - Loss: cross-entropy on actions with causal masking

    During evaluation:
    - Maintain growing buffer of (obs, prev_action, prev_reward, [prev_teammate_action])
    - At each step, run forward pass and predict next action
    """
    config: ADConfig

    def setup(self):
        cfg = self.config

        # Observation encoder
        self.obs_encoder = ObservationEncoder(
            embedding_dim=cfg.embedding_dim,
        )

        # Action embedding (discrete actions)
        self.action_embedding = nn.Embed(
            num_embeddings=cfg.num_actions,
            features=cfg.embedding_dim,
        )

        # Teammate action embedding (optional, same action space)
        if cfg.use_teammate_actions:
            self.teammate_action_embedding = nn.Embed(
                num_embeddings=cfg.num_actions,
                features=cfg.embedding_dim,
            )

        # Token embedding: combines (action_emb, [teammate_action_emb], reward, obs_emb) -> hidden_dim
        # Token format:
        #   - Without teammate: [action_emb, reward, obs_emb]
        #   - With teammate: [action_emb, teammate_action_emb, reward, obs_emb]
        # Input dimensions (inferred by Flax at runtime):
        #   - Without teammate: embedding_dim + 1 + embedding_dim = 2 * embedding_dim + 1
        #   - With teammate: 3 * embedding_dim + 1
        self.embed_token = nn.Dense(cfg.hidden_dim)

        # Transformer blocks
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

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,             # (batch_size, seq_len, H, W, C)
        prev_actions: jnp.ndarray,    # (batch_size, seq_len) int32
        prev_rewards: jnp.ndarray,    # (batch_size, seq_len)
        attention_mask: Optional[jnp.ndarray] = None,  # (batch_size, seq_len)
        prev_teammate_actions: Optional[jnp.ndarray] = None,  # (batch_size, seq_len) int32
        train: bool = True,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            obs: Observations at each timestep (batch_size, seq_len, H, W, C)
            prev_actions: Previous actions (batch_size, seq_len), action at t-1
            prev_rewards: Previous rewards (batch_size, seq_len), reward at t-1
            attention_mask: Padding mask (batch_size, seq_len), 1=valid, 0=padding
            prev_teammate_actions: Previous teammate actions (batch_size, seq_len), optional
            train: Whether in training mode

        Returns:
            Action logits of shape (batch_size, seq_len, num_actions)
        """
        cfg = self.config
        batch_size, seq_len = prev_rewards.shape

        # Encode observations: (batch, seq, H, W, C) -> (batch, seq, emb_dim)
        obs_emb = self.obs_encoder(obs, train=train)

        # Embed previous actions: (batch, seq) -> (batch, seq, emb_dim)
        action_emb = self.action_embedding(prev_actions)

        # Expand rewards: (batch, seq) -> (batch, seq, 1)
        reward_emb = prev_rewards[:, :, None]

        # Concatenate token based on whether teammate actions are used
        if cfg.use_teammate_actions and prev_teammate_actions is not None:
            # Embed teammate actions: (batch, seq) -> (batch, seq, emb_dim)
            teammate_action_emb = self.teammate_action_embedding(prev_teammate_actions)
            # Shape: (batch, seq, 3 * emb_dim + 1)
            sequence = jnp.concatenate([action_emb, teammate_action_emb, reward_emb, obs_emb], axis=-1)
        else:
            # Shape: (batch, seq, 2 * emb_dim + 1)
            sequence = jnp.concatenate([action_emb, reward_emb, obs_emb], axis=-1)

        # Project to hidden dimension
        sequence = self.embed_token(sequence)  # (batch, seq, hidden_dim)

        # Apply dropout to embeddings
        sequence = nn.Dropout(rate=cfg.embedding_dropout, deterministic=not train)(sequence)

        # Create causal attention mask
        # Shape: (1, 1, seq_len, seq_len)
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        causal_mask = causal_mask[None, None, :, :]  # Add batch and head dims

        # Combine with padding mask if provided
        if attention_mask is not None:
            # attention_mask: (batch, seq) -> (batch, 1, 1, seq)
            # This masks out padding positions in keys
            padding_mask = attention_mask[:, None, None, :]
            causal_mask = causal_mask * padding_mask

        # Apply transformer blocks
        x = sequence
        for block in self.blocks:
            x = block(x, mask=causal_mask, train=train)

        # Apply final norm if pre-norm
        if cfg.pre_norm:
            x = self.final_norm(x)

        # Compute action logits
        logits = self.action_head(x)  # (batch, seq, num_actions)

        return logits

    def get_action(
        self,
        params: Any,
        obs: jnp.ndarray,             # (H, W, C) or (batch, H, W, C)
        prev_action: jnp.ndarray,     # () or (batch,)
        prev_reward: jnp.ndarray,     # () or (batch,)
        context_obs: jnp.ndarray,     # (ctx_len, H, W, C) or (batch, ctx_len, H, W, C)
        context_prev_actions: jnp.ndarray,   # (ctx_len,) or (batch, ctx_len)
        context_prev_rewards: jnp.ndarray,   # (ctx_len,) or (batch, ctx_len)
        rng: Optional[jax.Array] = None,
        greedy: bool = True,
        prev_teammate_action: Optional[jnp.ndarray] = None,  # () or (batch,)
        context_prev_teammate_actions: Optional[jnp.ndarray] = None,  # (ctx_len,) or (batch, ctx_len)
    ) -> jnp.ndarray:
        """Get action given current step and context.

        This is the main inference function for AD evaluation.
        It concatenates the context with the current observation and predicts
        the next action.

        Args:
            params: Model parameters
            obs: Current observation (H, W, C) or (batch, H, W, C)
            prev_action: Action taken at t-1
            prev_reward: Reward received at t-1
            context_obs: Context observations (ctx_len, H, W, C) or (batch, ctx_len, H, W, C)
            context_prev_actions: Context previous actions
            context_prev_rewards: Context previous rewards
            rng: Random key for sampling (only used if not greedy)
            greedy: If True, return argmax action; if False, sample
            prev_teammate_action: Teammate action at t-1 (optional)
            context_prev_teammate_actions: Context previous teammate actions (optional)

        Returns:
            Action
        """
        cfg = self.config

        # Handle single sample (add batch dim)
        # Single sample has shape (H, W, C), batched has (batch, H, W, C)
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

        # Concatenate context with current step
        # Context: (batch, ctx_len, H, W, C)
        # Current: (batch, H, W, C) -> (batch, 1, H, W, C)
        full_obs = jnp.concatenate([context_obs, jnp.expand_dims(obs, axis=1)], axis=1)
        full_prev_actions = jnp.concatenate([context_prev_actions, prev_action[:, None]], axis=1)
        full_prev_rewards = jnp.concatenate([context_prev_rewards, prev_reward[:, None]], axis=1)

        # Handle teammate actions if configured
        full_prev_teammate_actions = None
        if cfg.use_teammate_actions and prev_teammate_action is not None and context_prev_teammate_actions is not None:
            full_prev_teammate_actions = jnp.concatenate(
                [context_prev_teammate_actions, prev_teammate_action[:, None]], axis=1
            )

        # Forward pass (inference mode)
        logits = self.apply(
            params,
            full_obs,
            full_prev_actions,
            full_prev_rewards,
            attention_mask=None,  # No padding during inference
            prev_teammate_actions=full_prev_teammate_actions,
            train=False,
        )

        # Get logits for the last position (current step)
        current_logits = logits[:, -1, :]  # (batch, num_actions)

        if greedy:
            action = jnp.argmax(current_logits, axis=-1)
        else:
            assert rng is not None, "Must provide rng for sampling"
            action = jax.random.categorical(rng, current_logits)

        if single:
            action = action[0]

        return action


def create_ad_model(config: ADConfig) -> Tuple[ADModel, Any]:
    """Create and initialize an AD model.

    Args:
        config: Model configuration

    Returns:
        Tuple of (model, init_params)
    """
    model = ADModel(config)

    # Initialize with dummy inputs
    rng = jax.random.PRNGKey(0)
    batch_size = 2
    seq_len = 32

    # Use obs_shape for (H, W, C) observations
    dummy_obs = jnp.zeros((batch_size, seq_len) + config.obs_shape)
    dummy_prev_actions = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    dummy_prev_rewards = jnp.zeros((batch_size, seq_len))

    # Include dummy teammate actions if configured
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


# ============================================================================
# Causal mask utilities (for explicit masking if needed)
# ============================================================================

def make_causal_mask(seq_len: int) -> jnp.ndarray:
    """Create a causal attention mask.

    Args:
        seq_len: Sequence length

    Returns:
        Mask of shape (1, 1, seq_len, seq_len) where 1=attend, 0=mask
    """
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    return mask[None, None, :, :]


def make_padding_mask(attention_mask: jnp.ndarray) -> jnp.ndarray:
    """Convert attention mask to broadcast shape for attention.

    Args:
        attention_mask: (batch, seq) with 1=valid, 0=padding

    Returns:
        Mask of shape (batch, 1, 1, seq) for key masking
    """
    return attention_mask[:, None, None, :]


def combine_masks(
    causal_mask: jnp.ndarray,
    padding_mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Combine causal and padding masks.

    Args:
        causal_mask: (1, 1, seq, seq)
        padding_mask: (batch, 1, 1, seq) or None

    Returns:
        Combined mask
    """
    if padding_mask is None:
        return causal_mask
    return causal_mask * padding_mask
