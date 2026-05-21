"""CNN+RNN Actor-Critic network for grid-based observations.

This architecture follows the JaxMARL IPPO recipe for OvercookedV2:
1. CNN encoder for grid-based observations (1x1 and 3x3 convs)
2. LayerNorm after CNN embedding
3. GRU for temporal modeling
4. FC layers for actor and critic heads
"""
import functools
import numpy as np
from typing import Sequence, Callable, Any

import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp


class ScannedRNN(nn.Module):
    """Scanned GRU for efficient sequence processing.

    Uses nn.scan for efficient sequential processing in JAX.
    Automatically resets hidden state on episode boundaries.
    """
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x

        new_carry = self.initialize_carry(ins.shape[0], ins.shape[1])

        rnn_state = jnp.where(
            resets[:, np.newaxis],
            new_carry,
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        """Initialize GRU hidden state to zeros."""
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class CNN(nn.Module):
    """CNN encoder for grid-based observations.

    Architecture from JaxMARL IPPO recipe:
    - 3x 1x1 convs (128, 128, 8 features) for pointwise feature extraction
    - 3x 3x3 convs (16, 32, 32 features) for spatial feature extraction
    - Dense layer to output size
    """
    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train=False):
        # Pointwise convolutions (1x1 kernels)
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=8,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        # Spatial convolutions (3x3 kernels)
        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        # Flatten spatial dimensions
        x = x.reshape((x.shape[0], -1))

        # Project to output size
        x = nn.Dense(
            features=self.output_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        return x


class CNNRNNActorCritic(nn.Module):
    """CNN+RNN Actor-Critic network for grid-based observations.

    Matches the JaxMARL IPPO architecture for OvercookedV2.

    Args:
        action_dim: Number of discrete actions
        fc_dim_size: Hidden dimension for actor/critic FC layers
        gru_hidden_dim: Hidden dimension for GRU
        activation: Activation function name ("relu" or "tanh")
    """
    action_dim: Sequence[int]
    fc_dim_size: int = 128
    gru_hidden_dim: int = 128
    activation: str = "relu"

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # CNN embedding (vmap over time dimension)
        embed_model = CNN(
            output_size=self.gru_hidden_dim,
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(obs)

        # LayerNorm after CNN embedding
        embedding = nn.LayerNorm()(embedding)

        # GRU
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # Actor head
        actor_mean = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        # Critic head
        critic = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class CNNRNNActorCriticWithConditionalCritic(nn.Module):
    """CNN+RNN Actor-Critic network with conditional critic for BRDiv.

    The actor uses CNN+RNN for grid-based observations.
    The critic is conditioned on an auxiliary input (partner ID) to predict
    values for different partner pairings.

    Args:
        action_dim: Number of discrete actions
        fc_dim_size: Hidden dimension for actor/critic FC layers
        gru_hidden_dim: Hidden dimension for GRU
        activation: Activation function name ("relu" or "tanh")
    """
    action_dim: Sequence[int]
    fc_dim_size: int = 128
    gru_hidden_dim: int = 128
    activation: str = "relu"

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions, aux_obs = x

        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # CNN embedding (vmap over time dimension)
        embed_model = CNN(
            output_size=self.gru_hidden_dim,
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(obs)

        # LayerNorm after CNN embedding
        embedding = nn.LayerNorm()(embedding)

        # GRU
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # Actor head (does not use aux_obs)
        actor_mean = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        # Apply action masking
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        # Conditional Critic head (uses aux_obs = partner ID one-hot encoding)
        # Concatenate embedding with aux_obs for the critic
        critic_input = jnp.concatenate([embedding, aux_obs], axis=-1)

        critic = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(critic_input)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class CNNRNNActorCriticWithAvailActions(nn.Module):
    """CNN+RNN Actor-Critic network with action masking support.

    Same as CNNRNNActorCritic but supports available action masking.

    Args:
        action_dim: Number of discrete actions
        fc_dim_size: Hidden dimension for actor/critic FC layers
        gru_hidden_dim: Hidden dimension for GRU
        activation: Activation function name ("relu" or "tanh")
    """
    action_dim: Sequence[int]
    fc_dim_size: int = 128
    gru_hidden_dim: int = 128
    activation: str = "relu"

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x

        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # CNN embedding (vmap over time dimension)
        embed_model = CNN(
            output_size=self.gru_hidden_dim,
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(obs)

        # LayerNorm after CNN embedding
        embedding = nn.LayerNorm()(embedding)

        # GRU
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # Actor head
        actor_mean = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        # Apply action masking
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        # Critic head
        critic = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)
