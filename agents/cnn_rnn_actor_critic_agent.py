"""CNN+RNN Actor-Critic Policy for grid-based observations.

This policy wraps the CNNRNNActorCritic network for use in training.
Designed for OvercookedV2 environments with grid-based observations.
"""
from functools import partial
from typing import Tuple, Optional

import jax
import jax.numpy as jnp

from agents.agent_interface import AgentPolicy
from agents.cnn_rnn_actor_critic import (
    CNNRNNActorCritic,
    CNNRNNActorCriticWithAvailActions,
    CNNRNNActorCriticWithConditionalCritic,
    ScannedRNN,
)


class CNNRNNActorCriticPolicy(AgentPolicy):
    """Policy wrapper for CNN+RNN Actor-Critic.

    This policy is designed for grid-based observations (H, W, C format).
    It uses CNN for spatial feature extraction followed by GRU for temporal modeling.

    Args:
        action_dim: Number of discrete actions
        obs_shape: Shape of the observation (H, W, C)
        activation: Activation function name ("relu" or "tanh")
        fc_dim_size: Hidden dimension for actor/critic FC layers
        gru_hidden_dim: Hidden dimension for GRU
        use_avail_actions: Whether to use action masking
    """

    def __init__(
        self,
        action_dim: int,
        obs_shape: Tuple[int, ...],
        activation: str = "relu",
        fc_dim_size: int = 128,
        gru_hidden_dim: int = 128,
        use_avail_actions: bool = True,
    ):
        # Store obs_shape as (H, W, C)
        self.obs_shape = obs_shape
        self.action_dim_int = action_dim
        # obs_dim is the flattened size for compatibility
        obs_dim = obs_shape[0] * obs_shape[1] * obs_shape[2] if len(obs_shape) == 3 else obs_shape[0]
        super().__init__(action_dim, obs_dim)

        self.use_avail_actions = use_avail_actions
        self.gru_hidden_dim = gru_hidden_dim
        self.fc_dim_size = fc_dim_size
        self.activation = activation

        if use_avail_actions:
            self.network = CNNRNNActorCriticWithAvailActions(
                action_dim=action_dim,
                fc_dim_size=fc_dim_size,
                gru_hidden_dim=gru_hidden_dim,
                activation=activation,
            )
        else:
            self.network = CNNRNNActorCritic(
                action_dim=action_dim,
                fc_dim_size=fc_dim_size,
                gru_hidden_dim=gru_hidden_dim,
                activation=activation,
            )

    @partial(jax.jit, static_argnums=(0,))
    def get_action(
        self,
        params,
        obs,
        done,
        avail_actions,
        hstate,
        rng,
        aux_obs=None,
        env_state=None,
        test_mode=False,
    ):
        """Get actions for the CNN+RNN policy.

        Args:
            params: Network parameters
            obs: Observations with shape (seq_len, batch_size, H, W, C)
            done: Done flags with shape (seq_len, batch_size)
            avail_actions: Available actions with shape (seq_len, batch_size, action_dim)
            hstate: Hidden state with shape (1, batch_size, gru_hidden_dim)
            rng: Random key
            test_mode: If True, use deterministic action selection

        Returns:
            action: Selected actions
            new_hstate: Updated hidden state
        """
        batch_size = obs.shape[1]

        if self.use_avail_actions:
            new_hstate, pi, _ = self.network.apply(
                params, hstate.squeeze(0), (obs, done, avail_actions)
            )
        else:
            new_hstate, pi, _ = self.network.apply(
                params, hstate.squeeze(0), (obs, done)
            )

        action = jax.lax.cond(
            test_mode, lambda: pi.mode(), lambda: pi.sample(seed=rng)
        )
        return action, new_hstate.reshape(1, batch_size, -1)

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(
        self,
        params,
        obs,
        done,
        avail_actions,
        hstate,
        rng,
        aux_obs=None,
        env_state=None,
    ):
        """Get actions, values, and policy for the CNN+RNN policy.

        Args:
            params: Network parameters
            obs: Observations with shape (seq_len, batch_size, H, W, C)
            done: Done flags with shape (seq_len, batch_size)
            avail_actions: Available actions with shape (seq_len, batch_size, action_dim)
            hstate: Hidden state with shape (1, batch_size, gru_hidden_dim)
            rng: Random key

        Returns:
            action: Selected actions
            val: Value estimates
            pi: Action distribution
            new_hstate: Updated hidden state
        """
        batch_size = obs.shape[1]

        if self.use_avail_actions:
            new_hstate, pi, val = self.network.apply(
                params, hstate.squeeze(0), (obs, done, avail_actions)
            )
        else:
            new_hstate, pi, val = self.network.apply(
                params, hstate.squeeze(0), (obs, done)
            )

        action = pi.sample(seed=rng)
        return action, val, pi, new_hstate.reshape(1, batch_size, -1)

    def init_hstate(self, batch_size, aux_info=None):
        """Initialize hidden state for the GRU.

        Args:
            batch_size: Number of parallel environments/actors

        Returns:
            Hidden state with shape (1, batch_size, gru_hidden_dim)
        """
        hstate = ScannedRNN.initialize_carry(batch_size, self.gru_hidden_dim)
        hstate = hstate.reshape(1, batch_size, self.gru_hidden_dim)
        return hstate

    def init_params(self, rng):
        """Initialize network parameters.

        Args:
            rng: Random key for initialization

        Returns:
            Initialized parameters
        """
        batch_size = 1
        seq_len = 1

        # Initialize hidden state
        init_hstate = self.init_hstate(batch_size)

        # Create dummy inputs with shape (seq_len, batch_size, H, W, C)
        dummy_obs = jnp.zeros((seq_len, batch_size) + self.obs_shape)
        dummy_done = jnp.zeros((seq_len, batch_size))

        if self.use_avail_actions:
            dummy_avail = jnp.ones((seq_len, batch_size, self.action_dim_int))
            dummy_x = (dummy_obs, dummy_done, dummy_avail)
        else:
            dummy_x = (dummy_obs, dummy_done)

        # Initialize model
        return self.network.init(rng, init_hstate.reshape(batch_size, -1), dummy_x)


class CNNRNNActorCriticWithConditionalCriticPolicy(AgentPolicy):
    """Policy wrapper for CNN+RNN Actor-Critic with Conditional Critic.

    This policy is designed for BRDiv training where the critic needs to be
    conditioned on the partner ID. It uses CNN+RNN for grid-based observations
    and has a conditional critic that takes an auxiliary input (partner ID).

    Args:
        action_dim: Number of discrete actions
        obs_shape: Shape of the observation (H, W, C)
        pop_size: Number of agents in the population for the conditional critic
        activation: Activation function name ("relu" or "tanh")
        fc_dim_size: Hidden dimension for actor/critic FC layers
        gru_hidden_dim: Hidden dimension for GRU
    """

    def __init__(
        self,
        action_dim: int,
        obs_shape: Tuple[int, ...],
        pop_size: int,
        activation: str = "relu",
        fc_dim_size: int = 128,
        gru_hidden_dim: int = 128,
    ):
        # Store obs_shape as (H, W, C)
        self.obs_shape = obs_shape
        self.action_dim_int = action_dim
        self.pop_size = pop_size
        # obs_dim is the flattened size for compatibility
        obs_dim = obs_shape[0] * obs_shape[1] * obs_shape[2] if len(obs_shape) == 3 else obs_shape[0]
        super().__init__(action_dim, obs_dim)

        self.gru_hidden_dim = gru_hidden_dim
        self.fc_dim_size = fc_dim_size
        self.activation = activation

        self.network = CNNRNNActorCriticWithConditionalCritic(
            action_dim=action_dim,
            fc_dim_size=fc_dim_size,
            gru_hidden_dim=gru_hidden_dim,
            activation=activation,
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_action(
        self,
        params,
        obs,
        done,
        avail_actions,
        hstate,
        rng,
        aux_obs=None,
        env_state=None,
        test_mode=False,
    ):
        """Get actions for the CNN+RNN policy with conditional critic.

        Args:
            params: Network parameters
            obs: Observations with shape (seq_len, batch_size, H, W, C)
            done: Done flags with shape (seq_len, batch_size)
            avail_actions: Available actions with shape (seq_len, batch_size, action_dim)
            hstate: Hidden state with shape (1, batch_size, gru_hidden_dim)
            rng: Random key
            aux_obs: Auxiliary observation (partner ID one-hot), shape (seq_len, batch_size, pop_size)
            test_mode: If True, use deterministic action selection

        Returns:
            action: Selected actions
            new_hstate: Updated hidden state
        """
        batch_size = obs.shape[1]

        # Use dummy aux_obs for action selection (critic doesn't affect action)
        if aux_obs is None:
            aux_obs = jnp.zeros(obs.shape[:2] + (self.pop_size,))

        new_hstate, pi, _ = self.network.apply(
            params, hstate.squeeze(0), (obs, done, avail_actions, aux_obs)
        )

        action = jax.lax.cond(
            test_mode, lambda: pi.mode(), lambda: pi.sample(seed=rng)
        )
        return action, new_hstate.reshape(1, batch_size, -1)

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(
        self,
        params,
        obs,
        done,
        avail_actions,
        hstate,
        rng,
        aux_obs=None,
        env_state=None,
    ):
        """Get actions, values, and policy for the CNN+RNN policy with conditional critic.

        Args:
            params: Network parameters
            obs: Observations with shape (seq_len, batch_size, H, W, C)
            done: Done flags with shape (seq_len, batch_size)
            avail_actions: Available actions with shape (seq_len, batch_size, action_dim)
            hstate: Hidden state with shape (1, batch_size, gru_hidden_dim)
            rng: Random key
            aux_obs: Auxiliary observation (partner ID one-hot), shape (seq_len, batch_size, pop_size)

        Returns:
            action: Selected actions
            val: Value estimates
            pi: Action distribution
            new_hstate: Updated hidden state
        """
        batch_size = obs.shape[1]

        # Use provided aux_obs or dummy for value estimation
        if aux_obs is None:
            aux_obs = jnp.zeros(obs.shape[:2] + (self.pop_size,))

        new_hstate, pi, val = self.network.apply(
            params, hstate.squeeze(0), (obs, done, avail_actions, aux_obs)
        )

        action = pi.sample(seed=rng)
        return action, val, pi, new_hstate.reshape(1, batch_size, -1)

    def init_hstate(self, batch_size, aux_info=None):
        """Initialize hidden state for the GRU.

        Args:
            batch_size: Number of parallel environments/actors

        Returns:
            Hidden state with shape (1, batch_size, gru_hidden_dim)
        """
        hstate = ScannedRNN.initialize_carry(batch_size, self.gru_hidden_dim)
        hstate = hstate.reshape(1, batch_size, self.gru_hidden_dim)
        return hstate

    def init_params(self, rng):
        """Initialize network parameters.

        Args:
            rng: Random key for initialization

        Returns:
            Initialized parameters
        """
        batch_size = 1
        seq_len = 1

        # Initialize hidden state
        init_hstate = self.init_hstate(batch_size)

        # Create dummy inputs with shape (seq_len, batch_size, H, W, C)
        dummy_obs = jnp.zeros((seq_len, batch_size) + self.obs_shape)
        dummy_done = jnp.zeros((seq_len, batch_size))
        dummy_avail = jnp.ones((seq_len, batch_size, self.action_dim_int))
        dummy_aux = jnp.zeros((seq_len, batch_size, self.pop_size))
        dummy_x = (dummy_obs, dummy_done, dummy_avail, dummy_aux)

        # Initialize model
        return self.network.init(rng, init_hstate.reshape(batch_size, -1), dummy_x)
