"""PPO utility functions for multi-agent reinforcement learning.

This module provides utilities for PPO training:
- Transition: NamedTuple for storing trajectory data
- Batching functions: batchify, batchify_spatial, unbatchify
- Minibatch creation for PPO updates
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def batchify_spatial(x: dict, agent_list, num_actors):
    """Batchify observations while preserving spatial dimensions for CNN.

    Unlike batchify() which flattens to (num_actors, -1), this preserves
    the spatial structure: (num_agents, num_envs, H, W, C) -> (num_actors, H, W, C)

    Args:
        x: Dict mapping agent names to observations of shape (num_envs, H, W, C)
        agent_list: List of agent names
        num_actors: Total number of actors (num_agents * num_envs)

    Returns:
        Batched observations of shape (num_actors, H, W, C)
    """
    x = jnp.stack([x[a] for a in agent_list])  # (num_agents, num_envs, H, W, C)
    # Reshape to (num_actors, H, W, C), preserving spatial dims
    return x.reshape((num_actors,) + x.shape[2:])

def batchify_info(x: dict, agent_list, num_actors):
    '''Handle special case that info has both per-agent and global information'''
    x = jnp.stack([x[a] for a in x if a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents):
    x = x.reshape((num_agents, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def _create_minibatches(traj_batch, advantages, targets, init_hstate, num_actors, num_minibatches, perm_rng, init_done=None):
    """Create minibatches for PPO updates, where each leaf has shape
        (num_minibatches, rollout_len, num_actors / num_minibatches, ...)
    This function ensures that the rollout (time) dimension is kept separate from the minibatch and num_actors
    dimensions, so that the minibatches are compatible with recurrent ActorCritics.

    Args:
        traj_batch: Trajectory batch pytree, obs shape (rollout_len, num_actors, feat_shape)
        advantages: shape (rollout_len, num_actors)
        targets: shape (rollout_len, num_actors)
        init_hstate: Initial hidden state, shape (1, num_actors, hidden_dim)
        num_actors: Total number of actors
        num_minibatches: Number of minibatches to create
        perm_rng: Random key for permutation
        init_done: Initial done signal for correct RNN reset during PPO updates, shape (1, num_actors).
                   If None, uses zeros (backward compatible).
    """
    # Create batch containing trajectory, advantages, targets, and init_done
    # init_done is needed to correctly reset RNN hidden states during PPO updates
    # when ROLLOUT_LENGTH != max_steps (episodes can span/end mid-rollout)
    if init_done is None:
        init_done = jnp.zeros((1, num_actors), dtype=jnp.bool_)

    batch = (
        init_hstate, # shape (1, num_actors, hidden_dim)
        init_done,   # shape (1, num_actors) - initial done for shifted done signal
        traj_batch,  # pytree: obs is shape (rollout_len, num_actors, feat_shape)
        advantages,  # shape (rollout_len, num_actors)
        targets      # shape (rollout_len, num_actors)
    )

    permutation = jax.random.permutation(perm_rng, num_actors)

    # each leaf of shuffled batch has shape (rollout_len, num_actors, feat_shape)
    # except for init_hstate which has shape (1, num_actors, hidden_dim)
    # and init_done which has shape (1, num_actors)
    shuffled_batch = jax.tree.map(
        lambda x: jnp.take(x, permutation, axis=1), batch
    )
    # each leaf has shape (num_minibatches, rollout_len, num_actors/num_minibatches, feat_shape)
    # except for init_hstate which has shape (num_minibatches, 1, num_actors/num_minibatches, hidden_dim)
    # and init_done which has shape (num_minibatches, 1, num_actors/num_minibatches)
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(
            jnp.reshape(
                x,
                [x.shape[0], num_minibatches, -1]
                + list(x.shape[2:]),
        ), 1, 0,),
        shuffled_batch,
    )

    return minibatches