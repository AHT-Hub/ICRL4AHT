"""Statistics computation utilities for training metrics.

This module provides JAX-jitted functions for computing statistics
from training metrics, including masked mean/std computation for
episode-level metrics within rollouts.
"""

import os
from functools import partial

import jax
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt

def get_metric_names(env_name):
    if env_name == 'overcooked-v2':
        return ("base_return", "returned_episode_returns")
    else:
        raise ValueError(f"Unknown env name {env_name} for getting metric names.")

@partial(jax.jit, static_argnames=['stats'])
def get_stats(metrics, stats: tuple):
    '''
    Computes mean and std of metrics of interest for each seed and update, 
    using only the final steps of episodes. Note that each rollout contains multiple episodes.

    metrics is a pytree where each leaf has shape 
        (..., rollout_length, num_envs)
    stats is a tuple of strings, each corresponding to a metric of interest in metrics
    '''
    # Get mask for final steps of episodes
    mask = metrics["returned_episode"]
    
    # Initialize output dictionary
    all_stats = {}
    stats = list(stats) # convert to list to correctly iterate if the tuple only has a single element
    for stat_name in stats:
        # Get the metric array
        metric_data = metrics[stat_name]  # Shape: (..., rollout_length, num_envs)

        # Compute means and stds for each seed and update
        # Use masked operations to only consider final episode steps
        means = jnp.where(mask, metric_data, 0).sum(axis=(-2, -1)) / mask.sum(axis=(-2, -1))
        # For std, first compute masked values
        masked_vals = jnp.where(mask, metric_data, 0)
        squared_diff = (masked_vals - means[..., None, None]) ** 2
        variance = jnp.where(mask, squared_diff, 0).sum(axis=(-2, -1)) / mask.sum(axis=(-2, -1))
        stds = jnp.sqrt(variance)
        # Stack means and stds
        all_stats[stat_name] = jnp.stack([means, stds], axis=-1)
    
    return all_stats