"""Wrapper for the OvercookedV2 environment to ensure compatibility with the benchmark framework.

This wrapper provides a standard interface for the JaxMARL OvercookedV2 environment,
including:
- Flattened observations
- Base return tracking (unshaped rewards)
- Available actions tracking
- Standard step/reset API compatible with BaseEnv
"""

from functools import partial
from typing import Dict, Tuple, Optional, Union, List

import chex
import jax
import jax.numpy as jnp
from envs.overcooked_v2 import spaces
from envs.overcooked_v2.overcooked import OvercookedV2
from envs.overcooked_v2.layouts import Layout, overcooked_v2_layouts

from envs.base_env import BaseEnv, WrappedEnvState


class OvercookedV2Wrapper(BaseEnv):
    """Wrapper for the OvercookedV2 environment to ensure compatibility with the benchmark framework.

    Main features:
    - Flattened observations (grid-based observations are flattened to 1D vectors)
    - Base return tracking (accumulates unshaped rewards for evaluation)
    - Available actions tracking
    - Standard BaseEnv interface

    Args:
        layout: Either a layout name (str) or a Layout object. If a string, it must be
               a valid key in overcooked_v2_layouts.
        max_steps: Maximum number of steps per episode.
        random_reset: Whether to randomize agent positions, inventories, and pot states on reset.
        random_agent_positions: Whether to only randomize agent positions (not inventories/pots).
        agent_view_size: Size of the agent's partial observation. None for full observability.
        start_cooking_interaction: If True, requires explicit interaction to start cooking.
        negative_rewards: If True, penalize incorrect deliveries.
        sample_recipe_on_delivery: If True, sample a new recipe after each successful delivery.
        indicate_successful_delivery: If True, include delivery success in observations.
        op_ingredient_permutations: List of ingredient indices to permute in observations.
        initial_state_buffer: Optional buffer of initial states to sample from on reset.
        force_path_planning: Force path planning even if not using featurized observations.
        observation_type: Type of observation to use (default, featurized, or list per agent).
        flatten_obs: Whether to flatten observations (default True for compatibility).
        **kwargs: Additional keyword arguments passed to OvercookedV2.
    """

    def __init__(
        self,
        layout: Union[str, Layout] = "cramped_room",
        max_steps: int = 400,
        random_reset: bool = False,
        random_agent_positions: bool = False,
        agent_view_size: Optional[int] = None,
        start_cooking_interaction: bool = False,
        negative_rewards: bool = False,
        sample_recipe_on_delivery: bool = False,
        indicate_successful_delivery: bool = False,
        op_ingredient_permutations: Optional[List[int]] = None,
        initial_state_buffer = None,
        force_path_planning: bool = False,
        observation_type = "default",
        flatten_obs: bool = True,
        **kwargs
    ):
        # Convert string layout to Layout object if needed
        if isinstance(layout, str):
            if layout not in overcooked_v2_layouts:
                raise ValueError(
                    f"Invalid layout: {layout}. Available layouts: {list(overcooked_v2_layouts.keys())}"
                )
            self._layout_name = layout
        else:
            self._layout_name = "custom"

        # Initialize the underlying OvercookedV2 environment
        self.env = OvercookedV2(
            layout=layout,
            max_steps=max_steps,
            random_reset=random_reset,
            random_agent_positions=random_agent_positions,
            agent_view_size=agent_view_size,
            start_cooking_interaction=start_cooking_interaction,
            negative_rewards=negative_rewards,
            sample_recipe_on_delivery=sample_recipe_on_delivery,
            indicate_successful_delivery=indicate_successful_delivery,
            op_ingredient_permutations=op_ingredient_permutations,
            initial_state_buffer=initial_state_buffer,
            force_path_planning=force_path_planning,
            observation_type=observation_type,
            **kwargs
        )

        self.agents = self.env.agents
        self.num_agents = len(self.agents)
        self.flatten_obs = flatten_obs

        # Store observation/action spaces
        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.agents}

        # Expose underlying environment properties
        self.agent_view_size = self.env.agent_view_size
        self.max_steps = self.env.max_steps
        self.height = self.env.height
        self.width = self.env.width
        self.layout = self.env.layout
        self.possible_recipes = self.env.possible_recipes

    def _get_flat_obs_shape(self) -> Tuple[int, ...]:
        """Compute the flattened observation shape."""
        obs_shape = self.env.obs_shape
        if isinstance(obs_shape, list):
            # Multiple observation types per agent - use the first one
            obs_shape = obs_shape[0]

        if len(obs_shape) == 1:
            # Already 1D (e.g., featurized observations)
            return obs_shape
        else:
            # Multi-dimensional (e.g., grid observations) - flatten
            flat_size = 1
            for dim in obs_shape:
                flat_size *= dim
            return (flat_size,)

    def observation_space(self, agent: str) -> spaces.Box:
        """Returns the (flattened) observation space.

        Args:
            agent: Agent identifier (e.g., "agent_0").

        Returns:
            Box space describing the observation dimensions.
        """
        if self.flatten_obs:
            flat_obs_shape = self._get_flat_obs_shape()
            return spaces.Box(0, 255, flat_obs_shape)
        else:
            obs_shape = self.env.obs_shape
            if isinstance(obs_shape, list):
                # Per-agent observation shapes
                agent_idx = int(agent.split("_")[1])
                return spaces.Box(0, 255, obs_shape[agent_idx])
            return spaces.Box(0, 255, obs_shape)

    def action_space(self, agent: str) -> spaces.Discrete:
        """Returns the action space for an agent.

        Args:
            agent: Agent identifier (e.g., "agent_0").

        Returns:
            Discrete space with 6 actions (right, down, left, up, stay, interact).
        """
        return self.env.action_space(agent)

    def _flatten_obs(self, obs: Dict[str, chex.Array]) -> Dict[str, chex.Array]:
        """Flatten observations if enabled."""
        if self.flatten_obs:
            return {agent: obs[agent].flatten() for agent in self.agents}
        return obs

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        """Reset the environment to an initial state.

        Args:
            key: JAX PRNG key for randomization.

        Returns:
            Tuple of (observations, wrapped_state) where:
            - observations: Dict mapping agent IDs to observation arrays
            - wrapped_state: WrappedEnvState containing the environment state
        """
        obs, env_state = self.env.reset(key)
        flat_obs = self._flatten_obs(obs)

        wrapped_state = WrappedEnvState(
            env_state=env_state,
            base_return_so_far=jnp.zeros(self.num_agents),
            avail_actions=jnp.ones((self.num_agents, self.env.num_actions)),
            step=jnp.array(0, dtype=jnp.int32)
        )

        return flat_obs, wrapped_state

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WrappedEnvState) -> Dict[str, jnp.ndarray]:
        """Returns the available actions for each agent.

        In OvercookedV2, all actions are always available.

        Args:
            state: Current wrapped environment state.

        Returns:
            Dict mapping agent IDs to action availability masks (all ones).
        """
        num_actions = self.env.num_actions
        return {agent: jnp.ones(num_actions) for agent in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.ndarray:
        """Returns the current step count in the episode.

        Args:
            state: Current wrapped environment state.

        Returns:
            Current time step as an integer array.
        """
        return state.env_state.time

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: WrappedEnvState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[WrappedEnvState] = None,
    ) -> Tuple[Dict[str, chex.Array], WrappedEnvState, Dict[str, float], Dict[str, bool], Dict]:
        """Execute one environment step.

        This method:
        1. Takes actions for all agents
        2. Computes rewards and observations
        3. Tracks base (unshaped) returns for evaluation
        4. Handles automatic episode reset

        Args:
            key: JAX PRNG key for randomization.
            state: Current wrapped environment state.
            actions: Dict mapping agent IDs to action indices.
            reset_state: Optional state to reset to (for curriculum learning).

        Returns:
            Tuple of (observations, new_state, rewards, dones, infos) where:
            - observations: Dict mapping agent IDs to observation arrays
            - new_state: New WrappedEnvState after the step
            - rewards: Dict mapping agent IDs to reward values
            - dones: Dict mapping agent IDs to done flags (includes "__all__")
            - infos: Dict with additional info including base_return and shaped_reward
        """
        # Execute step on the underlying environment
        obs, env_state, rewards, dones, infos = self.env.step(
            key, state.env_state, actions,
            reset_state.env_state if reset_state is not None else None
        )

        flat_obs = self._flatten_obs(obs)

        # Get base rewards for evaluation
        # Note: rewards from step_env is already the base reward (delivery + indicator cost only)
        # The shaped_reward in infos is a separate component (placement, pickup bonuses, etc.)
        base_reward = jnp.array([rewards[agent] for agent in self.agents])

        # Convert shaped_reward dict to array for IPPO compatibility
        # IPPO expects all info fields to have shape (num_agents,) for proper batching
        shaped_rewards_dict = infos.get('shaped_reward', {agent: 0.0 for agent in self.agents})
        shaped_reward_array = jnp.array([
            shaped_rewards_dict.get(agent, 0.0) for agent in self.agents
        ])

        # Accumulate base return
        base_return_so_far = base_reward + state.base_return_so_far

        # Create updated info dictionary
        # Replace shaped_reward dict with array to ensure consistent shapes for IPPO
        new_info = {
            k: v for k, v in infos.items() if k != 'shaped_reward'
        }
        new_info['shaped_reward'] = shaped_reward_array
        new_info['base_reward'] = base_reward
        new_info['base_return'] = base_return_so_far

        # Reset base return accumulator on episode termination
        base_return_so_far_reset = jax.lax.select(
            dones['__all__'],
            jnp.zeros(self.num_agents),
            base_return_so_far
        )

        # Update step counter
        new_step = jax.lax.select(
            dones['__all__'],
            jnp.array(0, dtype=jnp.int32),
            state.step + 1
        )

        new_state = WrappedEnvState(
            env_state=env_state,
            base_return_so_far=base_return_so_far_reset,
            avail_actions=jnp.ones((self.num_agents, self.env.num_actions)),
            step=new_step
        )

        return flat_obs, new_state, rewards, dones, new_info

    # Additional utility methods

    @property
    def name(self) -> str:
        """Environment name."""
        return f"OvercookedV2-{self._layout_name}"

    def get_obs(self, state: WrappedEnvState) -> Dict[str, chex.Array]:
        """Get observations from the current state.

        Args:
            state: Current wrapped environment state.

        Returns:
            Dict mapping agent IDs to observation arrays.
        """
        obs = self.env.get_obs(state.env_state)
        return self._flatten_obs(obs)

    def is_terminal(self, state: WrappedEnvState) -> bool:
        """Check if the episode has terminated.

        Args:
            state: Current wrapped environment state.

        Returns:
            True if the episode is done, False otherwise.
        """
        return self.env.is_terminal(state.env_state)
