"""Teammate Registry for constructing teammate policies from specs.

This module provides the main factory function `make_teammate` that constructs
a unified TeammatePolicy interface from either heuristic or RL specifications.

The TeammatePolicy interface is:
    - init(rng) -> carry: Initialize the policy carry state
    - act(carry, obs, done, rng, env_state=None, avail_actions=None) -> (new_carry, action)
    - name: Human-readable name including family/algo and theta_id/ckpt basename

All policies are JAX-compatible (work with jax.jit/vmap), or clearly document
any limitations.
"""

import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp

from teammate_wrapper.specs import (
    HeuristicTeammateSpec,
    RLTeammateSpec,
    TeammateSpec,
    THETA_PRESETS,
    VALID_HEURISTIC_FAMILIES,
    validate_spec,
)

# Heuristic agent imports
from agents.overcooked_v2.assembly_line_agent import AssemblyLineTheta
from agents.overcooked_v2.territory_agent import TerritoryTheta
from agents.overcooked_v2.utility_greedy_agent import UtilityGreedyTheta
from agents.overcooked_v2.recipe_aware_button_agent import RecipeAwareButtonTheta

from agents.overcooked_v2.agent_policy_wrappers import (
    OvercookedV2AssemblyLinePolicyWrapper,
    OvercookedV2TerritoryPolicyWrapper,
    OvercookedV2UtilityGreedyPolicyWrapper,
    OvercookedV2RecipeAwareButtonPolicyWrapper,
)


# Theta class mapping by family
THETA_CLASSES = {
    "assembly_line": AssemblyLineTheta,
    "territory": TerritoryTheta,
    "utility_greedy": UtilityGreedyTheta,
    "recipe_aware_button": RecipeAwareButtonTheta,
}

# Policy wrapper classes by family
POLICY_WRAPPERS = {
    "assembly_line": OvercookedV2AssemblyLinePolicyWrapper,
    "territory": OvercookedV2TerritoryPolicyWrapper,
    "utility_greedy": OvercookedV2UtilityGreedyPolicyWrapper,
    "recipe_aware_button": OvercookedV2RecipeAwareButtonPolicyWrapper,
}


@dataclass
class TeammatePolicy:
    """Unified interface for teammate policies.

    This is a dataclass wrapper providing a consistent interface for both
    heuristic and RL teammates.

    Attributes:
        name: Human-readable name (e.g., "assembly_line[default]" or "brdiv[ckpt_run1]")
        init: Callable (rng) -> carry, initializes policy state
        act: Callable (carry, obs, done, rng, env_state, avail_actions) -> (new_carry, action)
        _policy: The underlying policy object (for advanced use)
        _params: Policy parameters (only for RL policies, None for heuristics)
    """
    name: str
    init: Callable[[chex.PRNGKey], Any]
    act: Callable[..., Tuple[Any, jnp.ndarray]]
    _policy: Any = None
    _params: Any = None


def resolve_theta(family: str, theta_input: Union[str, Dict[str, Any], Any]) -> Any:
    """Resolve theta input to a theta dataclass instance.

    Args:
        family: The heuristic family name
        theta_input: Either:
            - A preset name (str): e.g., "default", "strict_vertical"
            - A dict of numeric fields
            - An already-constructed theta dataclass

    Returns:
        The resolved theta dataclass instance

    Raises:
        ValueError: If the theta_input is invalid
    """
    theta_class = THETA_CLASSES.get(family)
    if theta_class is None:
        raise ValueError(
            f"Unknown family '{family}'. Valid families: {sorted(THETA_CLASSES.keys())}"
        )

    # Case 1: Already a theta dataclass instance
    if isinstance(theta_input, theta_class):
        return theta_input

    # Case 2: Preset name string
    if isinstance(theta_input, str):
        presets = THETA_PRESETS.get(family, {})
        if theta_input not in presets:
            raise ValueError(
                f"Unknown theta preset '{theta_input}' for family '{family}'. "
                f"Valid presets: {sorted(presets.keys())}"
            )
        # Get the preset method from the theta class
        preset_method = getattr(theta_class, theta_input, None)
        if preset_method is None:
            raise ValueError(
                f"Theta class '{theta_class.__name__}' does not have method '{theta_input}'"
            )
        return preset_method()

    # Case 3: Dict of fields
    if isinstance(theta_input, dict):
        return _theta_from_dict(family, theta_class, theta_input)

    raise ValueError(
        f"Invalid theta_input type '{type(theta_input).__name__}'. "
        f"Expected str (preset name), dict (field values), or {theta_class.__name__} instance."
    )


def _theta_from_dict(family: str, theta_class: type, theta_dict: Dict[str, Any]) -> Any:
    """Construct theta dataclass from a dictionary of field values.

    Args:
        family: The heuristic family name (for error messages)
        theta_class: The theta dataclass class
        theta_dict: Dictionary mapping field names to values

    Returns:
        The constructed theta dataclass instance

    Raises:
        ValueError: If unknown fields are provided or required fields are missing
    """
    # Get the default theta to know expected fields
    default_theta = theta_class.default()

    # Check for unknown fields
    expected_fields = set(vars(default_theta).keys())
    provided_fields = set(theta_dict.keys())
    unknown_fields = provided_fields - expected_fields

    if unknown_fields:
        raise ValueError(
            f"Unknown theta fields for family '{family}': {sorted(unknown_fields)}. "
            f"Valid fields are: {sorted(expected_fields)}"
        )

    # Build kwargs starting from default, overriding with provided values
    kwargs = {}
    for field_name in expected_fields:
        if field_name in theta_dict:
            # Convert to jnp array with appropriate dtype
            default_val = getattr(default_theta, field_name)
            kwargs[field_name] = jnp.array(theta_dict[field_name], dtype=default_val.dtype)
        else:
            kwargs[field_name] = getattr(default_theta, field_name)

    return theta_class(**kwargs)


def _get_theta_id(theta_input: Union[str, Dict[str, Any], Any]) -> str:
    """Get a human-readable identifier for the theta configuration."""
    if isinstance(theta_input, str):
        return theta_input
    elif isinstance(theta_input, dict):
        # Create a short hash-like identifier from dict
        items = sorted(theta_input.items())
        return f"custom_{hash(tuple(items)) % 10000:04d}"
    else:
        # Dataclass instance - use class name
        return "custom"


def _make_heuristic_teammate(
    spec: HeuristicTeammateSpec,
    env,
    teammate_agent_id: str,
) -> TeammatePolicy:
    """Create a heuristic teammate policy from spec.

    Args:
        spec: The heuristic teammate specification
        env: The environment instance
        teammate_agent_id: The agent ID for this teammate (e.g., "agent_0" or "agent_1")

    Returns:
        TeammatePolicy with unified interface
    """
    # Resolve theta
    theta = resolve_theta(spec.family, spec.theta)

    # Get the policy wrapper class
    wrapper_class = POLICY_WRAPPERS[spec.family]

    # Get layout from env
    layout = env.layout if hasattr(env, 'layout') else env.env.layout

    # Create the policy wrapper
    policy = wrapper_class(
        layout=layout,
        theta=theta,
        using_log_wrapper=spec.use_log_wrapper,
        start_cooking_interaction=spec.start_cooking_interaction,
    )

    # Build the name
    theta_id = _get_theta_id(spec.theta)
    name = f"{spec.family}[{theta_id}]"

    # Create init function
    def init_fn(rng: chex.PRNGKey) -> Any:
        """Initialize the heuristic policy carry state."""
        # Parse agent_id to int (e.g., "agent_0" -> 0)
        if isinstance(teammate_agent_id, str) and teammate_agent_id.startswith("agent_"):
            agent_idx = int(teammate_agent_id.split("_")[1])
        else:
            agent_idx = int(teammate_agent_id)
        return policy.init_hstate(batch_size=1, aux_info={"agent_id": agent_idx})

    # Create act function
    def act_fn(
        carry: Any,
        obs: jnp.ndarray,
        done: jnp.ndarray,
        rng: Optional[chex.PRNGKey] = None,
        env_state: Any = None,
        avail_actions: Optional[jnp.ndarray] = None,
    ) -> Tuple[Any, jnp.ndarray]:
        """Get action from the heuristic policy.

        Args:
            carry: The policy carry state (hstate)
            obs: Observation (shape matching env obs)
            done: Done flag (bool or array)
            rng: Optional JAX random key (unused for most heuristics)
            env_state: Full environment state (required for heuristics)
            avail_actions: Available actions mask (optional)

        Returns:
            Tuple of (new_carry, action)
        """
        # Ensure done is a JAX array
        done = jnp.asarray(done)
        if done.ndim == 0:
            done = done.reshape(1)

        # Get available actions if not provided
        if avail_actions is None:
            avail_actions = jnp.ones((6,), dtype=jnp.float32)

        action, new_carry = policy.get_action(
            params=None,
            obs=obs,
            done=done,
            avail_actions=avail_actions,
            hstate=carry,
            rng=rng,
            env_state=env_state,
        )
        return new_carry, action

    return TeammatePolicy(
        name=name,
        init=init_fn,
        act=act_fn,
        _policy=policy,
        _params=None,
    )


def _make_rl_teammate(
    spec: RLTeammateSpec,
    env,
    teammate_agent_id: str,
) -> TeammatePolicy:
    """Create an RL teammate policy from spec.

    Args:
        spec: The RL teammate specification
        env: The environment instance
        teammate_agent_id: The agent ID for this teammate

    Returns:
        TeammatePolicy with unified interface

    Note:
        If the checkpoint file doesn't exist, this will raise FileNotFoundError.
        Use the dry_run mode in tests to skip actual loading.
    """
    # Import here to avoid circular imports and allow lazy loading
    from teammate_wrapper.rl_wrappers import make_rl_policy_wrapper

    # Get checkpoint basename for name
    ckpt_basename = os.path.basename(spec.ckpt_path.rstrip('/\\'))
    name = f"{spec.algo}[{ckpt_basename}]"

    # Create the RL policy wrapper
    policy, params = make_rl_policy_wrapper(
        algo=spec.algo,
        ckpt_path=spec.ckpt_path,
        env=env,
        use_log_wrapper=spec.use_log_wrapper,
        extra=spec.extra,
    )

    # Parse agent_id to int
    if isinstance(teammate_agent_id, str) and teammate_agent_id.startswith("agent_"):
        agent_idx = int(teammate_agent_id.split("_")[1])
    else:
        agent_idx = int(teammate_agent_id)

    # Get extra config
    test_mode = spec.extra.get("test_mode", True)

    # Create init function
    def init_fn(rng: chex.PRNGKey) -> Any:
        """Initialize the RL policy carry state (hidden state if RNN)."""
        return policy.init_hstate(batch_size=1, aux_info={"agent_id": agent_idx})

    # Create act function
    def act_fn(
        carry: Any,
        obs: jnp.ndarray,
        done: jnp.ndarray,
        rng: Optional[chex.PRNGKey] = None,
        env_state: Any = None,
        avail_actions: Optional[jnp.ndarray] = None,
    ) -> Tuple[Any, jnp.ndarray]:
        """Get action from the RL policy.

        Args:
            carry: The policy carry state (hidden state for RNN, None for MLP)
            obs: Observation
            done: Done flag
            rng: JAX random key for stochastic action selection
            env_state: Environment state (unused for RL policies)
            avail_actions: Available actions mask

        Returns:
            Tuple of (new_carry, action)
        """
        # Ensure done is a JAX array
        done = jnp.asarray(done)
        if done.ndim == 0:
            done = done.reshape(1)

        # Get available actions if not provided
        if avail_actions is None:
            avail_actions = jnp.ones((6,), dtype=jnp.float32)

        # Check if this is a CNN+RNN policy (has obs_shape attribute)
        is_cnn_rnn = hasattr(policy, 'obs_shape')

        # For RNN/CNN+RNN policies, need to handle sequence dimension
        if hasattr(policy, 'gru_hidden_dim'):
            if is_cnn_rnn:
                # CNN+RNN policy expects (seq_len, batch, H, W, C) shape
                # Add sequence and batch dimensions
                obs_seq = obs.reshape(1, 1, *obs.shape)  # (1, 1, H, W, C)
            else:
                # Standard RNN policy expects (seq_len, batch, obs_dim) shape
                obs_seq = obs.reshape(1, 1, -1)  # (1, 1, obs_dim)

            done_seq = done.reshape(1, 1)  # (1, 1)
            avail_seq = avail_actions.reshape(1, 1, -1)  # (1, 1, action_dim)

            action, new_carry = policy.get_action(
                params=params,
                obs=obs_seq,
                done=done_seq,
                avail_actions=avail_seq,
                hstate=carry,
                rng=rng,
                test_mode=test_mode,
            )
            # Remove sequence dimension from action
            action = action.squeeze(0)
        else:
            # MLP/S5 policy
            action, new_carry = policy.get_action(
                params=params,
                obs=obs,
                done=done,
                avail_actions=avail_actions,
                hstate=carry,
                rng=rng,
                test_mode=test_mode,
            )

        # Reset carry on done (for RNN policies with hidden state)
        if carry is not None and hasattr(policy, 'init_hstate'):
            new_carry = jax.lax.cond(
                done.squeeze().astype(bool),
                lambda: policy.init_hstate(batch_size=1, aux_info={"agent_id": agent_idx}),
                lambda: new_carry,
            )

        return new_carry, action

    return TeammatePolicy(
        name=name,
        init=init_fn,
        act=act_fn,
        _policy=policy,
        _params=params,
    )


def make_teammate(
    spec: TeammateSpec,
    env,
    teammate_agent_id: str,
) -> TeammatePolicy:
    """Create a teammate policy from a specification.

    This is the main factory function for the teammate registry. It takes a
    specification (heuristic or RL) and returns a unified TeammatePolicy interface.

    Args:
        spec: TeammateSpec (HeuristicTeammateSpec or RLTeammateSpec)
        env: The environment instance (used for layout and obs/action dims)
        teammate_agent_id: The agent ID string (e.g., "agent_0", "agent_1")

    Returns:
        TeammatePolicy with:
            - name: Human-readable identifier
            - init(rng) -> carry: Initialize policy state
            - act(carry, obs, done, rng, env_state, avail_actions) -> (new_carry, action)

    Raises:
        ValueError: If the spec is invalid
        TypeError: If the spec type is unknown
        FileNotFoundError: If an RL checkpoint doesn't exist

    Example:
        >>> from teammate_wrapper import make_teammate, HeuristicTeammateSpec
        >>> spec = HeuristicTeammateSpec(family="assembly_line", theta="default")
        >>> teammate = make_teammate(spec, env, "agent_1")
        >>> carry = teammate.init(jax.random.PRNGKey(0))
        >>> new_carry, action = teammate.act(carry, obs, done, rng, env_state=state)
    """
    # Validate spec first
    validate_spec(spec)

    if isinstance(spec, HeuristicTeammateSpec):
        return _make_heuristic_teammate(spec, env, teammate_agent_id)
    elif isinstance(spec, RLTeammateSpec):
        return _make_rl_teammate(spec, env, teammate_agent_id)
    else:
        raise TypeError(
            f"Unknown spec type '{type(spec).__name__}'. "
            f"Expected HeuristicTeammateSpec or RLTeammateSpec."
        )


class LoggingTeammateWrapper:
    """Wrapper that adds logging/debugging to a TeammatePolicy.

    This wrapper optionally:
    - Logs action distribution statistics (counts)
    - Asserts actions are within the environment action space

    Attributes:
        teammate: The underlying TeammatePolicy
        action_counts: Array tracking action frequency
        total_actions: Total number of actions taken
        validate_actions: Whether to validate action bounds
        action_dim: Size of action space (default 6 for Overcooked)
    """

    def __init__(
        self,
        teammate: TeammatePolicy,
        validate_actions: bool = True,
        action_dim: int = 6,
    ):
        """Initialize the logging wrapper.

        Args:
            teammate: The TeammatePolicy to wrap
            validate_actions: Whether to validate actions are in bounds
            action_dim: Size of the discrete action space
        """
        self.teammate = teammate
        self.validate_actions = validate_actions
        self.action_dim = action_dim
        self.reset_stats()

    @property
    def name(self) -> str:
        return f"Logged[{self.teammate.name}]"

    def reset_stats(self):
        """Reset action statistics."""
        self.action_counts = jnp.zeros(self.action_dim, dtype=jnp.int32)
        self.total_actions = 0

    def init(self, rng: chex.PRNGKey) -> Any:
        """Initialize the policy state."""
        return self.teammate.init(rng)

    def act(
        self,
        carry: Any,
        obs: jnp.ndarray,
        done: jnp.ndarray,
        rng: Optional[chex.PRNGKey] = None,
        env_state: Any = None,
        avail_actions: Optional[jnp.ndarray] = None,
    ) -> Tuple[Any, jnp.ndarray]:
        """Get action and update statistics."""
        new_carry, action = self.teammate.act(
            carry, obs, done, rng, env_state, avail_actions
        )

        # Convert to numpy for stats (this breaks jit but is for debug only)
        action_np = int(action)

        # Validate action
        if self.validate_actions:
            if action_np < 0 or action_np >= self.action_dim:
                raise ValueError(
                    f"Action {action_np} out of bounds [0, {self.action_dim})"
                )

        # Update stats (non-jittable)
        self.action_counts = self.action_counts.at[action_np].add(1)
        self.total_actions += 1

        return new_carry, action

    def get_stats(self) -> Dict[str, Any]:
        """Get action statistics.

        Returns:
            Dict with:
                - action_counts: Array of counts per action
                - total_actions: Total actions taken
                - action_distribution: Normalized distribution
        """
        dist = self.action_counts / max(self.total_actions, 1)
        return {
            "action_counts": self.action_counts,
            "total_actions": self.total_actions,
            "action_distribution": dist,
        }


def wrap_with_logging(
    teammate: TeammatePolicy,
    validate_actions: bool = True,
    action_dim: int = 6,
) -> LoggingTeammateWrapper:
    """Wrap a teammate with logging/debugging capabilities.

    Note: The logging wrapper is NOT jit-compatible as it tracks mutable state.
    Use this only for debugging/evaluation, not training.

    Args:
        teammate: TeammatePolicy to wrap
        validate_actions: Whether to validate action bounds
        action_dim: Size of discrete action space

    Returns:
        LoggingTeammateWrapper with same interface as TeammatePolicy
    """
    return LoggingTeammateWrapper(
        teammate=teammate,
        validate_actions=validate_actions,
        action_dim=action_dim,
    )
