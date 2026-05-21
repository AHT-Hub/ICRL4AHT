"""Teammate Registry module for unified teammate policy construction.

This module provides a unified interface for constructing teammate policies
from compact specifications, supporting both heuristic and RL teammates.

It also provides deterministic theta sampling for reproducible heuristic
teammate configurations across train/test splits.

Example Usage:
    >>> from teammate_wrapper import make_teammate, HeuristicTeammateSpec, RLTeammateSpec
    >>> from envs import make_env
    >>>
    >>> # Create environment
    >>> env = make_env('overcooked-v2', {'layout': 'cramped_room'})
    >>>
    >>> # Create heuristic teammate
    >>> spec = HeuristicTeammateSpec(family="assembly_line", theta="default")
    >>> teammate = make_teammate(spec, env, teammate_agent_id="agent_1")
    >>>
    >>> # Use the teammate
    >>> rng = jax.random.PRNGKey(0)
    >>> carry = teammate.init(rng)
    >>> action, carry = teammate.act(carry, obs, done, rng, env_state=state)
    >>>
    >>> # Theta sampling example
    >>> from teammate_wrapper import ThetaSpec, sample_theta, make_theta_set
    >>> theta_spec = ThetaSpec(family="territory", theta_id=0, split="train", base_seed=0)
    >>> theta = sample_theta(theta_spec)
    >>> train_specs = make_theta_set("territory", "train", n=10, base_seed=0)
"""

from teammate_wrapper.specs import (
    HeuristicTeammateSpec,
    RLTeammateSpec,
    TeammateSpec,
    VALID_HEURISTIC_FAMILIES,
    VALID_RL_ALGOS,
    THETA_PRESETS,
    validate_spec,
)
from teammate_wrapper.registry import make_teammate, TeammatePolicy

# Theta sampling imports
from teammate_wrapper.theta_sampling import (
    ThetaSpec,
    sample_theta,
    theta_to_json,
    theta_from_json,
    make_theta_set,
    make_all_theta_sets,
    assert_train_test_disjoint,
    verify_reproducibility,
    make_teammate_spec,
    make_teammate_from_theta,
    get_all_families,
    get_family_presets,
    get_theta_fields,
    summarize_theta,
    FAMILY_CONFIGS,
)

__all__ = [
    # Specs
    "HeuristicTeammateSpec",
    "RLTeammateSpec",
    "TeammateSpec",
    "VALID_HEURISTIC_FAMILIES",
    "VALID_RL_ALGOS",
    "THETA_PRESETS",
    "validate_spec",
    # Registry
    "make_teammate",
    "TeammatePolicy",
    # Theta Sampling
    "ThetaSpec",
    "sample_theta",
    "theta_to_json",
    "theta_from_json",
    "make_theta_set",
    "make_all_theta_sets",
    "assert_train_test_disjoint",
    "verify_reproducibility",
    "make_teammate_spec",
    "make_teammate_from_theta",
    "get_all_families",
    "get_family_presets",
    "get_theta_fields",
    "summarize_theta",
    "FAMILY_CONFIGS",
]
