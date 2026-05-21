"""Environment factory module for creating benchmark environments.

This module provides the `make_env` factory function that creates environments
with fixed benchmark configurations. Currently supports Overcooked V2 with
standardized settings for reproducible ad-hoc teamwork research.

The benchmark configuration enforces:
- Partial observability (agent_view_size=2)
- 400 max steps per episode
- Deterministic object/inventory state on reset
- Only agent positions are randomized
- Grid-based layered observations
"""

import copy
import numpy as np
from typing import Optional, Tuple, Union


def make_env(env_name: str, env_kwargs: dict = {}):
    if env_name == 'overcooked-v2':
        # =====================================================================
        # Overcooked V2 Benchmark Configuration
        # =====================================================================
        # These defaults are INTENTIONALLY FIXED for reproducible benchmarking.
        # They define a standard evaluation setting for ad-hoc teamwork research:
        #   - max_steps=400: Standard episode length for benchmarks
        #   - agent_view_size=2: Partial observability (2 blocks in each direction)
        #   - observation_type="default": Grid-based layered observations
        #   - op_ingredient_permutations=None: No ingredient permutation
        #   - random_reset=False: Deterministic object/inventory state on reset
        #   - random_agent_positions=True: Only agent positions are randomized
        #   - flatten_obs=False: Keep observations as (H, W, C) arrays
        #   - negative_rewards=True: Penalize incorrect deliveries
        #   - sample_recipe_on_delivery=True: Sample new recipe after delivery
        #   - indicate_successful_delivery: True for test_time layouts, False otherwise
        #
        # Only keys in ALLOWED_OVERRIDE_KEYS can be overridden by the caller.
        # All other keys are silently ignored to prevent accidental deviation.
        # =====================================================================

        from envs.overcooked_v2.overcooked_v2_wrapper import OvercookedV2Wrapper
        from envs.overcooked_v2.layouts import overcooked_v2_layouts

        # Fixed benchmark defaults (enforced, not overridable unless in whitelist)
        BENCHMARK_DEFAULTS = {
            "max_steps": 400,
            "agent_view_size": 2,              # Partial observability (2 blocks each direction)
            "observation_type": "default",     # Grid-based observations
            "op_ingredient_permutations": None,  # No ingredient permutation
            "random_reset": False,             # Deterministic object/inventory state
            "random_agent_positions": True,    # Randomize agent positions only
            "flatten_obs": False,              # Keep observations as (H, W, C) arrays
            "negative_rewards": True,          # Penalize incorrect deliveries
            "sample_recipe_on_delivery": True, # Sample new recipe after delivery
        }

        # Keys the caller is allowed to override from env_kwargs
        # max_steps: maximum steps per episode (default: 400)
        ALLOWED_OVERRIDE_KEYS = {"flatten_obs", "layout", "max_steps"}

        # --- Validate required 'layout' parameter ---
        if "layout" not in env_kwargs:
            raise ValueError(
                "Missing required 'layout' in env_kwargs for 'overcooked-v2'. "
                f"Available layouts: {list(overcooked_v2_layouts.keys())}"
            )

        layout = env_kwargs["layout"]
        if isinstance(layout, str) and layout not in overcooked_v2_layouts:
            raise ValueError(
                f"Unknown layout '{layout}' for 'overcooked-v2'. "
                f"Available layouts: {list(overcooked_v2_layouts.keys())}"
            )

        # --- Warn about unsupported keys ---
        unsupported_keys = set(env_kwargs.keys()) - ALLOWED_OVERRIDE_KEYS
        if unsupported_keys:
            import warnings
            warnings.warn(
                f"[overcooked-v2] Ignoring unsupported env_kwargs keys (benchmark config is fixed): "
                f"{sorted(unsupported_keys)}. Only {sorted(ALLOWED_OVERRIDE_KEYS)} can be overridden.",
                UserWarning
            )

        # --- Build final config: start with benchmark defaults ---
        env_kwargs_final = dict(copy.deepcopy(BENCHMARK_DEFAULTS))

        # Apply allowed overrides from caller
        for key in {"flatten_obs", "layout", "max_steps"}:
            if key in env_kwargs:
                env_kwargs_final[key] = env_kwargs[key]

        # --- Set indicate_successful_delivery for test_time layouts ---
        # For test_time_simple and test_time_wide, enable delivery success indicator
        TEST_TIME_LAYOUTS = {"test_time_simple", "test_time_wide"}
        layout_name = layout if isinstance(layout, str) else None
        env_kwargs_final["indicate_successful_delivery"] = layout_name in TEST_TIME_LAYOUTS

        env = OvercookedV2Wrapper(**env_kwargs_final)

    else:
        raise NotImplementedError(f"Environment {env_name} not implemented in make_env.")
    return env

if __name__ == "__main__":
    print("Overcooked V2 Benchmark (layout=cramped_room)")
    print("    Testing with fixed benchmark configuration...")
    test_layout = 'cramped_room'
    env = make_env('overcooked-v2', {'layout': test_layout})
    print(f"    Env: {env}")
    print(f"    Resolved benchmark config:")
    print(f"      - layout: {test_layout}")
    print(f"      - max_steps: {env.max_steps}")
    print(f"      - agent_view_size: {env.env.agent_view_size} (2 = partial observability)")
    print(f"      - observation_type: {env.env.observation_type}")
    print(f"      - op_ingredient_permutations: {env.env.op_ingredient_permutations}")
    print(f"      - random_reset: {env.env.random_reset}")
    print(f"      - random_agent_positions: {env.env.random_agent_positions}")
    print(f"      - flatten_obs: {env.flatten_obs}")
    print(f"      - negative_rewards: {env.env.negative_rewards}")
    print(f"      - sample_recipe_on_delivery: {env.env.sample_recipe_on_delivery}")
    print(f"      - indicate_successful_delivery: {env.env.indicate_successful_delivery}")