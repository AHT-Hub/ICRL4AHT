"""Teammate specification dataclasses for the Teammate Registry.

This module provides unified specification types for both heuristic and RL teammates,
allowing training/eval code to construct teammate policies from compact specs without
knowing the concrete policy type.

Example Usage:
    >>> from teammate_wrapper.specs import HeuristicTeammateSpec, RLTeammateSpec
    >>> # Heuristic spec with preset theta
    >>> spec = HeuristicTeammateSpec(family="assembly_line", theta="default")
    >>> # Heuristic spec with explicit theta dict
    >>> spec = HeuristicTeammateSpec(
    ...     family="territory",
    ...     theta={"split_mode": 0, "strictness": 0.8}
    ... )
    >>> # RL spec with checkpoint path
    >>> spec = RLTeammateSpec(algo="brdiv", ckpt_path="checkpoints/brdiv_run1")
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Union


@dataclass
class HeuristicTeammateSpec:
    """Specification for heuristic teammate policies.

    Attributes:
        family: The heuristic family name. One of:
            - "assembly_line": Role-based cooperative behavior
            - "territory": Territory-based split behavior
            - "utility_greedy": Utility-based greedy intent selection
            - "recipe_aware_button": Recipe-aware agent with L button
        theta: Hyperparameters, either:
            - A preset name (str): "default", "strict_vertical", etc.
            - A dict of numeric fields matching the theta dataclass
            - The actual theta dataclass instance
        use_log_wrapper: Whether the environment uses LogWrapper (affects state unwrapping)
        start_cooking_interaction: Whether explicit interaction is needed to start cooking
    """
    family: str
    theta: Union[str, Dict[str, Any], Any] = "default"
    use_log_wrapper: bool = True
    start_cooking_interaction: bool = False


@dataclass
class RLTeammateSpec:
    """Specification for RL teammate policies.

    Attributes:
        algo: The RL algorithm name. One of:
            - "fcp": Fictitious Co-Play
            - "brdiv": Best Response Diversity
            - "ippo": Independent PPO
        ckpt_path: Path to the checkpoint directory. For separated checkpoints, this is the
            run directory (e.g., "outputs/ippo_train_run"). For legacy checkpoints, this
            is the direct checkpoint path.
        use_log_wrapper: Whether the environment uses LogWrapper
        extra: Additional configuration options:
            Network architecture:
            - "actor_type": "mlp" | "rnn" | "cnn_rnn" | "s5" (default: "mlp")
            - "activation": Activation function (default: "tanh")
            - "fc_hidden_dim": FC hidden dimension for mlp/rnn (default: 64)
            - "fc_dim_size": FC dimension for cnn_rnn (default: 128)
            - "gru_hidden_dim": GRU hidden dimension for rnn/cnn_rnn (default: 64/128)
            - "use_avail_actions": Whether to use action masking for cnn_rnn (default: True)

            Checkpoint loading (separated format):
            - "use_separated_ckpt": Use separated checkpoint format (default: False)
            - "checkpoint_idx": Checkpoint index to load (default: -1 for last)
            - "population_idx": Population index for FCP/IPPO (default: 0)
            - "seed_idx": Seed index for multi-seed runs (default: 0)
            - "agent_type": For BRDiv, "conf" or "br" (default: "conf")

            Checkpoint loading (legacy format):
            - "ckpt_key": Key for checkpoint loading

            Inference:
            - "test_mode": Whether to use deterministic action selection (default: True)
    """
    algo: str
    ckpt_path: str
    use_log_wrapper: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


# Union type for all teammate specs
TeammateSpec = Union[HeuristicTeammateSpec, RLTeammateSpec]


# Valid heuristic family names
VALID_HEURISTIC_FAMILIES = frozenset({
    "assembly_line",
    "territory",
    "utility_greedy",
    "recipe_aware_button",
})

# Valid RL algorithm names
VALID_RL_ALGOS = frozenset({
    "fcp",
    "brdiv",
    "ippo",
})

# Preset theta names for each heuristic family
THETA_PRESETS = {
    "assembly_line": {
        "default": "default",
        "ingredient_runner": "ingredient_runner",
        "plater_deliverer": "plater_deliverer",
    },
    "territory": {
        "default": "default",
        "strict_vertical": "strict_vertical",
        "strict_horizontal": "strict_horizontal",
        "object_stations": "object_stations",
        "flexible": "flexible",
    },
    "utility_greedy": {
        "default": "default",
        "delivery_focused": "delivery_focused",
        "prep_focused": "prep_focused",
        "high_inertia": "high_inertia",
        "distance_sensitive": "distance_sensitive",
    },
    "recipe_aware_button": {
        "default": "default",
        "high_L_priority": "high_L_priority",
        "lazy_learner": "lazy_learner",
        "strict_adherent": "strict_adherent",
    },
}


def validate_heuristic_spec(spec: HeuristicTeammateSpec) -> None:
    """Validate a heuristic teammate spec.

    Raises:
        ValueError: If the spec has invalid family or theta configuration.
    """
    if spec.family not in VALID_HEURISTIC_FAMILIES:
        raise ValueError(
            f"Unknown heuristic family '{spec.family}'. "
            f"Valid families are: {sorted(VALID_HEURISTIC_FAMILIES)}"
        )

    # If theta is a string preset, validate it exists
    if isinstance(spec.theta, str):
        valid_presets = THETA_PRESETS.get(spec.family, {})
        if spec.theta not in valid_presets:
            raise ValueError(
                f"Unknown theta preset '{spec.theta}' for family '{spec.family}'. "
                f"Valid presets are: {sorted(valid_presets.keys())}"
            )


def validate_rl_spec(spec: RLTeammateSpec) -> None:
    """Validate an RL teammate spec.

    Raises:
        ValueError: If the spec has invalid algorithm or missing required fields.
    """
    if spec.algo not in VALID_RL_ALGOS:
        raise ValueError(
            f"Unknown RL algorithm '{spec.algo}'. "
            f"Valid algorithms are: {sorted(VALID_RL_ALGOS)}"
        )

    if not spec.ckpt_path:
        raise ValueError(
            f"Missing required 'ckpt_path' for RL teammate spec with algo='{spec.algo}'."
        )


def validate_spec(spec: TeammateSpec) -> None:
    """Validate any teammate spec.

    Raises:
        ValueError: If the spec is invalid.
        TypeError: If the spec is not a recognized type.
    """
    if isinstance(spec, HeuristicTeammateSpec):
        validate_heuristic_spec(spec)
    elif isinstance(spec, RLTeammateSpec):
        validate_rl_spec(spec)
    else:
        raise TypeError(
            f"Unknown spec type '{type(spec).__name__}'. "
            f"Expected HeuristicTeammateSpec or RLTeammateSpec."
        )
