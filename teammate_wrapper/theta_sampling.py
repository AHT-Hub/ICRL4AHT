"""Theta Sampling Module for Reproducible Heuristic Teammate Parameters.

This module provides deterministic, reproducible sampling of hyperparameters ("theta")
for the 4 heuristic teammate policy families. It supports train/test splits to ensure
disjoint parameter sets for Protocol-Generalization evaluation.

The module defines:
- ThetaSpec: A frozen dataclass representing a theta configuration
- Parameter spaces and presets for each family
- Sampling functions using stable hashing with JAX random
- JSON serialization for manifest generation
- Integration with the teammate registry

Example Usage:
    >>> from teammate_wrapper.theta_sampling import (
    ...     ThetaSpec, sample_theta, make_theta_set, theta_to_json, make_teammate_from_theta
    ... )
    >>>
    >>> # Sample a specific theta
    >>> spec = ThetaSpec(family="assembly_line", theta_id=0, split="train", base_seed=0)
    >>> theta = sample_theta(spec)
    >>>
    >>> # Generate train/test sets
    >>> train_specs = make_theta_set("territory", split="train", n=10, base_seed=0)
    >>> test_specs = make_theta_set("territory", split="test", n=5, base_seed=0)
    >>>
    >>> # Create teammate from theta
    >>> teammate_policy = make_teammate_from_theta(
    ...     layout_name="cramped_room", theta_spec=spec, env=env
    ... )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json

import jax
import jax.numpy as jnp


# Import theta classes
from agents.overcooked_v2.assembly_line_agent import (
    AssemblyLineTheta,
    ROLE_INGREDIENT_RUNNER,
    ROLE_PLATER_DELIVERER,
    ROLE_FLEX,
    HANDOFF_POT_ADJACENT,
    HANDOFF_CENTRAL,
    HANDOFF_TEAMMATE_NEARBY,
)
from agents.overcooked_v2.territory_agent import (
    TerritoryTheta,
    SPLIT_VERTICAL,
    SPLIT_HORIZONTAL,
    SPLIT_OBJECT_STATIONS,
)
from agents.overcooked_v2.utility_greedy_agent import UtilityGreedyTheta
from agents.overcooked_v2.recipe_aware_button_agent import RecipeAwareButtonTheta

from teammate_wrapper.specs import HeuristicTeammateSpec, THETA_PRESETS


# =============================================================================
# ThetaSpec Dataclass
# =============================================================================

@dataclass(frozen=True)
class ThetaSpec:
    """Specification for a sampled theta configuration.

    This is a frozen dataclass that uniquely identifies a theta configuration
    through deterministic sampling based on (family, theta_id, split, base_seed).

    Attributes:
        family: The heuristic family name (e.g., "assembly_line", "territory")
        theta_id: Integer ID within the split (enumeration index)
        split: Either "train" or "test" - determines parameter ranges
        base_seed: Benchmark-wide constant seed for reproducibility
        preset: Optional preset name to use as base (before jitter)
        overrides: Optional explicit field overrides
    """
    family: str
    theta_id: int
    split: str  # "train" or "test"
    base_seed: int = 0
    preset: Optional[str] = None
    overrides: Optional[Dict[str, Any]] = field(default=None, hash=False)

    def __post_init__(self):
        """Validate the spec."""
        if self.family not in FAMILY_CONFIGS:
            raise ValueError(
                f"Unknown family '{self.family}'. "
                f"Valid families: {sorted(FAMILY_CONFIGS.keys())}"
            )
        if self.split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{self.split}'")
        if self.theta_id < 0:
            raise ValueError(f"theta_id must be non-negative, got {self.theta_id}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict for serialization."""
        return {
            "family": self.family,
            "theta_id": self.theta_id,
            "split": self.split,
            "base_seed": self.base_seed,
            "preset": self.preset,
            "overrides": self.overrides,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ThetaSpec":
        """Create from a plain dict."""
        return cls(
            family=d["family"],
            theta_id=d["theta_id"],
            split=d["split"],
            base_seed=d.get("base_seed", 0),
            preset=d.get("preset"),
            overrides=d.get("overrides"),
        )


# =============================================================================
# Parameter Space Configurations
# =============================================================================

# Each family config defines:
# - theta_class: The theta dataclass class
# - train_presets: Presets used for training split (base configurations)
# - test_presets: Presets used for test split (disjoint from train)
# - continuous_fields: Dict of {field_name: (min, max)} for continuous params
# - discrete_fields: Dict of {field_name: allowed_values} for discrete params
# - train_ranges: Override ranges for train split (for disjoint sampling)
# - test_ranges: Override ranges for test split (for disjoint sampling)

FAMILY_CONFIGS = {
    "assembly_line": {
        "theta_class": AssemblyLineTheta,
        # Train gets FLEX and INGREDIENT_RUNNER roles
        "train_presets": ["default", "ingredient_runner"],
        # Test gets PLATER_DELIVERER role
        "test_presets": ["plater_deliverer"],
        "continuous_fields": {
            "plate_urgency": (0.0, 1.0),
            "prestage_bias": (0.0, 1.0),
            "start_cook_bias": (0.0, 1.0),
        },
        "discrete_fields": {
            "role_mode": [ROLE_INGREDIENT_RUNNER, ROLE_PLATER_DELIVERER, ROLE_FLEX],
            "handoff_style": [HANDOFF_POT_ADJACENT, HANDOFF_CENTRAL, HANDOFF_TEAMMATE_NEARBY],
        },
        # Train: use roles 0 (INGREDIENT_RUNNER) and 2 (FLEX)
        "train_ranges": {
            "role_mode": [ROLE_INGREDIENT_RUNNER, ROLE_FLEX],
            "plate_urgency": (0.0, 0.6),  # Lower urgency
            "prestage_bias": (0.2, 0.6),
        },
        # Test: use role 1 (PLATER_DELIVERER) and different continuous ranges
        "test_ranges": {
            "role_mode": [ROLE_PLATER_DELIVERER],
            "plate_urgency": (0.6, 1.0),  # Higher urgency
            "prestage_bias": (0.0, 0.3),
        },
    },

    "territory": {
        "theta_class": TerritoryTheta,
        # Train and test use low strictness since many layouts have split regions
        "train_presets": ["default", "flexible"],
        "test_presets": ["flexible", "object_stations"],
        "continuous_fields": {
            "strictness": (0.0, 1.0),
            "rescue_threshold": (0.0, 1.0),
            "yield_bias": (0.0, 1.0),
        },
        "discrete_fields": {
            "split_mode": [SPLIT_VERTICAL, SPLIT_HORIZONTAL, SPLIT_OBJECT_STATIONS],
            "shared_margin": [0, 1, 2, 3],
        },
        # Train: Very low strictness to work on split layouts
        "train_ranges": {
            "split_mode": [SPLIT_VERTICAL, SPLIT_HORIZONTAL],
            "strictness": (0.0, 0.3),  # Very low - essentially disabled
            "shared_margin": [1, 2, 3],
            "rescue_threshold": (0.2, 0.4),
        },
        # Test: Also low strictness
        "test_ranges": {
            "split_mode": [SPLIT_HORIZONTAL, SPLIT_OBJECT_STATIONS],
            "strictness": (0.0, 0.3),  # Very low
            "shared_margin": [1, 2, 3],
            "rescue_threshold": (0.2, 0.4),
        },
    },

    "utility_greedy": {
        "theta_class": UtilityGreedyTheta,
        "train_presets": ["default", "delivery_focused"],
        "test_presets": ["prep_focused", "high_inertia", "distance_sensitive"],
        "continuous_fields": {
            "w_deliver": (5.0, 20.0),
            "w_pickup_cooked": (3.0, 15.0),
            "w_get_plate": (2.0, 10.0),
            "w_add_ingredient": (3.0, 12.0),
            "w_fetch_ingredient": (2.0, 10.0),
            "w_stage_on_counter": (0.5, 5.0),
            "w_start_cooking": (3.0, 15.0),
            "w_press_L": (1.0, 8.0),
            "dist_weight": (0.1, 1.5),
            "inertia": (0.0, 1.0),
            "counter_preference": (0.0, 1.0),
        },
        "discrete_fields": {},
        # Train: bias toward delivery/plating weights being higher
        "train_ranges": {
            "w_deliver": (8.0, 18.0),
            "w_pickup_cooked": (6.0, 14.0),
            "w_add_ingredient": (3.0, 8.0),
            "dist_weight": (0.2, 0.7),
            "inertia": (0.1, 0.5),
        },
        # Test: bias toward prep weights being higher
        "test_ranges": {
            "w_deliver": (5.0, 12.0),
            "w_pickup_cooked": (3.0, 8.0),
            "w_add_ingredient": (7.0, 12.0),
            "dist_weight": (0.6, 1.2),
            "inertia": (0.5, 0.9),
        },
    },

    "recipe_aware_button": {
        "theta_class": RecipeAwareButtonTheta,
        "train_presets": ["default", "high_L_priority"],
        "test_presets": ["lazy_learner", "strict_adherent"],
        "continuous_fields": {
            "press_L_when_unknown": (0.0, 1.0),
            "cost_sensitivity": (0.0, 1.0),
            "strict_recipe": (0.0, 1.0),
            "exploration_bias": (0.0, 1.0),
            "plate_timing": (0.0, 1.0),
            "dist_weight": (0.1, 1.0),
            "inertia": (0.0, 1.0),
        },
        "discrete_fields": {
            "refresh_interval": [50, 80, 100, 150, 200],
        },
        # Train: High L priority modes
        "train_ranges": {
            "press_L_when_unknown": (0.6, 1.0),
            "cost_sensitivity": (0.0, 0.4),
            "strict_recipe": (0.7, 1.0),
            "refresh_interval": [50, 80, 100],
        },
        # Test: Low L priority / lazy modes
        "test_ranges": {
            "press_L_when_unknown": (0.2, 0.6),
            "cost_sensitivity": (0.4, 0.9),
            "strict_recipe": (0.2, 0.6),
            "refresh_interval": [100, 150, 200],
        },
    },
}


# =============================================================================
# Deterministic Seed Generation
# =============================================================================

def _stable_hash(family: str, theta_id: int, split: str, base_seed: int) -> int:
    """Generate a stable hash for deterministic seeding.

    Uses SHA256 for platform-independent hashing.

    Args:
        family: Family name
        theta_id: Theta ID within split
        split: "train" or "test"
        base_seed: Base seed

    Returns:
        A 32-bit integer seed
    """
    # Create deterministic string representation
    key_str = f"{family}:{theta_id}:{split}:{base_seed}"
    # Hash with SHA256
    h = hashlib.sha256(key_str.encode('utf-8')).hexdigest()
    # Take first 8 hex chars (32 bits)
    return int(h[:8], 16)


def _make_prng_key(spec: ThetaSpec) -> jax.Array:
    """Create a JAX PRNG key from a ThetaSpec."""
    seed = _stable_hash(spec.family, spec.theta_id, spec.split, spec.base_seed)
    return jax.random.PRNGKey(seed)


# =============================================================================
# Sampling Functions
# =============================================================================

def _sample_continuous(rng: jax.Array, low: float, high: float) -> float:
    """Sample a continuous value uniformly in [low, high]."""
    return float(jax.random.uniform(rng, minval=low, maxval=high))


def _sample_discrete(rng: jax.Array, values: List[Any]) -> Any:
    """Sample a discrete value uniformly from a list."""
    idx = int(jax.random.randint(rng, shape=(), minval=0, maxval=len(values)))
    return values[idx]


def _get_preset_theta(family: str, preset: str) -> Any:
    """Get a theta instance from a preset name."""
    theta_class = FAMILY_CONFIGS[family]["theta_class"]
    preset_method = getattr(theta_class, preset, None)
    if preset_method is None:
        raise ValueError(f"Unknown preset '{preset}' for family '{family}'")
    return preset_method()


def sample_theta(spec: ThetaSpec) -> Any:
    """Sample a theta configuration from a ThetaSpec.

    The sampling process is:
    1. If spec.preset is set, start from that preset
    2. Otherwise, pick a random preset from the split's preset list
    3. Apply jitter to continuous fields based on split-specific ranges
    4. Override discrete fields based on split-specific allowed values
    5. Apply any explicit overrides from spec.overrides

    Args:
        spec: The ThetaSpec defining what to sample

    Returns:
        A theta dataclass instance (e.g., AssemblyLineTheta, TerritoryTheta, etc.)
    """
    config = FAMILY_CONFIGS[spec.family]
    theta_class = config["theta_class"]

    # Get PRNG key
    rng = _make_prng_key(spec)

    # Determine base preset
    if spec.preset is not None:
        base_preset = spec.preset
    else:
        # Choose from split-appropriate presets
        presets = config.get(f"{spec.split}_presets", config.get("train_presets", ["default"]))
        rng, preset_rng = jax.random.split(rng)
        base_preset = _sample_discrete(preset_rng, presets)

    # Start from preset
    base_theta = _get_preset_theta(spec.family, base_preset)

    # Get split-specific ranges
    split_ranges = config.get(f"{spec.split}_ranges", {})
    continuous_fields = config.get("continuous_fields", {})
    discrete_fields = config.get("discrete_fields", {})

    # Build kwargs from base theta, applying jitter/sampling
    kwargs = {}

    for field_name in vars(base_theta).keys():
        base_val = getattr(base_theta, field_name)

        # Generate a field-specific RNG key
        rng, field_rng = jax.random.split(rng)

        if field_name in split_ranges:
            # Use split-specific range
            range_val = split_ranges[field_name]
            if isinstance(range_val, tuple) and len(range_val) == 2:
                # Continuous range: (low, high)
                val = _sample_continuous(field_rng, range_val[0], range_val[1])
                kwargs[field_name] = jnp.array(val, dtype=base_val.dtype)
            elif isinstance(range_val, list):
                # Discrete values list
                val = _sample_discrete(field_rng, range_val)
                kwargs[field_name] = jnp.array(val, dtype=base_val.dtype)
            else:
                kwargs[field_name] = base_val
        elif field_name in continuous_fields:
            # Apply jitter within full continuous range
            low, high = continuous_fields[field_name]
            # Jitter around base value: base +/- 20% of range
            base_float = float(base_val)
            range_size = (high - low) * 0.2
            jitter_low = max(low, base_float - range_size)
            jitter_high = min(high, base_float + range_size)
            val = _sample_continuous(field_rng, jitter_low, jitter_high)
            kwargs[field_name] = jnp.array(val, dtype=base_val.dtype)
        elif field_name in discrete_fields:
            # Use base value (no jitter for discrete unless in split_ranges)
            kwargs[field_name] = base_val
        else:
            # Unknown field, keep base value
            kwargs[field_name] = base_val

    # Apply explicit overrides
    if spec.overrides:
        for field_name, val in spec.overrides.items():
            if hasattr(base_theta, field_name):
                base_val = getattr(base_theta, field_name)
                kwargs[field_name] = jnp.array(val, dtype=base_val.dtype)

    return theta_class(**kwargs)


# =============================================================================
# JSON Serialization
# =============================================================================

def theta_to_json(theta: Any) -> Dict[str, Any]:
    """Convert a theta dataclass to a JSON-serializable dict.

    Args:
        theta: A theta dataclass instance

    Returns:
        Dict with field names mapped to Python native types
    """
    result = {}
    for field_name in vars(theta).keys():
        val = getattr(theta, field_name)
        # Convert JAX arrays to Python scalars
        if hasattr(val, 'item'):
            result[field_name] = val.item()
        elif hasattr(val, 'tolist'):
            result[field_name] = val.tolist()
        else:
            result[field_name] = val
    return result


def theta_from_json(family: str, d: Dict[str, Any]) -> Any:
    """Create a theta dataclass from a JSON dict.

    Args:
        family: The heuristic family name
        d: Dict with field names and values

    Returns:
        A theta dataclass instance
    """
    if family not in FAMILY_CONFIGS:
        raise ValueError(f"Unknown family '{family}'")

    theta_class = FAMILY_CONFIGS[family]["theta_class"]
    default_theta = theta_class.default()

    kwargs = {}
    for field_name in vars(default_theta).keys():
        if field_name in d:
            default_val = getattr(default_theta, field_name)
            kwargs[field_name] = jnp.array(d[field_name], dtype=default_val.dtype)
        else:
            kwargs[field_name] = getattr(default_theta, field_name)

    return theta_class(**kwargs)


# =============================================================================
# Theta Set Generation
# =============================================================================

def make_theta_set(
    family: str,
    split: str,
    n: int,
    base_seed: int = 0,
) -> List[ThetaSpec]:
    """Generate a list of ThetaSpecs for a family and split.

    Args:
        family: The heuristic family name
        split: "train" or "test"
        n: Number of specs to generate (theta_id = 0..n-1)
        base_seed: Base seed for reproducibility

    Returns:
        List of ThetaSpec instances
    """
    if family not in FAMILY_CONFIGS:
        raise ValueError(f"Unknown family '{family}'")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test'")
    if n < 0:
        raise ValueError(f"n must be non-negative")

    return [
        ThetaSpec(
            family=family,
            theta_id=i,
            split=split,
            base_seed=base_seed,
        )
        for i in range(n)
    ]


def make_all_theta_sets(
    n_train: int,
    n_test: int,
    base_seed: int = 0,
) -> Dict[str, Dict[str, List[ThetaSpec]]]:
    """Generate theta sets for all families.

    Args:
        n_train: Number of train specs per family
        n_test: Number of test specs per family
        base_seed: Base seed for reproducibility

    Returns:
        Nested dict: {family: {"train": [...], "test": [...]}}
    """
    result = {}
    for family in FAMILY_CONFIGS.keys():
        result[family] = {
            "train": make_theta_set(family, "train", n_train, base_seed),
            "test": make_theta_set(family, "test", n_test, base_seed),
        }
    return result


# =============================================================================
# Disjointness Verification
# =============================================================================

def assert_train_test_disjoint(
    family: str,
    train_specs: List[ThetaSpec],
    test_specs: List[ThetaSpec],
    tolerance: float = 1e-6,
) -> None:
    """Verify that train and test theta sets are disjoint.

    Compares JSON serializations of sampled thetas to detect duplicates.

    Args:
        family: The heuristic family name
        train_specs: List of train ThetaSpecs
        test_specs: List of test ThetaSpecs
        tolerance: Tolerance for floating-point comparison

    Raises:
        AssertionError: If any theta appears in both train and test sets
    """
    train_thetas = [theta_to_json(sample_theta(spec)) for spec in train_specs]
    test_thetas = [theta_to_json(sample_theta(spec)) for spec in test_specs]

    def theta_equal(t1: Dict, t2: Dict) -> bool:
        """Check if two theta dicts are equal within tolerance."""
        if t1.keys() != t2.keys():
            return False
        for key in t1.keys():
            v1, v2 = t1[key], t2[key]
            if isinstance(v1, float) and isinstance(v2, float):
                if abs(v1 - v2) > tolerance:
                    return False
            elif v1 != v2:
                return False
        return True

    for i, train_theta in enumerate(train_thetas):
        for j, test_theta in enumerate(test_thetas):
            if theta_equal(train_theta, test_theta):
                raise AssertionError(
                    f"Overlap detected for family '{family}':\n"
                    f"  train_specs[{i}] == test_specs[{j}]\n"
                    f"  theta = {train_theta}"
                )


def verify_reproducibility(spec: ThetaSpec, n_samples: int = 5) -> bool:
    """Verify that sampling the same spec produces identical results.

    Args:
        spec: ThetaSpec to test
        n_samples: Number of times to sample and compare

    Returns:
        True if all samples are identical
    """
    first_json = theta_to_json(sample_theta(spec))
    for _ in range(n_samples - 1):
        other_json = theta_to_json(sample_theta(spec))
        if first_json != other_json:
            return False
    return True


# =============================================================================
# Integration with Teammate Registry
# =============================================================================

def make_teammate_spec(
    theta_spec: ThetaSpec,
    use_log_wrapper: bool = True,
    start_cooking_interaction: bool = False,
) -> HeuristicTeammateSpec:
    """Create a HeuristicTeammateSpec from a ThetaSpec.

    This function samples the theta and creates a spec ready for use
    with the teammate registry.

    Args:
        theta_spec: The ThetaSpec to convert
        use_log_wrapper: Whether environment uses LogWrapper
        start_cooking_interaction: Whether explicit cooking interaction is needed

    Returns:
        HeuristicTeammateSpec ready for make_teammate()
    """
    theta = sample_theta(theta_spec)

    return HeuristicTeammateSpec(
        family=theta_spec.family,
        theta=theta,
        use_log_wrapper=use_log_wrapper,
        start_cooking_interaction=start_cooking_interaction,
    )


def make_teammate_from_theta(
    theta_spec: ThetaSpec,
    env,
    teammate_agent_id: str = "agent_1",
    use_log_wrapper: bool = True,
    start_cooking_interaction: bool = False,
):
    """Create a teammate policy directly from a ThetaSpec.

    This is a convenience function that combines sampling and policy creation.

    Args:
        theta_spec: The ThetaSpec defining the teammate parameters
        env: The environment instance
        teammate_agent_id: Agent ID for this teammate
        use_log_wrapper: Whether environment uses LogWrapper
        start_cooking_interaction: Whether explicit cooking interaction is needed

    Returns:
        TeammatePolicy from the registry
    """
    from teammate_wrapper.registry import make_teammate

    spec = make_teammate_spec(
        theta_spec,
        use_log_wrapper=use_log_wrapper,
        start_cooking_interaction=start_cooking_interaction,
    )

    return make_teammate(spec, env, teammate_agent_id)


# =============================================================================
# Utility Functions
# =============================================================================

def get_all_families() -> List[str]:
    """Get list of all supported heuristic families."""
    return list(FAMILY_CONFIGS.keys())


def get_family_presets(family: str, split: Optional[str] = None) -> List[str]:
    """Get available presets for a family.

    Args:
        family: The heuristic family name
        split: Optional - if provided, returns only presets for that split

    Returns:
        List of preset names
    """
    if family not in FAMILY_CONFIGS:
        raise ValueError(f"Unknown family '{family}'")

    config = FAMILY_CONFIGS[family]

    if split == "train":
        return config.get("train_presets", ["default"])
    elif split == "test":
        return config.get("test_presets", ["default"])
    else:
        # Return all presets
        return list(THETA_PRESETS.get(family, {}).keys())


def get_theta_fields(family: str) -> Dict[str, str]:
    """Get field names and types for a family's theta class.

    Args:
        family: The heuristic family name

    Returns:
        Dict mapping field names to type descriptions
    """
    if family not in FAMILY_CONFIGS:
        raise ValueError(f"Unknown family '{family}'")

    theta_class = FAMILY_CONFIGS[family]["theta_class"]
    default_theta = theta_class.default()

    result = {}
    for field_name in vars(default_theta).keys():
        val = getattr(default_theta, field_name)
        dtype_str = str(val.dtype) if hasattr(val, 'dtype') else type(val).__name__
        result[field_name] = dtype_str

    return result


def summarize_theta(theta: Any, family: str) -> str:
    """Create a human-readable summary of a theta configuration.

    Args:
        theta: A theta dataclass instance
        family: The family name (for context)

    Returns:
        Multi-line string summary
    """
    lines = [f"[{family}]"]

    theta_dict = theta_to_json(theta)
    for field_name, val in sorted(theta_dict.items()):
        if isinstance(val, float):
            lines.append(f"  {field_name}: {val:.4f}")
        else:
            lines.append(f"  {field_name}: {val}")

    return "\n".join(lines)
