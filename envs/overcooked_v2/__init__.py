"""OvercookedV2 environment module.

This module provides a self-contained migration of the JaxMARL OvercookedV2 environment
with a compatibility wrapper for the benchmark framework.
"""

from envs.overcooked_v2.overcooked_v2_wrapper import OvercookedV2Wrapper
from envs.overcooked_v2.overcooked import OvercookedV2, State, ObservationType
from envs.overcooked_v2.layouts import Layout, overcooked_v2_layouts
from envs.overcooked_v2 import spaces

__all__ = [
    "OvercookedV2Wrapper",
    "OvercookedV2",
    "State",
    "ObservationType",
    "Layout",
    "overcooked_v2_layouts",
    "spaces",
]
