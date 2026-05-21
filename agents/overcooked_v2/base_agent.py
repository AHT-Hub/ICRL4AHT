"""
Utility classes for simple rule-based agents in OvercookedV2.

These agents operate directly on the wrapped environment state and ignore the
flattened observation. The helpers below provide basic movement, nearest-target
queries and small utilities for checking inventories and pot status.
"""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from envs.overcooked_v2.common import (
    ACTION_TO_DIRECTION,
    Actions,
    Direction,
    DynamicObject,
    StaticObject,
)


# Action -> (dx, dy) where x is column index and y is row index.
ACTION_TO_DELTA = jnp.array([
    [1, 0],   # right
    [0, 1],   # down
    [-1, 0],  # left
    [0, -1],  # up
    [0, 0],   # stay
    [0, 0],   # interact
], dtype=jnp.int32)


@chex.dataclass
class AgentState:
    """JAX-compatible agent state for heuristic agents."""
    agent_id: jnp.ndarray  # int32 scalar
    rng_key: chex.PRNGKey


class BaseAgentV2:
    """Lightweight helper class shared by the OvercookedV2 heuristic agents."""

    def __init__(self, layout):
        static_objects = jnp.array(layout.static_objects)
        self.static_objects = static_objects
        self.height, self.width = static_objects.shape

        self.ingredient_pile_mask = static_objects >= StaticObject.INGREDIENT_PILE_BASE
        self.plate_pile_mask = static_objects == StaticObject.PLATE_PILE
        self.pot_mask = static_objects == StaticObject.POT
        self.goal_mask = static_objects == StaticObject.GOAL
        self.counter_mask = static_objects == StaticObject.WALL

    # ------------------------------------------------------------------ helpers
    def init_agent_state(self, agent_id: int) -> AgentState:
        return AgentState(
            agent_id=jnp.array(agent_id, dtype=jnp.int32),
            rng_key=jax.random.PRNGKey(agent_id)
        )

    def _get_agent_pos(self, state, agent_id: int) -> Tuple[int, int]:
        """Extract (x, y) for the given agent id from vectorised state.agents."""
        return int(state.agents.pos.x[agent_id]), int(state.agents.pos.y[agent_id])

    def _get_agent_dir(self, state, agent_id: int) -> int:
        return int(state.agents.dir[agent_id])

    def _get_inventory(self, state, agent_id: int) -> int:
        return int(state.agents.inventory[agent_id])

    def _nearest(self, mask: jnp.ndarray, pos: Tuple[int, int]) -> Tuple[Tuple[int, int], bool]:
        """Return coordinates of the nearest True cell to pos (Manhattan distance)."""
        if mask.ndim != 2:
            raise ValueError("Mask must be 2D (H, W)")

        x, y = pos
        ys = jnp.arange(mask.shape[0])[:, None]
        xs = jnp.arange(mask.shape[1])[None, :]
        dist = jnp.abs(xs - x) + jnp.abs(ys - y)
        dist = jnp.where(mask, dist, jnp.inf)

        flat_idx = jnp.argmin(dist)
        target_y = flat_idx // mask.shape[1]
        target_x = flat_idx % mask.shape[1]
        exists = bool(jnp.any(mask))
        return (int(target_x), int(target_y)), exists

    def _is_adjacent(self, pos: Tuple[int, int], target: Tuple[int, int]) -> bool:
        dx = abs(pos[0] - target[0])
        dy = abs(pos[1] - target[1])
        return dx + dy == 1

    def _direction_to(self, pos: Tuple[int, int], target: Tuple[int, int]) -> Actions:
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        if abs(dx) > abs(dy):
            return Actions.right if dx > 0 else Actions.left
        if dy != 0:
            return Actions.down if dy > 0 else Actions.up
        return Actions.stay

    def _is_walkable(self, x: int, y: int) -> bool:
        in_bounds = 0 <= x < self.width and 0 <= y < self.height
        if not in_bounds:
            return False
        return bool(self.static_objects[y, x] == StaticObject.EMPTY)

    def _move_towards(self, pos: Tuple[int, int], target: Tuple[int, int]) -> Actions:
        """Greedy step toward target avoiding non-empty static cells."""
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]

        candidates = []
        if dx > 0:
            candidates.append(Actions.right)
        if dx < 0:
            candidates.append(Actions.left)
        if dy > 0:
            candidates.append(Actions.down)
        if dy < 0:
            candidates.append(Actions.up)
        candidates.append(Actions.stay)

        for act in candidates:
            delta = ACTION_TO_DELTA.get(act, (0, 0))
            nx = pos[0] + delta[0]
            ny = pos[1] + delta[1]
            if self._is_walkable(nx, ny):
                return act
        return Actions.stay

    def _approach_and_interact(
        self,
        pos: Tuple[int, int],
        target_mask: jnp.ndarray,
        agent_dir: int,
    ) -> Actions:
        """Move toward the nearest target; interact when adjacent and facing it."""
        target, exists = self._nearest(target_mask, pos)
        if not exists:
            return Actions.stay

        if self._is_adjacent(pos, target):
            # _direction_to returns an Actions enum (movement action), but agent_dir
            # is a Direction enum. Use ACTION_TO_DIRECTION to convert the movement
            # action to the corresponding Direction before comparing.
            desired_move_action = self._direction_to(pos, target)
            desired_dir = int(ACTION_TO_DIRECTION[int(desired_move_action)])
            if agent_dir == desired_dir:
                return Actions.interact
            return desired_move_action

        return self._move_towards(pos, target)

    def _count_ingredients(self, pot_contents: jnp.ndarray) -> jnp.ndarray:
        """Vectorized ingredient counts for each cell (pot_contents already masked)."""
        count_fn = jax.vmap(jax.vmap(DynamicObject.ingredient_count))
        return count_fn(pot_contents)

    @staticmethod
    def _is_plate(inv: int) -> bool:
        return inv == DynamicObject.PLATE

    @staticmethod
    def _is_dish(inv: int) -> bool:
        return bool(inv & DynamicObject.COOKED) and bool(inv & DynamicObject.PLATE)

    @staticmethod
    def _is_ingredient(inv: int) -> bool:
        return bool(DynamicObject.is_ingredient(inv))
