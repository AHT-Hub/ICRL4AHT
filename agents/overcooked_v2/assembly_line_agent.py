"""Assembly Line Agent for OvercookedV2 - Fast NumPy Implementation.

This agent implements role-based cooperative behavior with tunable hyperparameters.
All pathfinding and decision logic is precomputed at initialization using NumPy,
making runtime execution extremely fast (simple array lookups).

The agent is JAX-compatible for training - it returns JAX arrays but all internal
computation uses NumPy for maximum speed.

Roles:
- INGREDIENT_RUNNER (0): Fetches ingredients and puts them in pots
- PLATER_DELIVERER (1): Fetches plates and delivers cooked dishes
- FLEX (2): Adapts behavior based on current game state
"""

import hashlib
import os
import pickle
from pathlib import Path
from typing import Tuple

import chex
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np

from envs.overcooked_v2.common import (
    Actions,
    Direction,
    DynamicObject,
    Position,
    StaticObject,
)
from envs.overcooked_v2.settings import POT_COOK_TIME

from .base_agent import AgentState, BaseAgentV2


# Role constants
ROLE_INGREDIENT_RUNNER = 0
ROLE_PLATER_DELIVERER = 1
ROLE_FLEX = 2

# Handoff style constants
HANDOFF_POT_ADJACENT = 0
HANDOFF_CENTRAL = 1
HANDOFF_TEAMMATE_NEARBY = 2

# Cache directory for precomputed data
CACHE_DIR = Path(__file__).parent / ".assembly_line_cache"


@flax.struct.dataclass
class AssemblyLineTheta:
    """Hyperparameters for the AssemblyLine agent family.

    All fields are JAX arrays to ensure compatibility with tracing.
    
    Cooperation-difficulty parameters:
    - hesitation_prob: Probability of pausing (stay action) instead of acting
    - wrong_action_prob: Probability of taking a random wrong action
    - task_abandon_prob: Probability of abandoning current task mid-way
    - stubbornness: How much the agent insists on its preferred spots/tasks
    - timing_mismatch: Delay factor that causes timing issues with teammate
    """
    role_mode: jnp.ndarray  # int32: 0=INGREDIENT_RUNNER, 1=PLATER_DELIVERER, 2=FLEX
    handoff_style: jnp.ndarray  # int32: 0=pot_adjacent, 1=central, 2=teammate_nearby
    plate_urgency: jnp.ndarray  # float32 in [0, 1]: how early to fetch plates
    prestage_bias: jnp.ndarray  # float32 in [0, 1]: how much to stage extra ingredients
    start_cook_bias: jnp.ndarray  # float32 in [0, 1]: how strongly to start cooking
    # New cooperation-difficulty parameters
    hesitation_prob: jnp.ndarray  # float32 in [0, 1]: probability of hesitating (stay)
    wrong_action_prob: jnp.ndarray  # float32 in [0, 1]: probability of random wrong action
    task_abandon_prob: jnp.ndarray  # float32 in [0, 1]: probability of dropping item unexpectedly
    stubbornness: jnp.ndarray  # float32 in [0, 1]: how much to prefer specific spots
    timing_mismatch: jnp.ndarray  # float32 in [0, 1]: extra delays causing timing issues

    @classmethod
    def default(cls) -> "AssemblyLineTheta":
        """Create default hyperparameters."""
        return cls(
            role_mode=jnp.array(ROLE_FLEX, dtype=jnp.int32),
            handoff_style=jnp.array(HANDOFF_POT_ADJACENT, dtype=jnp.int32),
            plate_urgency=jnp.array(0.5, dtype=jnp.float32),
            prestage_bias=jnp.array(0.3, dtype=jnp.float32),
            start_cook_bias=jnp.array(0.7, dtype=jnp.float32),
            hesitation_prob=jnp.array(0.0, dtype=jnp.float32),
            wrong_action_prob=jnp.array(0.0, dtype=jnp.float32),
            task_abandon_prob=jnp.array(0.0, dtype=jnp.float32),
            stubbornness=jnp.array(0.0, dtype=jnp.float32),
            timing_mismatch=jnp.array(0.0, dtype=jnp.float32),
        )

    @classmethod
    def ingredient_runner(cls) -> "AssemblyLineTheta":
        """Create hyperparameters for pure ingredient runner role."""
        return cls(
            role_mode=jnp.array(ROLE_INGREDIENT_RUNNER, dtype=jnp.int32),
            handoff_style=jnp.array(HANDOFF_POT_ADJACENT, dtype=jnp.int32),
            plate_urgency=jnp.array(0.0, dtype=jnp.float32),
            prestage_bias=jnp.array(0.5, dtype=jnp.float32),
            start_cook_bias=jnp.array(0.8, dtype=jnp.float32),
            hesitation_prob=jnp.array(0.0, dtype=jnp.float32),
            wrong_action_prob=jnp.array(0.0, dtype=jnp.float32),
            task_abandon_prob=jnp.array(0.0, dtype=jnp.float32),
            stubbornness=jnp.array(0.0, dtype=jnp.float32),
            timing_mismatch=jnp.array(0.0, dtype=jnp.float32),
        )

    @classmethod
    def plater_deliverer(cls) -> "AssemblyLineTheta":
        """Create hyperparameters for pure plater/deliverer role."""
        return cls(
            role_mode=jnp.array(ROLE_PLATER_DELIVERER, dtype=jnp.int32),
            handoff_style=jnp.array(HANDOFF_POT_ADJACENT, dtype=jnp.int32),
            plate_urgency=jnp.array(0.8, dtype=jnp.float32),
            prestage_bias=jnp.array(0.0, dtype=jnp.float32),
            start_cook_bias=jnp.array(0.3, dtype=jnp.float32),
            hesitation_prob=jnp.array(0.0, dtype=jnp.float32),
            wrong_action_prob=jnp.array(0.0, dtype=jnp.float32),
            task_abandon_prob=jnp.array(0.0, dtype=jnp.float32),
            stubbornness=jnp.array(0.0, dtype=jnp.float32),
            timing_mismatch=jnp.array(0.0, dtype=jnp.float32),
        )


class AssemblyLineAgentV2(BaseAgentV2):
    """Fast NumPy-based assembly line agent with role-based behavior.

    All pathfinding is precomputed at initialization. Runtime execution uses
    pure NumPy operations for maximum speed.
    """

    def __init__(self, layout, theta: AssemblyLineTheta = None, start_cooking_interaction: bool = False):
        """Initialize the assembly line agent.

        Args:
            layout: Layout object containing the static grid configuration.
            theta: Hyperparameters for the agent. If None, uses default.
            start_cooking_interaction: Whether the environment requires explicit
                interaction to start cooking (vs auto-cooking when pot is full).
        """
        super().__init__(layout)

        self.theta = theta if theta is not None else AssemblyLineTheta.default()
        self.start_cooking_interaction = start_cooking_interaction

        # Convert to numpy for fast computation
        self.static_objects_np = np.array(self.static_objects)
        self.walkable_mask_np = self.static_objects_np == StaticObject.EMPTY

        # Precompute all masks in numpy
        self.ingredient_pile_mask_np = self.static_objects_np >= StaticObject.INGREDIENT_PILE_BASE
        self.plate_pile_mask_np = self.static_objects_np == StaticObject.PLATE_PILE
        self.pot_mask_np = self.static_objects_np == StaticObject.POT
        self.goal_mask_np = self.static_objects_np == StaticObject.GOAL
        self.counter_mask_np = self.static_objects_np == StaticObject.WALL
        self.button_mask_np = self.static_objects_np == StaticObject.BUTTON_RECIPE_INDICATOR
        self.recipe_indicator_mask_np = self.static_objects_np == StaticObject.RECIPE_INDICATOR

        # Precompute center of walkable area
        walkable_positions = np.argwhere(self.walkable_mask_np)
        if len(walkable_positions) > 0:
            self.center_y = int(np.mean(walkable_positions[:, 0]))
            self.center_x = int(np.mean(walkable_positions[:, 1]))
        else:
            self.center_y, self.center_x = self.height // 2, self.width // 2

        # Precompute pot-adjacent counter mask
        self._precompute_pot_adjacent_counters_np()

        # Load or compute pathfinding data
        self._load_or_compute_pathfinding()

    def _get_layout_hash(self) -> str:
        """Get a hash of the layout for caching."""
        layout_bytes = self.static_objects_np.tobytes()
        return hashlib.md5(layout_bytes).hexdigest()[:16]

    def _load_or_compute_pathfinding(self):
        """Load precomputed pathfinding data from cache or compute it."""
        layout_hash = self._get_layout_hash()
        cache_file = CACHE_DIR / f"{layout_hash}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                self.dist_matrix = data["dist_matrix"]
                self.next_action_matrix = data["next_action_matrix"]
                self.pos_to_idx = data["pos_to_idx"]
                self.idx_to_pos = data["idx_to_pos"]
                self.adjacent_to_reachable = data["adjacent_to_reachable"]
                self.reachable_from = data["reachable_from"]
                return
            except Exception:
                pass  # Fall through to recompute

        # Compute pathfinding data
        self._precompute_pathfinding()

        # Save to cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump({
                    "dist_matrix": self.dist_matrix,
                    "next_action_matrix": self.next_action_matrix,
                    "pos_to_idx": self.pos_to_idx,
                    "idx_to_pos": self.idx_to_pos,
                    "adjacent_to_reachable": self.adjacent_to_reachable,
                    "reachable_from": self.reachable_from,
                }, f)
        except Exception:
            pass  # Ignore cache write errors

    def _precompute_pot_adjacent_counters_np(self):
        """Precompute masks for counters adjacent to pots using NumPy."""
        pot_mask = self.pot_mask_np

        up = np.roll(pot_mask, shift=-1, axis=0)
        down = np.roll(pot_mask, shift=1, axis=0)
        left = np.roll(pot_mask, shift=-1, axis=1)
        right = np.roll(pot_mask, shift=1, axis=1)

        up[-1, :] = False
        down[0, :] = False
        left[:, -1] = False
        right[:, 0] = False

        adjacent_to_pot = up | down | left | right
        self.pot_adjacent_counter_mask_np = adjacent_to_pot & self.counter_mask_np

    def _precompute_pathfinding(self):
        """Precompute all pathfinding data using NumPy.

        Computes:
        - Distance matrix between all walkable positions
        - Next action matrix for each (pos, direction, target) tuple
        - Reachability masks for each position
        """
        # Get all walkable positions
        walkable_positions = np.argwhere(self.walkable_mask_np)
        num_positions = len(walkable_positions)

        # Create position index mappings
        self.pos_to_idx = np.full((self.height, self.width), -1, dtype=np.int32)
        self.idx_to_pos = walkable_positions.copy()
        for idx, (y, x) in enumerate(walkable_positions):
            self.pos_to_idx[y, x] = idx

        # Direction deltas: UP=0, DOWN=1, RIGHT=2, LEFT=3
        # UP: dy=-1, dx=0; DOWN: dy=1, dx=0; RIGHT: dy=0, dx=1; LEFT: dy=0, dx=-1
        dir_deltas = np.array([
            [-1, 0],  # UP
            [1, 0],   # DOWN
            [0, 1],   # RIGHT
            [0, -1],  # LEFT
        ], dtype=np.int32)

        # Compute distance matrix using BFS from each position
        # dist_matrix[from_idx, to_y, to_x] = distance from position from_idx to (to_y, to_x)
        INF = 10000
        self.dist_matrix = np.full((num_positions, self.height, self.width), INF, dtype=np.int32)

        # next_action_matrix[from_idx, from_dir, to_y, to_x] = best action to reach (to_y, to_x)
        # Actions: right=0, down=1, left=2, up=3, stay=4, interact=5
        self.next_action_matrix = np.full((num_positions, 4, self.height, self.width), Actions.stay, dtype=np.int32)

        for start_idx in range(num_positions):
            start_y, start_x = walkable_positions[start_idx]

            # BFS to compute distances
            dist = np.full((self.height, self.width), INF, dtype=np.int32)
            dist[start_y, start_x] = 0

            # Also track the first step direction for each cell
            # first_step[y, x] = the action taken from start to begin path to (y, x)
            first_step = np.full((self.height, self.width), -1, dtype=np.int32)

            queue = [(start_y, start_x)]
            head = 0

            while head < len(queue):
                cy, cx = queue[head]
                head += 1
                curr_dist = dist[cy, cx]

                # Try all 4 directions
                for action, (dy, dx) in enumerate([(0, 1), (1, 0), (0, -1), (-1, 0)]):
                    # action: 0=right, 1=down, 2=left, 3=up
                    ny, nx = cy + dy, cx + dx

                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        if self.walkable_mask_np[ny, nx] and dist[ny, nx] == INF:
                            dist[ny, nx] = curr_dist + 1
                            queue.append((ny, nx))

                            # Track first step
                            if cy == start_y and cx == start_x:
                                first_step[ny, nx] = action
                            else:
                                first_step[ny, nx] = first_step[cy, cx]

            self.dist_matrix[start_idx] = dist

            # Fill next_action_matrix for each starting direction
            for start_dir in range(4):
                for ty in range(self.height):
                    for tx in range(self.width):
                        if ty == start_y and tx == start_x:
                            # Already at target - stay
                            self.next_action_matrix[start_idx, start_dir, ty, tx] = Actions.stay
                        elif dist[ty, tx] < INF:
                            # Can reach - use first step
                            fs = first_step[ty, tx]
                            if fs >= 0:
                                self.next_action_matrix[start_idx, start_dir, ty, tx] = fs
                        # else: unreachable, stays as Actions.stay

        # Compute reachable masks and adjacent-to-reachable masks for each position
        self.reachable_from = np.zeros((num_positions, self.height, self.width), dtype=np.bool_)
        self.adjacent_to_reachable = np.zeros((num_positions, self.height, self.width), dtype=np.bool_)

        for idx in range(num_positions):
            reachable = self.dist_matrix[idx] < INF
            self.reachable_from[idx] = reachable

            # Compute adjacent to reachable
            up = np.roll(reachable, shift=-1, axis=0)
            down = np.roll(reachable, shift=1, axis=0)
            left = np.roll(reachable, shift=-1, axis=1)
            right = np.roll(reachable, shift=1, axis=1)

            up[-1, :] = False
            down[0, :] = False
            left[:, -1] = False
            right[:, 0] = False

            self.adjacent_to_reachable[idx] = up | down | left | right

    def _get_closest_target_np(self, target_mask: np.ndarray, pos_y: int, pos_x: int) -> Tuple[int, int, bool]:
        """Find closest target cell using precomputed distances.

        For non-walkable targets (like pots, piles), we find the closest target
        by computing the minimum distance to any adjacent walkable cell.

        Returns:
            Tuple of (target_y, target_x, exists_flag)
        """
        idx = self.pos_to_idx[pos_y, pos_x]
        if idx < 0:
            return pos_y, pos_x, False

        if not np.any(target_mask):
            return pos_y, pos_x, False

        # For each target, compute the minimum distance to reach an adjacent walkable cell
        target_positions = np.argwhere(target_mask)
        best_dist = 10000
        best_target = None

        for ty, tx in target_positions:
            # Find adjacent walkable cells
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ay, ax = ty + dy, tx + dx
                if 0 <= ay < self.height and 0 <= ax < self.width:
                    if self.walkable_mask_np[ay, ax]:
                        dist = self.dist_matrix[idx, ay, ax]
                        if dist < best_dist:
                            best_dist = dist
                            best_target = (ty, tx)

        if best_target is None or best_dist >= 10000:
            return pos_y, pos_x, False

        return best_target[0], best_target[1], True

    def _get_action_to_adjacent_np(
        self,
        pos_y: int, pos_x: int,
        agent_dir: int,
        target_y: int, target_x: int,
        other_agents_positions: np.ndarray,
        agent_id: int,
    ) -> int:
        """Get action to move adjacent to target and interact.

        Args:
            pos_y, pos_x: Current position
            agent_dir: Current direction (0=UP, 1=DOWN, 2=RIGHT, 3=LEFT)
            target_y, target_x: Target position (non-walkable cell like pot/pile)
            other_agents_positions: (N, 2) array of other agent positions
            agent_id: Current agent ID for tie-breaking

        Returns:
            Action to take
        """
        # Find adjacent walkable cells to target
        adjacent_cells = []
        for dy, dx, face_dir in [(-1, 0, Direction.DOWN), (1, 0, Direction.UP),
                                  (0, -1, Direction.RIGHT), (0, 1, Direction.LEFT)]:
            ay, ax = target_y + dy, target_x + dx
            if 0 <= ay < self.height and 0 <= ax < self.width:
                if self.walkable_mask_np[ay, ax]:
                    adjacent_cells.append((ay, ax, face_dir))

        if not adjacent_cells:
            return Actions.stay

        pos_idx = self.pos_to_idx[pos_y, pos_x]
        if pos_idx < 0:
            return Actions.stay

        # Check if already adjacent to target
        dy = target_y - pos_y
        dx = target_x - pos_x
        if abs(dy) + abs(dx) == 1:
            # Already adjacent - face and interact
            # Determine required direction to face target
            if dy == -1:
                required_dir = Direction.UP
            elif dy == 1:
                required_dir = Direction.DOWN
            elif dx == 1:
                required_dir = Direction.RIGHT
            else:
                required_dir = Direction.LEFT

            if agent_dir == required_dir:
                return Actions.interact
            else:
                # Turn to face target
                # Direction -> movement action: UP->3, DOWN->1, RIGHT->0, LEFT->2
                dir_to_action = [3, 1, 0, 2]  # UP, DOWN, RIGHT, LEFT
                return dir_to_action[required_dir]

        # Find best adjacent cell to move to
        best_cell = None
        best_dist = 10000

        for ay, ax, face_dir in adjacent_cells:
            # Check if blocked by another agent
            blocked = False
            for other_pos in other_agents_positions:
                if other_pos[0] == ay and other_pos[1] == ax:
                    blocked = True
                    break

            dist = self.dist_matrix[pos_idx, ay, ax]
            if blocked:
                dist += 100  # Heavy penalty for blocked cells

            if dist < best_dist:
                best_dist = dist
                best_cell = (ay, ax)

        if best_cell is None or best_dist >= 10000:
            return Actions.stay

        # Get action to move towards best adjacent cell
        next_action = self.next_action_matrix[pos_idx, agent_dir, best_cell[0], best_cell[1]]

        # Collision avoidance: check if next step would collide with another agent
        # This handles the case where both agents try to move to the same cell simultaneously
        action_deltas = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0), (0, 0)]  # right, down, left, up, stay, interact
        dpy, dpx = action_deltas[next_action]
        next_y, next_x = pos_y + dpy, pos_x + dpx

        # Check if immediate next cell is blocked by another agent
        if next_action < 4:  # Only for movement actions
            for other_pos in other_agents_positions:
                other_y, other_x = int(other_pos[0]), int(other_pos[1])
                # If the cell we want to move to is occupied, yield if we have lower priority
                if other_y == next_y and other_x == next_x:
                    if agent_id > 0:  # Lower priority agent yields
                        return Actions.stay
                    # Higher priority agent (agent_id==0) can proceed
                    # The lower priority agent should yield or move away
                    break

        # Check if another agent might be trying to move to the same cell as us
        # This prevents collisions where both agents move to the same empty cell
        if next_action < 4:  # Only for movement actions
            for other_pos in other_agents_positions:
                other_y, other_x = int(other_pos[0]), int(other_pos[1])
                other_idx = self.pos_to_idx[other_y, other_x]
                if other_idx < 0:
                    continue

                # Check if the other agent is adjacent to our target cell
                other_dist_to_next = self.dist_matrix[other_idx, next_y, next_x]
                if other_dist_to_next == 1:
                    # Other agent could also be moving to the same cell
                    # Use agent_id as tie-breaker: lower id has priority
                    if agent_id > 0:
                        return Actions.stay

        # If our next position would be where another agent is trying to go (the best_cell),
        # and the other agent is also one step away from that cell, use agent_id as tie-breaker
        if len(adjacent_cells) == 1 and next_action < 4:  # Only for movement actions
            target_adj_y, target_adj_x = adjacent_cells[0][0], adjacent_cells[0][1]

            for other_pos in other_agents_positions:
                other_y, other_x = int(other_pos[0]), int(other_pos[1])
                other_idx = self.pos_to_idx[other_y, other_x]
                if other_idx < 0:
                    continue

                other_dist = self.dist_matrix[other_idx, target_adj_y, target_adj_x]
                my_dist = self.dist_matrix[pos_idx, target_adj_y, target_adj_x]

                # Only yield if:
                # 1. Other agent is also 1 step away from the same narrow target
                # 2. Both of us would collide trying to reach it
                # 3. I have higher agent_id (lower priority)
                if other_dist == 1 and my_dist == 1 and agent_id > 0:
                    return Actions.stay

        return next_action

    def _get_reachable_targets_np(self, target_mask: np.ndarray, pos_y: int, pos_x: int) -> np.ndarray:
        """Get targets that are adjacent to reachable walkable cells."""
        idx = self.pos_to_idx[pos_y, pos_x]
        if idx < 0:
            return np.zeros_like(target_mask, dtype=np.bool_)
        return target_mask & self.adjacent_to_reachable[idx]

    def _count_ingredients_np(self, content: int) -> int:
        """Count ingredients in a pot content encoding."""
        content = content >> 2  # Skip flag bits
        count = 0
        while content > 0:
            count += content & 0x3
            content >>= 2
        return count

    def _get_pot_states_np(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Analyze pot states using NumPy.

        Returns:
            - pot_mask: Which cells are pots
            - pot_non_full_mask: Pots that can accept more ingredients
            - pot_cooking_mask: Pots currently cooking
            - pot_cooked_mask: Pots with finished soup
        """
        static_layer = grid[:, :, 0]
        dynamic_layer = grid[:, :, 1]
        timer_layer = grid[:, :, 2]

        is_pot = static_layer == StaticObject.POT

        # Count ingredients in each pot
        ingredient_counts = np.zeros_like(dynamic_layer, dtype=np.int32)
        for y in range(self.height):
            for x in range(self.width):
                if is_pot[y, x]:
                    ingredient_counts[y, x] = self._count_ingredients_np(int(dynamic_layer[y, x]))

        pot_full = ingredient_counts >= 3
        pot_cooking = (timer_layer > 0) & is_pot
        pot_cooked = ((dynamic_layer & DynamicObject.COOKED) != 0) & is_pot

        pot_non_full = is_pot & ~pot_full & ~pot_cooking & ~pot_cooked

        return is_pot, pot_non_full, pot_cooking, pot_cooked

    def _should_start_cooking_np(self, grid: np.ndarray, pot_mask: np.ndarray) -> np.ndarray:
        """Check if any pot is ready to start cooking (full, timer==0, not cooked)."""
        dynamic_layer = grid[:, :, 1]
        timer_layer = grid[:, :, 2]

        # Count ingredients
        ingredient_counts = np.zeros_like(dynamic_layer, dtype=np.int32)
        for y in range(self.height):
            for x in range(self.width):
                if pot_mask[y, x]:
                    ingredient_counts[y, x] = self._count_ingredients_np(int(dynamic_layer[y, x]))

        pot_full = (ingredient_counts >= 3) & pot_mask
        not_cooking = timer_layer == 0
        not_cooked = (dynamic_layer & DynamicObject.COOKED) == 0

        return pot_full & not_cooking & not_cooked

    def _get_empty_counter_mask_np(self, grid: np.ndarray) -> np.ndarray:
        """Get mask of empty counters."""
        is_counter = self.counter_mask_np
        is_empty = grid[:, :, 1] == DynamicObject.EMPTY
        return is_counter & is_empty

    def _get_staged_ingredient_mask_np(self, grid: np.ndarray) -> np.ndarray:
        """Get mask of counters with ingredients staged on them."""
        is_counter = self.counter_mask_np
        has_ingredient = (grid[:, :, 1] >> 2) != 0
        no_plate = (grid[:, :, 1] & DynamicObject.PLATE) == 0
        return is_counter & has_ingredient & no_plate

    def _get_handoff_counter_mask_np(
        self,
        grid: np.ndarray,
        teammate_y: int,
        teammate_x: int,
        handoff_style: int,
    ) -> np.ndarray:
        """Get mask of suitable handoff counters based on handoff_style."""
        empty_counters = self._get_empty_counter_mask_np(grid)

        if handoff_style == HANDOFF_POT_ADJACENT:
            result = empty_counters & self.pot_adjacent_counter_mask_np
            if np.any(result):
                return result
            return empty_counters

        elif handoff_style == HANDOFF_CENTRAL:
            ys = np.arange(self.height)[:, None]
            xs = np.arange(self.width)[None, :]
            dist_to_center = np.abs(xs - self.center_x) + np.abs(ys - self.center_y)
            near_center = dist_to_center <= 3
            result = empty_counters & near_center
            if np.any(result):
                return result
            return empty_counters

        else:  # HANDOFF_TEAMMATE_NEARBY
            ys = np.arange(self.height)[:, None]
            xs = np.arange(self.width)[None, :]
            dist_to_teammate = np.abs(xs - teammate_x) + np.abs(ys - teammate_y)
            near_teammate = dist_to_teammate <= 2
            result = empty_counters & near_teammate
            if np.any(result):
                return result
            # Fallback to pot-adjacent
            result = empty_counters & self.pot_adjacent_counter_mask_np
            if np.any(result):
                return result
            return empty_counters

    def _decode_recipe(self, recipe: int) -> np.ndarray:
        """Decode recipe encoding into ingredient type counts.

        Recipe encoding: each ingredient type has 2 bits (0-3 count).
        Bits 0-1: unused (flags), Bits 2-3: ingredient 0, Bits 4-5: ingredient 1, etc.

        Returns:
            Array of shape (4,) with count of each ingredient type needed.
        """
        counts = np.zeros(4, dtype=np.int32)
        recipe = recipe >> 2  # Skip flag bits
        for i in range(4):
            counts[i] = recipe & 0x3
            recipe >>= 2
        return counts

    def _get_recipe_ingredient_pile_mask_np(
        self, grid: np.ndarray, recipe: int, carried_ingredients: np.ndarray = None
    ) -> np.ndarray:
        """Get mask for ingredient piles that match the recipe.

        Only returns piles for ingredients that are still needed based on
        what's already in the pot vs what the recipe requires. Also considers
        ingredients currently being carried by agents to avoid over-picking.
        
        Args:
            grid: Environment grid
            recipe: Recipe encoding
            carried_ingredients: Array of shape (4,) counting ingredients being carried
                                 by agents (treat as "in transit" to pot)
        """
        dynamic_layer = grid[:, :, 1]
        static_layer = grid[:, :, 0]
        is_pot = static_layer == StaticObject.POT

        # Decode what recipe needs
        recipe_counts = self._decode_recipe(recipe)

        # Count ingredients per type already in pots
        pot_counts = np.zeros(4, dtype=np.int32)
        for y in range(self.height):
            for x in range(self.width):
                if is_pot[y, x]:
                    content = int(dynamic_layer[y, x]) >> 2
                    for i in range(4):
                        pot_counts[i] += content & 0x3
                        content >>= 2

        # Add carried ingredients (in transit) to pot counts
        if carried_ingredients is not None:
            pot_counts = pot_counts + carried_ingredients

        # Compute how many more of each type we need
        needed = recipe_counts - pot_counts
        needed = np.maximum(needed, 0)

        # Build mask of piles for needed ingredients
        result_mask = np.zeros((self.height, self.width), dtype=np.bool_)
        for i in range(4):
            if needed[i] > 0:
                pile_type = StaticObject.INGREDIENT_PILE_BASE + i
                result_mask |= (static_layer == pile_type)

        if np.any(result_mask):
            return result_mask

        # If we don't need anything more, return empty mask
        return np.zeros((self.height, self.width), dtype=np.bool_)

    def _get_dominant_ingredient_pile_mask_np(self, grid: np.ndarray) -> np.ndarray:
        """Get mask for ingredient piles matching current pot contents.

        Returns all ingredient piles if pots are empty.
        """
        dynamic_layer = grid[:, :, 1]
        static_layer = grid[:, :, 0]
        is_pot = static_layer == StaticObject.POT

        # Count ingredients per type across all pots
        total_per_type = np.zeros(4, dtype=np.int32)
        for y in range(self.height):
            for x in range(self.width):
                if is_pot[y, x]:
                    content = int(dynamic_layer[y, x]) >> 2
                    for i in range(4):
                        total_per_type[i] += content & 0x3
                        content >>= 2

        if np.sum(total_per_type) > 0:
            # Use dominant ingredient type
            dominant_type = np.argmax(total_per_type)
            target_pile = StaticObject.INGREDIENT_PILE_BASE + dominant_type
            target_mask = static_layer == target_pile
            if np.any(target_mask):
                return target_mask

        # Return all ingredient piles
        return self.ingredient_pile_mask_np

    def _wander_action_np(
        self,
        pos_y: int, pos_x: int,
        agent_dir: int,
        other_agents_positions: np.ndarray,
    ) -> int:
        """Move to avoid blocking, preferring open spaces."""
        # Actions: right=0, down=1, left=2, up=3, stay=4
        deltas = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]

        best_action = Actions.stay
        best_score = 1000

        for action, (dy, dx) in enumerate(deltas):
            ny, nx = pos_y + dy, pos_x + dx

            if not (0 <= ny < self.height and 0 <= nx < self.width):
                continue

            if action < 4 and not self.walkable_mask_np[ny, nx]:
                continue

            # Check if blocked by another agent
            blocked = False
            for other_pos in other_agents_positions:
                if other_pos[0] == ny and other_pos[1] == nx:
                    blocked = True
                    break

            score = 0
            if blocked:
                score += 100
            if action == 4:  # stay
                score += 50

            if score < best_score:
                best_score = score
                best_action = action

        return best_action

    def _yield_to_teammate_np(
        self,
        pos_y: int, pos_x: int, agent_dir: int,
        teammate_y: int, teammate_x: int,
        other_agents_positions: np.ndarray,
    ) -> int:
        """Move away from teammate to let them pass.
        
        This is called when we're empty-handed and adjacent to a teammate
        who is carrying something useful.
        """
        # Direction away from teammate
        dy = pos_y - teammate_y
        dx = pos_x - teammate_x
        
        # Try to move away from teammate
        # Actions: right=0, down=1, left=2, up=3
        preferred_actions = []
        
        if dy < 0:  # teammate is below us, move up
            preferred_actions.append(3)  # up
        elif dy > 0:  # teammate is above us, move down
            preferred_actions.append(1)  # down
        
        if dx < 0:  # teammate is to the right, move left
            preferred_actions.append(2)  # left
        elif dx > 0:  # teammate is to the left, move right
            preferred_actions.append(0)  # right
        
        # Also try perpendicular moves
        if dy != 0:
            preferred_actions.extend([0, 2])  # right, left
        if dx != 0:
            preferred_actions.extend([1, 3])  # down, up
        
        action_deltas = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for action in preferred_actions:
            dpy, dpx = action_deltas[action]
            ny, nx = pos_y + dpy, pos_x + dpx
            
            if not (0 <= ny < self.height and 0 <= nx < self.width):
                continue
            if not self.walkable_mask_np[ny, nx]:
                continue
            
            # Check if blocked by another agent
            blocked = False
            for other_pos in other_agents_positions:
                if other_pos[0] == ny and other_pos[1] == nx:
                    blocked = True
                    break
            
            if not blocked:
                return action
        
        # Can't move away, just stay
        return Actions.stay

    def get_action(
        self,
        obs,
        env_state,
        agent_state: AgentState = None,
    ) -> Tuple[jnp.ndarray, AgentState]:
        """Get action for the agent (main entry point).

        Uses NumPy for all computation, returns JAX arrays for compatibility.

        Args:
            obs: Flattened observation (ignored, we use env_state directly)
            env_state: Full environment state
            agent_state: Agent's internal state

        Returns:
            Tuple of (action, new_agent_state)
        """
        if agent_state is None:
            agent_state = self.init_agent_state(0)

        agent_id = int(agent_state.agent_id)

        # Extract state as numpy arrays
        pos_x = int(env_state.agents.pos.x[agent_id])
        pos_y = int(env_state.agents.pos.y[agent_id])
        agent_dir = int(env_state.agents.dir[agent_id])
        inv = int(env_state.agents.inventory[agent_id])

        # Get teammate position
        num_agents = env_state.agents.pos.x.shape[0]
        other_agents_positions = []
        for i in range(num_agents):
            if i != agent_id:
                other_agents_positions.append([
                    int(env_state.agents.pos.y[i]),
                    int(env_state.agents.pos.x[i])
                ])
        other_agents_positions = np.array(other_agents_positions) if other_agents_positions else np.zeros((0, 2), dtype=np.int32)

        teammate_id = (agent_id + 1) % num_agents
        teammate_x = int(env_state.agents.pos.x[teammate_id])
        teammate_y = int(env_state.agents.pos.y[teammate_id])

        # Convert grid to numpy
        grid = np.array(env_state.grid)

        # Get recipe
        recipe = int(env_state.recipe) if hasattr(env_state, 'recipe') else 0

        # Get theta values as python scalars
        role_mode = int(self.theta.role_mode)
        handoff_style = int(self.theta.handoff_style)
        plate_urgency = float(self.theta.plate_urgency)
        start_cook_bias = float(self.theta.start_cook_bias)
        
        # Get cooperation-difficulty parameters
        hesitation_prob = float(self.theta.hesitation_prob)
        wrong_action_prob = float(self.theta.wrong_action_prob)
        task_abandon_prob = float(self.theta.task_abandon_prob)
        stubbornness = float(self.theta.stubbornness)
        timing_mismatch = float(self.theta.timing_mismatch)
        
        # Use agent_state RNG for stochastic behaviors
        rng = agent_state.rng_key
        rng, hesitate_key, wrong_key, abandon_key, timing_key = jax.random.split(rng, 5)
        
        # Apply hesitation: random chance to just stay in place
        if hesitation_prob > 0:
            hesitate_roll = float(jax.random.uniform(hesitate_key))
            if hesitate_roll < hesitation_prob:
                # Just stay in place (hesitate)
                new_state = AgentState(agent_id=jnp.array(agent_id, dtype=jnp.int32), rng_key=rng)
                return jnp.array(Actions.stay, dtype=jnp.int32), new_state
        
        # Apply timing mismatch: add delays based on position hash (deterministic per position)
        if timing_mismatch > 0:
            timing_roll = float(jax.random.uniform(timing_key))
            # Create position-based delay pattern
            pos_hash = (pos_x * 7 + pos_y * 13) % 10
            delay_threshold = timing_mismatch * (pos_hash / 10.0)
            if timing_roll < delay_threshold:
                # Delay action (stay in place)
                new_state = AgentState(agent_id=jnp.array(agent_id, dtype=jnp.int32), rng_key=rng)
                return jnp.array(Actions.stay, dtype=jnp.int32), new_state

        # Determine inventory state
        is_empty = inv == DynamicObject.EMPTY
        is_plate = inv == DynamicObject.PLATE
        is_dish = bool(inv & DynamicObject.COOKED) and bool(inv & DynamicObject.PLATE)
        is_ingredient = ((inv >> 2) != 0) and ((inv & DynamicObject.PLATE) == 0)

        # Get pot states
        pot_mask, pot_non_full, pot_cooking, pot_cooked = self._get_pot_states_np(grid)

        # Compute ingredients being carried by all agents (treat as "in transit")
        carried_ingredients = np.zeros(4, dtype=np.int32)
        for i in range(num_agents):
            agent_inv = int(env_state.agents.inventory[i])
            # Check if carrying an ingredient (not plate, not empty)
            if (agent_inv >> 2) != 0 and (agent_inv & DynamicObject.PLATE) == 0:
                # Decode ingredient type from inventory
                ingredient_bits = agent_inv >> 2
                for ing_type in range(4):
                    if ingredient_bits & (0x3 << (ing_type * 2)):
                        carried_ingredients[ing_type] += 1
                        break  # Each agent carries only one ingredient

        # Priority 0: If holding a dish, deliver it
        if is_dish:
            reachable_goals = self._get_reachable_targets_np(self.goal_mask_np, pos_y, pos_x)
            target_y, target_x, exists = self._get_closest_target_np(reachable_goals, pos_y, pos_x)
            if exists:
                action = self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)
            else:
                action = self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)
            return self._return_action(action, agent_state, agent_id)

        # Priority 1: If holding ingredient and pot is ready for pickup, drop ingredient
        if is_ingredient and np.any(pot_cooked):
            empty_counters = self._get_empty_counter_mask_np(grid)
            reachable_counters = self._get_reachable_targets_np(empty_counters, pos_y, pos_x)
            target_y, target_x, exists = self._get_closest_target_np(reachable_counters, pos_y, pos_x)
            if exists:
                action = self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)
            else:
                action = self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)
            return self._return_action(action, agent_state, agent_id)

        # Priority 2: Yield to teammate if they're carrying something useful and we're blocking
        # This helps resolve deadlocks in tight layouts
        if agent_id > 0 and is_empty and len(other_agents_positions) > 0:
            teammate_inv = int(env_state.agents.inventory[0])  # Agent 0's inventory
            teammate_has_useful = ((teammate_inv >> 2) != 0) or (teammate_inv == DynamicObject.PLATE)
            
            if teammate_has_useful:
                # Check if we're adjacent to the teammate
                t_pos_y, t_pos_x = int(other_agents_positions[0][0]), int(other_agents_positions[0][1])
                dist_to_teammate = abs(pos_y - t_pos_y) + abs(pos_x - t_pos_x)
                
                if dist_to_teammate == 1:
                    # We're adjacent - check if we might be blocking their path
                    # Move away from teammate to give them space
                    action = self._yield_to_teammate_np(
                        pos_y, pos_x, agent_dir, t_pos_y, t_pos_x, other_agents_positions
                    )
                    return self._return_action(action, agent_state, agent_id)

        # Role-based action selection
        if role_mode == ROLE_INGREDIENT_RUNNER:
            action = self._ingredient_runner_action_np(
                pos_y, pos_x, agent_dir, inv, is_empty, is_ingredient,
                grid, pot_non_full, pot_cooked, teammate_y, teammate_x,
                other_agents_positions, agent_id, handoff_style, start_cook_bias, recipe,
                carried_ingredients
            )
        elif role_mode == ROLE_PLATER_DELIVERER:
            action = self._plater_deliverer_action_np(
                pos_y, pos_x, agent_dir, inv, is_empty, is_plate, is_ingredient,
                grid, pot_mask, pot_non_full, pot_cooking, pot_cooked,
                teammate_y, teammate_x, other_agents_positions, agent_id,
                handoff_style, plate_urgency, recipe
            )
        else:  # ROLE_FLEX
            any_cooked = np.any(pot_cooked)
            any_cooking = np.any(pot_cooking)
            should_plate = any_cooked or (any_cooking and plate_urgency > 0.5)

            if should_plate:
                action = self._plater_deliverer_action_np(
                    pos_y, pos_x, agent_dir, inv, is_empty, is_plate, is_ingredient,
                    grid, pot_mask, pot_non_full, pot_cooking, pot_cooked,
                    teammate_y, teammate_x, other_agents_positions, agent_id,
                    handoff_style, plate_urgency, recipe
                )
            else:
                action = self._ingredient_runner_action_np(
                    pos_y, pos_x, agent_dir, inv, is_empty, is_ingredient,
                    grid, pot_non_full, pot_cooked, teammate_y, teammate_x,
                    other_agents_positions, agent_id, handoff_style, start_cook_bias, recipe,
                    carried_ingredients
                )

        # Apply task abandonment: randomly drop held items
        if task_abandon_prob > 0 and not is_empty:
            abandon_roll = float(jax.random.uniform(abandon_key))
            if abandon_roll < task_abandon_prob:
                # Try to drop item on nearest counter (abandon current task)
                empty_counters = self._get_empty_counter_mask_np(grid)
                reachable_counters = self._get_reachable_targets_np(empty_counters, pos_y, pos_x)
                target_y, target_x, exists = self._get_closest_target_np(reachable_counters, pos_y, pos_x)
                if exists:
                    action = self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)
        
        # Apply wrong action: randomly take a different action
        if wrong_action_prob > 0:
            wrong_roll = float(jax.random.uniform(wrong_key))
            if wrong_roll < wrong_action_prob:
                # Pick a random movement action (0-3: right, down, left, up)
                rng, action_key = jax.random.split(rng, 2)
                wrong_action = int(jax.random.randint(action_key, (), 0, 4))
                action = wrong_action
        
        # Apply stubbornness: prefer to stay near certain fixed positions
        if stubbornness > 0 and is_empty:
            # Stubborn agents prefer to hang around specific spots (corners/edges)
            stubborn_y = (agent_id * 3) % self.height
            stubborn_x = (agent_id * 5) % self.width
            dist_to_stubborn = abs(pos_y - stubborn_y) + abs(pos_x - stubborn_x)
            
            rng, stubborn_key = jax.random.split(rng, 2)
            stubborn_roll = float(jax.random.uniform(stubborn_key))
            
            if dist_to_stubborn > 2 and stubborn_roll < stubbornness:
                # Move towards stubborn spot instead of doing useful work
                if self.walkable_mask_np[stubborn_y, stubborn_x]:
                    idx = self.pos_to_idx[pos_y, pos_x]
                    if idx >= 0:
                        action = self.next_action_matrix[idx, agent_dir, stubborn_y, stubborn_x]

        # Update RNG in agent state before returning
        new_state = AgentState(agent_id=jnp.array(agent_id, dtype=jnp.int32), rng_key=rng)
        return jnp.array(action, dtype=jnp.int32), new_state

    def _return_action(self, action: int, agent_state: AgentState, agent_id: int) -> Tuple[jnp.ndarray, AgentState]:
        """Return action and updated state as JAX arrays."""
        # Update RNG state
        rng, _ = jax.random.split(agent_state.rng_key)
        new_state = AgentState(agent_id=jnp.array(agent_id, dtype=jnp.int32), rng_key=rng)
        return jnp.array(action, dtype=jnp.int32), new_state

    def _ingredient_runner_action_np(
        self,
        pos_y: int, pos_x: int, agent_dir: int, inv: int,
        is_empty: bool, is_ingredient: bool,
        grid: np.ndarray, pot_non_full: np.ndarray, pot_cooked: np.ndarray,
        teammate_y: int, teammate_x: int,
        other_agents_positions: np.ndarray, agent_id: int,
        handoff_style: int, start_cook_bias: float, recipe: int,
        carried_ingredients: np.ndarray = None,
    ) -> int:
        """Compute action for INGREDIENT_RUNNER role.

        Pure role: ONLY fetches ingredients and puts them in pots.
        Does NOT handle plating or delivery - that's PLATER_DELIVERER's job.
        
        Args:
            carried_ingredients: Counts of ingredients being carried by all agents
        """
        is_plate = inv == DynamicObject.PLATE

        # If accidentally holding a plate, put it down on a counter
        if is_plate:
            empty_counters = self._get_empty_counter_mask_np(grid)
            reachable_counters = self._get_reachable_targets_np(empty_counters, pos_y, pos_x)
            target_y, target_x, exists = self._get_closest_target_np(reachable_counters, pos_y, pos_x)
            if exists:
                return self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)
            return self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)

        if is_ingredient:
            # Try to put ingredient in pot
            reachable_pots = self._get_reachable_targets_np(pot_non_full, pos_y, pos_x)
            target_y, target_x, exists = self._get_closest_target_np(reachable_pots, pos_y, pos_x)
            if exists:
                return self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)

            # No pot available - try handoff counter
            handoff_counters = self._get_handoff_counter_mask_np(grid, teammate_y, teammate_x, handoff_style)
            reachable_handoff = self._get_reachable_targets_np(handoff_counters, pos_y, pos_x)
            target_y, target_x, exists = self._get_closest_target_np(reachable_handoff, pos_y, pos_x)
            if exists:
                return self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)

            return self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)

        # Empty handed - get ingredients (this is our main job!)

        # Help start cooking if needed (still ingredient-related)
        if self.start_cooking_interaction and start_cook_bias > 0.5:
            ready_to_cook = self._should_start_cooking_np(grid, self.pot_mask_np)
            reachable_cook = self._get_reachable_targets_np(ready_to_cook, pos_y, pos_x)
            target_y, target_x, exists = self._get_closest_target_np(reachable_cook, pos_y, pos_x)
            if exists:
                return self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)

        # Get ingredient from pile - use recipe-aware selection (considers carried ingredients)
        ingredient_pile_mask = self._get_recipe_ingredient_pile_mask_np(grid, recipe, carried_ingredients)
        reachable_piles = self._get_reachable_targets_np(ingredient_pile_mask, pos_y, pos_x)
        target_y, target_x, exists = self._get_closest_target_np(reachable_piles, pos_y, pos_x)
        if exists:
            return self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)

        # Fallback: hover near pot to wait for it to become available
        pot_mask = self.pot_mask_np
        reachable_pots = self._get_reachable_targets_np(pot_mask, pos_y, pos_x)
        target_y, target_x, exists = self._get_closest_target_np(reachable_pots, pos_y, pos_x)
        if exists:
            return self._hover_near_target_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)

        # Fallback: hover near button recipe indicator (don't spam interact)
        reachable_buttons = self._get_reachable_targets_np(self.button_mask_np, pos_y, pos_x)
        target_y, target_x, exists = self._get_closest_target_np(reachable_buttons, pos_y, pos_x)
        if exists:
            return self._hover_near_target_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)

        # Fallback: hover near recipe indicator (don't spam interact)
        reachable_recipe = self._get_reachable_targets_np(self.recipe_indicator_mask_np, pos_y, pos_x)
        target_y, target_x, exists = self._get_closest_target_np(reachable_recipe, pos_y, pos_x)
        if exists:
            return self._hover_near_target_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)

        return self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)

    def _plater_deliverer_action_np(
        self,
        pos_y: int, pos_x: int, agent_dir: int, inv: int,
        is_empty: bool, is_plate: bool, is_ingredient: bool,
        grid: np.ndarray, pot_mask: np.ndarray, pot_non_full: np.ndarray,
        pot_cooking: np.ndarray, pot_cooked: np.ndarray,
        teammate_y: int, teammate_x: int,
        other_agents_positions: np.ndarray, agent_id: int,
        handoff_style: int, plate_urgency: float, recipe: int,
    ) -> int:
        """Compute action for PLATER_DELIVERER role.

        Pure role: ONLY fetches plates, picks up cooked soup, and delivers dishes.
        Does NOT handle ingredient gathering - that's INGREDIENT_RUNNER's job.
        """
        if is_plate:
            # Try to pick up cooked soup (our main job!)
            reachable_cooked = self._get_reachable_targets_np(pot_cooked, pos_y, pos_x)
            target_y, target_x, exists = self._get_closest_target_np(reachable_cooked, pos_y, pos_x)
            if exists:
                return self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)

            # Hover near cooking/active pot waiting for soup
            pot_with_stuff = pot_cooking | (pot_mask & (grid[:, :, 1] != 0))
            reachable_active = self._get_reachable_targets_np(pot_with_stuff, pos_y, pos_x)
            target_y, target_x, exists = self._get_closest_target_np(reachable_active, pos_y, pos_x)
            if exists:
                return self._hover_near_pot_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)

            # Hover near any pot waiting for ingredients
            reachable_pots = self._get_reachable_targets_np(pot_mask, pos_y, pos_x)
            target_y, target_x, exists = self._get_closest_target_np(reachable_pots, pos_y, pos_x)
            if exists:
                return self._hover_near_target_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)

            # Fallback: hover near button
            reachable_buttons = self._get_reachable_targets_np(self.button_mask_np, pos_y, pos_x)
            target_y, target_x, exists = self._get_closest_target_np(reachable_buttons, pos_y, pos_x)
            if exists:
                return self._hover_near_target_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)

            return self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)

        if is_ingredient:
            # If accidentally holding an ingredient, put it down on a counter
            empty_counters = self._get_empty_counter_mask_np(grid)
            reachable_counters = self._get_reachable_targets_np(empty_counters, pos_y, pos_x)
            target_y, target_x, exists = self._get_closest_target_np(reachable_counters, pos_y, pos_x)
            if exists:
                return self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)
            return self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)

        # Empty handed - get a plate (this is our main job!)
        reachable_plates = self._get_reachable_targets_np(self.plate_pile_mask_np, pos_y, pos_x)
        target_y, target_x, exists = self._get_closest_target_np(reachable_plates, pos_y, pos_x)
        if exists:
            return self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)

        # Fallback: hover near pot waiting for delivery opportunity
        pot_with_stuff = pot_cooking | pot_cooked | (pot_mask & (grid[:, :, 1] != 0))
        reachable_active = self._get_reachable_targets_np(pot_with_stuff, pos_y, pos_x)
        target_y, target_x, exists = self._get_closest_target_np(reachable_active, pos_y, pos_x)
        if exists:
            return self._hover_near_target_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)

        # Fallback: hover near button recipe indicator (don't spam interact)
        reachable_buttons = self._get_reachable_targets_np(self.button_mask_np, pos_y, pos_x)
        target_y, target_x, exists = self._get_closest_target_np(reachable_buttons, pos_y, pos_x)
        if exists:
            return self._hover_near_target_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)

        # Fallback: hover near recipe indicator (don't spam interact)
        reachable_recipe = self._get_reachable_targets_np(self.recipe_indicator_mask_np, pos_y, pos_x)
        target_y, target_x, exists = self._get_closest_target_np(reachable_recipe, pos_y, pos_x)
        if exists:
            return self._hover_near_target_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)

        return self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)

    def _hover_near_pot_np(
        self,
        pos_y: int, pos_x: int, agent_dir: int,
        pot_y: int, pot_x: int,
        other_agents_positions: np.ndarray, agent_id: int,
    ) -> int:
        """Move adjacent to pot and wait (face it but don't interact)."""
        dy = pot_y - pos_y
        dx = pot_x - pos_x
        dist = abs(dy) + abs(dx)

        if dist == 1:
            # Already adjacent - face the pot but stay
            if dy == -1:
                required_dir = Direction.UP
            elif dy == 1:
                required_dir = Direction.DOWN
            elif dx == 1:
                required_dir = Direction.RIGHT
            else:
                required_dir = Direction.LEFT

            if agent_dir == required_dir:
                return Actions.stay
            else:
                dir_to_action = [3, 1, 0, 2]  # UP, DOWN, RIGHT, LEFT -> actions
                return dir_to_action[required_dir]

        # Move towards pot
        return self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, pot_y, pot_x, other_agents_positions, agent_id)

    def _hover_near_target_np(
        self,
        pos_y: int, pos_x: int, agent_dir: int,
        target_y: int, target_x: int,
        other_agents_positions: np.ndarray, agent_id: int,
    ) -> int:
        """Move adjacent to target and wait (face it but don't interact).

        Alias for _hover_near_pot_np that works with any target.
        """
        return self._hover_near_pot_np(
            pos_y, pos_x, agent_dir, target_y, target_x,
            other_agents_positions, agent_id
        )
