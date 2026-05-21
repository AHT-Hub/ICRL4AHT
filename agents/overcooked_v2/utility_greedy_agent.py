"""Utility Greedy Agent for OvercookedV2 - Fast NumPy Implementation.

This agent implements a "rational opportunist" that enumerates a fixed set of
candidate intents each step, scores them with a weighted utility, picks the best,
and executes it. Unlike role-split or territory-locked agents, this agent chooses
among micro-goals based on a weighted utility function.

All pathfinding and decision logic is precomputed at initialization using NumPy,
making runtime execution extremely fast (simple array lookups).

The agent is JAX-compatible for training - it returns JAX arrays but all internal
computation uses NumPy for maximum speed.
"""

import hashlib
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
    StaticObject,
)

from .base_agent import AgentState, BaseAgentV2


# Intent constants (fixed K=8 intents)
INTENT_DELIVER_DISH = 0
INTENT_PICKUP_COOKED = 1
INTENT_GET_PLATE = 2
INTENT_ADD_INGREDIENT = 3
INTENT_FETCH_INGREDIENT = 4
INTENT_STAGE_ON_COUNTER = 5
INTENT_START_COOKING = 6
INTENT_PRESS_L = 7
NUM_INTENTS = 8

# Cache directory for precomputed data
CACHE_DIR = Path(__file__).parent / ".utility_greedy_cache"


@chex.dataclass
class UtilityAgentState:
    """JAX-compatible agent state for UtilityGreedyAgent.

    Attributes:
        agent_id: Agent identifier
        rng_key: JAX random key for stochastic decisions
        last_intent: Last chosen intent (for inertia bonus)
        wait_counter: Counter for yielding to other agents (collision avoidance)
    """
    agent_id: jnp.ndarray  # int32 scalar
    rng_key: chex.PRNGKey
    last_intent: jnp.ndarray  # int32 scalar, -1 if no previous intent
    wait_counter: jnp.ndarray  # int32 scalar, counts down when yielding


@flax.struct.dataclass
class UtilityGreedyTheta:
    """Hyperparameters for the UtilityGreedy agent family.

    All fields are JAX arrays to ensure compatibility with tracing.
    """
    # Intent weights (higher = more valuable)
    w_deliver: jnp.ndarray        # float32: weight for delivering dish
    w_pickup_cooked: jnp.ndarray  # float32: weight for picking up cooked soup
    w_get_plate: jnp.ndarray      # float32: weight for getting a plate
    w_add_ingredient: jnp.ndarray # float32: weight for adding ingredient to pot
    w_fetch_ingredient: jnp.ndarray  # float32: weight for fetching ingredient
    w_stage_on_counter: jnp.ndarray  # float32: weight for staging on counter
    w_start_cooking: jnp.ndarray  # float32: weight for starting cooking
    w_press_L: jnp.ndarray        # float32: weight for pressing L button

    # Utility modifiers
    dist_weight: jnp.ndarray      # float32: penalty per distance unit
    inertia: jnp.ndarray          # float32 [0,1]: bonus for keeping same intent
    counter_preference: jnp.ndarray  # float32: preference for counter staging

    @classmethod
    def default(cls) -> "UtilityGreedyTheta":
        """Create default hyperparameters with balanced weights."""
        return cls(
            w_deliver=jnp.array(10.0, dtype=jnp.float32),
            w_pickup_cooked=jnp.array(8.0, dtype=jnp.float32),
            w_get_plate=jnp.array(5.0, dtype=jnp.float32),
            w_add_ingredient=jnp.array(6.0, dtype=jnp.float32),
            w_fetch_ingredient=jnp.array(4.0, dtype=jnp.float32),
            w_stage_on_counter=jnp.array(2.0, dtype=jnp.float32),
            w_start_cooking=jnp.array(7.0, dtype=jnp.float32),
            w_press_L=jnp.array(3.0, dtype=jnp.float32),
            dist_weight=jnp.array(0.5, dtype=jnp.float32),
            inertia=jnp.array(0.3, dtype=jnp.float32),
            counter_preference=jnp.array(0.5, dtype=jnp.float32),
        )

    @classmethod
    def delivery_focused(cls) -> "UtilityGreedyTheta":
        """Hyperparameters that prioritize delivery and plating."""
        return cls(
            w_deliver=jnp.array(15.0, dtype=jnp.float32),
            w_pickup_cooked=jnp.array(12.0, dtype=jnp.float32),
            w_get_plate=jnp.array(8.0, dtype=jnp.float32),
            w_add_ingredient=jnp.array(3.0, dtype=jnp.float32),
            w_fetch_ingredient=jnp.array(2.0, dtype=jnp.float32),
            w_stage_on_counter=jnp.array(1.0, dtype=jnp.float32),
            w_start_cooking=jnp.array(4.0, dtype=jnp.float32),
            w_press_L=jnp.array(2.0, dtype=jnp.float32),
            dist_weight=jnp.array(0.3, dtype=jnp.float32),
            inertia=jnp.array(0.4, dtype=jnp.float32),
            counter_preference=jnp.array(0.3, dtype=jnp.float32),
        )

    @classmethod
    def prep_focused(cls) -> "UtilityGreedyTheta":
        """Hyperparameters that prioritize ingredient prep and cooking."""
        return cls(
            w_deliver=jnp.array(8.0, dtype=jnp.float32),
            w_pickup_cooked=jnp.array(5.0, dtype=jnp.float32),
            w_get_plate=jnp.array(3.0, dtype=jnp.float32),
            w_add_ingredient=jnp.array(10.0, dtype=jnp.float32),
            w_fetch_ingredient=jnp.array(8.0, dtype=jnp.float32),
            w_stage_on_counter=jnp.array(4.0, dtype=jnp.float32),
            w_start_cooking=jnp.array(12.0, dtype=jnp.float32),
            w_press_L=jnp.array(5.0, dtype=jnp.float32),
            dist_weight=jnp.array(0.4, dtype=jnp.float32),
            inertia=jnp.array(0.5, dtype=jnp.float32),
            counter_preference=jnp.array(0.7, dtype=jnp.float32),
        )

    @classmethod
    def high_inertia(cls) -> "UtilityGreedyTheta":
        """Hyperparameters with high inertia to reduce oscillation."""
        return cls(
            w_deliver=jnp.array(10.0, dtype=jnp.float32),
            w_pickup_cooked=jnp.array(8.0, dtype=jnp.float32),
            w_get_plate=jnp.array(5.0, dtype=jnp.float32),
            w_add_ingredient=jnp.array(6.0, dtype=jnp.float32),
            w_fetch_ingredient=jnp.array(4.0, dtype=jnp.float32),
            w_stage_on_counter=jnp.array(2.0, dtype=jnp.float32),
            w_start_cooking=jnp.array(7.0, dtype=jnp.float32),
            w_press_L=jnp.array(3.0, dtype=jnp.float32),
            dist_weight=jnp.array(0.5, dtype=jnp.float32),
            inertia=jnp.array(0.7, dtype=jnp.float32),
            counter_preference=jnp.array(0.5, dtype=jnp.float32),
        )

    @classmethod
    def distance_sensitive(cls) -> "UtilityGreedyTheta":
        """Hyperparameters that heavily penalize distance."""
        return cls(
            w_deliver=jnp.array(10.0, dtype=jnp.float32),
            w_pickup_cooked=jnp.array(8.0, dtype=jnp.float32),
            w_get_plate=jnp.array(5.0, dtype=jnp.float32),
            w_add_ingredient=jnp.array(6.0, dtype=jnp.float32),
            w_fetch_ingredient=jnp.array(4.0, dtype=jnp.float32),
            w_stage_on_counter=jnp.array(2.0, dtype=jnp.float32),
            w_start_cooking=jnp.array(7.0, dtype=jnp.float32),
            w_press_L=jnp.array(3.0, dtype=jnp.float32),
            dist_weight=jnp.array(1.0, dtype=jnp.float32),
            inertia=jnp.array(0.3, dtype=jnp.float32),
            counter_preference=jnp.array(0.5, dtype=jnp.float32),
        )


class UtilityGreedyAgentV2(BaseAgentV2):
    """Fast NumPy-based utility greedy agent.

    All pathfinding is precomputed at initialization. Runtime execution uses
    pure NumPy operations for maximum speed.
    """

    def __init__(
        self,
        layout,
        theta: UtilityGreedyTheta = None,
        start_cooking_interaction: bool = False,
    ):
        """Initialize the utility greedy agent.

        Args:
            layout: Layout object containing the static grid configuration.
            theta: Hyperparameters for the agent. If None, uses default.
            start_cooking_interaction: Whether the environment requires explicit
                interaction to start cooking (vs auto-cooking when pot is full).
        """
        super().__init__(layout)

        self.theta = theta if theta is not None else UtilityGreedyTheta.default()
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
        self.has_button = np.any(self.button_mask_np)

        # Precompute pot-adjacent counter mask
        self._precompute_pot_adjacent_counters_np()

        # Load or compute pathfinding data
        self._load_or_compute_pathfinding()

        # Convert theta weights to numpy for fast scoring
        self.intent_weights_np = np.array([
            float(self.theta.w_deliver),
            float(self.theta.w_pickup_cooked),
            float(self.theta.w_get_plate),
            float(self.theta.w_add_ingredient),
            float(self.theta.w_fetch_ingredient),
            float(self.theta.w_stage_on_counter),
            float(self.theta.w_start_cooking),
            float(self.theta.w_press_L),
        ], dtype=np.float32)

        self.dist_weight_np = float(self.theta.dist_weight)
        self.inertia_np = float(self.theta.inertia)
        self.counter_preference_np = float(self.theta.counter_preference)

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

        # Compute distance matrix using BFS from each position
        INF = 10000
        self.dist_matrix = np.full((num_positions, self.height, self.width), INF, dtype=np.int32)

        # next_action_matrix[from_idx, from_dir, to_y, to_x] = best action to reach (to_y, to_x)
        self.next_action_matrix = np.full((num_positions, 4, self.height, self.width), Actions.stay, dtype=np.int32)

        for start_idx in range(num_positions):
            start_y, start_x = walkable_positions[start_idx]

            # BFS to compute distances
            dist = np.full((self.height, self.width), INF, dtype=np.int32)
            dist[start_y, start_x] = 0

            # Track the first step direction for each cell
            first_step = np.full((self.height, self.width), -1, dtype=np.int32)

            queue = [(start_y, start_x)]
            head = 0

            while head < len(queue):
                cy, cx = queue[head]
                head += 1
                curr_dist = dist[cy, cx]

                # Try all 4 directions: 0=right, 1=down, 2=left, 3=up
                for action, (dy, dx) in enumerate([(0, 1), (1, 0), (0, -1), (-1, 0)]):
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
                            self.next_action_matrix[start_idx, start_dir, ty, tx] = Actions.stay
                        elif dist[ty, tx] < INF:
                            fs = first_step[ty, tx]
                            if fs >= 0:
                                self.next_action_matrix[start_idx, start_dir, ty, tx] = fs

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

    # -------------------------------------------------------------------------
    # Agent state management
    # -------------------------------------------------------------------------

    def init_agent_state(self, agent_id: int) -> UtilityAgentState:
        """Initialize agent state with no previous intent."""
        return UtilityAgentState(
            agent_id=jnp.array(agent_id, dtype=jnp.int32),
            rng_key=jax.random.PRNGKey(agent_id),
            last_intent=jnp.array(-1, dtype=jnp.int32),
            wait_counter=jnp.array(0, dtype=jnp.int32),
        )

    # -------------------------------------------------------------------------
    # NumPy helper methods
    # -------------------------------------------------------------------------

    def _get_reachable_targets_np(self, target_mask: np.ndarray, pos_y: int, pos_x: int) -> np.ndarray:
        """Get targets that are adjacent to reachable walkable cells."""
        idx = self.pos_to_idx[pos_y, pos_x]
        if idx < 0:
            return np.zeros_like(target_mask, dtype=np.bool_)
        return target_mask & self.adjacent_to_reachable[idx]

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

    def _get_min_distance_to_target_np(self, target_mask: np.ndarray, pos_y: int, pos_x: int) -> float:
        """Get minimum distance to any target in the mask."""
        idx = self.pos_to_idx[pos_y, pos_x]
        if idx < 0:
            return 100.0

        if not np.any(target_mask):
            return 100.0

        target_positions = np.argwhere(target_mask)
        best_dist = 10000

        for ty, tx in target_positions:
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ay, ax = ty + dy, tx + dx
                if 0 <= ay < self.height and 0 <= ax < self.width:
                    if self.walkable_mask_np[ay, ax]:
                        dist = self.dist_matrix[idx, ay, ax]
                        if dist < best_dist:
                            best_dist = dist

        return float(best_dist) if best_dist < 10000 else 100.0

    def _count_ingredients_np(self, content: int) -> int:
        """Count ingredients in a pot content encoding."""
        content = content >> 2  # Skip flag bits
        count = 0
        while content > 0:
            count += content & 0x3
            content >>= 2
        return count

    def _get_pot_states_np(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Analyze pot states using NumPy.

        Returns:
            - pot_mask: Which cells are pots
            - pot_non_full_mask: Pots that can accept more ingredients
            - pot_cooking_mask: Pots currently cooking
            - pot_cooked_mask: Pots with finished soup
            - pot_ready_to_start: Pots full and ready to start cooking
            - ingredient_counts: Ingredient count per cell
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
        pot_ready_to_start = is_pot & pot_full & ~pot_cooking & ~pot_cooked

        return is_pot, pot_non_full, pot_cooking, pot_cooked, pot_ready_to_start, ingredient_counts

    def _get_empty_counter_mask_np(self, grid: np.ndarray) -> np.ndarray:
        """Get mask of empty counters."""
        is_counter = self.counter_mask_np
        is_empty = grid[:, :, 1] == DynamicObject.EMPTY
        return is_counter & is_empty

    def _get_button_active_np(self, grid: np.ndarray) -> bool:
        """Check if button is already pressed (recipe is visible)."""
        active_mask = (grid[:, :, 2] > 0) & self.button_mask_np
        return bool(np.any(active_mask))

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

    def _get_recipe_ingredient_pile_mask_np(self, grid: np.ndarray, recipe: int) -> np.ndarray:
        """Get mask for ingredient piles that match the recipe.

        Only returns piles for ingredients that are still needed based on
        what's already in the pot vs what the recipe requires.
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
                # Turn to face target: UP->3, DOWN->1, RIGHT->0, LEFT->2
                dir_to_action = [3, 1, 0, 2]
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
        return self.next_action_matrix[pos_idx, agent_dir, best_cell[0], best_cell[1]]

    def _wander_action_np(
        self,
        pos_y: int, pos_x: int,
        agent_dir: int,
        other_agents_positions: np.ndarray,
    ) -> int:
        """Move to avoid blocking, preferring open spaces."""
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

    # -------------------------------------------------------------------------
    # Intent scoring
    # -------------------------------------------------------------------------

    def _score_all_intents_np(
        self,
        grid: np.ndarray,
        pos_y: int, pos_x: int,
        inv: int,
        last_intent: int,
        recipe: int = 0,
    ) -> np.ndarray:
        """Score all intents using NumPy.

        Returns array of shape (NUM_INTENTS,) with scores.
        Invalid intents get -inf.
        """
        # Get pot states
        _, pot_non_full, pot_cooking, pot_cooked, pot_ready_to_start, ingredient_counts = self._get_pot_states_np(grid)
        empty_counters = self._get_empty_counter_mask_np(grid)

        # Inventory flags
        holding_dish = bool(inv & DynamicObject.COOKED) and bool(inv & DynamicObject.PLATE)
        holding_plate = inv == DynamicObject.PLATE
        holding_ingredient = ((inv >> 2) != 0) and ((inv & DynamicObject.PLATE) == 0)
        empty_handed = inv == DynamicObject.EMPTY

        # Check if any pot is cooking or cooked (active)
        any_pot_cooking = np.any(pot_cooking)
        any_pot_cooked = np.any(pot_cooked)
        any_pot_active = any_pot_cooking or any_pot_cooked

        # Compute scores for each intent
        scores = np.full(NUM_INTENTS, -np.inf, dtype=np.float32)

        # --- INTENT_DELIVER_DISH ---
        if holding_dish:
            reachable = self._get_reachable_targets_np(self.goal_mask_np, pos_y, pos_x)
            if np.any(reachable):
                dist = self._get_min_distance_to_target_np(reachable, pos_y, pos_x)
                num_cooked = np.sum(pot_cooked)
                value = 1.0 + num_cooked * 0.5
                scores[INTENT_DELIVER_DISH] = self.intent_weights_np[INTENT_DELIVER_DISH] * value - self.dist_weight_np * dist

        # --- INTENT_PICKUP_COOKED ---
        if holding_plate:
            reachable = self._get_reachable_targets_np(pot_cooked, pos_y, pos_x)
            if np.any(reachable):
                dist = self._get_min_distance_to_target_np(reachable, pos_y, pos_x)
                num_cooked = np.sum(pot_cooked)
                value = 1.0 + num_cooked * 0.5
                scores[INTENT_PICKUP_COOKED] = self.intent_weights_np[INTENT_PICKUP_COOKED] * value - self.dist_weight_np * dist
            elif any_pot_cooking:
                # Pot is cooking but not done - give a moderate score to hover/wait
                # Use the cooking pot as target to move towards
                reachable_cooking = self._get_reachable_targets_np(pot_cooking, pos_y, pos_x)
                if np.any(reachable_cooking):
                    dist = self._get_min_distance_to_target_np(reachable_cooking, pos_y, pos_x)
                    # Give a high score to wait near the pot
                    value = 1.5  # High value for waiting
                    scores[INTENT_PICKUP_COOKED] = self.intent_weights_np[INTENT_PICKUP_COOKED] * value - self.dist_weight_np * dist

        # --- INTENT_GET_PLATE ---
        if empty_handed:
            reachable = self._get_reachable_targets_np(self.plate_pile_mask_np, pos_y, pos_x)
            if np.any(reachable):
                dist = self._get_min_distance_to_target_np(reachable, pos_y, pos_x)
                num_cooking = np.sum(pot_cooking)
                num_cooked = np.sum(pot_cooked)
                num_non_full = np.sum(pot_non_full)
                # Only get plate when pot is active (cooking or cooked) or nearly full
                # Otherwise prioritize fetching ingredients
                if any_pot_active:
                    value = 1.0 + num_cooking * 0.3 + num_cooked * 0.5
                elif num_non_full == 0:
                    # All pots are full but not cooking yet - still might want a plate
                    value = 0.8
                else:
                    value = 0.2  # Low priority when we should be filling pots
                scores[INTENT_GET_PLATE] = self.intent_weights_np[INTENT_GET_PLATE] * value - self.dist_weight_np * dist

        # --- INTENT_ADD_INGREDIENT ---
        if holding_ingredient:
            reachable = self._get_reachable_targets_np(pot_non_full, pos_y, pos_x)
            if np.any(reachable):
                dist = self._get_min_distance_to_target_np(reachable, pos_y, pos_x)
                max_ingredients = np.max(np.where(self.pot_mask_np, ingredient_counts, 0))
                value = 1.0 + (3 - max_ingredients) * 0.2
                scores[INTENT_ADD_INGREDIENT] = self.intent_weights_np[INTENT_ADD_INGREDIENT] * value - self.dist_weight_np * dist

        # --- INTENT_FETCH_INGREDIENT ---
        if empty_handed:
            # Check if we can add to pot OR stage on counter (for layouts with separated agents)
            num_non_full = np.sum(pot_non_full)
            can_add = self._get_reachable_targets_np(pot_non_full, pos_y, pos_x)
            can_stage = np.any(self._get_reachable_targets_np(empty_counters, pos_y, pos_x))

            # Allow fetch if we can add to pot OR stage on counter
            if np.any(can_add) or can_stage:
                # Use recipe-aware ingredient selection
                ingredient_pile_mask = self._get_recipe_ingredient_pile_mask_np(grid, recipe)
                if not np.any(ingredient_pile_mask):
                    # No ingredients needed for recipe - DON'T fall back to all ingredients
                    # This happens when pot is already full with correct recipe
                    pass  # Leave score as -inf
                else:
                    reachable = self._get_reachable_targets_np(ingredient_pile_mask, pos_y, pos_x)
                    if np.any(reachable):
                        dist = self._get_min_distance_to_target_np(reachable, pos_y, pos_x)
                        # Reduce value if we can only stage (not add directly)
                        if np.any(can_add):
                            value = 1.0 + num_non_full * 0.2
                        else:
                            value = 0.8  # Slightly lower for staging-only scenario
                        scores[INTENT_FETCH_INGREDIENT] = self.intent_weights_np[INTENT_FETCH_INGREDIENT] * value - self.dist_weight_np * dist
                    else:
                        # Recipe ingredients not reachable from here
                        # Only fetch ingredients of types that the recipe requires
                        # (to avoid adding wrong ingredients and getting penalties)
                        recipe_counts = self._decode_recipe(recipe)
                        static_layer = grid[:, :, 0]
                        recipe_type_pile_mask = np.zeros((self.height, self.width), dtype=np.bool_)
                        for i in range(4):
                            if recipe_counts[i] > 0:  # Recipe requires this type
                                pile_type = StaticObject.INGREDIENT_PILE_BASE + i
                                recipe_type_pile_mask |= (static_layer == pile_type)

                        recipe_pile_reachable = self._get_reachable_targets_np(recipe_type_pile_mask, pos_y, pos_x)
                        if np.any(recipe_pile_reachable):
                            dist = self._get_min_distance_to_target_np(recipe_pile_reachable, pos_y, pos_x)
                            # Lower value since we may have some of this type already
                            if np.any(can_add):
                                value = 0.4 + num_non_full * 0.05
                            else:
                                # Even lower for staging-only (very low priority)
                                value = 0.15
                            scores[INTENT_FETCH_INGREDIENT] = self.intent_weights_np[INTENT_FETCH_INGREDIENT] * value - self.dist_weight_np * dist
                        # If no recipe-required ingredient types are reachable, don't fetch at all
                        # (fetching wrong ingredients causes -20 penalty)

        # --- INTENT_STAGE_ON_COUNTER ---
        # Only stage if we have no better options (pots full or unreachable)
        if (holding_ingredient or holding_plate) and not holding_dish:
            # Check if we actually need to stage
            should_stage = False
            if holding_ingredient:
                # Stage ingredient only if no pots are available to add to
                reachable_pots = self._get_reachable_targets_np(pot_non_full, pos_y, pos_x)
                should_stage = not np.any(reachable_pots)
            elif holding_plate:
                # Stage plate only if no active pots to wait for
                # AND no non-full pots (meaning we shouldn't have gotten the plate)
                num_non_full = np.sum(pot_non_full)
                if not any_pot_active and num_non_full == 0:
                    should_stage = True  # Nothing cooking and all pots full

            if should_stage:
                pot_adj_counters = empty_counters & self.pot_adjacent_counter_mask_np
                use_pot_adjacent = np.any(pot_adj_counters) and self.counter_preference_np > 0.5
                target = pot_adj_counters if use_pot_adjacent else empty_counters

                reachable = self._get_reachable_targets_np(target, pos_y, pos_x)
                if np.any(reachable):
                    dist = self._get_min_distance_to_target_np(reachable, pos_y, pos_x)
                    num_non_full = np.sum(pot_non_full)
                    value = 0.5 + num_non_full * 0.1
                    scores[INTENT_STAGE_ON_COUNTER] = self.intent_weights_np[INTENT_STAGE_ON_COUNTER] * value - self.dist_weight_np * dist

        # --- INTENT_START_COOKING ---
        if self.start_cooking_interaction and empty_handed:
            reachable = self._get_reachable_targets_np(pot_ready_to_start, pos_y, pos_x)
            if np.any(reachable):
                dist = self._get_min_distance_to_target_np(reachable, pos_y, pos_x)
                value = 1.5
                scores[INTENT_START_COOKING] = self.intent_weights_np[INTENT_START_COOKING] * value - self.dist_weight_np * dist

        # --- INTENT_PRESS_L ---
        # Skip entirely if w_press_L <= 0 (agent has direct state access and doesn't need to press L)
        if self.has_button and empty_handed and self.intent_weights_np[INTENT_PRESS_L] > 0:
            button_active = self._get_button_active_np(grid)
            if not button_active:
                reachable = self._get_reachable_targets_np(self.button_mask_np, pos_y, pos_x)
                if np.any(reachable):
                    dist = self._get_min_distance_to_target_np(reachable, pos_y, pos_x)
                    value = 1.0
                    scores[INTENT_PRESS_L] = self.intent_weights_np[INTENT_PRESS_L] * value - self.dist_weight_np * dist

        # Add inertia bonus
        if last_intent >= 0 and last_intent < NUM_INTENTS:
            if np.isfinite(scores[last_intent]):
                scores[last_intent] += self.inertia_np * 2.0

        return scores

    # -------------------------------------------------------------------------
    # Intent execution
    # -------------------------------------------------------------------------

    def _execute_intent_np(
        self,
        intent: int,
        grid: np.ndarray,
        pos_y: int, pos_x: int,
        agent_dir: int,
        inv: int,
        other_agents_positions: np.ndarray,
        agent_id: int,
        recipe: int = 0,
    ) -> int:
        """Execute the chosen intent by computing the appropriate action."""
        # Get pot states for target masks
        _, pot_non_full, pot_cooking, pot_cooked, pot_ready_to_start, _ = self._get_pot_states_np(grid)
        empty_counters = self._get_empty_counter_mask_np(grid)

        # Determine target mask based on intent
        if intent == INTENT_DELIVER_DISH:
            target_mask = self._get_reachable_targets_np(self.goal_mask_np, pos_y, pos_x)
        elif intent == INTENT_PICKUP_COOKED:
            # First try cooked pots, then cooking pots (to hover near)
            target_mask = self._get_reachable_targets_np(pot_cooked, pos_y, pos_x)
            if not np.any(target_mask):
                # Hover near cooking pot
                target_mask = self._get_reachable_targets_np(pot_cooking, pos_y, pos_x)
                if np.any(target_mask):
                    # For cooking pot, move adjacent but don't interact
                    target_y, target_x, exists = self._get_closest_target_np(target_mask, pos_y, pos_x)
                    if exists:
                        return self._hover_near_target_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)
        elif intent == INTENT_GET_PLATE:
            target_mask = self._get_reachable_targets_np(self.plate_pile_mask_np, pos_y, pos_x)
        elif intent == INTENT_ADD_INGREDIENT:
            target_mask = self._get_reachable_targets_np(pot_non_full, pos_y, pos_x)
        elif intent == INTENT_FETCH_INGREDIENT:
            # Use recipe-aware ingredient selection with fallback to recipe-type piles
            ingredient_pile_mask = self._get_recipe_ingredient_pile_mask_np(grid, recipe)
            if not np.any(ingredient_pile_mask):
                # No ingredients needed for recipe - wander instead
                return self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)
            target_mask = self._get_reachable_targets_np(ingredient_pile_mask, pos_y, pos_x)
            if not np.any(target_mask):
                # Recipe ingredients not reachable, fall back to piles of recipe-required types
                recipe_counts = self._decode_recipe(recipe)
                static_layer = grid[:, :, 0]
                recipe_type_pile_mask = np.zeros((self.height, self.width), dtype=np.bool_)
                for i in range(4):
                    if recipe_counts[i] > 0:  # Recipe requires this type
                        pile_type = StaticObject.INGREDIENT_PILE_BASE + i
                        recipe_type_pile_mask |= (static_layer == pile_type)
                target_mask = self._get_reachable_targets_np(recipe_type_pile_mask, pos_y, pos_x)
        elif intent == INTENT_STAGE_ON_COUNTER:
            pot_adj_counters = empty_counters & self.pot_adjacent_counter_mask_np
            use_pot_adjacent = np.any(pot_adj_counters) and self.counter_preference_np > 0.5
            target = pot_adj_counters if use_pot_adjacent else empty_counters
            target_mask = self._get_reachable_targets_np(target, pos_y, pos_x)
        elif intent == INTENT_START_COOKING:
            target_mask = self._get_reachable_targets_np(pot_ready_to_start, pos_y, pos_x)
        elif intent == INTENT_PRESS_L:
            target_mask = self._get_reachable_targets_np(self.button_mask_np, pos_y, pos_x)
        else:
            return Actions.stay

        # Get closest target
        target_y, target_x, exists = self._get_closest_target_np(target_mask, pos_y, pos_x)
        if not exists:
            return self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)

        # Get action to target
        return self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)

    def _hover_near_target_np(
        self,
        pos_y: int, pos_x: int,
        agent_dir: int,
        target_y: int, target_x: int,
        other_agents_positions: np.ndarray,
        agent_id: int,
    ) -> int:
        """Move adjacent to target and wait (face it but don't interact)."""
        dy = target_y - pos_y
        dx = target_x - pos_x
        dist = abs(dy) + abs(dx)

        if dist == 1:
            # Already adjacent - face the target but stay
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

        # Move towards target
        return self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def _get_move_target(self, action: int, pos_y: int, pos_x: int) -> Tuple[int, int]:
        """Get the target position for a movement action."""
        if action == Actions.right:
            return pos_y, pos_x + 1
        elif action == Actions.down:
            return pos_y + 1, pos_x
        elif action == Actions.left:
            return pos_y, pos_x - 1
        elif action == Actions.up:
            return pos_y - 1, pos_x
        else:
            return pos_y, pos_x  # stay or interact

    def _is_contested_move(
        self,
        action: int,
        pos_y: int, pos_x: int,
        other_agents_positions: np.ndarray,
    ) -> bool:
        """Check if this move targets a position another agent might also target.

        A position is contested if:
        1. It's a movement action (not stay/interact)
        2. Another agent is adjacent to the target position
        """
        if action == Actions.stay or action == Actions.interact:
            return False

        target_y, target_x = self._get_move_target(action, pos_y, pos_x)

        # Check if any other agent is adjacent to the target (within 1 step)
        for other_pos in other_agents_positions:
            oy, ox = other_pos[0], other_pos[1]
            dist = abs(oy - target_y) + abs(ox - target_x)
            if dist <= 1:  # Other agent is at or adjacent to target
                return True
        return False

    def get_action(
        self,
        obs,
        env_state,
        agent_state: UtilityAgentState = None,
    ) -> Tuple[jnp.ndarray, UtilityAgentState]:
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
        last_intent = int(agent_state.last_intent)
        wait_counter = int(agent_state.wait_counter)

        # Extract state as numpy arrays
        pos_x = int(env_state.agents.pos.x[agent_id])
        pos_y = int(env_state.agents.pos.y[agent_id])
        agent_dir = int(env_state.agents.dir[agent_id])
        inv = int(env_state.agents.inventory[agent_id])

        # Get other agent positions
        num_agents = env_state.agents.pos.x.shape[0]
        other_agents_positions = []
        for i in range(num_agents):
            if i != agent_id:
                other_agents_positions.append([
                    int(env_state.agents.pos.y[i]),
                    int(env_state.agents.pos.x[i])
                ])
        other_agents_positions = np.array(other_agents_positions) if other_agents_positions else np.zeros((0, 2), dtype=np.int32)

        # Convert grid to numpy
        grid = np.array(env_state.grid)

        # Get recipe
        recipe = int(env_state.recipe) if hasattr(env_state, 'recipe') else 0

        # If waiting, decrement counter and stay
        if wait_counter > 0:
            rng, _ = jax.random.split(agent_state.rng_key)
            new_state = UtilityAgentState(
                agent_id=jnp.array(agent_id, dtype=jnp.int32),
                rng_key=rng,
                last_intent=jnp.array(last_intent, dtype=jnp.int32),
                wait_counter=jnp.array(wait_counter - 1, dtype=jnp.int32),
            )
            return jnp.array(Actions.stay, dtype=jnp.int32), new_state

        # Score all intents
        scores = self._score_all_intents_np(grid, pos_y, pos_x, inv, last_intent, recipe)

        # Choose best intent
        if np.all(~np.isfinite(scores)):
            # No valid intents
            action = self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)
            best_intent = -1
        else:
            best_intent = int(np.argmax(scores))
            action = self._execute_intent_np(
                best_intent, grid, pos_y, pos_x, agent_dir, inv,
                other_agents_positions, agent_id, recipe
            )

        # Collision avoidance: randomly yield if contested position
        new_wait_counter = 0
        if self._is_contested_move(action, pos_y, pos_x, other_agents_positions):
            # Use agent_id for asymmetric yielding - higher ID yields more often
            # Also use randomness to break persistent deadlocks
            rng_key_np = np.array(agent_state.rng_key)
            rng_val = int(rng_key_np[0]) % 100

            # Base yield probability: agent 0 yields 20%, agent 1 yields 40%
            # This ensures agents eventually desynchronize
            yield_prob = 20 + agent_id * 20

            if rng_val < yield_prob:
                action = Actions.stay
                # Wait for 1-2 steps to desynchronize
                new_wait_counter = 1 + (rng_val % 2)

        # Update agent state
        rng, _ = jax.random.split(agent_state.rng_key)
        new_state = UtilityAgentState(
            agent_id=jnp.array(agent_id, dtype=jnp.int32),
            rng_key=rng,
            last_intent=jnp.array(best_intent, dtype=jnp.int32),
            wait_counter=jnp.array(new_wait_counter, dtype=jnp.int32),
        )

        return jnp.array(action, dtype=jnp.int32), new_state
