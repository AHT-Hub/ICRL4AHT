"""Recipe-Aware Button Agent for OvercookedV2 - Fast NumPy Implementation.

This agent maintains a belief/memory of the recipe encoding and uses the 'L' button
(BUTTON_RECIPE_INDICATOR) to learn the recipe when unknown. It implements an
interpretable policy: "When I don't know the recipe, I press L (depending on theta),
then I cook the correct recipe."

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
    MAX_INGREDIENTS,
    StaticObject,
)
from envs.overcooked_v2.settings import POT_COOK_TIME

from .base_agent import AgentState, BaseAgentV2


# Cache directory for precomputed data
CACHE_DIR = Path(__file__).parent / ".recipe_aware_button_cache"


@chex.dataclass
class RecipeAwareButtonAgentState:
    """JAX-compatible agent state for RecipeAwareButtonAgent.

    Attributes:
        agent_id: Agent identifier
        rng_key: JAX random key for stochastic decisions
        known_recipe: Remembered recipe encoding (0 means unknown)
        last_recipe_time: Step index when recipe was last observed
        last_intent: Last chosen intent (for inertia/stability)
    """
    agent_id: jnp.ndarray  # int32 scalar
    rng_key: chex.PRNGKey
    known_recipe: jnp.ndarray  # int32 scalar, 0 = unknown
    last_recipe_time: jnp.ndarray  # int32 scalar
    last_intent: jnp.ndarray  # int32 scalar, -1 if no previous intent


@flax.struct.dataclass
class RecipeAwareButtonTheta:
    """Hyperparameters for the RecipeAwareButton agent family.

    All fields are JAX arrays to ensure compatibility with tracing.
    """
    # Core recipe-learning parameters
    press_L_when_unknown: jnp.ndarray  # float32 [0,1]: probability/strength to press L when recipe unknown
    refresh_interval: jnp.ndarray  # int32: steps before considering recipe stale
    cost_sensitivity: jnp.ndarray  # float32: penalty for pressing L (tradeoff vs productivity)

    # Recipe adherence parameters
    strict_recipe: jnp.ndarray  # float32 [0,1]: how strictly to match recipe (1=strict; 0=default ingredient 0)
    exploration_bias: jnp.ndarray  # float32 [0,1]: if recipe unknown and strict_recipe high, wander vs default cooking

    # Timing parameters
    plate_timing: jnp.ndarray  # float32 [0,1]: how early to fetch plate (1=very early, 0=wait until cooked)

    # Distance/utility modifiers
    dist_weight: jnp.ndarray  # float32: penalty per distance unit
    inertia: jnp.ndarray  # float32 [0,1]: bonus for keeping same intent

    # Non-cooperative behavior parameters (for harder-to-cooperate policies)
    task_specialization: jnp.ndarray  # float32 [0,1]: 0=flexible, 1=only do one task type (cook OR deliver)
    stubbornness: jnp.ndarray  # float32 [0,1]: how much to block/hold position vs yield
    wrong_ingredient_prob: jnp.ndarray  # float32 [0,1]: probability of choosing wrong ingredient
    idle_prob: jnp.ndarray  # float32 [0,1]: probability of doing nothing useful each step
    path_blocking: jnp.ndarray  # float32 [0,1]: tendency to stand in critical locations

    @classmethod
    def default(cls) -> "RecipeAwareButtonTheta":
        """Create default hyperparameters with balanced behavior."""
        return cls(
            press_L_when_unknown=jnp.array(0.8, dtype=jnp.float32),
            refresh_interval=jnp.array(100, dtype=jnp.int32),
            cost_sensitivity=jnp.array(0.2, dtype=jnp.float32),
            strict_recipe=jnp.array(0.9, dtype=jnp.float32),
            exploration_bias=jnp.array(0.1, dtype=jnp.float32),
            plate_timing=jnp.array(0.5, dtype=jnp.float32),
            dist_weight=jnp.array(0.3, dtype=jnp.float32),
            inertia=jnp.array(0.4, dtype=jnp.float32),
            task_specialization=jnp.array(0.0, dtype=jnp.float32),
            stubbornness=jnp.array(0.0, dtype=jnp.float32),
            wrong_ingredient_prob=jnp.array(0.0, dtype=jnp.float32),
            idle_prob=jnp.array(0.0, dtype=jnp.float32),
            path_blocking=jnp.array(0.0, dtype=jnp.float32),
        )

    @classmethod
    def high_L_priority(cls) -> "RecipeAwareButtonTheta":
        """Hyperparameters that strongly prioritize pressing L to learn recipe."""
        return cls(
            press_L_when_unknown=jnp.array(1.0, dtype=jnp.float32),
            refresh_interval=jnp.array(50, dtype=jnp.int32),
            cost_sensitivity=jnp.array(0.0, dtype=jnp.float32),
            strict_recipe=jnp.array(1.0, dtype=jnp.float32),
            exploration_bias=jnp.array(0.0, dtype=jnp.float32),
            plate_timing=jnp.array(0.6, dtype=jnp.float32),
            dist_weight=jnp.array(0.3, dtype=jnp.float32),
            inertia=jnp.array(0.3, dtype=jnp.float32),
            task_specialization=jnp.array(0.0, dtype=jnp.float32),
            stubbornness=jnp.array(0.0, dtype=jnp.float32),
            wrong_ingredient_prob=jnp.array(0.0, dtype=jnp.float32),
            idle_prob=jnp.array(0.0, dtype=jnp.float32),
            path_blocking=jnp.array(0.0, dtype=jnp.float32),
        )

    @classmethod
    def lazy_learner(cls) -> "RecipeAwareButtonTheta":
        """Hyperparameters that prefer cooking default ingredient over learning recipe."""
        return cls(
            press_L_when_unknown=jnp.array(0.3, dtype=jnp.float32),
            refresh_interval=jnp.array(200, dtype=jnp.int32),
            cost_sensitivity=jnp.array(0.8, dtype=jnp.float32),
            strict_recipe=jnp.array(0.3, dtype=jnp.float32),
            exploration_bias=jnp.array(0.5, dtype=jnp.float32),
            plate_timing=jnp.array(0.4, dtype=jnp.float32),
            dist_weight=jnp.array(0.4, dtype=jnp.float32),
            inertia=jnp.array(0.5, dtype=jnp.float32),
            task_specialization=jnp.array(0.0, dtype=jnp.float32),
            stubbornness=jnp.array(0.0, dtype=jnp.float32),
            wrong_ingredient_prob=jnp.array(0.0, dtype=jnp.float32),
            idle_prob=jnp.array(0.0, dtype=jnp.float32),
            path_blocking=jnp.array(0.0, dtype=jnp.float32),
        )

    @classmethod
    def strict_adherent(cls) -> "RecipeAwareButtonTheta":
        """Hyperparameters that strictly follow the known recipe."""
        return cls(
            press_L_when_unknown=jnp.array(0.9, dtype=jnp.float32),
            refresh_interval=jnp.array(80, dtype=jnp.int32),
            cost_sensitivity=jnp.array(0.1, dtype=jnp.float32),
            strict_recipe=jnp.array(1.0, dtype=jnp.float32),
            exploration_bias=jnp.array(0.0, dtype=jnp.float32),
            plate_timing=jnp.array(0.7, dtype=jnp.float32),
            dist_weight=jnp.array(0.25, dtype=jnp.float32),
            inertia=jnp.array(0.5, dtype=jnp.float32),
            task_specialization=jnp.array(0.0, dtype=jnp.float32),
            stubbornness=jnp.array(0.0, dtype=jnp.float32),
            wrong_ingredient_prob=jnp.array(0.0, dtype=jnp.float32),
            idle_prob=jnp.array(0.0, dtype=jnp.float32),
            path_blocking=jnp.array(0.0, dtype=jnp.float32),
        )


class RecipeAwareButtonAgentV2(BaseAgentV2):
    """Fast NumPy-based recipe-aware button agent.

    All pathfinding is precomputed at initialization. Runtime execution uses
    pure NumPy operations for maximum speed.

    This agent:
    1. Maintains a memory of the recipe encoding
    2. Uses the L button (BUTTON_RECIPE_INDICATOR) to learn the recipe when unknown
    3. Chooses ingredients matching the known recipe
    4. Compares pot contents vs recipe and fills missing ingredients
    5. Plates and delivers cooked dishes
    """

    def __init__(
        self,
        layout,
        theta: RecipeAwareButtonTheta = None,
        start_cooking_interaction: bool = False,
    ):
        """Initialize the recipe-aware button agent.

        Args:
            layout: Layout object containing the static grid configuration.
            theta: Hyperparameters for the agent. If None, uses default.
            start_cooking_interaction: Whether the environment requires explicit
                interaction to start cooking (vs auto-cooking when pot is full).
        """
        super().__init__(layout)

        self.theta = theta if theta is not None else RecipeAwareButtonTheta.default()
        self.start_cooking_interaction = start_cooking_interaction
        self.num_ingredients = layout.num_ingredients

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

        # Check if layout has button or always-visible recipe indicator
        self.has_button = np.any(self.button_mask_np)
        self.has_recipe_indicator = np.any(self.recipe_indicator_mask_np)

        # Detect cramped layouts (few walkable cells) for special coordination logic
        # Cramped layouts need agent-id based ingredient selection to avoid deadlocks
        walkable_count = np.sum(self.walkable_mask_np)
        self.is_cramped_layout = walkable_count <= 8

        # Precompute ingredient pile masks per ingredient type
        self.ingredient_masks_np = []
        for i in range(self.num_ingredients):
            pile_type = StaticObject.INGREDIENT_PILE_BASE + i
            mask = self.static_objects_np == pile_type
            self.ingredient_masks_np.append(mask)

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

                # Try all 4 directions: right=0, down=1, left=2, up=3
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

    def init_agent_state(self, agent_id: int) -> RecipeAwareButtonAgentState:
        """Initialize agent state with unknown recipe."""
        return RecipeAwareButtonAgentState(
            agent_id=jnp.array(agent_id, dtype=jnp.int32),
            rng_key=jax.random.PRNGKey(agent_id),
            known_recipe=jnp.array(0, dtype=jnp.int32),  # 0 = unknown
            last_recipe_time=jnp.array(-1000, dtype=jnp.int32),  # Very old
            last_intent=jnp.array(-1, dtype=jnp.int32),
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
        my_inventory: int = 0,
    ) -> int:
        """Get action to move adjacent to target and interact.

        Args:
            my_inventory: Current inventory to determine priority for collision resolution
        """
        # Calculate priority based on inventory for collision resolution
        # Priority: Dish (3) > Plate (2) > Ingredient (1) > Empty (0)
        is_dish = bool(my_inventory & DynamicObject.COOKED) and bool(my_inventory & DynamicObject.PLATE)
        is_plate = (my_inventory == DynamicObject.PLATE)
        is_ingredient = ((my_inventory >> 2) != 0) and ((my_inventory & DynamicObject.PLATE) == 0)

        if is_dish:
            my_priority = 3
        elif is_plate:
            my_priority = 2
        elif is_ingredient:
            my_priority = 1
        else:
            my_priority = 0

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
                dir_to_action = [3, 1, 0, 2]  # UP, DOWN, RIGHT, LEFT
                return dir_to_action[required_dir]

        # Find best adjacent cell to move to
        best_cell = None
        best_dist = 10000

        for ay, ax, face_dir in adjacent_cells:
            blocked = False
            for other_pos in other_agents_positions:
                if other_pos[0] == ay and other_pos[1] == ax:
                    blocked = True
                    break

            dist = self.dist_matrix[pos_idx, ay, ax]
            if blocked:
                dist += 100

            # Add agent-based territory preference to break symmetry
            # Only in cramped layouts to prevent deadlocks
            if self.is_cramped_layout:
                # Agent 0 prefers upper-left cells, Agent 1 prefers lower-right cells
                if agent_id == 0:
                    # Agent 0: slight penalty for moving right or down
                    dist += 0.1 * (ay + ax)
                else:
                    # Agent 1: slight penalty for moving left or up
                    dist += 0.1 * ((self.height - ay) + (self.width - ax))

            if dist < best_dist:
                best_dist = dist
                best_cell = (ay, ax)

        if best_cell is None or best_dist >= 10000:
            return Actions.stay

        action = self.next_action_matrix[pos_idx, agent_dir, best_cell[0], best_cell[1]]

        # Collision avoidance for movement actions - only in cramped layouts
        if self.is_cramped_layout and action < 4:  # Movement action (not stay or interact)
            deltas = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
            dy, dx = deltas[action]
            next_y, next_x = pos_y + dy, pos_x + dx

            # Check if next cell is blocked by another agent
            blocked_by_agent = False
            for other_pos in other_agents_positions:
                if other_pos[0] == next_y and other_pos[1] == next_x:
                    blocked_by_agent = True
                    break

            if blocked_by_agent:
                # Use priority and agent_id to decide who yields
                # Priority: Dish (3) > Plate (2) > Ingredient (1) > Empty (0)
                # Within same priority, agent_id=0 has right of way

                # Find alternative actions first
                alt_actions = []
                for alt_action, (alt_dy, alt_dx) in enumerate(deltas):
                    if alt_action == action:
                        continue
                    alt_y, alt_x = pos_y + alt_dy, pos_x + alt_dx
                    if 0 <= alt_y < self.height and 0 <= alt_x < self.width:
                        if self.walkable_mask_np[alt_y, alt_x]:
                            # Check not blocked
                            alt_blocked = False
                            for op in other_agents_positions:
                                if op[0] == alt_y and op[1] == alt_x:
                                    alt_blocked = True
                                    break
                            if not alt_blocked:
                                alt_actions.append(alt_action)

                # Decision logic based on priority and agent_id
                if my_priority >= 3:
                    # Dish carrier - very high priority
                    # Try alternatives first, only stay if no other option
                    if alt_actions:
                        return alt_actions[0]
                    # If no alternatives, wait (other agent should move)
                    return Actions.stay
                elif my_priority == 2:
                    # Plate carrier - yield if lower agent_id or if alternatives exist
                    if agent_id > 0 or alt_actions:
                        if alt_actions:
                            return alt_actions[0]
                        return Actions.stay
                elif my_priority >= 1:
                    # Ingredient carrier - yield based on agent_id
                    if agent_id > 0:
                        # Agent 1 with ingredient yields
                        if alt_actions:
                            return alt_actions[0]
                        return Actions.stay
                    # Agent 0 with ingredient can be more persistent, but still yield if alternatives
                    if alt_actions and len(alt_actions) > 1:
                        return alt_actions[0]
                else:
                    # Empty agent - most likely to yield
                    if agent_id > 0:
                        # Agent 1 empty always yields
                        if alt_actions:
                            return alt_actions[0]
                        return Actions.stay
                    # Agent 0 empty tries alternatives if available
                    if alt_actions:
                        return alt_actions[0]

        return action

    def _hover_near_target_np(
        self,
        pos_y: int, pos_x: int, agent_dir: int,
        target_y: int, target_x: int,
        other_agents_positions: np.ndarray, agent_id: int,
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
                dir_to_action = [3, 1, 0, 2]
                return dir_to_action[required_dir]

        # Move towards target
        return self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id, my_inventory=0)

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

            blocked = False
            for other_pos in other_agents_positions:
                if other_pos[0] == ny and other_pos[1] == nx:
                    blocked = True
                    break

            score = 0
            if blocked:
                score += 100
            if action == 4:
                score += 50

            if score < best_score:
                best_score = score
                best_action = action

        return best_action

    # -------------------------------------------------------------------------
    # Recipe/pot analysis helpers
    # -------------------------------------------------------------------------

    def _count_ingredients_np(self, content: int) -> int:
        """Count ingredients in a pot content encoding."""
        content = content >> 2
        count = 0
        while content > 0:
            count += content & 0x3
            content >>= 2
        return count

    def _decode_recipe_np(self, recipe: int) -> np.ndarray:
        """Decode recipe encoding into ingredient type counts."""
        counts = np.zeros(MAX_INGREDIENTS, dtype=np.int32)
        recipe = recipe >> 2
        for i in range(MAX_INGREDIENTS):
            counts[i] = recipe & 0x3
            recipe >>= 2
        return counts

    def _decode_pot_contents_np(self, pot_contents: int) -> np.ndarray:
        """Decode pot contents to ingredient counts."""
        return self._decode_recipe_np(pot_contents)

    def _get_pot_states_np(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Analyze pot states using NumPy.

        Returns:
            - pot_mask: Which cells are pots
            - pot_non_full_mask: Pots that can accept more ingredients
            - pot_cooking_mask: Pots currently cooking
            - pot_cooked_mask: Pots with finished soup
            - pot_contents: Dynamic layer values at pot positions
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

        return is_pot, pot_non_full, pot_cooking, pot_cooked, dynamic_layer

    def _is_button_active_np(self, grid: np.ndarray) -> bool:
        """Check if button L is active (recipe visible via button)."""
        timer_layer = grid[:, :, 2]
        active_mask = (timer_layer > 0) & self.button_mask_np
        return np.any(active_mask)

    def _is_recipe_visible_np(self, grid: np.ndarray) -> bool:
        """Check if recipe is currently visible."""
        has_always_visible = self.has_recipe_indicator
        button_active = self._is_button_active_np(grid)
        return has_always_visible or button_active

    def _compute_missing_ingredients_np(
        self,
        known_recipe: int,
        pot_contents: int,
    ) -> np.ndarray:
        """Compute which ingredients are missing for a pot to match recipe."""
        recipe_counts = self._decode_recipe_np(known_recipe)
        pot_counts = self._decode_pot_contents_np(pot_contents)
        missing = np.maximum(recipe_counts - pot_counts, 0)
        return missing

    def _get_most_needed_ingredient_np(
        self,
        known_recipe: int,
        grid: np.ndarray,
        pot_non_full: np.ndarray,
        dynamic_layer: np.ndarray,
    ) -> Tuple[int, bool]:
        """Determine which ingredient type is most needed across all pots.

        Returns:
            Tuple of (ingredient_index, exists_flag)
        """
        # For each non-full pot, compute missing ingredients
        scores = np.zeros(MAX_INGREDIENTS, dtype=np.float32)

        pot_positions = np.argwhere(pot_non_full)
        for py, px in pot_positions:
            contents = int(dynamic_layer[py, px])
            missing = self._compute_missing_ingredients_np(known_recipe, contents)
            scores += missing.astype(np.float32)

        # If recipe is unknown (0), fall back to ingredient 0
        if known_recipe == 0:
            scores = np.zeros(MAX_INGREDIENTS, dtype=np.float32)
            scores[0] = 1.0

        best_idx = int(np.argmax(scores))
        exists = scores[best_idx] > 0

        return best_idx, exists

    def _get_best_ingredient_mask_np(
        self,
        known_recipe: int,
        grid: np.ndarray,
        pot_non_full: np.ndarray,
        dynamic_layer: np.ndarray,
        pos_y: int, pos_x: int,
        strict_recipe: float,
        agent_id: int = 0,
    ) -> Tuple[np.ndarray, bool]:
        """Get mask for the best ingredient to fetch based on recipe and pot status.

        When recipe is known, always follow it regardless of strict_recipe.
        When recipe is unknown, use strict_recipe to decide whether to default
        to ingredient 0 or wait/explore.
        
        To avoid collisions, different agents prefer different ingredients when
        multiple are needed.
        """
        # Get the missing ingredient counts
        scores = np.zeros(MAX_INGREDIENTS, dtype=np.float32)
        pot_positions = np.argwhere(pot_non_full)
        for py, px in pot_positions:
            contents = int(dynamic_layer[py, px])
            missing = self._compute_missing_ingredients_np(known_recipe, contents)
            scores += missing.astype(np.float32)

        recipe_known = known_recipe != 0

        if recipe_known:
            # Recipe is known - find all needed ingredients
            needed_ingredients = np.where(scores > 0)[0]
            
            if len(needed_ingredients) == 0:
                # No ingredients needed (pot is full or recipe satisfied)
                return np.zeros((self.height, self.width), dtype=np.bool_), False
            
            if len(needed_ingredients) == 1:
                # Only one ingredient type needed
                final_idx = int(needed_ingredients[0])
            elif self.is_cramped_layout and len(needed_ingredients) > 1:
                # Multiple ingredients needed in cramped layout - agents prefer different ones
                # Agent 0 prefers earlier indices, Agent 1 prefers later indices
                if agent_id == 0:
                    final_idx = int(needed_ingredients[0])
                else:
                    # Agent 1 picks from the end or wraps around
                    pick_idx = min(agent_id, len(needed_ingredients) - 1)
                    final_idx = int(needed_ingredients[pick_idx])
            else:
                # Non-cramped layout: use normal logic (first needed ingredient)
                final_idx = int(needed_ingredients[0])
        else:
            # Recipe is unknown - use strict_recipe to decide
            # Low strict_recipe = default to ingredient 0
            # High strict_recipe = don't fetch (wait for recipe)
            if strict_recipe < 0.5:
                # Only use agent_id based selection in cramped layouts to avoid deadlock
                if self.is_cramped_layout and agent_id > 0 and self.num_ingredients > 1:
                    final_idx = 1  # Agent 1 prefers ingredient 1 in cramped layouts
                else:
                    final_idx = 0  # Default to ingredient 0
            else:
                # Don't fetch anything if recipe unknown and strict
                return np.zeros((self.height, self.width), dtype=np.bool_), False

        # Clamp to valid range
        final_idx = max(0, min(final_idx, self.num_ingredients - 1))

        # Get ingredient mask
        ingredient_mask = self.ingredient_masks_np[final_idx]
        reachable = self._get_reachable_targets_np(ingredient_mask, pos_y, pos_x)

        return reachable, np.any(reachable)

    def _get_pot_for_ingredient_np(
        self,
        inv: int,
        known_recipe: int,
        pot_non_full: np.ndarray,
        dynamic_layer: np.ndarray,
        pos_y: int, pos_x: int,
    ) -> Tuple[np.ndarray, bool]:
        """Find the best pot to add the held ingredient to."""
        # Get ingredient type from inventory
        inv_shifted = inv >> 2
        ing_type = -1
        for i in range(MAX_INGREDIENTS):
            count = (inv_shifted >> (2 * i)) & 0x3
            if count > 0:
                ing_type = i
                break

        # Find pots that need this ingredient
        pot_positions = np.argwhere(pot_non_full)
        wants_mask = np.zeros((self.height, self.width), dtype=np.bool_)

        for py, px in pot_positions:
            contents = int(dynamic_layer[py, px])
            missing = self._compute_missing_ingredients_np(known_recipe, contents)
            if ing_type >= 0 and ing_type < len(missing):
                wants = missing[ing_type] > 0
            else:
                wants = True  # Unknown ingredient, any pot is fine
            if wants:
                wants_mask[py, px] = True

        # Fall back to any non-full pot
        if not np.any(wants_mask):
            wants_mask = pot_non_full

        reachable = self._get_reachable_targets_np(wants_mask, pos_y, pos_x)
        return reachable, np.any(reachable)

    def _should_get_plate_np(
        self,
        pot_cooking: np.ndarray,
        pot_cooked: np.ndarray,
        pot_mask: np.ndarray,
        timer_layer: np.ndarray,
        plate_timing: float,
    ) -> bool:
        """Decide if agent should fetch a plate based on pot status and plate_timing."""
        any_cooked = np.any(pot_cooked)
        any_cooking = np.any(pot_cooking)

        # Calculate cooking progress (0-1) for cooking pots
        cooking_progress = np.where(
            pot_cooking,
            1.0 - (timer_layer.astype(np.float32) / POT_COOK_TIME),
            0.0
        )
        max_progress = np.max(cooking_progress)

        # Decision based on plate_timing theta
        if any_cooked:
            urgency = 1.0
        elif any_cooking:
            urgency = max_progress
        else:
            urgency = 0.0

        return urgency >= (1.0 - plate_timing)

    def _get_empty_counter_mask_np(self, grid: np.ndarray) -> np.ndarray:
        """Get mask of empty counters."""
        is_counter = self.counter_mask_np
        is_empty = grid[:, :, 1] == DynamicObject.EMPTY
        return is_counter & is_empty

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def get_action(
        self,
        obs,
        env_state,
        agent_state: RecipeAwareButtonAgentState = None,
    ) -> Tuple[jnp.ndarray, RecipeAwareButtonAgentState]:
        """Get action for the agent (main entry point).

        Uses NumPy for all computation, returns JAX arrays for compatibility.
        """
        if agent_state is None:
            agent_state = self.init_agent_state(0)

        agent_id = int(agent_state.agent_id)

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

        # Get current time
        current_time = int(env_state.time) if hasattr(env_state, 'time') else 0

        # Get recipe and button state
        env_recipe = int(env_state.recipe) if hasattr(env_state, 'recipe') else 0
        recipe_visible = self._is_recipe_visible_np(grid)

        # Update recipe memory
        known_recipe = int(agent_state.known_recipe)
        last_recipe_time = int(agent_state.last_recipe_time)

        if recipe_visible:
            known_recipe = env_recipe
            last_recipe_time = current_time

        # Get theta values as python scalars
        press_L_when_unknown = float(self.theta.press_L_when_unknown)
        refresh_interval = int(self.theta.refresh_interval)
        cost_sensitivity = float(self.theta.cost_sensitivity)
        strict_recipe = float(self.theta.strict_recipe)
        plate_timing = float(self.theta.plate_timing)
        
        # Get non-cooperative behavior parameters
        task_specialization = float(self.theta.task_specialization)
        stubbornness = float(self.theta.stubbornness)
        wrong_ingredient_prob = float(self.theta.wrong_ingredient_prob)
        idle_prob = float(self.theta.idle_prob)
        path_blocking = float(self.theta.path_blocking)

        # Random check for idle behavior
        rng_key = agent_state.rng_key
        rng_key, idle_key, wrong_ing_key, block_key = jax.random.split(rng_key, 4)
        idle_roll = float(jax.random.uniform(idle_key))
        
        # If idle_prob is high and we roll under it, just do nothing or wander
        if idle_roll < idle_prob:
            action = Actions.stay
            return self._return_action_with_key(action, agent_state, agent_id, known_recipe, last_recipe_time, rng_key)

        # Determine inventory state
        is_empty = inv == DynamicObject.EMPTY
        is_plate = inv == DynamicObject.PLATE
        is_dish = bool(inv & DynamicObject.COOKED) and bool(inv & DynamicObject.PLATE)
        is_ingredient = ((inv >> 2) != 0) and ((inv & DynamicObject.PLATE) == 0)

        # Get pot states
        pot_mask, pot_non_full, pot_cooking, pot_cooked, dynamic_layer = self._get_pot_states_np(grid)
        timer_layer = grid[:, :, 2]

        # Check if recipe is stale
        recipe_age = current_time - last_recipe_time
        recipe_stale = recipe_age > refresh_interval
        recipe_unknown = known_recipe == 0

        # Task specialization logic
        # task_specialization > 0.5 means agent only does one type of task:
        # - agent_id 0: only cooks (fetches ingredients, adds to pot)
        # - agent_id 1: only delivers (fetches plates, delivers dishes)
        # This requires coordination but a random partner won't help
        is_cook_specialist = (task_specialization > 0.5) and (agent_id == 0)
        is_delivery_specialist = (task_specialization > 0.5) and (agent_id == 1)
        
        # Path blocking: with high path_blocking, agent tends to hover near critical locations
        block_roll = float(jax.random.uniform(block_key))
        should_block_path = block_roll < path_blocking

        # Priority A: Holding dish -> deliver
        # BUT if cook specialist, just drop the dish and go back to cooking
        if is_dish:
            if is_cook_specialist:
                # Cook specialists don't deliver - drop on counter and go back to cooking
                empty_counters = self._get_empty_counter_mask_np(grid)
                reachable_counters = self._get_reachable_targets_np(empty_counters, pos_y, pos_x)
                target_y, target_x, exists = self._get_closest_target_np(reachable_counters, pos_y, pos_x)
                if exists:
                    action = self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id, my_inventory=inv)
                else:
                    action = self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)
                return self._return_action_with_key(action, agent_state, agent_id, known_recipe, last_recipe_time, rng_key)
            else:
                reachable_goals = self._get_reachable_targets_np(self.goal_mask_np, pos_y, pos_x)
                target_y, target_x, exists = self._get_closest_target_np(reachable_goals, pos_y, pos_x)
                if exists:
                    action = self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id, my_inventory=inv)
                else:
                    action = self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)
                return self._return_action_with_key(action, agent_state, agent_id, known_recipe, last_recipe_time, rng_key)

        # Priority B: Holding plate -> pickup cooked
        if is_plate:
            reachable_cooked = self._get_reachable_targets_np(pot_cooked, pos_y, pos_x)
            target_y, target_x, exists = self._get_closest_target_np(reachable_cooked, pos_y, pos_x)
            if exists:
                action = self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id, my_inventory=inv)
            else:
                # Check if another agent has a dish - if so, get out of their way (only in cramped layouts)
                other_has_dish = False
                if self.is_cramped_layout:
                    dish_carrier_id = -1
                    for i in range(num_agents):
                        if i != agent_id:
                            other_inv = int(env_state.agents.inventory[i])
                            if bool(other_inv & DynamicObject.COOKED) and bool(other_inv & DynamicObject.PLATE):
                                other_has_dish = True
                                dish_carrier_id = i
                                break
                
                if other_has_dish:
                    # Get out of dish carrier's path to goal
                    # Find goal-adjacent cells - these are the delivery points
                    goal_pos = np.argwhere(self.goal_mask_np)
                    goal_adj_cells = set()
                    if len(goal_pos) > 0:
                        for gy, gx in goal_pos:
                            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                ay, ax = gy + dy, gx + dx
                                if 0 <= ay < self.height and 0 <= ax < self.width:
                                    if self.walkable_mask_np[ay, ax]:
                                        goal_adj_cells.add((ay, ax))
                    
                    # Find the dish carrier's position
                    dish_carrier_pos = None
                    for i in range(num_agents):
                        if i != agent_id:
                            other_inv = int(env_state.agents.inventory[i])
                            if bool(other_inv & DynamicObject.COOKED) and bool(other_inv & DynamicObject.PLATE):
                                dish_carrier_pos = (int(env_state.agents.pos.y[i]), int(env_state.agents.pos.x[i]))
                                break
                    
                    # Include all cells that might be on path from dish carrier to goal
                    # Use BFS-like approach: any cell that is between dish carrier and goal
                    cells_to_avoid = set(goal_adj_cells)
                    if dish_carrier_pos:
                        dcy, dcx = dish_carrier_pos
                        for gac in goal_adj_cells:
                            gy, gx = gac
                            # Include all cells in the rectangle between dish carrier and goal
                            min_y, max_y = min(dcy, gy), max(dcy, gy)
                            min_x, max_x = min(dcx, gx), max(dcx, gx)
                            for cy in range(min_y, max_y + 1):
                                for cx in range(min_x, max_x + 1):
                                    if 0 <= cy < self.height and 0 <= cx < self.width:
                                        if self.walkable_mask_np[cy, cx]:
                                            cells_to_avoid.add((cy, cx))
                        # Also add cells adjacent to dish carrier (it might move)
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ay, ax = dcy + dy, dcx + dx
                            if 0 <= ay < self.height and 0 <= ax < self.width:
                                if self.walkable_mask_np[ay, ax]:
                                    cells_to_avoid.add((ay, ax))
                    
                    # Move to a cell not on the path and stay there
                    if (pos_y, pos_x) in cells_to_avoid:
                        best_action = Actions.stay
                        best_dist = -1
                        deltas = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                        for action_idx, (dy, dx) in enumerate(deltas):
                            ny, nx = pos_y + dy, pos_x + dx
                            if 0 <= ny < self.height and 0 <= nx < self.width:
                                if self.walkable_mask_np[ny, nx]:
                                    blocked = any(op[0] == ny and op[1] == nx for op in other_agents_positions)
                                    if not blocked and (ny, nx) not in cells_to_avoid:
                                        # Calculate min distance to any goal-adjacent cell
                                        min_dist = min(abs(ny - cy) + abs(nx - cx) for cy, cx in goal_adj_cells) if goal_adj_cells else 0
                                        if min_dist > best_dist:
                                            best_dist = min_dist
                                            best_action = action_idx
                        
                        # If no direct escape found, try to move toward any walkable cell
                        # furthest from the goal to create space for dish carrier
                        if best_action == Actions.stay:
                            # Find furthest walkable cell from goal
                            furthest_cell = None
                            max_dist_from_goal = -1
                            for wy in range(self.height):
                                for wx in range(self.width):
                                    if self.walkable_mask_np[wy, wx] and (wy, wx) not in cells_to_avoid:
                                        # Calculate min distance to any goal-adjacent cell
                                        min_d = min(abs(wy - cy) + abs(wx - cx) for cy, cx in goal_adj_cells) if goal_adj_cells else 0
                                        if min_d > max_dist_from_goal:
                                            max_dist_from_goal = min_d
                                            furthest_cell = (wy, wx)
                            
                            if furthest_cell:
                                # Use pathfinding to move toward furthest cell
                                fy, fx = furthest_cell
                                # Simple greedy: move in direction that reduces manhattan distance
                                best_action = Actions.stay
                                best_dist_to_target = abs(pos_y - fy) + abs(pos_x - fx)
                                for action_idx, (dy, dx) in enumerate(deltas):
                                    ny, nx = pos_y + dy, pos_x + dx
                                    if 0 <= ny < self.height and 0 <= nx < self.width:
                                        if self.walkable_mask_np[ny, nx]:
                                            blocked = any(op[0] == ny and op[1] == nx for op in other_agents_positions)
                                            if not blocked:
                                                dist_to_target = abs(ny - fy) + abs(nx - fx)
                                                if dist_to_target < best_dist_to_target:
                                                    best_dist_to_target = dist_to_target
                                                    best_action = action_idx
                        
                        action = best_action
                    else:
                        # Already out of the way, stay put
                        action = Actions.stay
                else:
                    # Hover near cooking pot
                    pot_with_stuff = pot_cooking | (pot_mask & (dynamic_layer != 0))
                    reachable_active = self._get_reachable_targets_np(pot_with_stuff, pos_y, pos_x)
                    target_y, target_x, exists = self._get_closest_target_np(reachable_active, pos_y, pos_x)
                    if exists:
                        action = self._hover_near_target_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)
                    else:
                        action = self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)
            return self._return_action(action, agent_state, agent_id, known_recipe, last_recipe_time)

        # Priority C: Holding ingredient -> add to pot
        if is_ingredient:
            pot_for_ing_mask, can_add = self._get_pot_for_ingredient_np(
                inv, known_recipe, pot_non_full, dynamic_layer, pos_y, pos_x
            )
            if can_add:
                target_y, target_x, exists = self._get_closest_target_np(pot_for_ing_mask, pos_y, pos_x)
                if exists:
                    # Check if other agent is also holding ingredient and heading to same pot
                    # In cramped layouts only, yield to agent 0 to prevent deadlock
                    should_yield = False
                    if self.is_cramped_layout and agent_id > 0:
                        # Check if agent 0 is also carrying an ingredient
                        for i in range(num_agents):
                            if i != agent_id:
                                other_inv = int(env_state.agents.inventory[i])
                                other_has_ingredient = ((other_inv >> 2) != 0) and ((other_inv & DynamicObject.PLATE) == 0)
                                if other_has_ingredient:
                                    # Check if we're competing for the same pot-adjacent cell
                                    other_y = int(env_state.agents.pos.y[i])
                                    other_x = int(env_state.agents.pos.x[i])
                                    # Find adjacent cells to target pot
                                    pot_adjacent_cells = []
                                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                        ay, ax = target_y + dy, target_x + dx
                                        if 0 <= ay < self.height and 0 <= ax < self.width:
                                            if self.walkable_mask_np[ay, ax]:
                                                pot_adjacent_cells.append((ay, ax))
                                    
                                    # If only one pot-adjacent cell, yield to agent 0
                                    if len(pot_adjacent_cells) <= 1:
                                        # Check if agent 0 is close to that cell too
                                        if pot_adjacent_cells:
                                            adj_y, adj_x = pot_adjacent_cells[0]
                                            my_dist = abs(pos_y - adj_y) + abs(pos_x - adj_x)
                                            other_dist = abs(other_y - adj_y) + abs(other_x - adj_x)
                                            # Agent 1 yields if agent 0 is closer or same distance
                                            if other_dist <= my_dist + 1:
                                                should_yield = True
                    
                    if should_yield:
                        # Yield by moving AWAY from the pot-adjacent area
                        # Try to find a cell that doesn't block agent 0's path
                        best_yield_action = Actions.stay
                        best_yield_dist = -1
                        
                        # Find the pot-adjacent cell
                        pot_adj_y, pot_adj_x = None, None
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ay, ax = target_y + dy, target_x + dx
                            if 0 <= ay < self.height and 0 <= ax < self.width:
                                if self.walkable_mask_np[ay, ax]:
                                    pot_adj_y, pot_adj_x = ay, ax
                                    break
                        
                        if pot_adj_y is not None:
                            # Try each direction and pick the one that moves us furthest from pot_adj
                            deltas = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
                            for action, (dy, dx) in enumerate(deltas):
                                ny, nx = pos_y + dy, pos_x + dx
                                if 0 <= ny < self.height and 0 <= nx < self.width:
                                    if self.walkable_mask_np[ny, nx]:
                                        # Check not blocked by other agent
                                        blocked = False
                                        for other_pos in other_agents_positions:
                                            if other_pos[0] == ny and other_pos[1] == nx:
                                                blocked = True
                                                break
                                        if not blocked:
                                            # Calculate distance from pot-adjacent cell
                                            new_dist = abs(ny - pot_adj_y) + abs(nx - pot_adj_x)
                                            if new_dist > best_yield_dist:
                                                best_yield_dist = new_dist
                                                best_yield_action = action
                        
                        action = best_yield_action
                    else:
                        action = self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id, my_inventory=inv)
                    return self._return_action(action, agent_state, agent_id, known_recipe, last_recipe_time)

            # No pot available - drop on counter
            empty_counters = self._get_empty_counter_mask_np(grid)
            reachable_counters = self._get_reachable_targets_np(empty_counters, pos_y, pos_x)
            target_y, target_x, exists = self._get_closest_target_np(reachable_counters, pos_y, pos_x)
            if exists:
                action = self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id, my_inventory=inv)
            else:
                action = self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)
            return self._return_action(action, agent_state, agent_id, known_recipe, last_recipe_time)

        # Priority D: Empty-handed options

        # D1: Press L if recipe unknown/stale and button available
        if self.has_button and (recipe_unknown or recipe_stale):
            button_active = self._is_button_active_np(grid)
            if not button_active:
                # Decide whether to press L based on theta
                rng_key = agent_state.rng_key
                rng, sample_key = jax.random.split(rng_key)
                rand_val = float(jax.random.uniform(sample_key))
                prob_threshold = press_L_when_unknown * (1.0 - cost_sensitivity)

                if rand_val < prob_threshold:
                    reachable_buttons = self._get_reachable_targets_np(self.button_mask_np, pos_y, pos_x)
                    target_y, target_x, exists = self._get_closest_target_np(reachable_buttons, pos_y, pos_x)
                    if exists:
                        action = self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id, my_inventory=inv)
                        # Update RNG in returned state
                        new_state = RecipeAwareButtonAgentState(
                            agent_id=jnp.array(agent_id, dtype=jnp.int32),
                            rng_key=rng,
                            known_recipe=jnp.array(known_recipe, dtype=jnp.int32),
                            last_recipe_time=jnp.array(last_recipe_time, dtype=jnp.int32),
                            last_intent=jnp.array(5, dtype=jnp.int32),  # INTENT_PRESS_L
                        )
                        return jnp.array(action, dtype=jnp.int32), new_state

        # D2: Get plate if should plate
        # Delivery specialists always try to get plates
        # Cook specialists skip plate fetching
        should_plate = self._should_get_plate_np(pot_cooking, pot_cooked, pot_mask, timer_layer, plate_timing)
        if is_cook_specialist:
            should_plate = False  # Cook specialists don't get plates
        if is_delivery_specialist:
            should_plate = True  # Delivery specialists always try to get plates
            
        if should_plate:
            # In cramped layouts only, with limited access, coordinate who gets plates
            # Check if plate pile has limited access (only one adjacent walkable cell)
            should_skip_plate = False
            if self.is_cramped_layout:
                plate_pile_pos = np.argwhere(self.plate_pile_mask_np)
                if len(plate_pile_pos) > 0:
                    py, px = plate_pile_pos[0]
                    plate_adj_cells = []
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ay, ax = py + dy, px + dx
                        if 0 <= ay < self.height and 0 <= ax < self.width:
                            if self.walkable_mask_np[ay, ax]:
                                plate_adj_cells.append((ay, ax))
                    
                    # If only one adjacent cell, only agent 0 should fetch plate
                    # unless agent 0 already has a plate/dish
                    if len(plate_adj_cells) <= 1 and agent_id > 0:
                        # Check if agent 0 already has plate or dish
                        agent0_inv = int(env_state.agents.inventory[0])
                        agent0_has_plate = (agent0_inv == DynamicObject.PLATE)
                        agent0_has_dish = bool(agent0_inv & DynamicObject.COOKED) and bool(agent0_inv & DynamicObject.PLATE)
                        
                        # Agent 1 skips plate if agent 0 doesn't have plate/dish yet
                        if not agent0_has_plate and not agent0_has_dish:
                            should_skip_plate = True
            
            if not should_skip_plate:
                reachable_plates = self._get_reachable_targets_np(self.plate_pile_mask_np, pos_y, pos_x)
                target_y, target_x, exists = self._get_closest_target_np(reachable_plates, pos_y, pos_x)
                if exists:
                    action = self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id, my_inventory=inv)
                    return self._return_action_with_key(action, agent_state, agent_id, known_recipe, last_recipe_time, rng_key)

        # D3: Fetch ingredient
        # Delivery specialists don't fetch ingredients - they just hover near pot or wander
        if is_delivery_specialist:
            # Delivery specialist: hover near pot waiting for dishes
            reachable_pots = self._get_reachable_targets_np(pot_mask, pos_y, pos_x)
            target_y, target_x, exists = self._get_closest_target_np(reachable_pots, pos_y, pos_x)
            if exists:
                action = self._hover_near_target_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)
            else:
                action = self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)
            return self._return_action_with_key(action, agent_state, agent_id, known_recipe, last_recipe_time, rng_key)
        
        # Path blocking behavior - hover near critical locations
        if should_block_path:
            # Find critical locations (pot-adjacent, goal-adjacent, or ingredient pile adjacent)
            critical_mask = np.zeros((self.height, self.width), dtype=np.bool_)
            # Add pot-adjacent cells
            pot_positions = np.argwhere(pot_mask)
            for py, px in pot_positions:
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ay, ax = py + dy, px + dx
                    if 0 <= ay < self.height and 0 <= ax < self.width:
                        if self.walkable_mask_np[ay, ax]:
                            critical_mask[ay, ax] = True
            # Add goal-adjacent cells  
            goal_positions = np.argwhere(self.goal_mask_np)
            for gy, gx in goal_positions:
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ay, ax = gy + dy, gx + dx
                    if 0 <= ay < self.height and 0 <= ax < self.width:
                        if self.walkable_mask_np[ay, ax]:
                            critical_mask[ay, ax] = True
            
            # Move toward and stay at a critical location
            reachable_critical = self._get_reachable_targets_np(critical_mask, pos_y, pos_x)
            if np.any(reachable_critical):
                # Already at a critical location? Just stay
                if critical_mask[pos_y, pos_x]:
                    action = Actions.stay
                else:
                    # Move to closest critical cell
                    pos_idx = self.pos_to_idx[pos_y, pos_x]
                    if pos_idx >= 0:
                        best_dist = 10000
                        best_target = None
                        for cy in range(self.height):
                            for cx in range(self.width):
                                if critical_mask[cy, cx] and self.walkable_mask_np[cy, cx]:
                                    dist = self.dist_matrix[pos_idx, cy, cx]
                                    if dist < best_dist:
                                        best_dist = dist
                                        best_target = (cy, cx)
                        if best_target:
                            action = self.next_action_matrix[pos_idx, agent_dir, best_target[0], best_target[1]]
                        else:
                            action = Actions.stay
                    else:
                        action = Actions.stay
                return self._return_action_with_key(action, agent_state, agent_id, known_recipe, last_recipe_time, rng_key)
        
        # Check for wrong ingredient probability
        wrong_ing_roll = float(jax.random.uniform(wrong_ing_key))
        use_wrong_ingredient = wrong_ing_roll < wrong_ingredient_prob
        
        if use_wrong_ingredient and self.num_ingredients > 1:
            # Pick a random wrong ingredient
            rng_key, ing_key = jax.random.split(rng_key)
            # Get the correct ingredient index
            correct_mask, _ = self._get_best_ingredient_mask_np(
                known_recipe, grid, pot_non_full, dynamic_layer, pos_y, pos_x, strict_recipe, agent_id
            )
            # Find which ingredient type we would normally pick
            correct_ing_idx = -1
            for i, mask in enumerate(self.ingredient_masks_np):
                if np.any(correct_mask & mask):
                    correct_ing_idx = i
                    break
            # Pick a different ingredient
            wrong_ing_idx = (correct_ing_idx + 1) % self.num_ingredients
            wrong_ing_mask = self.ingredient_masks_np[wrong_ing_idx]
            reachable_wrong = self._get_reachable_targets_np(wrong_ing_mask, pos_y, pos_x)
            if np.any(reachable_wrong):
                target_y, target_x, exists = self._get_closest_target_np(reachable_wrong, pos_y, pos_x)
                if exists:
                    action = self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id, my_inventory=inv)
                    return self._return_action_with_key(action, agent_state, agent_id, known_recipe, last_recipe_time, rng_key)
        
        ingredient_mask, can_fetch = self._get_best_ingredient_mask_np(
            known_recipe, grid, pot_non_full, dynamic_layer, pos_y, pos_x, strict_recipe, agent_id
        )
        if can_fetch:
            target_y, target_x, exists = self._get_closest_target_np(ingredient_mask, pos_y, pos_x)
            if exists:
                action = self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id, my_inventory=inv)
                return self._return_action_with_key(action, agent_state, agent_id, known_recipe, last_recipe_time, rng_key)

        # Fallback: hover near pot or wander
        reachable_pots = self._get_reachable_targets_np(pot_mask, pos_y, pos_x)
        target_y, target_x, exists = self._get_closest_target_np(reachable_pots, pos_y, pos_x)
        if exists:
            action = self._hover_near_target_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)
        else:
            action = self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)

        return self._return_action_with_key(action, agent_state, agent_id, known_recipe, last_recipe_time, rng_key)

    def _return_action(
        self,
        action: int,
        agent_state: RecipeAwareButtonAgentState,
        agent_id: int,
        known_recipe: int,
        last_recipe_time: int,
    ) -> Tuple[jnp.ndarray, RecipeAwareButtonAgentState]:
        """Return action and updated state as JAX arrays."""
        rng, _ = jax.random.split(agent_state.rng_key)
        new_state = RecipeAwareButtonAgentState(
            agent_id=jnp.array(agent_id, dtype=jnp.int32),
            rng_key=rng,
            known_recipe=jnp.array(known_recipe, dtype=jnp.int32),
            last_recipe_time=jnp.array(last_recipe_time, dtype=jnp.int32),
            last_intent=jnp.array(-1, dtype=jnp.int32),
        )
        return jnp.array(action, dtype=jnp.int32), new_state

    def _return_action_with_key(
        self,
        action: int,
        agent_state: RecipeAwareButtonAgentState,
        agent_id: int,
        known_recipe: int,
        last_recipe_time: int,
        rng_key: chex.PRNGKey,
    ) -> Tuple[jnp.ndarray, RecipeAwareButtonAgentState]:
        """Return action and updated state with provided RNG key."""
        new_state = RecipeAwareButtonAgentState(
            agent_id=jnp.array(agent_id, dtype=jnp.int32),
            rng_key=rng_key,
            known_recipe=jnp.array(known_recipe, dtype=jnp.int32),
            last_recipe_time=jnp.array(last_recipe_time, dtype=jnp.int32),
            last_intent=jnp.array(-1, dtype=jnp.int32),
        )
        return jnp.array(action, dtype=jnp.int32), new_state
