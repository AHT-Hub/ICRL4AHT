"""Territory Agent for OvercookedV2 - Fast NumPy Implementation.

This agent implements territory-based behavior where each agent is biased to operate
within a predefined region/territory of the kitchen. The agent can be configured to
use different split modes (vertical, horizontal, or object-based stations).

All pathfinding and decision logic is precomputed at initialization using NumPy,
making runtime execution extremely fast (simple array lookups).

The agent is JAX-compatible for training - it returns JAX arrays but all internal
computation uses NumPy for maximum speed.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Tuple

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


@flax.struct.dataclass
class TerritoryAgentState:
    """Extended agent state with deadlock detection for Territory agent."""
    agent_id: jnp.ndarray  # int32 scalar
    rng_key: jax.Array  # PRNGKey
    last_pos_y: jnp.ndarray  # int32: last known Y position
    last_pos_x: jnp.ndarray  # int32: last known X position
    stuck_counter: jnp.ndarray  # int32: how many steps agent hasn't moved


# Split mode constants
SPLIT_VERTICAL = 0
SPLIT_HORIZONTAL = 1
SPLIT_OBJECT_STATIONS = 2

# Behavior mode constants (for difficult-to-cooperate strategies)
BEHAVIOR_NORMAL = 0        # Standard cooperative behavior
BEHAVIOR_BLOCKER = 1       # Positions in chokepoints, prefers staying
BEHAVIOR_HOARDER = 2       # Picks up items but drops on counters, never completes
BEHAVIOR_LAZY = 3          # Rarely acts, mostly stays still
BEHAVIOR_COUNTER = 4       # Ignores urgent tasks, does opposite of helpful
BEHAVIOR_INVADER = 5       # Operates in partner's territory, causes collisions

# Cache directory for precomputed data
CACHE_DIR = Path(__file__).parent / ".territory_cache"


@flax.struct.dataclass
class TerritoryTheta:
    """Hyperparameters for the Territory agent family.

    All fields are JAX arrays to ensure compatibility with tracing.
    """
    split_mode: jnp.ndarray  # int32: 0=VERTICAL, 1=HORIZONTAL, 2=OBJECT_STATIONS
    strictness: jnp.ndarray  # float32 in [0, 1]: how strongly to avoid acting outside territory
    shared_margin: jnp.ndarray  # int32 >= 0: columns/rows treated as "shared corridor"
    rescue_threshold: jnp.ndarray  # float32 in [0, 1]: when to break territory for urgent events
    yield_bias: jnp.ndarray  # float32 in [0, 1]: tendency to yield to avoid deadlocks
    behavior_mode: jnp.ndarray  # int32: behavior strategy (0=NORMAL, 1=BLOCKER, 2=HOARDER, etc.)
    action_probability: jnp.ndarray  # float32 in [0, 1]: probability of taking action vs staying (for LAZY mode)

    @classmethod
    def default(cls) -> "TerritoryTheta":
        """Create default hyperparameters."""
        return cls(
            split_mode=jnp.array(SPLIT_VERTICAL, dtype=jnp.int32),
            strictness=jnp.array(0.7, dtype=jnp.float32),
            shared_margin=jnp.array(1, dtype=jnp.int32),
            rescue_threshold=jnp.array(0.6, dtype=jnp.float32),
            yield_bias=jnp.array(0.3, dtype=jnp.float32),
            behavior_mode=jnp.array(BEHAVIOR_NORMAL, dtype=jnp.int32),
            action_probability=jnp.array(1.0, dtype=jnp.float32),
        )

    @classmethod
    def strict_vertical(cls) -> "TerritoryTheta":
        """Create hyperparameters for strict vertical split."""
        return cls(
            split_mode=jnp.array(SPLIT_VERTICAL, dtype=jnp.int32),
            strictness=jnp.array(1.0, dtype=jnp.float32),
            shared_margin=jnp.array(0, dtype=jnp.int32),
            rescue_threshold=jnp.array(0.8, dtype=jnp.float32),
            yield_bias=jnp.array(0.5, dtype=jnp.float32),
            behavior_mode=jnp.array(BEHAVIOR_NORMAL, dtype=jnp.int32),
            action_probability=jnp.array(1.0, dtype=jnp.float32),
        )

    @classmethod
    def strict_horizontal(cls) -> "TerritoryTheta":
        """Create hyperparameters for strict horizontal split."""
        return cls(
            split_mode=jnp.array(SPLIT_HORIZONTAL, dtype=jnp.int32),
            strictness=jnp.array(1.0, dtype=jnp.float32),
            shared_margin=jnp.array(0, dtype=jnp.int32),
            rescue_threshold=jnp.array(0.8, dtype=jnp.float32),
            yield_bias=jnp.array(0.5, dtype=jnp.float32),
            behavior_mode=jnp.array(BEHAVIOR_NORMAL, dtype=jnp.int32),
            action_probability=jnp.array(1.0, dtype=jnp.float32),
        )

    @classmethod
    def object_stations(cls) -> "TerritoryTheta":
        """Create hyperparameters for object-based station split."""
        return cls(
            split_mode=jnp.array(SPLIT_OBJECT_STATIONS, dtype=jnp.int32),
            strictness=jnp.array(0.8, dtype=jnp.float32),
            shared_margin=jnp.array(1, dtype=jnp.int32),
            rescue_threshold=jnp.array(0.5, dtype=jnp.float32),
            yield_bias=jnp.array(0.4, dtype=jnp.float32),
            behavior_mode=jnp.array(BEHAVIOR_NORMAL, dtype=jnp.int32),
            action_probability=jnp.array(1.0, dtype=jnp.float32),
        )

    @classmethod
    def flexible(cls) -> "TerritoryTheta":
        """Create hyperparameters for flexible territory (low strictness)."""
        return cls(
            split_mode=jnp.array(SPLIT_VERTICAL, dtype=jnp.int32),
            strictness=jnp.array(0.3, dtype=jnp.float32),
            shared_margin=jnp.array(2, dtype=jnp.int32),
            rescue_threshold=jnp.array(0.3, dtype=jnp.float32),
            yield_bias=jnp.array(0.2, dtype=jnp.float32),
            behavior_mode=jnp.array(BEHAVIOR_NORMAL, dtype=jnp.int32),
            action_probability=jnp.array(1.0, dtype=jnp.float32),
        )


class TerritoryAgentV2(BaseAgentV2):
    """Fast NumPy-based territory agent with territory-based behavior.

    All pathfinding is precomputed at initialization. Runtime execution uses
    pure NumPy operations for maximum speed.

    Split modes:
    - VERTICAL_SPLIT: agent 0 gets left side, agent 1 gets right side
    - HORIZONTAL_SPLIT: agent 0 gets top, agent 1 gets bottom
    - OBJECT_STATIONS: agent 0 gets prep side (pots/ingredients), agent 1 gets service side (goal/plates)
    """

    def __init__(self, layout, theta: TerritoryTheta = None, start_cooking_interaction: bool = False):
        """Initialize the territory agent.

        Args:
            layout: Layout object containing the static grid configuration.
            theta: Hyperparameters for the agent. If None, uses default.
            start_cooking_interaction: Whether the environment requires explicit
                interaction to start cooking (vs auto-cooking when pot is full).
        """
        super().__init__(layout)

        self.theta = theta if theta is not None else TerritoryTheta.default()
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

        # Get theta values as python scalars for precomputation
        self.split_mode_val = int(self.theta.split_mode)
        self.shared_margin_val = int(self.theta.shared_margin)
        self.strictness_val = float(self.theta.strictness)
        self.rescue_threshold_val = float(self.theta.rescue_threshold)
        self.yield_bias_val = float(self.theta.yield_bias)
        self.behavior_mode_val = int(self.theta.behavior_mode)
        self.action_probability_val = float(self.theta.action_probability)

        # Precompute pot-adjacent counter mask
        self._precompute_pot_adjacent_counters_np()

        # Precompute territory masks
        self._precompute_territory_masks_np()

        # Load or compute pathfinding data
        self._load_or_compute_pathfinding()

    def init_agent_state(self, agent_id: int) -> TerritoryAgentState:
        """Initialize territory agent state with deadlock tracking."""
        return TerritoryAgentState(
            agent_id=jnp.array(agent_id, dtype=jnp.int32),
            rng_key=jax.random.PRNGKey(agent_id),
            last_pos_y=jnp.array(-1, dtype=jnp.int32),
            last_pos_x=jnp.array(-1, dtype=jnp.int32),
            stuck_counter=jnp.array(0, dtype=jnp.int32),
        )

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

    def _precompute_territory_masks_np(self):
        """Precompute territory masks based on split_mode using NumPy."""
        height, width = self.height, self.width
        shared_margin = self.shared_margin_val

        # Create coordinate grids
        ys_grid, xs_grid = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

        # VERTICAL_SPLIT: left vs right
        mid_x = width // 2

        # Left territory (agent 0)
        left_strict = xs_grid < mid_x
        left_with_margin = xs_grid < (mid_x + shared_margin)

        # Right territory (agent 1)
        right_strict = xs_grid >= mid_x
        right_with_margin = xs_grid >= (mid_x - shared_margin)

        # Shared zone
        vertical_shared = (xs_grid >= (mid_x - shared_margin)) & (xs_grid < (mid_x + shared_margin))

        self.vertical_territory_0_strict_np = left_strict
        self.vertical_territory_0_with_margin_np = left_with_margin
        self.vertical_territory_1_strict_np = right_strict
        self.vertical_territory_1_with_margin_np = right_with_margin
        self.vertical_shared_np = vertical_shared

        # HORIZONTAL_SPLIT: top vs bottom
        mid_y = height // 2

        # Top territory (agent 0)
        top_strict = ys_grid < mid_y
        top_with_margin = ys_grid < (mid_y + shared_margin)

        # Bottom territory (agent 1)
        bottom_strict = ys_grid >= mid_y
        bottom_with_margin = ys_grid >= (mid_y - shared_margin)

        # Shared zone
        horizontal_shared = (ys_grid >= (mid_y - shared_margin)) & (ys_grid < (mid_y + shared_margin))

        # Check if this split would put agents outside their territories
        # This can happen in small layouts where both agents start in the same row
        # For now, we'll keep the split as is since strictness <= 0.5 will ignore territory anyway
        # But for better behavior, we could fall back to vertical split
        self.horizontal_territory_0_strict_np = top_strict
        self.horizontal_territory_0_with_margin_np = top_with_margin
        self.horizontal_territory_1_strict_np = bottom_strict
        self.horizontal_territory_1_with_margin_np = bottom_with_margin
        self.horizontal_shared_np = horizontal_shared

        # OBJECT_STATIONS: prep side vs service side
        self._precompute_object_station_territories_np(ys_grid, xs_grid)

    def _precompute_object_station_territories_np(self, ys_grid: np.ndarray, xs_grid: np.ndarray):
        """Precompute object-based station territories using NumPy.

        Territory A (agent 0): prep side - closer to pots + ingredient piles
        Territory B (agent 1): service side - closer to goals + plate piles
        """
        # Compute center of mass for prep objects (pots + ingredients)
        prep_mask = self.pot_mask_np | self.ingredient_pile_mask_np
        prep_positions = np.argwhere(prep_mask)
        if len(prep_positions) > 0:
            prep_center_y = np.mean(prep_positions[:, 0])
            prep_center_x = np.mean(prep_positions[:, 1])
        else:
            prep_center_y = self.height // 2
            prep_center_x = self.width // 4

        # Compute center of mass for service objects (goals + plates)
        service_mask = self.goal_mask_np | self.plate_pile_mask_np
        service_positions = np.argwhere(service_mask)
        if len(service_positions) > 0:
            service_center_y = np.mean(service_positions[:, 0])
            service_center_x = np.mean(service_positions[:, 1])
        else:
            service_center_y = self.height // 2
            service_center_x = 3 * self.width // 4

        # Compute distance to each center for each cell
        dist_to_prep = np.abs(xs_grid - prep_center_x) + np.abs(ys_grid - prep_center_y)
        dist_to_service = np.abs(xs_grid - service_center_x) + np.abs(ys_grid - service_center_y)

        # Territory is where distance to one center is less than the other
        margin_tolerance = self.shared_margin_val * 1.5

        self.object_territory_0_strict_np = dist_to_prep < dist_to_service
        self.object_territory_0_with_margin_np = dist_to_prep < (dist_to_service + margin_tolerance)
        self.object_territory_1_strict_np = dist_to_service <= dist_to_prep
        self.object_territory_1_with_margin_np = dist_to_service < (dist_to_prep + margin_tolerance)
        self.object_shared_np = np.abs(dist_to_prep - dist_to_service) < margin_tolerance

        # Check if territories are too imbalanced (one agent has no walkable cells)
        # This can happen in very small layouts where objects are asymmetrically placed
        walkable_0_strict = np.sum(self.object_territory_0_strict_np & self.walkable_mask_np)
        walkable_1_strict = np.sum(self.object_territory_1_strict_np & self.walkable_mask_np)

        if walkable_0_strict == 0 or walkable_1_strict == 0:
            # Fallback to vertical split when object-based split is too imbalanced
            import warnings
            warnings.warn(f"Object-based territory split is imbalanced (walkable cells: {walkable_0_strict} vs {walkable_1_strict}). Falling back to vertical split.")
            mid_x = self.width // 2
            shared_margin = self.shared_margin_val  # Use shared_margin directly, not margin_tolerance
            self.object_territory_0_strict_np = xs_grid < mid_x
            self.object_territory_0_with_margin_np = xs_grid < (mid_x + shared_margin)
            self.object_territory_1_strict_np = xs_grid >= mid_x
            self.object_territory_1_with_margin_np = xs_grid >= (mid_x - shared_margin)
            self.object_shared_np = (xs_grid >= (mid_x - shared_margin)) & (xs_grid < (mid_x + shared_margin))

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
        # Actions: right=0, down=1, left=2, up=3, stay=4, interact=5
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

    def _get_territory_mask_np(self, agent_id: int, strict: bool = True) -> np.ndarray:
        """Get the territory mask for the given agent based on split_mode.

        Args:
            agent_id: Agent ID (0 or 1)
            strict: If True, return strict territory. If False, include shared margin.

        Returns:
            Boolean mask of shape (height, width) indicating agent's territory
        """
        if self.split_mode_val == SPLIT_VERTICAL:
            if agent_id == 0:
                return self.vertical_territory_0_strict_np if strict else self.vertical_territory_0_with_margin_np
            else:
                return self.vertical_territory_1_strict_np if strict else self.vertical_territory_1_with_margin_np
        elif self.split_mode_val == SPLIT_HORIZONTAL:
            if agent_id == 0:
                return self.horizontal_territory_0_strict_np if strict else self.horizontal_territory_0_with_margin_np
            else:
                return self.horizontal_territory_1_strict_np if strict else self.horizontal_territory_1_with_margin_np
        else:  # SPLIT_OBJECT_STATIONS
            if agent_id == 0:
                return self.object_territory_0_strict_np if strict else self.object_territory_0_with_margin_np
            else:
                return self.object_territory_1_strict_np if strict else self.object_territory_1_with_margin_np

    def _get_shared_zone_mask_np(self) -> np.ndarray:
        """Get the shared zone mask based on split_mode."""
        if self.split_mode_val == SPLIT_VERTICAL:
            return self.vertical_shared_np
        elif self.split_mode_val == SPLIT_HORIZONTAL:
            return self.horizontal_shared_np
        else:  # SPLIT_OBJECT_STATIONS
            return self.object_shared_np

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

        # Collision avoidance: if another agent is also 1 step away from the same target cell,
        # agent with higher ID yields based on yield_bias to prevent deadlock
        if len(adjacent_cells) == 1 and agent_id > 0:
            # Single access point - check if other agent is also close
            for other_pos in other_agents_positions:
                other_y, other_x = int(other_pos[0]), int(other_pos[1])
                other_idx = self.pos_to_idx[other_y, other_x]
                if other_idx >= 0:
                    other_dist = self.dist_matrix[other_idx, best_cell[0], best_cell[1]]
                    # If both agents are close to the same cell, yield to avoid collision
                    if other_dist <= 1 and best_dist <= 2:
                        # Use yield_bias to determine if we should wait
                        # Higher yield_bias = more likely to yield
                        if self.yield_bias_val > 0.1:
                            return Actions.stay

        # Get action to move towards best adjacent cell
        action = self.next_action_matrix[pos_idx, agent_dir, best_cell[0], best_cell[1]]

        # Check if this action would move us into a cell occupied by another agent
        if action < 4:  # Movement action (not stay/interact)
            # Compute next position if we take this action
            move_deltas = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # RIGHT, DOWN, LEFT, UP
            dy, dx = move_deltas[action]
            next_y, next_x = pos_y + dy, pos_x + dx

            # Check if blocked by another agent
            for other_pos in other_agents_positions:
                if other_pos[0] == next_y and other_pos[1] == next_x:
                    # Would collide - agent with higher ID or higher yield_bias should move away
                    if agent_id > 0 or self.yield_bias_val > 0.3:
                        # Try to find an alternate cell to move to (move away from conflict)
                        for alt_action in range(4):
                            if alt_action == action:
                                continue
                            alt_dy, alt_dx = move_deltas[alt_action]
                            alt_y, alt_x = pos_y + alt_dy, pos_x + alt_dx
                            if 0 <= alt_y < self.height and 0 <= alt_x < self.width:
                                if self.walkable_mask_np[alt_y, alt_x]:
                                    # Check this alternate cell is not blocked
                                    alt_blocked = False
                                    for op in other_agents_positions:
                                        if op[0] == alt_y and op[1] == alt_x:
                                            alt_blocked = True
                                            break
                                    if not alt_blocked:
                                        return alt_action
                        # No alternate move found - stay
                        return Actions.stay
                    break

        return action

    def _get_reachable_targets_np(self, target_mask: np.ndarray, pos_y: int, pos_x: int) -> np.ndarray:
        """Get targets that are adjacent to reachable walkable cells."""
        idx = self.pos_to_idx[pos_y, pos_x]
        if idx < 0:
            return np.zeros_like(target_mask, dtype=np.bool_)
        return target_mask & self.adjacent_to_reachable[idx]

    def _get_effective_territory_mask_np(self, agent_id: int, strict: bool = True) -> np.ndarray:
        """Get territory mask based on behavior mode.

        For INVADER mode, returns partner's territory.
        For other modes, returns own territory.
        """
        if self.behavior_mode_val == BEHAVIOR_INVADER:
            # Use partner's territory (invade their space)
            partner_id = 1 - agent_id
            return self._get_territory_mask_np(partner_id, strict)
        return self._get_territory_mask_np(agent_id, strict)

    def _get_territory_filtered_targets_np(
        self,
        target_mask: np.ndarray,
        agent_id: int,
        pos_y: int, pos_x: int,
        urgency: float,
        allow_fallback: bool = False,
    ) -> np.ndarray:
        """Filter target mask based on territory and urgency.

        If strictness is high, prefer targets in own territory.
        If urgency exceeds rescue_threshold, allow global targets.

        Args:
            target_mask: Mask of potential targets
            agent_id: Agent ID (0 or 1)
            pos_y, pos_x: Current agent position
            urgency: Urgency score (0-1)
            allow_fallback: If True, fall back to global targets when none in territory

        Returns:
            Filtered target mask (may be empty if strict and no targets in territory)
        """
        # Get territory masks (using effective territory based on behavior mode)
        territory_strict = self._get_effective_territory_mask_np(agent_id, strict=True)
        territory_relaxed = self._get_effective_territory_mask_np(agent_id, strict=False)

        # Get reachable targets
        reachable_targets = self._get_reachable_targets_np(target_mask, pos_y, pos_x)

        # Territory-filtered targets
        territory_targets_strict = reachable_targets & territory_strict
        territory_targets_relaxed = reachable_targets & territory_relaxed

        # Decide based on strictness and urgency
        should_break_territory = urgency > self.rescue_threshold_val

        if should_break_territory:
            return reachable_targets

        if self.strictness_val <= 0.5:
            return reachable_targets

        # High strictness: prefer strict territory, then relaxed
        if np.any(territory_targets_strict):
            return territory_targets_strict
        elif np.any(territory_targets_relaxed):
            return territory_targets_relaxed
        elif allow_fallback:
            return reachable_targets  # Fallback only when explicitly allowed
        else:
            return np.zeros_like(target_mask, dtype=np.bool_)  # Return empty - no valid targets

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

    def _compute_urgency_np(self, pot_cooked: np.ndarray) -> float:
        """Compute urgency score for breaking territory rules.

        Higher urgency when:
        - Pot is cooked and waiting
        """
        num_cooked = np.sum(pot_cooked)
        urgency = min(num_cooked / 2.0, 1.0)
        return urgency

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
                score += 150  # Heavy penalty for staying - prefer moving

            if score < best_score:
                best_score = score
                best_action = action

        return best_action

    def _get_blocker_action_np(
        self,
        pos_y: int, pos_x: int, agent_dir: int,
        other_agents_positions: np.ndarray,
        rng_val: float,
    ) -> int:
        """Get action for BLOCKER behavior - prefer staying in chokepoints."""
        # 70% chance to just stay still
        if rng_val < 0.7:
            return Actions.stay

        # 30% chance to move towards center/chokepoint
        center_y = self.height // 2
        center_x = self.width // 2

        # Move towards center if not there
        dy = np.sign(center_y - pos_y)
        dx = np.sign(center_x - pos_x)

        if abs(dy) > abs(dx) and dy != 0:
            target_y = pos_y + dy
            if 0 <= target_y < self.height and self.walkable_mask_np[target_y, pos_x]:
                return 3 if dy < 0 else 1  # UP or DOWN
        elif dx != 0:
            target_x = pos_x + dx
            if 0 <= target_x < self.width and self.walkable_mask_np[pos_y, target_x]:
                return 2 if dx < 0 else 0  # LEFT or RIGHT

        return Actions.stay

    def _get_hoarder_action_np(
        self,
        pos_y: int, pos_x: int, agent_dir: int,
        inv: int, grid: np.ndarray,
        other_agents_positions: np.ndarray, agent_id: int,
        rng_val: float,
    ) -> Tuple[int, bool]:
        """Get action for HOARDER behavior - pick up items but never complete tasks.

        Returns:
            (action, should_override): action to take, and whether to override normal behavior
        """
        is_empty = inv == DynamicObject.EMPTY
        is_plate = inv == DynamicObject.PLATE
        is_dish = bool(inv & DynamicObject.COOKED) and bool(inv & DynamicObject.PLATE)
        is_ingredient = ((inv >> 2) != 0) and ((inv & DynamicObject.PLATE) == 0)

        # If holding a dish, drop it on a counter (don't deliver!)
        if is_dish:
            empty_counters = self._get_empty_counter_mask_np(grid)
            target_y, target_x, exists = self._get_closest_target_np(empty_counters, pos_y, pos_x)
            if exists:
                return self._get_action_to_adjacent_np(
                    pos_y, pos_x, agent_dir, target_y, target_x,
                    other_agents_positions, agent_id
                ), True
            return Actions.stay, True

        # If holding ingredient, 80% chance to drop on counter instead of using pot
        if is_ingredient and rng_val < 0.8:
            empty_counters = self._get_empty_counter_mask_np(grid)
            target_y, target_x, exists = self._get_closest_target_np(empty_counters, pos_y, pos_x)
            if exists:
                return self._get_action_to_adjacent_np(
                    pos_y, pos_x, agent_dir, target_y, target_x,
                    other_agents_positions, agent_id
                ), True

        # If holding plate, 70% chance to just wander instead of getting soup
        if is_plate and rng_val < 0.7:
            return self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions), True

        # Empty handed - pick something up (normal behavior)
        return Actions.stay, False

    def _get_counter_action_np(
        self,
        pos_y: int, pos_x: int, agent_dir: int,
        inv: int, grid: np.ndarray, recipe: int,
        pot_non_full: np.ndarray, pot_cooked: np.ndarray,
        other_agents_positions: np.ndarray, agent_id: int,
    ) -> Tuple[int, bool]:
        """Get action for COUNTER behavior - ignore urgent tasks, do wrong things.

        Returns:
            (action, should_override): action to take, and whether to override normal behavior
        """
        is_empty = inv == DynamicObject.EMPTY
        is_dish = bool(inv & DynamicObject.COOKED) and bool(inv & DynamicObject.PLATE)

        # If holding dish, drop on counter instead of delivering
        if is_dish:
            empty_counters = self._get_empty_counter_mask_np(grid)
            target_y, target_x, exists = self._get_closest_target_np(empty_counters, pos_y, pos_x)
            if exists:
                return self._get_action_to_adjacent_np(
                    pos_y, pos_x, agent_dir, target_y, target_x,
                    other_agents_positions, agent_id
                ), True
            return Actions.stay, True

        # When empty handed and soup is ready, fetch MORE ingredients instead of plate
        any_cooked = np.any(pot_cooked)
        if is_empty and any_cooked:
            # Fetch ingredients (opposite of what we should do)
            recipe_pile_mask = self._get_recipe_ingredient_pile_mask_np(grid, recipe)
            if np.any(recipe_pile_mask):
                target_y, target_x, exists = self._get_closest_target_np(recipe_pile_mask, pos_y, pos_x)
                if exists:
                    return self._get_action_to_adjacent_np(
                        pos_y, pos_x, agent_dir, target_y, target_x,
                        other_agents_positions, agent_id
                    ), True

        return Actions.stay, False

    def _get_invader_territory_mask_np(self, agent_id: int, strict: bool = True) -> np.ndarray:
        """Get the PARTNER's territory mask (opposite of normal) for INVADER behavior."""
        # Return partner's territory (opposite agent_id)
        partner_id = 1 - agent_id
        return self._get_territory_mask_np(partner_id, strict)

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

    def get_action(
        self,
        obs,
        env_state,
        agent_state: TerritoryAgentState = None,
    ) -> Tuple[jnp.ndarray, TerritoryAgentState]:
        """Get action for the agent (main entry point).

        Uses NumPy for all computation, returns JAX arrays for compatibility.

        Args:
            obs: Flattened observation (ignored, we use env_state directly)
            env_state: Full environment state
            agent_state: Agent's internal state with deadlock tracking

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

        # Get other agents positions
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

        # Get recipe for ingredient selection
        recipe = int(env_state.recipe) if hasattr(env_state, 'recipe') else 0

        # Determine inventory state
        is_empty = inv == DynamicObject.EMPTY
        is_plate = inv == DynamicObject.PLATE
        is_dish = bool(inv & DynamicObject.COOKED) and bool(inv & DynamicObject.PLATE)
        is_ingredient = ((inv >> 2) != 0) and ((inv & DynamicObject.PLATE) == 0)

        # Get pot states
        pot_mask, pot_non_full, pot_cooking, pot_cooked = self._get_pot_states_np(grid)

        # Compute urgency for territory-breaking decisions
        urgency = self._compute_urgency_np(pot_cooked)

        # Generate random value for stochastic behavior modes
        rng, subkey = jax.random.split(agent_state.rng_key)
        rng_val = float(jax.random.uniform(subkey))

        # === BEHAVIOR MODE HANDLING ===
        behavior_mode = self.behavior_mode_val

        # BEHAVIOR_LAZY: Skip action with probability (1 - action_probability)
        if behavior_mode == BEHAVIOR_LAZY:
            if rng_val > self.action_probability_val:
                return self._return_action(Actions.stay, agent_state, agent_id, pos_y, pos_x)

        # BEHAVIOR_BLOCKER: Prefer staying in chokepoints
        if behavior_mode == BEHAVIOR_BLOCKER:
            action = self._get_blocker_action_np(pos_y, pos_x, agent_dir, other_agents_positions, rng_val)
            return self._return_action(action, agent_state, agent_id, pos_y, pos_x)

        # BEHAVIOR_HOARDER: Pick up items but drop on counters instead of completing
        if behavior_mode == BEHAVIOR_HOARDER:
            action, should_override = self._get_hoarder_action_np(
                pos_y, pos_x, agent_dir, inv, grid,
                other_agents_positions, agent_id, rng_val
            )
            if should_override:
                return self._return_action(action, agent_state, agent_id, pos_y, pos_x)
            # Fall through to normal behavior for picking up items

        # BEHAVIOR_COUNTER: Ignore urgent tasks, do wrong things
        if behavior_mode == BEHAVIOR_COUNTER:
            action, should_override = self._get_counter_action_np(
                pos_y, pos_x, agent_dir, inv, grid, recipe,
                pot_non_full, pot_cooked, other_agents_positions, agent_id
            )
            if should_override:
                return self._return_action(action, agent_state, agent_id, pos_y, pos_x)
            # Fall through to normal behavior

        # DEADLOCK DETECTION: If stuck for too long, force a yield/wander action
        stuck_count = int(agent_state.stuck_counter)
        if stuck_count >= 3:  # Stuck for 3+ steps
            # Force wander to break deadlock
            action = self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)
            return self._return_action(action, agent_state, agent_id, pos_y, pos_x)

        # Priority 0: If holding a dish, deliver it (must deliver - allow fallback)
        if is_dish:
            goal_targets = self._get_territory_filtered_targets_np(
                self.goal_mask_np, agent_id, pos_y, pos_x, urgency, allow_fallback=True
            )
            target_y, target_x, exists = self._get_closest_target_np(goal_targets, pos_y, pos_x)
            if exists:
                action = self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)
            else:
                action = self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)
            return self._return_action(action, agent_state, agent_id, pos_y, pos_x)

        # Priority 1: If holding an ingredient, go to pot in territory or drop on counter
        if is_ingredient:
            # Try to find pot in territory first
            pot_targets = self._get_territory_filtered_targets_np(
                pot_non_full, agent_id, pos_y, pos_x, urgency, allow_fallback=False
            )
            target_y, target_x, exists = self._get_closest_target_np(pot_targets, pos_y, pos_x)

            if not exists:
                # No pot in territory - try breaking territory to find any pot
                pot_targets = self._get_territory_filtered_targets_np(
                    pot_non_full, agent_id, pos_y, pos_x, urgency, allow_fallback=True
                )
                target_y, target_x, exists = self._get_closest_target_np(pot_targets, pos_y, pos_x)

            if exists:
                action = self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)
            else:
                # No pot available - drop ingredient on counter for other agent
                # Prefer counters near the shared zone / boundary
                empty_counters = self._get_empty_counter_mask_np(grid)
                # Use allow_fallback=True to ensure we can drop somewhere
                counter_targets = self._get_territory_filtered_targets_np(
                    empty_counters, agent_id, pos_y, pos_x, urgency, allow_fallback=True
                )
                target_y, target_x, exists = self._get_closest_target_np(counter_targets, pos_y, pos_x)
                if exists:
                    action = self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)
                else:
                    action = self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)
            return self._return_action(action, agent_state, agent_id, pos_y, pos_x)

        # Priority 2: If holding a plate, get cooked soup or wait near cooking pot
        if is_plate:
            # Try to find cooked pot in territory
            cooked_targets = self._get_territory_filtered_targets_np(
                pot_cooked, agent_id, pos_y, pos_x, urgency, allow_fallback=False
            )
            target_y, target_x, exists = self._get_closest_target_np(cooked_targets, pos_y, pos_x)
            if exists:
                action = self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)
            else:
                # Try cooking pot in territory
                cooking_targets = self._get_territory_filtered_targets_np(
                    pot_cooking, agent_id, pos_y, pos_x, urgency, allow_fallback=False
                )
                target_y, target_x, exists = self._get_closest_target_np(cooking_targets, pos_y, pos_x)
                if exists:
                    action = self._hover_near_pot_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)
                else:
                    # No pot activity in territory - drop plate on counter and do something else
                    empty_counters = self._get_empty_counter_mask_np(grid)
                    counter_targets = self._get_territory_filtered_targets_np(
                        empty_counters, agent_id, pos_y, pos_x, urgency, allow_fallback=True
                    )
                    target_y, target_x, exists = self._get_closest_target_np(counter_targets, pos_y, pos_x)
                    if exists:
                        action = self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)
                    else:
                        action = self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)
            return self._return_action(action, agent_state, agent_id, pos_y, pos_x)

        # Empty handed: decide what to fetch
        any_cooked = np.any(pot_cooked)
        any_cooking = np.any(pot_cooking)
        any_non_full = np.any(pot_non_full)

        # If soup is ready, get plate
        if any_cooked:
            plate_targets = self._get_territory_filtered_targets_np(
                self.plate_pile_mask_np, agent_id, pos_y, pos_x, urgency
            )
            target_y, target_x, exists = self._get_closest_target_np(plate_targets, pos_y, pos_x)
            if exists:
                action = self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)
                return self._return_action(action, agent_state, agent_id, pos_y, pos_x)

        # If pots need ingredients, fetch ingredients
        if any_non_full or not any_cooking:
            # First check for staged ingredients
            staged = self._get_staged_ingredient_mask_np(grid)
            staged_targets = self._get_territory_filtered_targets_np(
                staged, agent_id, pos_y, pos_x, urgency
            )
            target_y, target_x, exists = self._get_closest_target_np(staged_targets, pos_y, pos_x)
            if exists:
                action = self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)
                return self._return_action(action, agent_state, agent_id, pos_y, pos_x)

            # Fetch from pile - use recipe-aware selection
            recipe_pile_mask = self._get_recipe_ingredient_pile_mask_np(grid, recipe)
            pile_targets = self._get_territory_filtered_targets_np(
                recipe_pile_mask, agent_id, pos_y, pos_x, urgency, allow_fallback=True
            )
            target_y, target_x, exists = self._get_closest_target_np(pile_targets, pos_y, pos_x)
            if exists:
                action = self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)
                return self._return_action(action, agent_state, agent_id, pos_y, pos_x)

        # Default: get plate for when cooking finishes
        plate_targets = self._get_territory_filtered_targets_np(
            self.plate_pile_mask_np, agent_id, pos_y, pos_x, urgency
        )
        target_y, target_x, exists = self._get_closest_target_np(plate_targets, pos_y, pos_x)
        if exists:
            action = self._get_action_to_adjacent_np(pos_y, pos_x, agent_dir, target_y, target_x, other_agents_positions, agent_id)
        else:
            action = self._wander_action_np(pos_y, pos_x, agent_dir, other_agents_positions)

        return self._return_action(action, agent_state, agent_id, pos_y, pos_x)

    def _return_action(
        self,
        action: int,
        agent_state: TerritoryAgentState,
        agent_id: int,
        pos_y: int,
        pos_x: int,
    ) -> Tuple[jnp.ndarray, TerritoryAgentState]:
        """Return action and updated state with deadlock tracking."""
        # Update RNG state
        rng, _ = jax.random.split(agent_state.rng_key)

        # Check if agent moved
        last_y = int(agent_state.last_pos_y)
        last_x = int(agent_state.last_pos_x)
        moved = (last_y != pos_y or last_x != pos_x) if last_y >= 0 else True

        # Update stuck counter
        new_stuck = 0 if moved else int(agent_state.stuck_counter) + 1

        new_state = TerritoryAgentState(
            agent_id=jnp.array(agent_id, dtype=jnp.int32),
            rng_key=rng,
            last_pos_y=jnp.array(pos_y, dtype=jnp.int32),
            last_pos_x=jnp.array(pos_x, dtype=jnp.int32),
            stuck_counter=jnp.array(new_stuck, dtype=jnp.int32),
        )
        return jnp.array(action, dtype=jnp.int32), new_state
