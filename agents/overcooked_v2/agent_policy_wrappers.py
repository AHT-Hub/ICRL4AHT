"""Policy wrappers for OvercookedV2 heuristic agents."""

import jax
import jax.numpy as jnp

from agents.agent_interface import AgentPolicy
from .base_agent import AgentState

from .assembly_line_agent import AssemblyLineAgentV2, AssemblyLineTheta
from .random_agent import RandomAgentV2
from .static_agent import StaticAgentV2
from .territory_agent import TerritoryAgentV2, TerritoryTheta, TerritoryAgentState
from .utility_greedy_agent import UtilityGreedyAgentV2, UtilityGreedyTheta, UtilityAgentState
from .recipe_aware_button_agent import RecipeAwareButtonAgentV2, RecipeAwareButtonTheta, RecipeAwareButtonAgentState


class _BaseWrapper(AgentPolicy):
    def __init__(self, policy, using_log_wrapper: bool = False):
        super().__init__(action_dim=6, obs_dim=None)
        self.policy = policy
        self.using_log_wrapper = using_log_wrapper

    def init_hstate(self, batch_size: int, aux_info=None):
        return self.policy.init_agent_state(aux_info["agent_id"])

    def _unwrap_env_state(self, env_state):
        return env_state.env_state if self.using_log_wrapper else env_state


class OvercookedV2RandomPolicyWrapper(_BaseWrapper):
    def __init__(self, layout, using_log_wrapper: bool = False):
        super().__init__(RandomAgentV2(layout), using_log_wrapper)

    def get_action(self, params, obs, done, avail_actions, hstate, rng, env_state=None, aux_obs=None, test_mode=False):
        env_state = self._unwrap_env_state(env_state)
        action, new_hstate = self.policy.get_action(obs, env_state, hstate)
        new_hstate = jax.lax.cond(done.squeeze(), lambda: self.policy.init_agent_state(hstate.agent_id), lambda: new_hstate)
        return action, new_hstate


class OvercookedV2StaticPolicyWrapper(_BaseWrapper):
    def __init__(self, layout, using_log_wrapper: bool = False):
        super().__init__(StaticAgentV2(layout), using_log_wrapper)

    def get_action(self, params, obs, done, avail_actions, hstate, rng, env_state=None, aux_obs=None, test_mode=False):
        env_state = self._unwrap_env_state(env_state)
        action, new_hstate = self.policy.get_action(obs, env_state, hstate)
        new_hstate = jax.lax.cond(done.squeeze(), lambda: self.policy.init_agent_state(hstate.agent_id), lambda: new_hstate)
        return action, new_hstate


class OvercookedV2AssemblyLinePolicyWrapper(_BaseWrapper):
    """Policy wrapper for the AssemblyLine agent family.

    This wrapper is fully JAX-traceable and supports jit/vmap/scan.

    Args:
        layout: Layout object or string name of layout
        theta: AssemblyLineTheta hyperparameters (optional, uses defaults if None)
        using_log_wrapper: Whether the env state is wrapped in LogWrapper
        start_cooking_interaction: Whether explicit interaction is needed to start cooking
    """

    def __init__(
        self,
        layout,
        theta: AssemblyLineTheta = None,
        using_log_wrapper: bool = False,
        start_cooking_interaction: bool = False,
    ):
        super().__init__(
            AssemblyLineAgentV2(
                layout,
                theta=theta,
                start_cooking_interaction=start_cooking_interaction,
            ),
            using_log_wrapper,
        )

    def get_action(
        self,
        params,
        obs,
        done,
        avail_actions,
        hstate,
        rng,
        env_state=None,
        aux_obs=None,
        test_mode=False,
    ):
        """Get action from the assembly line agent.

        Args:
            params: Network parameters (unused for heuristic agents)
            obs: Flattened observation
            done: Done flag
            avail_actions: Available actions mask (unused)
            hstate: AgentState
            rng: JAX random key
            env_state: Full environment state
            aux_obs: Auxiliary observations (unused)
            test_mode: Test/eval mode flag (unused)

        Returns:
            Tuple of (action, new_hstate)
        """
        # Use pure_callback to escape JAX tracing for the heuristic agent
        # This allows NumPy/Python operations inside get_action
        def _eager_get_action(obs, env_state, hstate):
            # Unwrap inside callback where we have concrete values
            # LogEnvState -> WrappedEnvState -> actual env state
            unwrapped = env_state.env_state if self.using_log_wrapper else env_state
            # WrappedEnvState -> actual env state (with .agents)
            actual_env_state = unwrapped.env_state
            action, new_hstate = self.policy.get_action(obs, actual_env_state, hstate)
            return action, new_hstate.agent_id, new_hstate.rng_key

        # Define output shape for pure_callback
        result_shape = (
            jax.ShapeDtypeStruct((), jnp.int32),  # action
            jax.ShapeDtypeStruct((), jnp.int32),  # agent_id
            jax.ShapeDtypeStruct((2,), jnp.uint32),  # rng_key
        )

        action, agent_id, rng_key = jax.pure_callback(
            _eager_get_action,
            result_shape,
            obs, env_state, hstate,
        )
        new_hstate = AgentState(agent_id=agent_id, rng_key=rng_key)

        # Reset state on episode done
        new_hstate = jax.lax.cond(
            done.squeeze(),
            lambda: self.policy.init_agent_state(hstate.agent_id),
            lambda: new_hstate,
        )
        return action, new_hstate


class OvercookedV2TerritoryPolicyWrapper(_BaseWrapper):
    """Policy wrapper for the Territory agent family.

    This wrapper is fully JAX-traceable and supports jit/vmap/scan.

    Args:
        layout: Layout object or string name of layout
        theta: TerritoryTheta hyperparameters (optional, uses defaults if None)
        using_log_wrapper: Whether the env state is wrapped in LogWrapper
        start_cooking_interaction: Whether explicit interaction is needed to start cooking
    """

    def __init__(
        self,
        layout,
        theta: TerritoryTheta = None,
        using_log_wrapper: bool = False,
        start_cooking_interaction: bool = False,
    ):
        super().__init__(
            TerritoryAgentV2(
                layout,
                theta=theta,
                start_cooking_interaction=start_cooking_interaction,
            ),
            using_log_wrapper,
        )

    def get_action(
        self,
        params,
        obs,
        done,
        avail_actions,
        hstate,
        rng,
        env_state=None,
        aux_obs=None,
        test_mode=False,
    ):
        """Get action from the territory agent.

        Args:
            params: Network parameters (unused for heuristic agents)
            obs: Flattened observation
            done: Done flag
            avail_actions: Available actions mask (unused)
            hstate: TerritoryAgentState
            rng: JAX random key
            env_state: Full environment state
            aux_obs: Auxiliary observations (unused)
            test_mode: Test/eval mode flag (unused)

        Returns:
            Tuple of (action, new_hstate)
        """
        # Use pure_callback to escape JAX tracing for the heuristic agent
        # This allows NumPy/Python operations inside get_action
        def _eager_get_action(obs, env_state, hstate):
            # Unwrap inside callback where we have concrete values
            # LogEnvState -> WrappedEnvState -> actual env state
            unwrapped = env_state.env_state if self.using_log_wrapper else env_state
            # WrappedEnvState -> actual env state (with .agents)
            actual_env_state = unwrapped.env_state
            action, new_hstate = self.policy.get_action(obs, actual_env_state, hstate)
            return (
                action,
                new_hstate.agent_id,
                new_hstate.rng_key,
                new_hstate.last_pos_y,
                new_hstate.last_pos_x,
                new_hstate.stuck_counter,
            )

        # Define output shape for pure_callback
        result_shape = (
            jax.ShapeDtypeStruct((), jnp.int32),  # action
            jax.ShapeDtypeStruct((), jnp.int32),  # agent_id
            jax.ShapeDtypeStruct((2,), jnp.uint32),  # rng_key
            jax.ShapeDtypeStruct((), jnp.int32),  # last_pos_y
            jax.ShapeDtypeStruct((), jnp.int32),  # last_pos_x
            jax.ShapeDtypeStruct((), jnp.int32),  # stuck_counter
        )

        action, agent_id, rng_key, last_pos_y, last_pos_x, stuck_counter = jax.pure_callback(
            _eager_get_action,
            result_shape,
            obs, env_state, hstate,
        )
        new_hstate = TerritoryAgentState(
            agent_id=agent_id,
            rng_key=rng_key,
            last_pos_y=last_pos_y,
            last_pos_x=last_pos_x,
            stuck_counter=stuck_counter,
        )

        # Reset state on episode done
        new_hstate = jax.lax.cond(
            done.squeeze(),
            lambda: self.policy.init_agent_state(int(hstate.agent_id)),
            lambda: new_hstate,
        )
        return action, new_hstate


class OvercookedV2UtilityGreedyPolicyWrapper(_BaseWrapper):
    """Policy wrapper for the UtilityGreedy agent family.

    This wrapper is fully JAX-traceable and supports jit/vmap/scan.
    The agent enumerates candidate intents, scores them with a weighted utility,
    picks the best, and executes it.

    Args:
        layout: Layout object or string name of layout
        theta: UtilityGreedyTheta hyperparameters (optional, uses defaults if None)
        using_log_wrapper: Whether the env state is wrapped in LogWrapper
        start_cooking_interaction: Whether explicit interaction is needed to start cooking
    """

    def __init__(
        self,
        layout,
        theta: UtilityGreedyTheta = None,
        using_log_wrapper: bool = False,
        start_cooking_interaction: bool = False,
    ):
        super().__init__(
            UtilityGreedyAgentV2(
                layout,
                theta=theta,
                start_cooking_interaction=start_cooking_interaction,
            ),
            using_log_wrapper,
        )

    def get_action(
        self,
        params,
        obs,
        done,
        avail_actions,
        hstate,
        rng,
        env_state=None,
        aux_obs=None,
        test_mode=False,
    ):
        """Get action from the utility greedy agent.

        Args:
            params: Network parameters (unused for heuristic agents)
            obs: Flattened observation
            done: Done flag
            avail_actions: Available actions mask (unused)
            hstate: UtilityAgentState
            rng: JAX random key
            env_state: Full environment state
            aux_obs: Auxiliary observations (unused)
            test_mode: Test/eval mode flag (unused)

        Returns:
            Tuple of (action, new_hstate)
        """
        # Use pure_callback to escape JAX tracing for the heuristic agent
        # This allows NumPy/Python operations inside get_action
        def _eager_get_action(obs, env_state, hstate):
            # Unwrap inside callback where we have concrete values
            # LogEnvState -> WrappedEnvState -> actual env state
            unwrapped = env_state.env_state if self.using_log_wrapper else env_state
            # WrappedEnvState -> actual env state (with .agents)
            actual_env_state = unwrapped.env_state
            action, new_hstate = self.policy.get_action(obs, actual_env_state, hstate)
            return action, new_hstate.agent_id, new_hstate.rng_key, new_hstate.last_intent, new_hstate.wait_counter

        # Define output shape for pure_callback
        result_shape = (
            jax.ShapeDtypeStruct((), jnp.int32),  # action
            jax.ShapeDtypeStruct((), jnp.int32),  # agent_id
            jax.ShapeDtypeStruct((2,), jnp.uint32),  # rng_key
            jax.ShapeDtypeStruct((), jnp.int32),  # last_intent
            jax.ShapeDtypeStruct((), jnp.int32),  # wait_counter
        )

        action, agent_id, rng_key, last_intent, wait_counter = jax.pure_callback(
            _eager_get_action,
            result_shape,
            obs, env_state, hstate,
        )
        new_hstate = UtilityAgentState(agent_id=agent_id, rng_key=rng_key, last_intent=last_intent, wait_counter=wait_counter)

        # Reset state on episode done
        new_hstate = jax.lax.cond(
            done.squeeze(),
            lambda: self.policy.init_agent_state(hstate.agent_id),
            lambda: new_hstate,
        )
        return action, new_hstate


class OvercookedV2RecipeAwareButtonPolicyWrapper(_BaseWrapper):
    """Policy wrapper for the RecipeAwareButton agent family.

    This wrapper is fully JAX-traceable and supports jit/vmap/scan.
    The agent maintains a belief/memory of the recipe encoding and uses
    the L button (BUTTON_RECIPE_INDICATOR) to learn the recipe when unknown.

    Args:
        layout: Layout object or string name of layout
        theta: RecipeAwareButtonTheta hyperparameters (optional, uses defaults if None)
        using_log_wrapper: Whether the env state is wrapped in LogWrapper
        start_cooking_interaction: Whether explicit interaction is needed to start cooking
    """

    def __init__(
        self,
        layout,
        theta: RecipeAwareButtonTheta = None,
        using_log_wrapper: bool = False,
        start_cooking_interaction: bool = False,
    ):
        super().__init__(
            RecipeAwareButtonAgentV2(
                layout,
                theta=theta,
                start_cooking_interaction=start_cooking_interaction,
            ),
            using_log_wrapper,
        )

    def get_action(
        self,
        params,
        obs,
        done,
        avail_actions,
        hstate,
        rng,
        env_state=None,
        aux_obs=None,
        test_mode=False,
    ):
        """Get action from the recipe-aware button agent.

        Args:
            params: Network parameters (unused for heuristic agents)
            obs: Flattened observation
            done: Done flag
            avail_actions: Available actions mask (unused)
            hstate: RecipeAwareButtonAgentState
            rng: JAX random key
            env_state: Full environment state
            aux_obs: Auxiliary observations (unused)
            test_mode: Test/eval mode flag (unused)

        Returns:
            Tuple of (action, new_hstate)
        """
        # Use pure_callback to escape JAX tracing for the heuristic agent
        # This allows NumPy/Python operations inside get_action
        def _eager_get_action(obs, env_state, hstate):
            # Unwrap inside callback where we have concrete values
            # LogEnvState -> WrappedEnvState -> actual env state
            unwrapped = env_state.env_state if self.using_log_wrapper else env_state
            # WrappedEnvState -> actual env state (with .agents)
            actual_env_state = unwrapped.env_state
            action, new_hstate = self.policy.get_action(obs, actual_env_state, hstate)
            return (
                action,
                new_hstate.agent_id,
                new_hstate.rng_key,
                new_hstate.known_recipe,
                new_hstate.last_recipe_time,
                new_hstate.last_intent,
            )

        # Define output shape for pure_callback
        result_shape = (
            jax.ShapeDtypeStruct((), jnp.int32),  # action
            jax.ShapeDtypeStruct((), jnp.int32),  # agent_id
            jax.ShapeDtypeStruct((2,), jnp.uint32),  # rng_key
            jax.ShapeDtypeStruct((), jnp.int32),  # known_recipe
            jax.ShapeDtypeStruct((), jnp.int32),  # last_recipe_time
            jax.ShapeDtypeStruct((), jnp.int32),  # last_intent
        )

        action, agent_id, rng_key, known_recipe, last_recipe_time, last_intent = jax.pure_callback(
            _eager_get_action,
            result_shape,
            obs, env_state, hstate,
        )
        new_hstate = RecipeAwareButtonAgentState(
            agent_id=agent_id,
            rng_key=rng_key,
            known_recipe=known_recipe,
            last_recipe_time=last_recipe_time,
            last_intent=last_intent,
        )

        # Reset state on episode done
        new_hstate = jax.lax.cond(
            done.squeeze(),
            lambda: self.policy.init_agent_state(hstate.agent_id),
            lambda: new_hstate,
        )
        return action, new_hstate
