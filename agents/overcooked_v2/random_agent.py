"""Random policy for OvercookedV2."""

import jax
import jax.numpy as jnp

from envs.overcooked_v2.common import Actions

from .base_agent import AgentState, BaseAgentV2


class RandomAgentV2(BaseAgentV2):
    def get_action(self, obs, env_state, agent_state: AgentState | None = None):
        if agent_state is None:
            agent_state = self.init_agent_state(0)
        rng, sub = jax.random.split(agent_state.rng_key)
        action = jax.random.randint(sub, (), 0, len(Actions), dtype=jnp.int32)
        return action, AgentState(agent_id=agent_state.agent_id, rng_key=rng)
