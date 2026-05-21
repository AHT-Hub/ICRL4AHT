"""Static agent that always stays in place."""

from envs.overcooked_v2.common import Actions

from .base_agent import AgentState, BaseAgentV2


class StaticAgentV2(BaseAgentV2):
    def get_action(self, obs, env_state, agent_state: AgentState | None = None):
        if agent_state is None:
            agent_state = self.init_agent_state(0)
        return int(Actions.stay), agent_state
