from functools import partial
import jax
import jax.numpy as jnp


class AgentPopulation:
    '''Base class for a population of homogeneous agents
    TODO: develop more complex population classes that can handle heterogeneous agents
    '''
    def __init__(self, pop_size, policy_cls):
        '''
        Args:
            pop_size: int, number of agents in the population
            policy_cls: an instance of the AgentPolicy class. The policy class for the population of agents
        '''
        self.pop_size = pop_size
        self.policy_cls = policy_cls # AgentPolicy class

    def sample_agent_indices(self, n, rng):
        '''Sample n indices from the population, with replacement.'''
        return jax.random.randint(rng, (n,), 0, self.pop_size)
    
    def gather_agent_params(self, pop_params, agent_indices):
        '''Gather the parameters of the agents specified by agent_indices.

        Args:
            pop_params: pytree of parameters for the population of agents of shape (pop_size, ...).
            agent_indices: indices with shape (num_envs,), each in [0, pop_size)
        '''
        def gather_leaf(leaf):
            # leaf shape: (num_envs,  ...)
            return jax.vmap(lambda idx: leaf[idx])(agent_indices)
        return jax.tree.map(gather_leaf, pop_params)
    
    def get_actions(self, pop_params, agent_indices, obs, done, avail_actions, hstate, rng, 
                    env_state=None, aux_obs=None, test_mode=False):
        '''
        Get the actions of the agents specified by agent_indices. 
        
        Args:
            pop_params: pytree of parameters for the population of agents of shape (pop_size, ...).
            agent_indices: indices with shape (num_envs,), each in [0, pop_size)
            obs: observations with shape (num_envs, ...) 
            done: done flags with shape (num_envs,)
            avail_actions: available actions with shape (num_envs, num_actions)
            hstate: hidden state with shape (num_envs, ...) or None if policy doesn't use hidden state
            rng: random key
            env_state: environment state with shape (num_envs, ...) or None if policy doesn't use env state
            aux_obs: an optional auxiliary vector to append to the observation
        Returns:
            actions: actions with shape (num_envs,)
            new_hstate: new hidden state with shape (num_envs, ...) or None
        '''
        gathered_params = self.gather_agent_params(pop_params, agent_indices)
        num_envs = agent_indices.squeeze().shape[0]
        rngs_batched = jax.random.split(rng, num_envs)
        vmapped_get_action = jax.vmap(partial(self.policy_cls.get_action, 
                                              aux_obs=aux_obs, 
                                              env_state=env_state, 
                                              test_mode=test_mode))
        actions, new_hstate = vmapped_get_action(
            gathered_params, obs, done, avail_actions, hstate, 
            rngs_batched)
        return actions, new_hstate
    
    def init_hstate(self, n: int, aux_info: dict=None):
        '''Initialize the hidden state for n members of the population.'''
        return self.policy_cls.init_hstate(n, aux_info)

class DummyPolicyPopulation(AgentPopulation):
    '''A wrapper around the AgentPopulation class that allows for a single policy to be used.
    The main difference from the AgentPopulation is that the test mode is a class attribute, 
    so it remains static for the lifetime of the object
    '''
    def __init__(self, policy_cls, test_mode=False):
        super().__init__(pop_size=1, policy_cls=policy_cls)
        self.test_mode = test_mode
    
    def get_actions(self, pop_params, agent_indices, obs, done, avail_actions, hstate, rng, 
                    env_state=None, aux_obs=None):
        '''
        Get the actions of the agents specified by agent_indices. Does not support agents that 
        require auxiliary observations.
        Returns:
            actions: actions with shape (num_envs,)
            new_hstate: new hidden state with shape (num_envs, ...) or None
        '''
        gathered_params = self.gather_agent_params(pop_params, agent_indices)
        num_envs = agent_indices.squeeze().shape[0]
        rngs_batched = jax.random.split(rng, num_envs)
        vmapped_get_action = jax.vmap(partial(self.policy_cls.get_action, 
                                              aux_obs=aux_obs, 
                                              env_state=env_state, 
                                              test_mode=self.test_mode))
        actions, new_hstate = vmapped_get_action(
            gathered_params, obs, done, avail_actions, hstate, 
            rngs_batched)
        return actions, new_hstate

    def init_hstate(self, n: int):
        '''Initialize the hidden state for n members of the population.

        For vmap compatibility, we need shape (n, 1, 1, hidden_dim) so each
        vmapped call gets (1, 1, hidden_dim) = (1, batch_size=1, hidden_dim).
        '''
        # Get hstate for single env with batch_size=1: (1, 1, hidden_dim)
        single_hstate = self.policy_cls.init_hstate(1)
        if single_hstate is not None:
            # Stack n copies along new axis 0: (n, 1, 1, hidden_dim)
            # We add a new axis then repeat, since tile repeats along existing dims
            hstate = jax.tree.map(
                lambda x: jnp.repeat(x[jnp.newaxis, ...], n, axis=0),
                single_hstate
            )
            return hstate
        return None

class HeuristicPolicyPopulation(AgentPopulation):
    '''A wrapper around the AgentPopulation class that allows for a heuristic policy to be used.
    The main difference from the AgentPopulation is that:
    - test mode is not used b/c heuristic agents do not have a test mode
    - get_actions requires the environment state
    - the init_hstate method is overridden to vmap over the hidden state initialization.
    '''
    def __init__(self, policy_cls):
        super().__init__(pop_size=1, policy_cls=policy_cls)

    def get_actions(self, pop_params, agent_indices, obs, done, avail_actions, hstate, rng, 
                    env_state, aux_obs=None):
        '''
        Get the actions of the agents specified by agent_indices. Requires env_state. 
        Does not support agents that require auxiliary observations.
        Returns:
            actions: actions with shape (num_envs,)
            new_hstate: new hidden state with shape (num_envs, ...) or None
        '''
        gathered_params = self.gather_agent_params(pop_params, agent_indices)
        num_envs = agent_indices.squeeze().shape[0]
        rngs_batched = jax.random.split(rng, num_envs)
        
        def _policy_cls_get_action(params, obs, done, avail_actions, hstate, rng, env_state
                                   ):
            return self.policy_cls.get_action(params=params, obs=obs, done=done, 
                                              avail_actions=avail_actions, hstate=hstate, 
                                              rng=rng, env_state=env_state, 
                                              aux_obs=None, test_mode=False)
        vmapped_get_action = jax.vmap(_policy_cls_get_action)
        actions, new_hstate = vmapped_get_action(
            params=gathered_params, 
            obs=obs, 
            done=done, 
            avail_actions=avail_actions, 
            hstate=hstate, 
            rng=rngs_batched, 
            env_state=env_state)
        return actions, new_hstate

    def init_hstate(self, n: int):
        '''Initialize the hidden state for n members of the population.'''
        # partner agent is always agent 1 in the ppo_ego training code
        vmap_dummy_input = jnp.ones(n)
        return jax.vmap(partial(self.policy_cls.init_hstate, aux_info={"agent_id": 1}))(vmap_dummy_input)