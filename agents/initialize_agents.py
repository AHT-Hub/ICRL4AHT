import jax

from agents.mlp_actor_critic_agent import MLPActorCriticPolicy, ActorWithDoubleCriticPolicy, \
    ActorWithConditionalCriticPolicy, PseudoActorWithDoubleCriticPolicy, \
    PseudoActorWithConditionalCriticPolicy
from agents.rnn_actor_critic_agent import RNNActorCriticPolicy
from agents.cnn_rnn_actor_critic_agent import CNNRNNActorCriticPolicy
from agents.s5_actor_critic_agent import S5ActorCriticPolicy


def initialize_s5_agent(config, env, rng):
    """Initialize an S5 agent with the given config.

    Args:
        config: dict, config for the agent
        env: gymnasium environment
        rng: jax.random.PRNGKey, random key for initialization

    Returns:
        policy: S5ActorCriticPolicy, the policy object
        params: dict, initial parameters for the agent
    """
    # Create the S5 policy with direct parameters
    policy = S5ActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        d_model=config.get("S5_D_MODEL", 128),
        ssm_size=config.get("S5_SSM_SIZE", 128),
        # d_model=config.get("S5_D_MODEL", 16),
        # ssm_size=config.get("S5_SSM_SIZE", 16),
        ssm_n_layers=config.get("S5_N_LAYERS", 2),
        blocks=config.get("S5_BLOCKS", 1),
        fc_hidden_dim=config.get("S5_ACTOR_CRITIC_HIDDEN_DIM", 1024),
        fc_n_layers=config.get("FC_N_LAYERS", 3),
        # fc_hidden_dim=config.get("S5_ACTOR_CRITIC_HIDDEN_DIM", 64),
        # fc_n_layers=config.get("FC_N_LAYERS", 2),
        s5_activation=config.get("S5_ACTIVATION", "full_glu"),
        s5_do_norm=config.get("S5_DO_NORM", True),
        s5_prenorm=config.get("S5_PRENORM", True),
        s5_do_gtrxl_norm=config.get("S5_DO_GTRXL_NORM", True),
    )

    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_rnn_agent(config, env, rng):
    """Initialize an RNN agent with the given config.

    Args:
        config: dict, config for the agent
        env: gymnasium environment
        rng: jax.random.PRNGKey, random key for initialization

    Returns:
        policy: RNNActorCriticPolicy, the policy object
        params: dict, initial parameters for the agent
    """
    # Create the RNN policy
    policy = RNNActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        activation=config.get("ACTIVATION", "tanh"),
        fc_hidden_dim=config.get("FC_HIDDEN_DIM", 64),
        gru_hidden_dim=config.get("GRU_HIDDEN_DIM", 64),
    )

    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params


def initialize_cnn_rnn_agent(config, env, rng):
    """Initialize a CNN+RNN agent with the given config.

    This agent uses CNN for spatial feature extraction from grid-based observations,
    followed by GRU for temporal modeling. Architecture follows JaxMARL IPPO recipe.

    Args:
        config: dict, config for the agent. Expected keys:
            - ACTIVATION: "relu" or "tanh" (default: "relu")
            - FC_DIM_SIZE: Hidden dim for actor/critic FC layers (default: 128)
            - GRU_HIDDEN_DIM: Hidden dim for GRU (default: 128)
        env: gymnasium environment with grid-based observations
        rng: jax.random.PRNGKey, random key for initialization

    Returns:
        policy: CNNRNNActorCriticPolicy, the policy object
        params: dict, initial parameters for the agent
    """
    # Get observation shape - should be (H, W, C) for grid-based observations
    obs_shape = env.observation_space(env.agents[0]).shape

    # Create the CNN+RNN policy
    policy = CNNRNNActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_shape=obs_shape,
        activation=config.get("ACTIVATION", "relu"),
        fc_dim_size=config.get("FC_DIM_SIZE", 128),
        gru_hidden_dim=config.get("GRU_HIDDEN_DIM", 128),
        use_avail_actions=True,  # Support action masking
    )

    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_mlp_agent(config, env, rng):
    """
    Initialize an MLP agent with the given config.
    """
    policy = MLPActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        activation=config.get("ACTIVATION", "tanh"),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_actor_with_double_critic(config, env, rng):
    """Initialize an actor with double critic with the given config."""
    policy = ActorWithDoubleCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        activation=config.get("ACTIVATION", "tanh"),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_pseudo_actor_with_double_critic(config, env, rng):
    """Initialize a pseudo actor with double critic with the given config."""
    policy = PseudoActorWithDoubleCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        activation=config.get("ACTIVATION", "tanh"),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_actor_with_conditional_critic(config, env, rng):
    """Initialize an actor with conditional critic with the given config."""
    policy = ActorWithConditionalCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        pop_size=config["POP_SIZE"],
        activation=config.get("ACTIVATION", "tanh"),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_pseudo_actor_with_conditional_critic(config, env, rng):
    """Initialize a pseudo actor with conditional critic with the given config."""
    policy = PseudoActorWithConditionalCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        pop_size=config["POP_SIZE"],
        activation=config.get("ACTIVATION", "tanh"),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params


def initialize_ego_agent(algorithm_config, env, init_rng):
    if algorithm_config["EGO_ACTOR_TYPE"] == "s5":
        ego_policy, init_ego_params = initialize_s5_agent(algorithm_config, env, init_rng)
    elif algorithm_config["EGO_ACTOR_TYPE"] == "mlp":
        ego_policy, init_ego_params = initialize_mlp_agent(algorithm_config, env, init_rng)
    elif algorithm_config["EGO_ACTOR_TYPE"] == "rnn":
        ego_policy, init_ego_params = initialize_rnn_agent(algorithm_config, env, init_rng)
    elif algorithm_config["EGO_ACTOR_TYPE"] == "cnn_rnn":
        ego_policy, init_ego_params = initialize_cnn_rnn_agent(algorithm_config, env, init_rng)
    else:
        raise ValueError(f"Unknown EGO_ACTOR_TYPE: {algorithm_config['EGO_ACTOR_TYPE']}")
    return ego_policy, init_ego_params