"""RL teammate policy wrappers for FCP, BRDiv, and IPPO.

This module provides utilities for loading RL checkpoints and wrapping them
in a uniform interface compatible with the TeammatePolicy specification.

FCP, BRDiv, and IPPO use the same underlying policy architectures (MLP, RNN, S5, CNN_RNN),
so the wrappers are shared. The main difference is in checkpoint structure.
"""

import os
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from common.save_load_utils import load_train_run, load_checkpoints, load_individual_checkpoint

# Policy classes
from agents.mlp_actor_critic_agent import MLPActorCriticPolicy
from agents.rnn_actor_critic_agent import RNNActorCriticPolicy
from agents.cnn_rnn_actor_critic_agent import CNNRNNActorCriticPolicy


def make_rl_policy_wrapper(
    algo: str,
    ckpt_path: str,
    env,
    use_log_wrapper: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    """Create an RL policy and load parameters from checkpoint.

    Args:
        algo: Algorithm name ("fcp", "brdiv", or "ippo")
        ckpt_path: Path to checkpoint directory. For separated checkpoints, this is the
            run directory (e.g., "outputs/ippo_train_run"). For legacy checkpoints, this
            is the direct checkpoint path.
        env: Environment instance for getting obs/action dims
        use_log_wrapper: Whether environment uses LogWrapper
        extra: Additional configuration:
            - actor_type: "mlp" | "rnn" | "s5" | "cnn_rnn" (default: "mlp")
            - activation: Activation function (default: "tanh")
            - fc_hidden_dim: FC hidden dimension (default: 64)
            - fc_dim_size: FC dimension for cnn_rnn (default: 128)
            - gru_hidden_dim: GRU hidden dimension for RNN/cnn_rnn (default: 64/128)
            - use_separated_ckpt: Whether to use separated checkpoint format (default: False)
            - checkpoint_idx: Checkpoint index for separated checkpoints (default: -1, meaning last)
            - population_idx: Population index for FCP/IPPO (default: 0)
            - seed_idx: Seed index for multi-seed training (default: 0)
            - agent_type: Agent type for BRDiv ("conf" or "br") (default: None)
            - ckpt_key: Key for accessing params in checkpoint (legacy format)

    Returns:
        Tuple of (policy, params) where policy is an AgentPolicy instance
        and params are the loaded network parameters.

    Raises:
        FileNotFoundError: If checkpoint path doesn't exist
        ValueError: If actor_type is unsupported
    """
    extra = extra or {}

    # Check checkpoint exists
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint path does not exist: {ckpt_path}. "
            f"Please provide a valid checkpoint path or use dry_run mode for testing."
        )

    # Get environment dimensions
    action_dim = env.action_space(env.agents[0]).n
    obs_space = env.observation_space(env.agents[0])
    obs_shape = obs_space.shape

    # Get config from extra
    actor_type = extra.get("actor_type", "mlp")
    activation = extra.get("activation", "tanh")
    fc_hidden_dim = extra.get("fc_hidden_dim", 64)
    fc_dim_size = extra.get("fc_dim_size", 128)
    gru_hidden_dim = extra.get("gru_hidden_dim", 64 if actor_type != "cnn_rnn" else 128)

    # Create policy based on actor type
    if actor_type == "mlp":
        obs_dim = obs_shape[0] if len(obs_shape) == 1 else int(jnp.prod(jnp.array(obs_shape)))
        policy = MLPActorCriticPolicy(
            action_dim=action_dim,
            obs_dim=obs_dim,
            activation=activation,
        )
    elif actor_type == "rnn":
        obs_dim = obs_shape[0] if len(obs_shape) == 1 else int(jnp.prod(jnp.array(obs_shape)))
        policy = RNNActorCriticPolicy(
            action_dim=action_dim,
            obs_dim=obs_dim,
            activation=activation,
            fc_hidden_dim=fc_hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
        )
    elif actor_type == "cnn_rnn":
        # CNN+RNN requires spatial observation shape (H, W, C)
        if len(obs_shape) != 3:
            raise ValueError(
                f"cnn_rnn actor_type requires 3D observation shape (H, W, C), "
                f"got {obs_shape}. Make sure flatten_obs=False in environment."
            )
        policy = CNNRNNActorCriticPolicy(
            action_dim=action_dim,
            obs_shape=obs_shape,
            activation=activation,
            fc_dim_size=fc_dim_size,
            gru_hidden_dim=gru_hidden_dim,
            use_avail_actions=extra.get("use_avail_actions", True),
        )
    elif actor_type == "s5":
        # S5 requires additional config
        from agents.s5_actor_critic_agent import S5ActorCriticPolicy
        obs_dim = obs_shape[0] if len(obs_shape) == 1 else int(jnp.prod(jnp.array(obs_shape)))
        policy = S5ActorCriticPolicy(
            action_dim=action_dim,
            obs_dim=obs_dim,
            d_model=extra.get("s5_d_model", 128),
            ssm_size=extra.get("s5_ssm_size", 128),
            ssm_n_layers=extra.get("s5_n_layers", 2),
            blocks=extra.get("s5_blocks", 1),
            fc_hidden_dim=extra.get("s5_fc_hidden_dim", 1024),
            fc_n_layers=extra.get("s5_fc_n_layers", 3),
        )
    else:
        raise ValueError(
            f"Unsupported actor_type '{actor_type}'. "
            f"Valid types: 'mlp', 'rnn', 'cnn_rnn', 's5'"
        )

    # Load checkpoint and extract params
    use_separated_ckpt = extra.get("use_separated_ckpt", False)
    if use_separated_ckpt:
        params = _load_params_from_separated_checkpoint(ckpt_path, algo, extra)
    else:
        params = _load_params_from_checkpoint(ckpt_path, algo, extra)

    return policy, params


def _load_params_from_checkpoint(
    ckpt_path: str,
    algo: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Any:
    """Load parameters from checkpoint.

    Handles different checkpoint structures for FCP and BRDiv.

    Args:
        ckpt_path: Path to checkpoint
        algo: Algorithm name
        extra: Additional config with optional ckpt_key

    Returns:
        Network parameters (pytree)
    """
    extra = extra or {}

    # Load the full checkpoint
    restored = load_train_run(ckpt_path)

    # Determine how to extract params based on algo and checkpoint structure
    ckpt_key = extra.get("ckpt_key", None)

    if ckpt_key is not None:
        # User specified explicit key
        if ckpt_key in restored:
            params = restored[ckpt_key]
        else:
            raise KeyError(
                f"Key '{ckpt_key}' not found in checkpoint. "
                f"Available keys: {list(restored.keys())}"
            )
    elif algo == "fcp":
        # FCP checkpoint structure typically contains partner population params
        # The structure can vary based on how FCP was saved
        if "final_params" in restored:
            params = restored["final_params"]
        elif "partner_params" in restored:
            # FCP stores multiple partners; take first one by default
            partner_idx = extra.get("partner_idx", 0)
            partner_params = restored["partner_params"]
            if hasattr(partner_params, '__getitem__'):
                params = jax.tree_util.tree_map(
                    lambda x: x[partner_idx] if hasattr(x, '__getitem__') else x,
                    partner_params
                )
            else:
                params = partner_params
        elif "checkpoints" in restored:
            checkpoints = restored["checkpoints"]
            if isinstance(checkpoints, list):
                params = checkpoints[-1]
            else:
                params = checkpoints
        else:
            params = restored

    elif algo == "brdiv":
        # BRDiv checkpoint structure:
        # - "final_params_conf": final confederate params
        # - "final_params_br": final best-response params
        # - "checkpoints_conf": list of confederate checkpoints
        # - "checkpoints_br": list of best-response checkpoints
        # By default, use confederate params (these are the diverse teammates)
        param_type = extra.get("brdiv_param_type", "conf")  # "conf" or "br"

        if param_type == "conf":
            if "final_params_conf" in restored:
                params = restored["final_params_conf"]
            elif "checkpoints_conf" in restored:
                checkpoints = restored["checkpoints_conf"]
                if isinstance(checkpoints, list):
                    params = checkpoints[-1]
                else:
                    params = checkpoints
            else:
                # Fall back to generic keys
                params = restored.get("final_params", restored)
        else:  # param_type == "br"
            if "final_params_br" in restored:
                params = restored["final_params_br"]
            elif "checkpoints_br" in restored:
                checkpoints = restored["checkpoints_br"]
                if isinstance(checkpoints, list):
                    params = checkpoints[-1]
                else:
                    params = checkpoints
            else:
                params = restored.get("final_params", restored)

        # BRDiv may have population dimension - select specific partner if needed
        partner_idx = extra.get("partner_idx", 0)
        if hasattr(params, 'shape') and len(params.shape) > 0:
            # If params have a leading population dimension, select one
            params = jax.tree_util.tree_map(
                lambda x: x[partner_idx] if hasattr(x, '__getitem__') and len(x.shape) > 1 else x,
                params
            )

    elif algo == "ippo":
        # IPPO checkpoint structure:
        # - "final_params": final trained parameters
        # - "checkpoints": array of checkpoint parameters
        if "final_params" in restored:
            params = restored["final_params"]
        elif "checkpoints" in restored:
            checkpoints = restored["checkpoints"]
            if isinstance(checkpoints, list):
                params = checkpoints[-1]
            else:
                params = checkpoints
        else:
            params = restored

    else:
        # Unknown algo - try common keys
        if "final_params" in restored:
            params = restored["final_params"]
        elif "final_params_conf" in restored:
            params = restored["final_params_conf"]
        elif "params" in restored:
            params = restored["params"]
        else:
            params = restored

    return params


def _load_params_from_separated_checkpoint(
    run_dir: str,
    algo: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Any:
    """Load parameters from separated checkpoint format using load_individual_checkpoint.

    This function handles the new separated checkpoint format where each checkpoint
    is saved individually, allowing efficient loading of specific checkpoints without
    loading the entire training run.

    Args:
        run_dir: Path to the run directory (e.g., "outputs/ippo_train_run")
        algo: Algorithm name ("fcp", "brdiv", "ippo")
        extra: Additional configuration:
            - checkpoint_idx: Index of checkpoint to load (default: -1 for last)
            - population_idx: Population index for FCP/IPPO (default: 0)
            - seed_idx: Seed index for multi-seed runs (default: 0)
            - agent_type: For BRDiv, "conf" or "br" (default: "conf")

    Returns:
        Network parameters (pytree)
    """
    import json
    from pathlib import Path

    extra = extra or {}

    run_path = Path(run_dir)

    # Load config to get checkpoint count
    config_path = run_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Make sure the checkpoint was saved using save_separated_checkpoints."
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    num_checkpoints = config.get("NUM_CHECKPOINTS", 1)

    # Get checkpoint index (default to last checkpoint)
    checkpoint_idx = extra.get("checkpoint_idx", -1)
    if checkpoint_idx < 0:
        checkpoint_idx = num_checkpoints + checkpoint_idx  # Handle negative indexing

    if checkpoint_idx < 0 or checkpoint_idx >= num_checkpoints:
        raise ValueError(
            f"checkpoint_idx {checkpoint_idx} out of range. "
            f"Valid range: 0 to {num_checkpoints - 1}"
        )

    population_idx = extra.get("population_idx", 0)
    seed_idx = extra.get("seed_idx", 0)

    # Determine agent_type for BRDiv
    agent_type = None
    if algo == "brdiv":
        agent_type = extra.get("agent_type", "conf")

    # Load using load_individual_checkpoint
    params = load_individual_checkpoint(
        run_dir=run_dir,
        checkpoint_idx=checkpoint_idx,
        population_idx=population_idx,
        seed_idx=seed_idx,
        agent_type=agent_type,
    )

    return params


def create_dummy_rl_policy(
    env,
    actor_type: str = "mlp",
    activation: str = "tanh",
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    """Create a dummy RL policy with random params for testing.

    This is useful for testing the registry without needing actual checkpoints.

    Args:
        env: Environment instance
        actor_type: Policy architecture type
        activation: Activation function
        extra: Additional configuration for cnn_rnn

    Returns:
        Tuple of (policy, params) with randomly initialized parameters
    """
    extra = extra or {}
    action_dim = env.action_space(env.agents[0]).n
    obs_space = env.observation_space(env.agents[0])
    obs_shape = obs_space.shape

    if actor_type == "mlp":
        obs_dim = obs_shape[0] if len(obs_shape) == 1 else int(jnp.prod(jnp.array(obs_shape)))
        policy = MLPActorCriticPolicy(
            action_dim=action_dim,
            obs_dim=obs_dim,
            activation=activation,
        )
    elif actor_type == "rnn":
        obs_dim = obs_shape[0] if len(obs_shape) == 1 else int(jnp.prod(jnp.array(obs_shape)))
        policy = RNNActorCriticPolicy(
            action_dim=action_dim,
            obs_dim=obs_dim,
            activation=activation,
        )
    elif actor_type == "cnn_rnn":
        if len(obs_shape) != 3:
            raise ValueError(
                f"cnn_rnn actor_type requires 3D observation shape (H, W, C), "
                f"got {obs_shape}. Make sure flatten_obs=False in environment."
            )
        fc_dim_size = extra.get("fc_dim_size", 128)
        gru_hidden_dim = extra.get("gru_hidden_dim", 128)
        policy = CNNRNNActorCriticPolicy(
            action_dim=action_dim,
            obs_shape=obs_shape,
            activation=activation,
            fc_dim_size=fc_dim_size,
            gru_hidden_dim=gru_hidden_dim,
            use_avail_actions=extra.get("use_avail_actions", True),
        )
    else:
        raise ValueError(f"Unsupported actor_type for dummy: {actor_type}")

    # Initialize with random params
    rng = jax.random.PRNGKey(0)
    params = policy.init_params(rng)

    return policy, params


class DryRunRLSpec:
    """Marker class for dry-run RL testing.

    When the registry encounters an RLTeammateSpec with ckpt_path set to
    this special value, it will create a dummy policy instead of loading
    from checkpoint.
    """
    DRY_RUN_PATH = "__DRY_RUN__"


def make_rl_policy_wrapper_dry_run(
    algo: str,
    env,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    """Create a dummy RL policy for testing without checkpoints.

    This creates a policy with randomly initialized parameters,
    useful for testing the registry interface without needing
    actual trained checkpoints.

    Args:
        algo: Algorithm name (used for logging only)
        env: Environment instance
        extra: Additional configuration (including actor_type: "mlp", "rnn", "cnn_rnn", "s5")

    Returns:
        Tuple of (policy, params)
    """
    extra = extra or {}
    actor_type = extra.get("actor_type", "mlp")
    activation = extra.get("activation", "tanh")

    return create_dummy_rl_policy(env, actor_type, activation, extra)
