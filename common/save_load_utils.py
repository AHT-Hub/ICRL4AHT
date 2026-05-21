"""Checkpoint save/load utilities for training runs.

This module provides utilities for saving and loading training checkpoints
in both Orbax and pickle formats, supporting:
- Single-file checkpoints (Orbax PyTreeCheckpointer)
- Separated checkpoints (per-population, per-checkpoint-index)
- Multi-agent checkpoints (e.g., BRDiv with conf/br agents)
- Checkpoint returns and config JSON files
"""

import os
import pickle
import json
import orbax.checkpoint
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# suppress logging from orbax
import logging
logger = logging.getLogger("absl")
logger.setLevel(logging.ERROR)

# Setup module logger
log = logging.getLogger(__name__)

# compute path to repo root by using this file's path
REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_json_serializable(obj):
    """Convert an object to be JSON serializable.

    Handles numpy arrays, numpy scalar types, and nested dicts/lists.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    else:
        return obj


def _convert_json_to_numpy(obj):
    """Convert JSON-loaded data back to numpy arrays where appropriate.

    Specifically converts lists under 'base_returns' and 'shaped_returns' keys.
    """
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in ('base_returns', 'shaped_returns') and isinstance(v, list):
                result[k] = np.array(v)
            else:
                result[k] = _convert_json_to_numpy(v)
        return result
    elif isinstance(obj, list):
        return [_convert_json_to_numpy(v) for v in obj]
    else:
        return obj


def save_train_run(out, savedir, savename):
    '''Save train run as orbax checkpoint. 
    Orbax requires absolute paths, so we compute the absolute path to the repo root.'''
    # determine whether savedir is relative or absolute
    if not os.path.isabs(savedir):
        savedir = os.path.join(REPO_PATH, savedir)
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)
    savepath = os.path.join(savedir, savename)
    
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(out)
    
    # Save the checkpoint
    checkpointer.save(savepath, out, save_args=save_args)
    return savepath

def load_checkpoints(path, ckpt_key="checkpoints", custom_loader_cfg: dict=None):
    '''Load checkpoints from orbax checkpoint. 
    Orbax requires absolute paths, so we compute the absolute path to the repo root.'''
    restored = load_train_run(path)
    if custom_loader_cfg is None:
        return restored[ckpt_key]
    elif custom_loader_cfg["name"] == "open_ended":
        partner_out, ego_out = restored
        out = ego_out if custom_loader_cfg["type"] == "ego" else partner_out
        if ckpt_key == "final_buffer":
            return out["final_buffer"]["params"]
        else:
            return out[ckpt_key]
    else:
        raise ValueError(f"Invalid custom loader name: {custom_loader_cfg['name']}")

def load_train_run(path):
    '''Load checkpoints from orbax checkpoint. 
    Orbax requires absolute paths, so we compute the absolute path to the repo root.'''
    # determine whether path is relative or absolute
    if not os.path.isabs(path):
        path = os.path.join(REPO_PATH, path)
    # load the checkpoint
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = checkpointer.restore(path)
    # convert pytree leaves from np arrays to jax arrays
    restored = jax.tree_util.tree_map(
        lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x,
        restored
    )
    return restored

def save_train_run_as_pickle(out, savedir, savename):
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)
        
    savepath = f"{savedir}/{savename}.pkl"
    with open(savepath, "wb") as f:
        pickle.dump(out, f)
    return savepath

def load_checkpoints_from_pickle(path, ckpt_key="checkpoints"):
    out = load_train_run_from_pickle(path)
    return out[ckpt_key]

def load_train_run_from_pickle(path):
    with open(path, "rb") as f:
        out = pickle.load(f)
    return out


# =============================================================================
# Separated Checkpoint Saving/Loading Functions
# =============================================================================

def save_separated_checkpoints(
    checkpoints: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: str,
    savename: str = "train_run",
    checkpoint_key: str = "checkpoints",
    pop_size_key: str = "PARTNER_POP_SIZE",
    num_ckpts_key: str = "NUM_CHECKPOINTS",
    checkpoint_returns: Optional[Dict[str, Any]] = None,
) -> str:
    """Save checkpoints separated by population index and checkpoint index.

    This saves each population member's checkpoints in separate directories, allowing
    individual loading of any population/checkpoint combination.

    Supports different checkpoint structures:
    - 2D: (num_seeds, pop_size, num_checkpoints, ...) - FCP style
    - 1D: (num_seeds, num_checkpoints, ...) - IPPO style (set pop_size_key=None)

    Directory structure:
        {output_dir}/{savename}/
            pi_0/
                ckpt_0/  (orbax checkpoint)
                ckpt_1/
                ...
            pi_1/
                ckpt_0/
                ...
            metrics/  (if present in checkpoints)
            checkpoint_returns.json  (if checkpoint_returns provided)
            config.json  (training config)

    Args:
        checkpoints: Dictionary containing checkpoint data and optionally 'metrics'.
        config: Training configuration dictionary.
        output_dir: Base output directory.
        savename: Name for the run directory.
        checkpoint_key: Key for checkpoints in the data dict (default: "checkpoints").
        pop_size_key: Config key for population size. Set to None for 1D structure.
        num_ckpts_key: Config key for number of checkpoints.
        checkpoint_returns: Optional dict with 'base_returns' and 'shaped_returns' arrays
            containing the returns at each checkpoint interval. Shape should be
            (num_seeds, num_checkpoints) or (num_seeds, pop_size, num_checkpoints).

    Returns:
        Path to the saved run directory.
    """
    output_path = Path(output_dir) / savename
    output_path.mkdir(parents=True, exist_ok=True)

    num_seeds = config.get("NUM_SEEDS", 1)
    num_checkpoints = config[num_ckpts_key]

    # Determine if we have 2D (pop_size x num_checkpoints) or 1D (num_checkpoints) structure
    if pop_size_key is not None and pop_size_key in config:
        pop_size = config[pop_size_key]
        is_2d = True
    else:
        pop_size = 1
        is_2d = False

    ckpt_data = checkpoints[checkpoint_key]

    # Log shape info
    sample_leaf = jax.tree_util.tree_leaves(ckpt_data)[0]
    log.info(f"Checkpoint sample shape: {sample_leaf.shape}")
    if is_2d:
        log.info(f"Expected: (num_seeds={num_seeds}, pop_size={pop_size}, "
                 f"num_checkpoints={num_checkpoints}, ...)")
    else:
        log.info(f"Expected: (num_seeds={num_seeds}, num_checkpoints={num_checkpoints}, ...)")

    # Save each checkpoint separately
    for pi_idx in range(pop_size):
        pi_dir = output_path / f"pi_{pi_idx}"
        pi_dir.mkdir(parents=True, exist_ok=True)

        for ckpt_idx in range(num_checkpoints):
            # Extract this specific checkpoint
            if is_2d:
                # 2D structure: (num_seeds, pop_size, num_checkpoints, ...)
                single_ckpt = jax.tree.map(
                    lambda x, pi=pi_idx, ck=ckpt_idx: x[:, pi, ck, ...],
                    ckpt_data
                )
            else:
                # 1D structure: (num_seeds, num_checkpoints, ...)
                single_ckpt = jax.tree.map(
                    lambda x, ck=ckpt_idx: x[:, ck, ...],
                    ckpt_data
                )

            # Save using orbax
            save_train_run({"params": single_ckpt}, str(pi_dir), f"ckpt_{ckpt_idx}")

    log.info(f"Saved {pop_size} x {num_checkpoints} = "
             f"{pop_size * num_checkpoints} checkpoints to: {output_path}")

    # Save metrics in a single file (not separated)
    if "metrics" in checkpoints:
        save_train_run({"metrics": checkpoints["metrics"]}, str(output_path), "metrics")
        log.info(f"Saved metrics to: {output_path / 'metrics'}")

    # Save checkpoint returns if provided (as JSON for readability)
    if checkpoint_returns is not None:
        returns_path = output_path / "checkpoint_returns.json"
        with open(returns_path, "w") as f:
            json.dump(_make_json_serializable(checkpoint_returns), f, indent=2)
        log.info(f"Saved checkpoint returns to: {returns_path}")

    # Save config as JSON for readability
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(_make_json_serializable(config), f, indent=2)
    log.info(f"Saved config to: {config_path}")

    return str(output_path)


def save_separated_checkpoints_multi(
    checkpoints: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: str,
    savename: str = "train_run",
    checkpoint_keys: Dict[str, str] = None,
    num_ckpts_key: str = "NUM_CHECKPOINTS",
    checkpoint_returns: Optional[Dict[str, Any]] = None,
) -> str:
    """Save multiple checkpoint types separated by checkpoint index.

    This is for algorithms with multiple agent types (e.g., BRDiv with conf/br).
    Each agent type's checkpoints are saved in separate subdirectories.

    Directory structure:
        {output_dir}/{savename}/
            conf/
                ckpt_0/
                ckpt_1/
                ...
            br/
                ckpt_0/
                ...
            metrics/
            checkpoint_returns.json  (if checkpoint_returns provided)
            config.json

    Args:
        checkpoints: Dictionary containing checkpoint data for multiple agent types.
        config: Training configuration dictionary.
        output_dir: Base output directory.
        savename: Name for the run directory.
        checkpoint_keys: Dict mapping agent type name to checkpoint key in data.
            E.g., {"conf": "checkpoints_conf", "br": "checkpoints_br"}
        num_ckpts_key: Config key for number of checkpoints.
        checkpoint_returns: Optional dict with 'base_returns' and 'shaped_returns' arrays
            containing the returns at each checkpoint interval. Shape should be
            (num_seeds, num_checkpoints).

    Returns:
        Path to the saved run directory.
    """
    if checkpoint_keys is None:
        checkpoint_keys = {"conf": "checkpoints_conf", "br": "checkpoints_br"}

    output_path = Path(output_dir) / savename
    output_path.mkdir(parents=True, exist_ok=True)

    num_seeds = config.get("NUM_SEEDS", 1)
    num_checkpoints = config[num_ckpts_key]

    # Save each agent type's checkpoints
    for agent_type, ckpt_key in checkpoint_keys.items():
        if ckpt_key not in checkpoints:
            log.warning(f"Checkpoint key '{ckpt_key}' not found in data, skipping {agent_type}")
            continue

        ckpt_data = checkpoints[ckpt_key]
        agent_dir = output_path / agent_type
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Log shape info
        sample_leaf = jax.tree_util.tree_leaves(ckpt_data)[0]
        log.info(f"{agent_type} checkpoint sample shape: {sample_leaf.shape}")

        for ckpt_idx in range(num_checkpoints):
            # Extract this checkpoint: (num_seeds, num_checkpoints, ...) -> (num_seeds, ...)
            single_ckpt = jax.tree.map(
                lambda x, ck=ckpt_idx: x[:, ck, ...],
                ckpt_data
            )
            save_train_run({"params": single_ckpt}, str(agent_dir), f"ckpt_{ckpt_idx}")

        log.info(f"Saved {num_checkpoints} {agent_type} checkpoints to: {agent_dir}")

    # Save metrics in a single file
    if "metrics" in checkpoints:
        save_train_run({"metrics": checkpoints["metrics"]}, str(output_path), "metrics")
        log.info(f"Saved metrics to: {output_path / 'metrics'}")

    # Save checkpoint returns if provided (as JSON for readability)
    if checkpoint_returns is not None:
        returns_path = output_path / "checkpoint_returns.json"
        with open(returns_path, "w") as f:
            json.dump(_make_json_serializable(checkpoint_returns), f, indent=2)
        log.info(f"Saved checkpoint returns to: {returns_path}")

    # Save config as JSON for readability
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(_make_json_serializable(config), f, indent=2)
    log.info(f"Saved config to: {config_path}")

    return str(output_path)


def load_individual_checkpoint(
    run_dir: str,
    checkpoint_idx: int,
    population_idx: int = 0,
    seed_idx: int = 0,
    agent_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Load an individual checkpoint from a separated checkpoint directory.

    Args:
        run_dir: Path to the run directory (e.g., "outputs/fcp_train_run").
        checkpoint_idx: Index of the checkpoint (0 to num_checkpoints-1).
        population_idx: Index of the population member (default: 0).
            For 1D structure (IPPO), this is always 0.
        seed_idx: Index of the seed to load (default: 0).
        agent_type: For multi-agent algorithms (BRDiv), specify agent type
            (e.g., "conf" or "br"). None for single-agent algorithms.

    Returns:
        Dictionary containing the checkpoint params for the specified seed.

    Examples:
        # Load FCP checkpoint
        params = load_individual_checkpoint(
            "outputs/fcp_train_run", checkpoint_idx=2, population_idx=0
        )

        # Load IPPO checkpoint
        params = load_individual_checkpoint(
            "outputs/ippo_train_run", checkpoint_idx=3
        )

        # Load BRDiv checkpoint
        params = load_individual_checkpoint(
            "outputs/brdiv_train_run", checkpoint_idx=1, agent_type="conf"
        )
    """
    run_path = Path(run_dir)

    # Build path to checkpoint
    if agent_type is not None:
        ckpt_path = run_path / agent_type / f"ckpt_{checkpoint_idx}"
    else:
        ckpt_path = run_path / f"pi_{population_idx}" / f"ckpt_{checkpoint_idx}"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    restored = load_train_run(str(ckpt_path))
    params = restored["params"]

    # Extract the specific seed
    params = jax.tree.map(lambda x: x[seed_idx], params)

    return params


def load_all_checkpoints_from_separated(
    run_dir: str,
    seed_idx: int = 0,
    agent_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Load all checkpoints from a separated checkpoint directory.

    This reconstructs the full checkpoint array from individually saved checkpoints.

    Args:
        run_dir: Path to the run directory (e.g., "outputs/fcp_train_run").
        seed_idx: Index of the seed to load (default: 0).
        agent_type: For multi-agent algorithms (BRDiv), specify agent type.
            None loads all population members (FCP/IPPO style).

    Returns:
        Dictionary with 'params' containing stacked checkpoint params and 'config'.
    """
    run_path = Path(run_dir)

    # Load config - try JSON first (new format), fall back to pickle (old format)
    config_json_path = run_path / "config.json"

    if config_json_path.exists():
        with open(config_json_path, "r") as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(f"Config not found: {config_json_path}")

    num_checkpoints = config["NUM_CHECKPOINTS"]

    if agent_type is not None:
        # Load single agent type (BRDiv style)
        all_params = []
        for ckpt_idx in range(num_checkpoints):
            params = load_individual_checkpoint(
                run_dir, ckpt_idx, agent_type=agent_type, seed_idx=seed_idx
            )
            all_params.append(params)
    else:
        # Load all population members (FCP/IPPO style)
        pop_size = config.get("PARTNER_POP_SIZE", 1)
        all_params = []
        for pi_idx in range(pop_size):
            for ckpt_idx in range(num_checkpoints):
                params = load_individual_checkpoint(
                    run_dir, ckpt_idx, population_idx=pi_idx, seed_idx=seed_idx
                )
                all_params.append(params)

    # Stack all params
    stacked_params = jax.tree.map(
        lambda *xs: jnp.stack(xs, axis=0),
        *all_params
    )

    return {"params": stacked_params, "config": config}


def load_checkpoint_returns(run_dir: str) -> Optional[Dict[str, Any]]:
    """Load checkpoint returns from a separated checkpoint directory.

    Args:
        run_dir: Path to the run directory (e.g., "outputs/ippo_train_run").

    Returns:
        Dictionary containing 'base_returns' and 'shaped_returns' arrays,
        or None if checkpoint_returns.json doesn't exist.
    """
    run_path = Path(run_dir)
    returns_path = run_path / "checkpoint_returns.json"

    # Try JSON first (new format), fall back to pickle (old format)
    if returns_path.exists():
        with open(returns_path, "r") as f:
            checkpoint_returns = json.load(f)
        return _convert_json_to_numpy(checkpoint_returns)

    return None