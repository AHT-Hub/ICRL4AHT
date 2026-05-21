'''
Based on the IPPO implementation from JaxMarl. Trains a parameter-shared, MLP IPPO agent on a
fully cooperative multi-agent environment. Note that this code is only compatible with MLP policies.
'''
import shutil

import hydra
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from agents.initialize_agents import initialize_s5_agent, initialize_mlp_agent, \
    initialize_rnn_agent, initialize_cnn_rnn_agent, initialize_pseudo_actor_with_double_critic, initialize_pseudo_actor_with_conditional_critic
from common.stats_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_utils import Transition, batchify, batchify_spatial, unbatchify, _create_minibatches


def initialize_agent(actor_type, algorithm_config, env, init_rng):
    if actor_type == "s5":
        policy, init_params = initialize_s5_agent(algorithm_config, env, init_rng)
    elif actor_type == "mlp":
        policy, init_params = initialize_mlp_agent(algorithm_config, env, init_rng)
    elif actor_type == "rnn":
        policy, init_params = initialize_rnn_agent(algorithm_config, env, init_rng)
    elif actor_type == "cnn_rnn":
        policy, init_params = initialize_cnn_rnn_agent(algorithm_config, env, init_rng)
    elif actor_type == "pseudo_actor_with_double_critic":
        policy, init_params = initialize_pseudo_actor_with_double_critic(algorithm_config, env, init_rng)
    elif actor_type == "pseudo_actor_with_conditional_critic":
        policy, init_params = initialize_pseudo_actor_with_conditional_critic(algorithm_config, env, init_rng)
    return policy, init_params

def make_train(config, env):
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
    )

    # Validate that NUM_ACTORS is divisible by NUM_MINIBATCHES
    if config["NUM_ACTORS"] % config["NUM_MINIBATCHES"] != 0:
        raise ValueError(
            f"NUM_ACTORS ({config['NUM_ACTORS']} = {env.num_agents} agents × {config['NUM_ENVS']} envs) "
            f"must be divisible by NUM_MINIBATCHES ({config['NUM_MINIBATCHES']}). "
            f"Suggested NUM_MINIBATCHES values that divide {config['NUM_ACTORS']}: "
            f"{[i for i in [8, 10, 16, 20, 25, 32, 40, 50, 64, 80, 100] if config['NUM_ACTORS'] % i == 0]}"
        )

    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"] // config["NUM_MINIBATCHES"]
    )

    # Linear schedule (original)
    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    # Cosine decay with warmup schedule (from JaxMARL)
    def create_warmup_cosine_schedule():
        base_learning_rate = config["LR"]
        lr_warmup = config.get("LR_WARMUP", 0.0)
        update_steps = config["NUM_UPDATES"]
        warmup_steps = int(lr_warmup * update_steps)

        steps_per_epoch = config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]

        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=warmup_steps * steps_per_epoch,
        )
        cosine_epochs = max(update_steps - warmup_steps, 1)

        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate,
            decay_steps=cosine_epochs * steps_per_epoch
        )
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps * steps_per_epoch],
        )
        return schedule_fn

    # Reward shaping annealing schedule (from JaxMARL)
    # Only used if REW_SHAPING_HORIZON is in config and > 0
    rew_shaping_horizon = config.get("REW_SHAPING_HORIZON", 0)
    if rew_shaping_horizon > 0:
        rew_shaping_anneal = optax.linear_schedule(
            init_value=1.0,
            end_value=0.0,
            transition_steps=int(rew_shaping_horizon)
        )
    else:
        rew_shaping_anneal = None

    def train(rng):
        # INIT NETWORK
        rng, init_rng = jax.random.split(rng)
        policy, init_params = initialize_agent(config["ACTOR_TYPE"], config, env, init_rng)

        if config["ANNEAL_LR"]:
            # Use warmup + cosine decay if LR_WARMUP is specified, otherwise use linear
            if config.get("LR_WARMUP", 0.0) > 0:
                lr_schedule = create_warmup_cosine_schedule()
            else:
                lr_schedule = linear_schedule
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=lr_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5))
        train_state = TrainState.create(
            apply_fn=policy.network.apply,
            params=init_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            # Compute reward shaping annealing factor for this update step
            # (from JaxMARL recipe - same factor for all steps in a rollout)
            if rew_shaping_anneal is not None:
                current_timestep = update_steps * config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
                anneal_factor = rew_shaping_anneal(current_timestep)
            else:
                anneal_factor = 0.0  # No reward shaping

            # Save the initial hidden state and done signal BEFORE the rollout for PPO update epochs
            # This is important when ROLLOUT_LENGTH != max_steps (episodes span multiple rollouts)
            initial_hstate_for_update = runner_state[4]  # hstate is at index 4
            # Save initial done state for correct RNN hidden state reset during PPO updates
            # During rollout, RNN uses last_done (done from previous step) for reset
            # We need to preserve this for replay to avoid off-by-one errors
            initial_done_dict = runner_state[3]  # last_done is at index 3
            initial_done_for_update = batchify(initial_done_dict, env.agents, config["NUM_ACTORS"]).squeeze()
            initial_done_for_update = initial_done_for_update.reshape(1, config["NUM_ACTORS"])  # (1, num_actors)

            def _env_step(runner_state, unused):
                # Extended runner_state includes shaped reward tracking:
                # (train_state, env_state, last_obs, last_done, last_hstate, rng,
                #  cumulative_shaped_reward, returned_shaped_returns)
                (train_state, env_state, last_obs, last_done, last_hstate, rng,
                 cumulative_shaped_reward, returned_shaped_returns) = runner_state

                rng, act_rng = jax.random.split(rng)

                # Handle observation batchify based on actor type
                # CNN+RNN needs spatial structure preserved (num_actors, H, W, C)
                # MLP/RNN needs flattened (num_actors, obs_dim)
                if config.get("ACTOR_TYPE", "mlp") == "cnn_rnn":
                    last_obs_batch = batchify_spatial(last_obs, env.agents, config["NUM_ACTORS"])
                else:
                    last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                last_done_batch = batchify(last_done, env.agents, config["NUM_ACTORS"])

                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(batchify(avail_actions,
                    env.agents, config["NUM_ACTORS"]).astype(jnp.float32))

                # Add sequence dimension for policy input
                # CNN+RNN: (1, NUM_ACTORS, H, W, C)
                # MLP/RNN: (1, NUM_ACTORS, obs_dim)
                obs_input = last_obs_batch[np.newaxis, :]

                action, value, pi, new_hstate = policy.get_action_value_policy(
                    params=train_state.params,
                    obs=obs_input,
                    done=last_done_batch.reshape(1, config["NUM_ACTORS"]),
                    avail_actions=avail_actions.reshape(1, config["NUM_ACTORS"], -1),
                    hstate=last_hstate,
                    rng=act_rng
                )
                log_prob = pi.log_prob(action)

                action = action.squeeze()
                log_prob = log_prob.squeeze()
                value = value.squeeze()

                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k:v.flatten() for k,v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                new_obs, new_env_state, reward, new_done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )

                # Get shaped reward from env info (before applying to training reward)
                # shaped_reward from env has shape (NUM_ENVS, num_agents)
                shaped_reward_raw = info.get("shaped_reward", jnp.zeros((config["NUM_ENVS"], env.num_agents)))

                # Compute the annealed shaping component for this step
                # Flatten to (NUM_ACTORS,): stack agents then flatten
                annealed_shaping = anneal_factor * batchify(
                    {agent: shaped_reward_raw[:, i] for i, agent in enumerate(env.agents)},
                    env.agents, config["NUM_ACTORS"]
                ).squeeze()

                # Update cumulative shaped reward (adds to running total)
                new_cumulative_shaped = cumulative_shaped_reward + annealed_shaping

                # Apply reward shaping with annealing (from JaxMARL recipe)
                # reward = base_reward + anneal_factor * shaped_reward
                if rew_shaping_anneal is not None:
                    shaped_reward_dict = {agent: shaped_reward_raw[:, i] for i, agent in enumerate(env.agents)}
                    reward = jax.tree.map(
                        lambda r, sr: r + anneal_factor * sr,
                        reward,
                        shaped_reward_dict
                    )

                # Compute shaped reward for transition (base + annealed shaping)
                shaped_reward_for_transition = batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze()

                # Get global done signal
                ep_done = jnp.tile(new_done["__all__"], env.num_agents)

                # Get base episode returns from LogWrapper (tracks BASE rewards only)
                base_episode_returns = info["returned_episode_returns"]  # Shape: (NUM_ENVS, num_agents)
                base_episode_returns_flat = batchify(
                    {agent: base_episode_returns[:, i] for i, agent in enumerate(env.agents)},
                    env.agents, config["NUM_ACTORS"]
                ).squeeze()

                # Update returned_shaped_returns when episode ends
                # Total shaped return = base return (from LogWrapper) + cumulative shaped bonus
                new_returned_shaped = jnp.where(
                    ep_done,
                    base_episode_returns_flat + new_cumulative_shaped,
                    returned_shaped_returns
                )

                # Reset cumulative shaped reward on episode end
                new_cumulative_shaped = jnp.where(ep_done, 0.0, new_cumulative_shaped)

                # Reshape info and add shaped returns tracking
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                info["returned_episode_shaped_returns"] = new_returned_shaped

                # Use global done signal for transition (from JaxMARL recipe)
                done_for_transition = ep_done

                transition = Transition(
                    done_for_transition,
                    action,
                    value,
                    shaped_reward_for_transition,
                    log_prob,
                    last_obs_batch,
                    info,
                    avail_actions
                )
                runner_state = (train_state, new_env_state, new_obs, new_done, new_hstate, rng,
                               new_cumulative_shaped, new_returned_shaped)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
            )

            # Get final value estimate for completed trajectory
            # Unpack extended runner_state
            (train_state, env_state, last_obs, last_done, last_hstate, rng,
             cumulative_shaped_reward, returned_shaped_returns) = runner_state
            # Use spatial batchify for CNN+RNN to preserve (H, W, C) structure
            if config.get("ACTOR_TYPE", "mlp") == "cnn_rnn":
                last_obs_batch = batchify_spatial(last_obs, env.agents, config["NUM_ACTORS"])
            else:
                last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            last_done_batch = batchify(last_done, env.agents, config["NUM_ACTORS"])
            last_done_batch = last_done_batch.reshape(1, config["NUM_ACTORS"])
            last_avail_batch = jax.vmap(env.get_avail_actions)(env_state.env_state)
            last_avail_batch = jax.lax.stop_gradient(batchify(last_avail_batch,
                env.agents, config["NUM_ACTORS"]).astype(jnp.float32))

            # Add sequence dimension for policy input
            last_obs_input = last_obs_batch[np.newaxis, :]

            _, last_val, _, _ = policy.get_action_value_policy(
                params=train_state.params,
                obs=last_obs_input,
                done=last_done_batch,
                avail_actions=last_avail_batch,
                hstate=last_hstate,
                rng=jax.random.PRNGKey(0)  # Dummy key since we're just extracting the value
            )
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, init_done, traj_batch, advantages, targets = batch_info
                    def _loss_fn(params, traj_batch, gae, targets):
                        # Create shifted done signal for correct RNN hidden state reset
                        # During rollout: RNN uses last_done (done from step t-1) for reset at step t
                        # traj_batch.done[t] = done AFTER step t's action (i.e., for step t+1's reset)
                        # So we need: shifted_done[0] = init_done, shifted_done[t] = done[t-1] for t > 0
                        # This ensures RNN resets at the START of new episodes, not at the END of old ones
                        shifted_done = jnp.concatenate([init_done, traj_batch.done[:-1]], axis=0)

                        # RERUN NETWORK with corrected done signal
                        _, value, pi, _ = policy.get_action_value_policy(
                            params=params,
                            obs=traj_batch.obs,
                            done=shifted_done,
                            avail_actions=traj_batch.avail_actions,
                            hstate=init_hstate,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is unused here
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, init_done, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                minibatches = _create_minibatches(traj_batch, advantages, targets, init_hstate,
                                                  config["NUM_ACTORS"], config["NUM_MINIBATCHES"], perm_rng,
                                                  init_done=init_done)

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, init_hstate, init_done, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            # Use the ACTUAL initial hidden state and done signal from the start of the rollout
            # (not fresh zeros) - this is critical when ROLLOUT_LENGTH != max_steps
            update_state = (train_state, initial_hstate_for_update, initial_done_for_update, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            metric["update_steps"] = update_steps

            rng = update_state[-1]
            update_steps += 1
            # Include shaped tracking state in runner_state
            runner_state = (train_state, env_state, last_obs, last_done, last_hstate, rng,
                           cumulative_shaped_reward, returned_shaped_returns)
            return (runner_state, update_steps), metric

        ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1)
        num_ckpts = config["NUM_CHECKPOINTS"]

        # build a pytree that can hold the parameters for all checkpoints.
        def init_ckpt_array(params_pytree):
            return jax.tree.map(
                lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                params_pytree
            )

        def _update_step_with_checkpoint(update_with_ckpt_runner_state, unused):
            (update_runner_state, checkpoint_array, ckpt_idx) = update_with_ckpt_runner_state
            # update_runner_state is ((train_state, env_state, obs, done, hstate, rng,
            #                          cumulative_shaped, returned_shaped), update_steps)
            # Run one PPO update step
            update_runner_state, metric = _update_step(update_runner_state, None)
            _, update_steps = update_runner_state
            # update steps is 1-indexed because it was incremented at the end of the update step
            to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps-1, ckpt_and_eval_interval), 0),
                                      jnp.equal(update_steps, config["NUM_UPDATES"]))

            def store_ckpt_fn(args):
                # Write current runner_state[0].params into checkpoint_array at ckpt_idx
                # and increment ckpt_idx
                _checkpoint_array, _ckpt_idx = args
                new_checkpoint_array = jax.tree.map(
                    lambda c_arr, p: c_arr.at[_ckpt_idx].set(p),
                    _checkpoint_array,
                    update_runner_state[0][0].params
                )
                return new_checkpoint_array, _ckpt_idx + 1 
            # TODO: potential issue is that if this function is always executed regardless of whether to_store is true or false, then _ckpt_idx will be wrong

            def skip_ckpt_fn(args):
                return args  # No changes if we don't store

            checkpoint_array, ckpt_idx = jax.lax.cond(
                to_store, # if to_store, execute true function(operand). else, execute false function(operand).
                store_ckpt_fn, # true fn
                skip_ckpt_fn, # false fn
                (checkpoint_array, ckpt_idx),
            )

            runner_state = (update_runner_state, checkpoint_array, ckpt_idx)
            return runner_state, metric

        # (5) Use lax.scan over NUM_UPDATES
        rng, _rng = jax.random.split(rng)
        update_steps = 0
        init_hstate = policy.init_hstate(config["NUM_ACTORS"])
        init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
        # Initialize shaped reward tracking state (zeros)
        init_cumulative_shaped = jnp.zeros((config["NUM_ACTORS"],))
        init_returned_shaped = jnp.zeros((config["NUM_ACTORS"],))
        update_runner_state = ((train_state, env_state, obsv, init_done, init_hstate, _rng,
                                init_cumulative_shaped, init_returned_shaped), update_steps)
        checkpoint_array = init_ckpt_array(train_state.params)
        ckpt_idx = 0
        update_with_ckpt_runner_state = (update_runner_state, checkpoint_array, ckpt_idx)

        runner_state, metrics = jax.lax.scan(
            _update_step_with_checkpoint,
            update_with_ckpt_runner_state,
            xs=None,  # No per-step input data
            length=config["NUM_UPDATES"],
        )

        update_runner_state, checkpoint_array, final_ckpt_idx = runner_state

        return {
            "final_params": update_runner_state[0][0].params,
            "metrics": metrics,
            "checkpoints": checkpoint_array,
            "final_ckpt_idx": final_ckpt_idx # CLEANUP FLAG
        }
    return train

def run_ippo(config, logger):
    algorithm_config = dict(config.algorithm)
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rngs = jax.random.split(rng, algorithm_config["NUM_SEEDS"])
    
    with jax.disable_jit(False):
        train_jit = jax.jit(jax.vmap(make_train(algorithm_config, env)))
        out = train_jit(rngs)

    log_metrics(config, out, logger)
    return out

def make_train_chunked(config, env):
    """Create training functions for chunked training to reduce GPU memory usage.

    Instead of running all updates in a single jax.lax.scan (which requires
    pre-allocating memory for all metrics upfront), this splits training into chunks.
    After each chunk, metrics can be moved to CPU to free GPU memory.

    Args:
        config: Training configuration dict. Must include NUM_CHUNKS.
        env: Environment instance.

    Returns:
        Tuple of (init_fn, train_one_chunk_fn, updated_config):
            - init_fn(rng) -> initial_state: Initialize training state
            - train_one_chunk_fn(state) -> (new_state, metrics): Run one chunk of updates
            - updated_config: Config with computed values like NUM_UPDATES
    """
    config = config.copy()  # Don't modify original
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
    )

    # Validate that NUM_ACTORS is divisible by NUM_MINIBATCHES
    if config["NUM_ACTORS"] % config["NUM_MINIBATCHES"] != 0:
        raise ValueError(
            f"NUM_ACTORS ({config['NUM_ACTORS']} = {env.num_agents} agents × {config['NUM_ENVS']} envs) "
            f"must be divisible by NUM_MINIBATCHES ({config['NUM_MINIBATCHES']}). "
            f"Suggested NUM_MINIBATCHES values that divide {config['NUM_ACTORS']}: "
            f"{[i for i in [8, 10, 16, 20, 25, 32, 40, 50, 64, 80, 100] if config['NUM_ACTORS'] % i == 0]}"
        )

    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"] // config["NUM_MINIBATCHES"]
    )

    num_chunks = config.get("NUM_CHUNKS", 1)
    updates_per_chunk = config["NUM_UPDATES"] // num_chunks
    # Actual total updates may be less than NUM_UPDATES due to integer division
    actual_total_updates = updates_per_chunk * num_chunks

    ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1)
    num_ckpts = config["NUM_CHECKPOINTS"]

    # Linear schedule (original)
    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    # Cosine decay with warmup schedule (from JaxMARL)
    def create_warmup_cosine_schedule():
        base_learning_rate = config["LR"]
        lr_warmup = config.get("LR_WARMUP", 0.0)
        update_steps = config["NUM_UPDATES"]
        warmup_steps = int(lr_warmup * update_steps)

        steps_per_epoch = config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]

        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=warmup_steps * steps_per_epoch,
        )
        cosine_epochs = max(update_steps - warmup_steps, 1)

        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate,
            decay_steps=cosine_epochs * steps_per_epoch
        )
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps * steps_per_epoch],
        )
        return schedule_fn

    # Reward shaping annealing schedule (from JaxMARL)
    # Only used if REW_SHAPING_HORIZON is in config and > 0
    rew_shaping_horizon = config.get("REW_SHAPING_HORIZON", 0)
    if rew_shaping_horizon > 0:
        rew_shaping_anneal = optax.linear_schedule(
            init_value=1.0,
            end_value=0.0,
            transition_steps=int(rew_shaping_horizon)
        )
    else:
        rew_shaping_anneal = None

    def init_train_state(rng):
        """Initialize training state.

        Args:
            rng: JAX random key.

        Returns:
            Dictionary containing all training state needed across chunks.
        """
        # INIT NETWORK
        rng, init_rng = jax.random.split(rng)
        policy, init_params = initialize_agent(config["ACTOR_TYPE"], config, env, init_rng)

        if config["ANNEAL_LR"]:
            # Use warmup + cosine decay if LR_WARMUP is specified, otherwise use linear
            if config.get("LR_WARMUP", 0.0) > 0:
                lr_schedule = create_warmup_cosine_schedule()
            else:
                lr_schedule = linear_schedule
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=lr_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5))
        train_state = TrainState.create(
            apply_fn=policy.network.apply,
            params=init_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # Initialize checkpoint array
        def init_ckpt_array(params_pytree):
            return jax.tree.map(
                lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                params_pytree
            )

        rng, _rng = jax.random.split(rng)
        init_hstate = policy.init_hstate(config["NUM_ACTORS"])
        init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
        # Initialize shaped reward tracking state (zeros)
        init_cumulative_shaped = jnp.zeros((config["NUM_ACTORS"],))
        init_returned_shaped = jnp.zeros((config["NUM_ACTORS"],))
        checkpoint_array = init_ckpt_array(train_state.params)

        # Return state dict
        return {
            "train_state": train_state,
            "env_state": env_state,
            "obs": obsv,
            "done": init_done,
            "hstate": init_hstate,
            "rng": _rng,
            "cumulative_shaped": init_cumulative_shaped,
            "returned_shaped": init_returned_shaped,
            "update_steps": jnp.array(0, dtype=jnp.int32),
            "checkpoint_array": checkpoint_array,
            "ckpt_idx": jnp.array(0, dtype=jnp.int32),
        }

    def train_one_chunk(state):
        """Run one chunk of training updates.

        Args:
            state: Training state dictionary from init_train_state or previous chunk.

        Returns:
            Tuple of (new_state, metrics_for_this_chunk)
        """
        # Unpack state
        train_state = state["train_state"]
        env_state = state["env_state"]
        obsv = state["obs"]
        done = state["done"]
        hstate = state["hstate"]
        rng = state["rng"]
        cumulative_shaped = state["cumulative_shaped"]
        returned_shaped = state["returned_shaped"]
        update_steps = state["update_steps"]
        checkpoint_array = state["checkpoint_array"]
        ckpt_idx = state["ckpt_idx"]

        # Re-initialize policy for apply functions (stateless)
        policy, _ = initialize_agent(config["ACTOR_TYPE"], config, env, jax.random.PRNGKey(0))

        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            # Compute reward shaping annealing factor for this update step
            # (from JaxMARL recipe - same factor for all steps in a rollout)
            if rew_shaping_anneal is not None:
                current_timestep = update_steps * config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
                anneal_factor = rew_shaping_anneal(current_timestep)
            else:
                anneal_factor = 0.0  # No reward shaping

            # Save the initial hidden state and done signal BEFORE the rollout for PPO update epochs
            # This is important when ROLLOUT_LENGTH != max_steps (episodes span multiple rollouts)
            initial_hstate_for_update = runner_state[4]  # hstate is at index 4
            # Save initial done state for correct RNN hidden state reset during PPO updates
            # During rollout, RNN uses last_done (done from previous step) for reset
            # We need to preserve this for replay to avoid off-by-one errors
            initial_done_dict = runner_state[3]  # last_done is at index 3
            initial_done_for_update = batchify(initial_done_dict, env.agents, config["NUM_ACTORS"]).squeeze()
            initial_done_for_update = initial_done_for_update.reshape(1, config["NUM_ACTORS"])  # (1, num_actors)

            def _env_step(runner_state, unused):
                # Extended runner_state includes shaped reward tracking:
                # (train_state, env_state, last_obs, last_done, last_hstate, rng,
                #  cumulative_shaped_reward, returned_shaped_returns)
                (train_state, env_state, last_obs, last_done, last_hstate, rng,
                 cumulative_shaped_reward, returned_shaped_returns) = runner_state

                rng, act_rng = jax.random.split(rng)

                # Handle observation batchify based on actor type
                # CNN+RNN needs spatial structure preserved (num_actors, H, W, C)
                # MLP/RNN needs flattened (num_actors, obs_dim)
                if config.get("ACTOR_TYPE", "mlp") == "cnn_rnn":
                    last_obs_batch = batchify_spatial(last_obs, env.agents, config["NUM_ACTORS"])
                else:
                    last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                last_done_batch = batchify(last_done, env.agents, config["NUM_ACTORS"])

                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(batchify(avail_actions,
                    env.agents, config["NUM_ACTORS"]).astype(jnp.float32))

                # Add sequence dimension for policy input
                # CNN+RNN: (1, NUM_ACTORS, H, W, C)
                # MLP/RNN: (1, NUM_ACTORS, obs_dim)
                obs_input = last_obs_batch[np.newaxis, :]

                action, value, pi, new_hstate = policy.get_action_value_policy(
                    params=train_state.params,
                    obs=obs_input,
                    done=last_done_batch.reshape(1, config["NUM_ACTORS"]),
                    avail_actions=avail_actions.reshape(1, config["NUM_ACTORS"], -1),
                    hstate=last_hstate,
                    rng=act_rng
                )
                log_prob = pi.log_prob(action)

                action = action.squeeze()
                log_prob = log_prob.squeeze()
                value = value.squeeze()

                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k:v.flatten() for k,v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                new_obs, new_env_state, reward, new_done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )

                # Get shaped reward from env info (before applying to training reward)
                # shaped_reward from env has shape (NUM_ENVS, num_agents)
                shaped_reward_raw = info.get("shaped_reward", jnp.zeros((config["NUM_ENVS"], env.num_agents)))

                # Compute the annealed shaping component for this step
                # Flatten to (NUM_ACTORS,): stack agents then flatten
                annealed_shaping = anneal_factor * batchify(
                    {agent: shaped_reward_raw[:, i] for i, agent in enumerate(env.agents)},
                    env.agents, config["NUM_ACTORS"]
                ).squeeze()

                # Update cumulative shaped reward (adds to running total)
                new_cumulative_shaped = cumulative_shaped_reward + annealed_shaping

                # Apply reward shaping with annealing (from JaxMARL recipe)
                # reward = base_reward + anneal_factor * shaped_reward
                if rew_shaping_anneal is not None:
                    shaped_reward_dict = {agent: shaped_reward_raw[:, i] for i, agent in enumerate(env.agents)}
                    reward = jax.tree.map(
                        lambda r, sr: r + anneal_factor * sr,
                        reward,
                        shaped_reward_dict
                    )

                # Compute shaped reward for transition (base + annealed shaping)
                shaped_reward_for_transition = batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze()

                # Get global done signal
                ep_done = jnp.tile(new_done["__all__"], env.num_agents)

                # Get base episode returns from LogWrapper (tracks BASE rewards only)
                base_episode_returns = info["returned_episode_returns"]  # Shape: (NUM_ENVS, num_agents)
                base_episode_returns_flat = batchify(
                    {agent: base_episode_returns[:, i] for i, agent in enumerate(env.agents)},
                    env.agents, config["NUM_ACTORS"]
                ).squeeze()

                # Update returned_shaped_returns when episode ends
                # Total shaped return = base return (from LogWrapper) + cumulative shaped bonus
                new_returned_shaped = jnp.where(
                    ep_done,
                    base_episode_returns_flat + new_cumulative_shaped,
                    returned_shaped_returns
                )

                # Reset cumulative shaped reward on episode end
                new_cumulative_shaped = jnp.where(ep_done, 0.0, new_cumulative_shaped)

                # Reshape info and add shaped returns tracking
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                info["returned_episode_shaped_returns"] = new_returned_shaped

                # Use global done signal for transition (from JaxMARL recipe)
                done_for_transition = ep_done

                transition = Transition(
                    done_for_transition,
                    action,
                    value,
                    shaped_reward_for_transition,
                    log_prob,
                    last_obs_batch,
                    info,
                    avail_actions
                )
                runner_state = (train_state, new_env_state, new_obs, new_done, new_hstate, rng,
                               new_cumulative_shaped, new_returned_shaped)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
            )

            # Unpack extended runner_state
            (train_state, env_state, last_obs, last_done, last_hstate, rng,
             cumulative_shaped_reward, returned_shaped_returns) = runner_state
            # Use spatial batchify for CNN+RNN to preserve (H, W, C) structure
            if config.get("ACTOR_TYPE", "mlp") == "cnn_rnn":
                last_obs_batch = batchify_spatial(last_obs, env.agents, config["NUM_ACTORS"])
            else:
                last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            last_done_batch = batchify(last_done, env.agents, config["NUM_ACTORS"])
            last_done_batch = last_done_batch.reshape(1, config["NUM_ACTORS"])
            last_avail_batch = jax.vmap(env.get_avail_actions)(env_state.env_state)
            last_avail_batch = jax.lax.stop_gradient(batchify(last_avail_batch,
                env.agents, config["NUM_ACTORS"]).astype(jnp.float32))

            # Add sequence dimension for policy input
            last_obs_input = last_obs_batch[np.newaxis, :]

            _, last_val, _, _ = policy.get_action_value_policy(
                params=train_state.params,
                obs=last_obs_input,
                done=last_done_batch,
                avail_actions=last_avail_batch,
                hstate=last_hstate,
                rng=jax.random.PRNGKey(0)
            )
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, init_done, traj_batch, advantages, targets = batch_info
                    def _loss_fn(params, traj_batch, gae, targets):
                        # Create shifted done signal for correct RNN hidden state reset
                        # During rollout: RNN uses last_done (done from step t-1) for reset at step t
                        # traj_batch.done[t] = done AFTER step t's action (i.e., for step t+1's reset)
                        # So we need: shifted_done[0] = init_done, shifted_done[t] = done[t-1] for t > 0
                        # This ensures RNN resets at the START of new episodes, not at the END of old ones
                        shifted_done = jnp.concatenate([init_done, traj_batch.done[:-1]], axis=0)

                        # RERUN NETWORK with corrected done signal
                        _, value, pi, _ = policy.get_action_value_policy(
                            params=params,
                            obs=traj_batch.obs,
                            done=shifted_done,
                            avail_actions=traj_batch.avail_actions,
                            hstate=init_hstate,
                            rng=jax.random.PRNGKey(0)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, init_done, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                minibatches = _create_minibatches(traj_batch, advantages, targets, init_hstate,
                                                  config["NUM_ACTORS"], config["NUM_MINIBATCHES"], perm_rng,
                                                  init_done=init_done)

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, init_hstate, init_done, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            # Use the ACTUAL initial hidden state and done signal from the start of the rollout
            # (not fresh zeros) - this is critical when ROLLOUT_LENGTH != max_steps
            update_state = (train_state, initial_hstate_for_update, initial_done_for_update, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            metric["update_steps"] = update_steps
            # Note: returned_episode_shaped_returns is now properly computed in _env_step
            # and stored in traj_batch.info

            rng = update_state[-1]
            update_steps += 1
            # Include shaped tracking state in runner_state
            runner_state = (train_state, env_state, last_obs, last_done, last_hstate, rng,
                           cumulative_shaped_reward, returned_shaped_returns)
            return (runner_state, update_steps), metric

        def _update_step_with_checkpoint(update_with_ckpt_runner_state, unused):
            (update_runner_state, checkpoint_array, ckpt_idx) = update_with_ckpt_runner_state
            update_runner_state, metric = _update_step(update_runner_state, None)
            _, update_steps = update_runner_state

            to_store = jnp.logical_or(
                jnp.equal(jnp.mod(update_steps-1, ckpt_and_eval_interval), 0),
                jnp.equal(update_steps, actual_total_updates)  # Use actual total, not config value
            )

            def store_ckpt_fn(args):
                _checkpoint_array, _ckpt_idx = args
                new_checkpoint_array = jax.tree.map(
                    lambda c_arr, p: c_arr.at[_ckpt_idx].set(p),
                    _checkpoint_array,
                    update_runner_state[0][0].params
                )
                return new_checkpoint_array, _ckpt_idx + 1

            def skip_ckpt_fn(args):
                return args

            checkpoint_array, ckpt_idx = jax.lax.cond(
                to_store,
                store_ckpt_fn,
                skip_ckpt_fn,
                (checkpoint_array, ckpt_idx),
            )

            runner_state = (update_runner_state, checkpoint_array, ckpt_idx)
            return runner_state, metric

        # Run the chunk - include shaped tracking in runner_state
        runner_state = (train_state, env_state, obsv, done, hstate, rng,
                       cumulative_shaped, returned_shaped)
        update_runner_state = (runner_state, update_steps)
        update_with_ckpt_runner_state = (update_runner_state, checkpoint_array, ckpt_idx)

        # Run exactly updates_per_chunk updates
        final_state, metrics = jax.lax.scan(
            _update_step_with_checkpoint,
            update_with_ckpt_runner_state,
            xs=None,
            length=updates_per_chunk,
        )

        # Unpack final state (including shaped tracking)
        (runner_state_final, new_update_steps), new_checkpoint_array, new_ckpt_idx = final_state
        (train_state_final, env_state_final, obs_final, done_final, hstate_final, rng_final,
         cumulative_shaped_final, returned_shaped_final) = runner_state_final

        # Pack new state
        new_state = {
            "train_state": train_state_final,
            "env_state": env_state_final,
            "obs": obs_final,
            "done": done_final,
            "hstate": hstate_final,
            "rng": rng_final,
            "cumulative_shaped": cumulative_shaped_final,
            "returned_shaped": returned_shaped_final,
            "update_steps": new_update_steps,
            "checkpoint_array": new_checkpoint_array,
            "ckpt_idx": new_ckpt_idx,
        }

        return new_state, metrics

    return init_train_state, train_one_chunk, config


def log_metrics(config, out, logger):
    '''Save train run output and log to wandb as artifact.'''    
    train_metrics = out["metrics"]
    metric_names = get_metric_names(config["ENV_NAME"])
    train_stats = get_stats(train_metrics, metric_names)

    # each key in train_stats is a metric name, and the value is an array of shape (num_seeds, num_updates, num_agents_per_game)
    # where the last dimension contains the mean and std of the metric
    train_stats = {k: np.mean(np.array(v), axis=0) for k, v in train_stats.items()}

    # Log metrics for each update step
    num_updates = train_metrics["returned_episode"].shape[1] # shape is (num_seeds, num_updates, rollout_len, num_envs*num_agents_per_game)
    for step in range(num_updates):
        for stat_name, stat_data in train_stats.items():
            # second dimension contains the mean and std of the metric
            stat_mean = stat_data[step, 0]
            logger.log_item(f"Train/{stat_name}", stat_mean, train_step=step, commit=True)

    logger.commit()

    # save artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_savepath = save_train_run(out, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
        # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)
