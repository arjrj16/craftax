"""8-GPU PPO training for Craftax-Classic from scratch.

Single-host 8xH100 usage:

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py

Quick smoke test:

    python train.py --require-num-devices 0 --total-timesteps 1048576 \
        --num-envs-per-device 128 --num-steps 32 --checkpoint-interval-updates 0

The implementation uses jax.pmap. Each GPU owns a shard of environments and
rollout data; gradients are averaged across GPUs with lax.pmean.
"""

from __future__ import annotations

# These must be set before importing JAX. They are safe defaults for large GPUs.
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, NamedTuple

from flax import jax_utils, serialization
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from craftax.craftax_env import make_craftax_env_from_name

from config import CONFIG, Config, ModelConfig, PPOConfig
from model import ActorCritic


AXIS_NAME = "devices"


class RunnerState(NamedTuple):
    env_state: Any
    obs: jnp.ndarray
    rng: jax.Array
    episode_return: jnp.ndarray
    episode_length: jnp.ndarray


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray
    terminal_return: jnp.ndarray
    terminal_length: jnp.ndarray
    terminal: jnp.ndarray


class PPOBatch(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    old_log_prob: jnp.ndarray
    old_value: jnp.ndarray
    advantage: jnp.ndarray
    target: jnp.ndarray


def _parse_int_like(x: str) -> int:
    return int(float(x))


def parse_args() -> Config:
    """Parse flat CLI overrides into the nested config dataclasses."""

    parser = argparse.ArgumentParser()
    defaults = CONFIG

    # PPOConfig fields: --total-timesteps maps to total_timesteps.
    for f in fields(PPOConfig):
        default = getattr(defaults.ppo, f.name)
        arg = "--" + f.name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=default)
        elif isinstance(default, int):
            parser.add_argument(arg, type=_parse_int_like, default=default)
        elif isinstance(default, float):
            parser.add_argument(arg, type=float, default=default)
        else:
            parser.add_argument(arg, type=type(default), default=default)

    # ModelConfig fields use --model-*. Example: --model-hidden-size 1024.
    for f in fields(ModelConfig):
        default = getattr(defaults.model, f.name)
        arg = "--model-" + f.name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=default)
        elif isinstance(default, int):
            parser.add_argument(arg, type=_parse_int_like, default=default)
        elif isinstance(default, float):
            parser.add_argument(arg, type=float, default=default)
        else:
            parser.add_argument(arg, type=type(default), default=default)

    args = vars(parser.parse_args())

    ppo_kwargs = {f.name: args[f.name] for f in fields(PPOConfig)}
    model_kwargs = {f.name: args["model_" + f.name] for f in fields(ModelConfig)}
    return Config(model=replace(defaults.model, **model_kwargs), ppo=replace(defaults.ppo, **ppo_kwargs))


def categorical_log_prob(logits: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, action[..., None], axis=-1).squeeze(-1)


def categorical_entropy(logits: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs, axis=-1)


def make_lr_schedule(cfg: PPOConfig, num_updates: int):
    total_optimizer_steps = num_updates * cfg.update_epochs * cfg.num_minibatches
    if not cfg.anneal_lr:
        return cfg.lr

    def schedule(count: jnp.ndarray) -> jnp.ndarray:
        frac = 1.0 - (count / float(max(total_optimizer_steps, 1)))
        return cfg.lr * jnp.clip(frac, 0.0, 1.0)

    return schedule


def create_train_state(
    rng: jax.Array,
    obs_shape: tuple[int, ...],
    num_actions: int,
    cfg: Config,
    num_updates: int,
) -> TrainState:
    model = ActorCritic(
        num_actions=num_actions,
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        activation=cfg.model.activation,
        orthogonal_init=cfg.model.orthogonal_init,
        actor_output_scale=cfg.model.actor_output_scale,
        critic_output_scale=cfg.model.critic_output_scale,
    )
    dummy_obs = jnp.zeros((1, *obs_shape), dtype=jnp.float32)
    params = model.init(rng, dummy_obs)["params"]
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.ppo.max_grad_norm),
        optax.adam(make_lr_schedule(cfg.ppo, num_updates), eps=cfg.ppo.adam_eps),
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def make_pmapped_fns(cfg: Config, env: Any, env_params: Any):
    """Build pmapped init/update functions around a Craftax env closure."""

    num_envs = cfg.ppo.num_envs_per_device
    num_steps = cfg.ppo.num_steps
    batch_size = num_envs * num_steps
    assert batch_size % cfg.ppo.num_minibatches == 0
    minibatch_size = batch_size // cfg.ppo.num_minibatches

    reset_one = lambda key: env.reset(key, env_params)
    step_one = lambda key, state, action: env.step(key, state, action, env_params)
    reset_batch = jax.vmap(reset_one)
    step_batch = jax.vmap(step_one)

    def init_runner_state(rng: jax.Array) -> RunnerState:
        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, num_envs)
        obs, env_state = reset_batch(reset_keys)
        return RunnerState(
            env_state=env_state,
            obs=obs,
            rng=rng,
            episode_return=jnp.zeros((num_envs,), dtype=jnp.float32),
            episode_length=jnp.zeros((num_envs,), dtype=jnp.int32),
        )

    def calculate_gae(transitions: Transition, last_value: jnp.ndarray):
        def scan_fn(carry, transition):
            gae, next_value = carry
            not_done = 1.0 - transition.done.astype(jnp.float32)
            delta = transition.reward + cfg.ppo.gamma * next_value * not_done - transition.value
            gae = delta + cfg.ppo.gamma * cfg.ppo.gae_lambda * not_done * gae
            return (gae, transition.value), gae

        _, advantages = jax.lax.scan(
            scan_fn,
            (jnp.zeros_like(last_value), last_value),
            transitions,
            reverse=True,
            unroll=16,
        )
        targets = advantages + transitions.value
        return advantages, targets

    def update_once(train_state: TrainState, runner_state: RunnerState):
        def env_step(state: tuple[TrainState, RunnerState], _: Any):
            train_state, runner_state = state
            rng, action_rng, step_rng = jax.random.split(runner_state.rng, 3)

            logits, value = train_state.apply_fn({"params": train_state.params}, runner_state.obs)
            action = jax.random.categorical(action_rng, logits, axis=-1)
            log_prob = categorical_log_prob(logits, action)

            step_keys = jax.random.split(step_rng, num_envs)
            next_obs, next_env_state, reward, done, _info = step_batch(
                step_keys, runner_state.env_state, action
            )
            reward = reward.astype(jnp.float32)
            done_f = done.astype(jnp.float32)

            updated_episode_return = runner_state.episode_return + reward
            updated_episode_length = runner_state.episode_length + 1
            terminal_return = jnp.where(done, updated_episode_return, 0.0)
            terminal_length = jnp.where(done, updated_episode_length, 0)

            next_runner_state = RunnerState(
                env_state=next_env_state,
                obs=next_obs,
                rng=rng,
                episode_return=updated_episode_return * (1.0 - done_f),
                episode_length=updated_episode_length * (1 - done.astype(jnp.int32)),
            )
            transition = Transition(
                obs=runner_state.obs,
                action=action,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob,
                terminal_return=terminal_return,
                terminal_length=terminal_length,
                terminal=done,
            )
            return (train_state, next_runner_state), transition

        (train_state, runner_state), transitions = jax.lax.scan(
            env_step, (train_state, runner_state), None, length=num_steps
        )

        _, last_value = train_state.apply_fn({"params": train_state.params}, runner_state.obs)
        advantages, targets = calculate_gae(transitions, last_value)

        def flatten_time_env(x: jnp.ndarray) -> jnp.ndarray:
            return x.reshape((batch_size,) + x.shape[2:])

        batch = PPOBatch(
            obs=flatten_time_env(transitions.obs),
            action=flatten_time_env(transitions.action),
            old_log_prob=flatten_time_env(transitions.log_prob),
            old_value=flatten_time_env(transitions.value),
            advantage=flatten_time_env(advantages),
            target=flatten_time_env(targets),
        )

        if cfg.ppo.normalize_advantages:
            adv_mean = batch.advantage.mean()
            adv_var = jnp.mean(jnp.square(batch.advantage - adv_mean))
            # Use all replicas' rollout advantages for normalization.
            adv_mean = jax.lax.pmean(adv_mean, AXIS_NAME)
            adv_var = jax.lax.pmean(adv_var, AXIS_NAME)
            batch = batch._replace(
                advantage=(batch.advantage - adv_mean) / jnp.sqrt(adv_var + 1e-8)
            )

        def loss_fn(params: Any, mb: PPOBatch):
            logits, value = train_state.apply_fn({"params": params}, mb.obs)
            log_prob = categorical_log_prob(logits, mb.action)
            entropy = categorical_entropy(logits).mean()

            ratio = jnp.exp(log_prob - mb.old_log_prob)
            policy_loss_1 = ratio * mb.advantage
            policy_loss_2 = (
                jnp.clip(ratio, 1.0 - cfg.ppo.clip_eps, 1.0 + cfg.ppo.clip_eps)
                * mb.advantage
            )
            policy_loss = -jnp.minimum(policy_loss_1, policy_loss_2).mean()

            if cfg.ppo.clip_value_loss:
                value_clipped = mb.old_value + jnp.clip(
                    value - mb.old_value, -cfg.ppo.clip_eps, cfg.ppo.clip_eps
                )
                value_loss = jnp.maximum(
                    jnp.square(value - mb.target),
                    jnp.square(value_clipped - mb.target),
                ).mean()
            else:
                value_loss = jnp.square(value - mb.target).mean()
            value_loss = 0.5 * value_loss

            total_loss = policy_loss + cfg.ppo.vf_coef * value_loss - cfg.ppo.ent_coef * entropy
            approx_kl = (mb.old_log_prob - log_prob).mean()
            clip_fraction = (jnp.abs(ratio - 1.0) > cfg.ppo.clip_eps).mean()
            explained_var = 1.0 - jnp.var(mb.target - value) / (jnp.var(mb.target) + 1e-8)
            aux = {
                "loss": total_loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
                "explained_var": explained_var,
            }
            return total_loss, aux

        def update_minibatch(state: TrainState, mb: PPOBatch):
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, mb)
            grads = jax.lax.pmean(grads, AXIS_NAME)
            aux = jax.lax.pmean(aux, AXIS_NAME)
            state = state.apply_gradients(grads=grads)
            return state, aux

        def update_epoch(state_and_rng: tuple[TrainState, jax.Array], _: Any):
            state, rng = state_and_rng
            rng, perm_rng = jax.random.split(rng)
            permutation = jax.random.permutation(perm_rng, batch_size)
            shuffled = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = jax.tree.map(
                lambda x: x.reshape((cfg.ppo.num_minibatches, minibatch_size) + x.shape[1:]),
                shuffled,
            )
            state, aux = jax.lax.scan(update_minibatch, state, minibatches)
            return (state, rng), aux

        (train_state, rng), aux = jax.lax.scan(
            update_epoch,
            (train_state, runner_state.rng),
            None,
            length=cfg.ppo.update_epochs,
        )
        runner_state = runner_state._replace(rng=rng)

        # Rollout-level metrics. Sum/count are psum'd so replica 0 sees global values.
        local_episode_count = transitions.terminal.astype(jnp.float32).sum()
        local_return_sum = transitions.terminal_return.sum()
        local_length_sum = transitions.terminal_length.astype(jnp.float32).sum()
        global_episode_count = jax.lax.psum(local_episode_count, AXIS_NAME)
        global_return_sum = jax.lax.psum(local_return_sum, AXIS_NAME)
        global_length_sum = jax.lax.psum(local_length_sum, AXIS_NAME)
        denom = jnp.maximum(global_episode_count, 1.0)

        metrics = jax.tree.map(lambda x: x.mean(), aux)
        metrics["episodes"] = global_episode_count
        metrics["mean_episode_return"] = global_return_sum / denom
        metrics["mean_episode_length"] = global_length_sum / denom
        metrics["mean_rollout_reward"] = jax.lax.pmean(transitions.reward.mean(), AXIS_NAME)
        return train_state, runner_state, metrics

    return (
        jax.pmap(init_runner_state, axis_name=AXIS_NAME),
        jax.pmap(update_once, axis_name=AXIS_NAME),
    )


def unreplicate(tree: Any) -> Any:
    return jax.tree.map(lambda x: np.asarray(x[0]), tree)


def save_checkpoint(path: Path, train_state: TrainState, cfg: Config, update: int, global_step: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "update": update,
        "global_step": global_step,
        "config": asdict(cfg),
    }
    with (path / "metadata.json").open("w") as f:
        json.dump(payload, f, indent=2)
    params = unreplicate(train_state.params)
    with (path / "params.msgpack").open("wb") as f:
        f.write(serialization.to_bytes(params))


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    cfg = parse_args()
    ppo = cfg.ppo

    num_devices = jax.local_device_count()
    devices = jax.local_devices()
    if ppo.require_num_devices and num_devices != ppo.require_num_devices:
        raise RuntimeError(
            f"Expected {ppo.require_num_devices} JAX devices, but found {num_devices}: {devices}. "
            "Set --require-num-devices 0 for smoke tests or non-8-GPU runs."
        )

    env = make_craftax_env_from_name(ppo.env_name, auto_reset=True)
    env_params = env.default_params
    obs_shape = tuple(env.observation_space(env_params).shape)
    num_actions = int(env.action_space(env_params).n)

    global_envs = ppo.num_envs_per_device * num_devices
    global_steps_per_update = global_envs * ppo.num_steps
    num_updates = int(ppo.total_timesteps // global_steps_per_update)
    if num_updates <= 0:
        raise ValueError(
            f"total_timesteps={ppo.total_timesteps} is smaller than one update "
            f"({global_steps_per_update} steps)."
        )
    effective_total_timesteps = num_updates * global_steps_per_update

    batch_size_per_device = ppo.num_envs_per_device * ppo.num_steps
    if batch_size_per_device % ppo.num_minibatches != 0:
        raise ValueError(
            "num_envs_per_device * num_steps must be divisible by num_minibatches. "
            f"Got {batch_size_per_device} and {ppo.num_minibatches}."
        )

    run_dir = Path(ppo.output_dir) / ppo.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("Craftax PPO pmap training")
    print(f"  env_name:                 {ppo.env_name}")
    print(f"  devices:                  {num_devices} {[d.platform + ':' + str(d.id) for d in devices]}")
    print(f"  obs_shape:                {obs_shape}")
    print(f"  num_actions:              {num_actions}")
    print(f"  per_device_envs:          {ppo.num_envs_per_device}")
    print(f"  global_envs:              {global_envs}")
    print(f"  num_steps/update:         {ppo.num_steps}")
    print(f"  global_steps/update:      {global_steps_per_update:,}")
    print(f"  num_updates:              {num_updates:,}")
    print(f"  effective_total_steps:    {effective_total_timesteps:,}")
    print(f"  run_dir:                  {run_dir}")

    wandb_run = None
    if ppo.use_wandb:
        import wandb
        init_kwargs = dict(
            project=ppo.wandb_project,
            name=ppo.run_name,
            config=asdict(cfg),
            mode=ppo.wandb_mode,
            dir=str(run_dir),
        )
        if ppo.wandb_entity:
            init_kwargs["entity"] = ppo.wandb_entity
        wandb_run = wandb.init(**init_kwargs)

    rng = jax.random.PRNGKey(ppo.seed)
    rng, init_rng, runner_rng = jax.random.split(rng, 3)

    train_state = create_train_state(init_rng, obs_shape, num_actions, cfg, num_updates)
    train_state = jax_utils.replicate(train_state)

    p_init_runner_state, p_update_once = make_pmapped_fns(cfg, env, env_params)
    runner_keys = jax.random.split(runner_rng, num_devices)
    runner_state = p_init_runner_state(runner_keys)

    # Trigger compilation before timing steady-state throughput.
    print("Compiling first update...")
    t_compile = time.time()
    train_state, runner_state, metrics = p_update_once(train_state, runner_state)
    jax.block_until_ready(metrics["loss"])
    compile_seconds = time.time() - t_compile
    print(f"First update including compile: {compile_seconds:.2f}s")

    global_step = global_steps_per_update
    start_time = time.time()
    last_log_time = start_time
    last_log_step = global_step

    def host_metric(x: jnp.ndarray | np.ndarray | float) -> float:
        return float(np.asarray(x)[0])

    # Log first update too.
    first_row = {
        "update": 1,
        "global_step": global_step,
        "sps_since_last_log": math.nan,
        "sps_total_after_compile": math.nan,
        **{k: host_metric(v) for k, v in metrics.items()},
    }
    append_csv(run_dir / "metrics.csv", first_row)
    print(json.dumps(first_row, indent=2))
    if wandb_run is not None:
        wandb_run.log(first_row, step=global_step)

    for update in range(2, num_updates + 1):
        train_state, runner_state, metrics = p_update_once(train_state, runner_state)
        global_step += global_steps_per_update

        should_log = (update % ppo.log_interval_updates == 0) or (update == num_updates)
        if should_log:
            jax.block_until_ready(metrics["loss"])
            now = time.time()
            elapsed = now - start_time
            delta_steps = global_step - last_log_step
            delta_time = now - last_log_time
            row = {
                "update": update,
                "global_step": global_step,
                "sps_since_last_log": delta_steps / max(delta_time, 1e-9),
                "sps_total_after_compile": (global_step - global_steps_per_update)
                / max(elapsed, 1e-9),
                **{k: host_metric(v) for k, v in metrics.items()},
            }
            append_csv(run_dir / "metrics.csv", row)
            print(json.dumps(row, indent=2))
            if wandb_run is not None:
                wandb_run.log(row, step=global_step)
            last_log_time = now
            last_log_step = global_step

        if ppo.checkpoint_interval_updates > 0 and update % ppo.checkpoint_interval_updates == 0:
            jax.block_until_ready(train_state.step)
            ckpt_dir = run_dir / "checkpoints" / f"update_{update:08d}"
            save_checkpoint(ckpt_dir, train_state, cfg, update, global_step)
            print(f"Saved checkpoint: {ckpt_dir}")

    if ppo.save_final_checkpoint:
        jax.block_until_ready(train_state.step)
        ckpt_dir = run_dir / "checkpoints" / "final"
        save_checkpoint(ckpt_dir, train_state, cfg, num_updates, global_step)
        print(f"Saved final checkpoint: {ckpt_dir}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
