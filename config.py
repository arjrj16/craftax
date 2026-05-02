"""Configuration for PPO training on Craftax-Classic.

This file intentionally only contains model and training hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """Actor-critic network hyperparameters."""

    hidden_size: int = 512
    num_hidden_layers: int = 3
    activation: str = "tanh"  # "tanh", "relu", or "silu"
    orthogonal_init: bool = True
    actor_output_scale: float = 0.01
    critic_output_scale: float = 1.0


@dataclass(frozen=True)
class PPOConfig:
    """PPO and runtime hyperparameters.

    The batch is sharded as:
        global_envs = num_envs_per_device * number_of_visible_JAX_devices
        global_steps_per_update = global_envs * num_steps

    On an 8xH100 single host, the default uses 8 shards via jax.pmap.
    """

    # Environment.
    env_name: str = "Craftax-Classic-Symbolic-v1"
    seed: int = 0
    env_max_timesteps: int = 10000  # Craftax-Classic default; raise to extend timeout.

    # Scale. For a quick smoke test, override total_timesteps to e.g. 1048576.
    total_timesteps: int = 1_000_000_000
    num_envs_per_device: int = 2048
    num_steps: int = 64

    # PPO update.
    update_epochs: int = 4
    num_minibatches: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.80
    clip_eps: float = 0.20
    vf_coef: float = 0.50
    ent_coef: float = 0.01
    max_grad_norm: float = 1.0
    normalize_advantages: bool = True
    clip_value_loss: bool = True

    # Optimizer.
    lr: float = 2e-4
    adam_eps: float = 1e-5
    anneal_lr: bool = False  # Paper note: anneal hurts at long horizons.

    # Self-Imitation Learning auxiliary loss (Oh et al. 2018).
    # Enable by setting sil_coef > 0. Buffer is a per-device circular buffer of
    # rollout transitions; the SIL loss masks by clip(advantage, 0, inf) so only
    # transitions where the agent outperformed its value estimate contribute.
    sil_coef: float = 0.0  # Set e.g. 0.1 to enable.
    sil_vf_coef: float = 0.5
    sil_buffer_capacity_per_device: int = 16384  # transitions per device
    sil_minibatch_size: int = 0  # 0 -> match PPO minibatch size

    # Logging/checkpointing.
    run_name: str = "craftax_classic_ppo_pmap"
    output_dir: str = "runs"
    log_interval_updates: int = 1
    checkpoint_interval_updates: int = 25
    save_final_checkpoint: bool = True

    # Weights & Biases.
    use_wandb: bool = False
    wandb_project: str = "craftax-ppo"
    wandb_entity: str = ""
    wandb_mode: str = "online"  # "online", "offline", or "disabled"

    # Safety checks.
    require_num_devices: int = 8  # set to 0 to allow any number of devices


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)


CONFIG = Config()

__all__ = ["ModelConfig", "PPOConfig", "Config", "CONFIG"]
