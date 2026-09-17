"""Minimal, dependency-light PPO (Schulman et al., 2017) for continuous control.

Written from scratch on top of plain PyTorch so the update rule stays readable:
clipped surrogate objective + clipped value loss + entropy bonus, GAE(lambda)
advantages, and a diagonal Gaussian policy whose log-std is a learned parameter
independent of the state (the standard choice for low-dimensional control).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..utils import RunningMeanStd


@dataclass
class PPOConfig:
    obs_dim: int = 6
    action_dim: int = 1

    # optimisation
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip_range: float | None = 0.2
    entropy_coef: float = 0.005
    entropy_coef_final: float = 5e-4
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs_per_update: int = 10
    minibatches: int = 16
    target_kl: float = 0.05          # early-stop an update when KL blows up
    normalize_advantage: bool = True
    normalize_obs: bool = True

    # architecture
    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: str = "tanh"
    #: initial log standard deviation of the Gaussian policy.  Actions are in
    #: units of the actuator limit, and a good balance policy uses only a few
    #: percent of that limit, so exploration noise must start *small*: with
    #: log_std = -3 (sigma = 0.05) the initial dither is ~5 % of full scale.
    #: Starting at sigma ~ 0.6 (the common default) buries the tiny optimal
    #: action under noise about an order of magnitude larger than the signal
    #: and the policy never learns.  It is a learnable parameter, so the agent
    #: can still widen it later if the task needs larger moves.
    log_std_init: float = -3.0
    log_std_min: float = -6.0
    log_std_max: float = 1.0

    def lr_lambda(self, progress: float) -> float:
        """Linear decay of the entropy bonus over training."""
        progress = float(np.clip(progress, 0.0, 1.0))
        self.entropy_coef = self.entropy_coef_final + (self.entropy_coef_original - self.entropy_coef_final) * (
            1.0 - progress
        )
        return 1.0

    def __post_init__(self) -> None:
        self.entropy_coef_original = self.entropy_coef


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, activation: str = "tanh") -> nn.Sequential:
    act = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[activation]
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    net = nn.Sequential(*layers)
    for module in net.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(module.bias)
    return net


class ActorCritic(nn.Module):
    """Shared-input actor / critic with a state-independent diagonal Gaussian."""

    def __init__(self, cfg: PPOConfig):
        super().__init__()
        self.cfg = cfg
        self.actor = _mlp(cfg.obs_dim, cfg.hidden_sizes, cfg.action_dim, cfg.activation)
        self.critic = _mlp(cfg.obs_dim, cfg.hidden_sizes, 1, cfg.activation)
        self.log_std = nn.Parameter(torch.full((cfg.action_dim,), float(cfg.log_std_init)))
        # Keep the initial policy's outputs small so early rollouts are informative.
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(obs)
        log_std = self.log_std.clamp(self.cfg.log_std_min, self.cfg.log_std_max).expand_as(mean)
        return torch.distributions.Normal(mean, log_std.exp())

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        return action, dist.log_prob(action).sum(-1), self.value(obs)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, self.value(obs)


class RolloutBuffer:
    """One PPO iteration worth of experience, stored as flat numpy arrays."""

    def __init__(self, steps: int, num_envs: int, obs_dim: int, action_dim: int):
        size = steps * num_envs
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.ptr = 0

    def __len__(self) -> int:
        return self.ptr

    def add(self, obs, actions, log_probs, values, rewards, dones) -> None:
        n = len(obs)
        sl = slice(self.ptr, self.ptr + n)
        self.obs[sl] = obs
        self.actions[sl] = np.asarray(actions).reshape(n, -1)
        self.log_probs[sl] = log_probs
        self.values[sl] = values
        self.rewards[sl] = rewards
        self.dones[sl] = dones
        self.ptr += n

    def compute_gae(self, last_values: np.ndarray, steps: int, num_envs: int, cfg: PPOConfig) -> tuple[np.ndarray, np.ndarray]:
        """GAE(lambda) over a (steps, num_envs) time-major view of the buffer."""
        advantages = np.zeros_like(self.rewards)
        last_gae = np.zeros(num_envs, dtype=np.float32)
        rewards = self.rewards.reshape(steps, num_envs)
        values = self.values.reshape(steps, num_envs)
        dones = self.dones.reshape(steps, num_envs)
        adv = advantages.reshape(steps, num_envs)
        for t in reversed(range(steps)):
            next_value = last_values if t == steps - 1 else values[t + 1]
            next_non_terminal = 1.0 - (dones[t] if t == steps - 1 else dones[t + 1])
            # `dones[t]` marks the transition t -> t+1, so bootstrap only when
            # the next state is not terminal.
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + cfg.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
            adv[t] = last_gae
        returns = advantages + self.values
        return advantages, returns

    def minibatches(self, batch_size: int, generator: np.random.Generator):
        indices = np.arange(self.ptr)
        generator.shuffle(indices)
        for start in range(0, self.ptr, batch_size):
            yield indices[start : start + batch_size]


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Accept 'auto' / 'gpu' as well as anything ``torch.device`` understands."""
    if isinstance(device, torch.device):
        return device
    name = (device or "auto").lower()
    if name in ("auto", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class PPOAgent:
    """Ties together the network, the optimiser and the PPO loss computation."""

    def __init__(self, cfg: PPOConfig, device: str | torch.device = "cpu"):
        self.cfg = cfg
        self.device = resolve_device(device)
        self.model = ActorCritic(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.obs_rms = RunningMeanStd(shape=(cfg.obs_dim,))
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------- rollout io
    @torch.no_grad()
    def sample_actions(self, obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_n = self.obs_rms.normalize(obs)
        obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=self.device)
        action, log_prob, value = self.model.act(obs_t, deterministic=deterministic)
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> np.ndarray:
        obs_n = self.obs_rms.normalize(obs)
        obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=self.device)
        return self.model.value(obs_t).cpu().numpy()

    # ----------------------------------------------------------------- update
    def update(self, buffer: RolloutBuffer, steps: int, num_envs: int) -> dict[str, float]:
        cfg = self.cfg
        last_values = self.value(buffer.obs[-num_envs:])
        advantages, returns = buffer.compute_gae(last_values, steps, num_envs, cfg)

        obs = torch.as_tensor(self.obs_rms.normalize(buffer.obs), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(buffer.actions, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(buffer.log_probs, dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(buffer.values, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        batch_size = max(1, len(buffer) // cfg.minibatches)
        stats: dict[str, float] = {}
        n_updates = 0
        early_stop = False
        for _ in range(cfg.epochs_per_update):
            for idx in buffer.minibatches(batch_size, self._rng):
                idx_t = torch.as_tensor(idx, dtype=torch.long, device=self.device)
                mb_adv = adv_t[idx_t]
                if cfg.normalize_advantage and mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                log_prob, entropy, value = self.model.evaluate_actions(obs[idx_t], actions[idx_t])

                # --- clipped surrogate policy loss -----------------------
                ratio = torch.exp(log_prob - old_log_probs[idx_t])
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                # --- clipped value loss ---------------------------------
                if cfg.value_clip_range is None:
                    value_loss = 0.5 * (value - ret_t[idx_t]).pow(2).mean()
                else:
                    v_clipped = old_values[idx_t] + torch.clamp(
                        value - old_values[idx_t], -cfg.value_clip_range, cfg.value_clip_range
                    )
                    value_loss = 0.5 * torch.max(
                        (value - ret_t[idx_t]).pow(2), (v_clipped - ret_t[idx_t]).pow(2)
                    ).mean()

                entropy_bonus = entropy.mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_bonus

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - (log_prob - old_log_probs[idx_t])).mean()
                    clip_fraction = ((ratio - 1.0).abs() > cfg.clip_range).float().mean()
                stats = {
                    "loss": float(loss.detach()),
                    "policy_loss": float(policy_loss.detach()),
                    "value_loss": float(value_loss.detach()),
                    "entropy": float(entropy_bonus.detach()),
                    "approx_kl": float(approx_kl.detach()),
                    "clip_fraction": float(clip_fraction.detach()),
                    "std": float(self.model.log_std.detach().exp().mean()),
                }
                n_updates += 1
                if cfg.target_kl is not None and stats["approx_kl"] > cfg.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break
        stats["updates"] = float(n_updates)
        stats["early_stop"] = float(early_stop)
        return stats

    # ------------------------------------------------------------ (de)serialise
    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_rms": self.obs_rms.state_dict(),
            "config": vars(self.cfg).copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        model_state = state.get("model", state)
        self.model.load_state_dict(model_state)
        if "optimizer" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except ValueError:  # parameter groups differ -> fresh optimiser is fine
                pass
        if "obs_rms" in state:
            self.obs_rms.load_state_dict(state["obs_rms"])

    def freeze_normalizer(self) -> None:
        self.obs_rms.freeze = True
