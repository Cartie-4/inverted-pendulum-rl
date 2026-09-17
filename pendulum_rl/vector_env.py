"""Synchronous vectorised env + episode bookkeeping (import-light, no Lightning).

Thin wrapper around :class:`pendulum_rl.batched_env.BatchedPendulum`: the whole
batch is integrated in NumPy, so training does not pay Python-loop overhead per
environment per physics substep.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .batched_env import BatchedPendulum
from .envs.inverted_pendulum import EnvConfig


class SyncVectorEnv:
    """Run several plants side by side and record episode outcomes."""

    def __init__(self, cfgs: list[EnvConfig], seeds: list[int]):
        if not cfgs:
            raise ValueError("need at least one env config")
        self.cfgs = list(cfgs)
        self.cfg = cfgs[0]
        self.seeds = list(seeds)
        self.num_envs = len(cfgs)
        self.rngs = [np.random.default_rng(s) for s in seeds]
        self.plant = BatchedPendulum(self.cfg, self.num_envs)
        self.obs = np.zeros((self.num_envs, self.cfg.obs_dim), dtype=np.float32)
        self.finished_returns: list[float] = []
        self.finished_lengths: list[int] = []
        self.finished_success: list[bool] = []
        self.last_info: dict[str, Any] = {}
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self, seeds: list[int] | None = None) -> np.ndarray:
        if seeds is not None:
            self.rngs = [np.random.default_rng(s) for s in seeds]
            self.seeds = list(seeds)
        self.plant.reset(np.ones(self.num_envs, dtype=bool), rngs=self.rngs)
        self.clear_episodes()
        self.obs = self.plant.obs()
        return self.obs

    def clear_episodes(self) -> None:
        self.finished_returns.clear()
        self.finished_lengths.clear()
        self.finished_success.clear()

    # ------------------------------------------------------------------- step
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs, rewards, done, info = self.plant.step(actions)
        for i in np.flatnonzero(done):
            self.finished_returns.append(float(info["episode_return"][i]))
            self.finished_lengths.append(int(self.plant.steps[i]))
            self.finished_success.append(bool(info["is_success"][i]))
        if done.any():
            # Fresh episodes start from a newly drawn state.
            self.plant.reset(done, rngs=self.rngs)
            obs = self.plant.obs()
        self.obs = obs
        self.last_info = info
        return obs, rewards.astype(np.float32), done.astype(np.float32)

    # ---------------------------------------------------------------- helpers
    @property
    def mean_abs_theta(self) -> float:
        """Mean |angle error| across envs, in radians."""
        return float(np.mean(np.abs(_wrap(self.plant.state.theta))))

    def energies(self) -> np.ndarray:
        return self.plant.energies()

    def close(self) -> None:
        """Nothing to release (kept for API symmetry with gymnasium)."""

    def __len__(self) -> int:
        return self.num_envs


def _wrap(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2.0 * np.pi) - np.pi
