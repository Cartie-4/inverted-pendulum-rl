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

    def set_init_mode(self, mode: str) -> None:
        """Switch the initial-state curriculum for this whole vector env.

        ``BatchedPendulum`` holds a single config object (``cfgs[0]``) and resets
        through it, so every plant in the batch shares one initial-state
        distribution.  Switching therefore means mutating that one object, which
        is what the training loop used to reach into and do by hand.  Keeping the
        operation here means the shared-object assumption lives next to the code
        that depends on it, instead of in the training loop.
        """
        self.cfg.init_mode = mode  # type: ignore[assignment]
        self.plant.cfg.init_mode = mode  # type: ignore[assignment]

    # ------------------------------------------------------------------- step
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs, rewards, done, info = self.plant.step(actions)
        for i in np.flatnonzero(done):
            self.finished_returns.append(float(info["episode_return"][i]))
            self.finished_lengths.append(int(self.plant.steps[i]))
            self.finished_success.append(bool(info["is_success"][i]))
        # The observation handed back for a finished env is the *fresh* episode's state,
        # which is what the next action must be chosen from.  Keep the state the episode
        # actually ended in as well: bootstrapping a truncated transition with
        # V(fresh start) tells the critic that running into the rail teleports the agent
        # into a good episode, and the policy goes looking for the rail (run
        # p5_swing_fixed degraded exactly that way: value loss 700 -> 5800, mean episode
        # length 265 -> 50, val success 0.72 -> 0.31).
        info["terminal_obs"] = obs.copy()
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
