"""Batched (vectorised) version of the same plant used by the scalar env.

The dynamics are small enough to run for *every* environment at once in NumPy,
which matters because the plant is integrated 10 times per agent step: a Python
loop over environments would dominate training time.  The equations here are
kept literally identical to
:meth:`pendulum_rl.envs.inverted_pendulum.InvertedPendulumEnv._accel` and the
sanity tests assert the two agree, so there is a single source of truth for the
physics and the fast path cannot silently drift from the reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .envs.inverted_pendulum import MODEL_CART, MODEL_PIVOT, EnvConfig, wrap_angle


@dataclass
class BatchArrays:
    """Mutable state of a batch of plants, laid out as flat arrays."""

    x: np.ndarray
    x_dot: np.ndarray
    theta: np.ndarray
    theta_dot: np.ndarray
    u: np.ndarray


class BatchedPendulum:
    """Vectorised dynamics / reward / bookkeeping for ``n`` identical plants."""

    def __init__(self, cfg: EnvConfig, n: int):
        self.cfg = cfg
        self.n = n
        self.state = BatchArrays(
            x=np.zeros(n),
            x_dot=np.zeros(n),
            theta=np.zeros(n),
            theta_dot=np.zeros(n),
            u=np.zeros(n),
        )
        self.steps = np.zeros(n, dtype=np.int64)
        self.success_streak = np.zeros(n, dtype=np.int64)
        self.max_success_streak = np.zeros(n, dtype=np.int64)
        self.episode_return = np.zeros(n)

    # ------------------------------------------------------------------ reset
    def reset(self, mask: np.ndarray, init_mode: str | None = None, rngs: list[np.random.Generator] | None = None) -> None:
        """Reset the entries selected by ``mask`` (a boolean array of length n)."""
        cfg = self.cfg
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return
        mode = init_mode or cfg.init_mode
        k = idx.size
        if rngs is not None:
            def draw(lo: float, hi: float) -> np.ndarray:
                return np.array([rngs[i].uniform(lo, hi) for i in idx])

            def draw_norm(scale: float) -> np.ndarray:
                return np.array([rngs[i].normal(0.0, scale) for i in idx])
        else:  # pragma: no cover - only used by explicit tests
            def draw(lo: float, hi: float) -> np.ndarray:
                return np.random.uniform(lo, hi, size=k)

            def draw_norm(scale: float) -> np.ndarray:
                return np.random.normal(0.0, scale, size=k)

        if mode == "upright":
            x = draw(-cfg.init_pos_range, cfg.init_pos_range)
            theta = draw(-cfg.init_angle_range, cfg.init_angle_range)
            x_dot = np.zeros(k)
            theta_dot = np.zeros(k)
        elif mode == "hanging":
            x = draw(-0.2, 0.2)
            theta = np.pi + draw(-0.2, 0.2)
            x_dot = np.zeros(k)
            theta_dot = draw(-0.2, 0.2)
        elif mode == "random":
            if cfg.init_angle_limit is not None:
                theta = cfg.init_angle_center + draw(-cfg.init_angle_limit, cfg.init_angle_limit)
                rate = cfg.init_rate_limit if cfg.init_rate_limit is not None else 0.5
                x = draw(-1.0, 1.0)
                x_dot = draw(-rate, rate)
                theta_dot = draw(-rate, rate)
            else:
                x = draw(-1.0, 1.0)
                theta = draw(-np.pi, np.pi)
                x_dot = draw(-0.5, 0.5)
                theta_dot = draw(-0.5, 0.5)
        else:
            raise ValueError(f"unknown init mode: {mode!r}")

        if cfg.model == MODEL_PIVOT:
            x = np.zeros(k)
            x_dot = np.zeros(k)

        s = self.state
        s.x[idx], s.theta[idx] = x, theta
        s.x_dot[idx], s.theta_dot[idx] = x_dot, theta_dot
        s.u[idx] = 0.0
        self.steps[idx] = 0
        self.success_streak[idx] = 0
        self.max_success_streak[idx] = 0
        self.episode_return[idx] = 0.0

    # -------------------------------------------------------------------- obs
    def obs(self) -> np.ndarray:
        cfg = self.cfg
        s = self.state
        theta = wrap_angle(s.theta)
        if cfg.model == MODEL_PIVOT:
            return np.stack(
                [np.cos(theta), np.sin(theta), _scale(s.theta_dot, 8.0), _scale(theta, np.pi)], axis=-1
            ).astype(np.float32)
        return np.stack(
            [
                _scale(s.x, cfg.x_limit),
                _scale(s.x_dot, 10.0),
                np.cos(theta),
                np.sin(theta),
                _scale(s.theta_dot, 8.0),
                _scale(theta, np.pi),
            ],
            axis=-1,
        ).astype(np.float32)

    # ------------------------------------------------------------------- step
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Advance every plant one agent step.

        Returns ``(obs, reward, done, info)`` where ``done = terminated or truncated``.
        """
        cfg = self.cfg
        u = np.clip(np.asarray(actions, dtype=np.float64).reshape(self.n), -cfg.action_limit, cfg.action_limit)
        self.state.u = u

        for _ in range(cfg.frame_skip):
            self._integrate()

        self.steps += 1
        s = self.state
        theta = wrap_angle(s.theta)
        theta_dot, x, x_dot = s.theta_dot, s.x, s.x_dot

        u_norm = u / cfg.action_limit
        reward = (
            np.cos(theta) ** cfg.a_theta_power
            + cfg.a_theta_offset
            - cfg.w_theta * theta**2
            - cfg.w_x * x**2
            - cfg.w_v * x_dot**2
            - cfg.w_omega * theta_dot**2
            - cfg.w_u * u_norm**2
        )
        self.episode_return += reward

        out_of_rail = np.abs(x) > cfg.x_limit if cfg.model == MODEL_CART else np.zeros(self.n, dtype=bool)
        diverged = (~np.isfinite(s.x)) | (~np.isfinite(s.theta)) | (~np.isfinite(theta_dot)) | (np.abs(theta_dot) > 1e3)
        # "the pole fell over" (optional): a real failure -> terminated, so the
        # value function stops bootstrapping instead of drifting along.
        fell_over = (
            np.abs(theta) > cfg.terminate_angle
            if cfg.terminate_angle is not None
            else np.zeros(self.n, dtype=bool)
        )
        failed = out_of_rail | diverged | fell_over
        terminated = failed & (fell_over | cfg.terminate_on_limit)
        truncated = failed & ~terminated
        if cfg.max_episode_steps is not None:
            truncated = truncated | (self.steps >= cfg.max_episode_steps)

        balanced = (np.abs(theta) < cfg.success_angle) & (np.abs(x) < cfg.success_x_range) & (np.abs(theta_dot) < 1.5)
        self.success_streak = np.where(balanced, self.success_streak + 1, 0)
        self.max_success_streak = np.maximum(self.max_success_streak, self.success_streak)

        done = terminated | truncated
        info = {
            "theta": theta,
            "x": x,
            "theta_dot": theta_dot,
            "x_dot": x_dot,
            "out_of_rail": out_of_rail,
            "diverged": diverged,
            "fell_over": fell_over,
            # Exposed separately from ``done`` so the PPO update can bootstrap
            # through truncations (rail limit, divergence, step cap) and stop
            # bootstrapping only at a true termination.
            "terminated": terminated,
            "truncated": truncated,
            "is_success": self.max_success_streak >= cfg.success_steps,
            "episode_return": self.episode_return.copy(),
            "balanced_streak": self.success_streak.copy(),
        }
        return self.obs(), reward, done, info

    # -------------------------------------------------------------- dynamics
    def _integrate(self) -> None:
        """One RK4 step for the whole batch (mirrors the scalar implementation)."""
        cfg = self.cfg
        dt = cfg.sim_dt
        s = self.state

        def deriv(y: np.ndarray, yd: np.ndarray) -> np.ndarray:
            return self._accel(y, yd)

        y = np.stack([s.x, s.theta], axis=-1)
        yd = np.stack([s.x_dot, s.theta_dot], axis=-1)
        k1y, k1v = yd, deriv(y, yd)
        k2y, k2v = yd + 0.5 * dt * k1v, deriv(y + 0.5 * dt * k1y, yd + 0.5 * dt * k1v)
        k3y, k3v = yd + 0.5 * dt * k2v, deriv(y + 0.5 * dt * k2y, yd + 0.5 * dt * k2v)
        k4y, k4v = yd + dt * k3v, deriv(y + dt * k3y, yd + dt * k3v)

        y = y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        yd = yd + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

        if cfg.model == MODEL_PIVOT:
            y[:, 0] = 0.0
            yd[:, 0] = 0.0
        s.x, s.theta = y[:, 0], y[:, 1]
        s.x_dot, s.theta_dot = yd[:, 0], yd[:, 1]

    def _accel(self, y: np.ndarray, yd: np.ndarray) -> np.ndarray:
        """Return ``[x_ddot, theta_ddot]`` for every plant in the batch."""
        cfg = self.cfg
        theta = y[:, 1]
        theta_dot = yd[:, 1]
        x_dot = yd[:, 0]
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        m, l, M, g = cfg.m_pole, cfg.l_pole, cfg.m_cart, cfg.gravity
        u = self.state.u

        if cfg.model == MODEL_PIVOT:
            theta_ddot = (g * sin_t + u / (m * l)) / l
            return np.stack([np.zeros_like(theta_ddot), theta_ddot], axis=-1)

        if cfg.action_mode == "force":
            f = u
        else:
            f = u * (M + m * (1.0 - 0.75 * cos_t**2))
        a11, a12 = M + m, m * l * cos_t
        a21, a22 = m * l * cos_t, m * l**2
        b1 = f + m * l * sin_t * theta_dot**2
        b2 = m * g * l * sin_t
        det = a11 * a22 - a12 * a21
        det = np.where(np.abs(det) < 1e-12, np.sign(det + 1e-30) * 1e-12, det)
        x_ddot = (b1 * a22 - a12 * b2) / det
        theta_ddot = (a11 * b2 - b1 * a21) / det
        return np.stack([x_ddot, theta_ddot], axis=-1)

    # ----------------------------------------------------------------- extras
    def energies(self) -> np.ndarray:
        """``env.energy()`` for the whole batch."""
        cfg = self.cfg
        theta, theta_dot, x_dot = self.state.theta, self.state.theta_dot, self.state.x_dot
        if cfg.model == MODEL_CART:
            pole_ke = 0.5 * cfg.m_pole * (
                x_dot**2 + (cfg.l_pole * theta_dot) ** 2 + 2.0 * cfg.l_pole * x_dot * theta_dot * np.cos(theta)
            )
            kinetic = pole_ke + 0.5 * cfg.m_cart * x_dot**2
        else:
            kinetic = 0.5 * cfg.m_pole * (cfg.l_pole * theta_dot) ** 2
        return kinetic + cfg.m_pole * cfg.gravity * cfg.l_pole * (np.cos(theta) + 1.0)


def _scale(value: np.ndarray, half_range: float) -> np.ndarray:
    return np.clip(value / half_range, -1.0, 1.0)
