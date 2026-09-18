"""A single (one-link) inverted pendulum, Gymnasium API compatible.

Two plant models are supported through :class:`EnvConfig`:

``model="cart"``
    Classic cart-pole.  A pendulum is attached to a cart that slides on a
    horizontal rail; the RL agent commands a horizontal force on the cart and
    the pendulum has to be swung up and then balanced in the unstable upright
    equilibrium.  This is the plant people usually mean by *inverted pendulum*.

``model="pivot"``
    Torque driven pendulum whose pivot is fixed in space (the same plant as
    ``gymnasium.make("Pendulum-v1")``).  The agent directly commands a joint
    torque.  Cheaper to learn and handy as a sanity check for the RL stack.

Generalised coordinates: ``q = [x, theta]`` with ``theta = 0`` at the upright
(unstable) equilibrium and ``theta`` growing counter clockwise, so ``theta``
also equals the pendulum's angular deviation from upright.

Dynamics (spong-style, no friction):
    cart :  (M + m) x'' + m l cos(theta) theta'' - m l sin(theta) theta'^2 = F
            m l cos(theta) x'' + m l^2 theta'' - m g l sin(theta)        = 0
    pivot:  m l^2 theta'' - m g l sin(theta) = tau

Integrated with classical RK4 at ``sim_dt`` and optionally sub-sampled to
``control_dt`` (frame skip), which is the standard trick for making control
problems learnable: the agent decides at 50 Hz while physics runs at 250 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ModelName = Literal["cart", "pivot"]
InitMode = Literal["upright", "hanging", "random"]
ShapingMode = Literal["none", "energy"]
ActionMode = Literal["force", "acceleration"]

MODEL_CART: ModelName = "cart"
MODEL_PIVOT: ModelName = "pivot"


def wrap_angle(theta: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to [-pi, pi)."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class EnvConfig:
    """Physical / task configuration of the pendulum."""

    model: ModelName = MODEL_CART

    # --- mechanics -------------------------------------------------------
    m_pole: float = 0.2          # pendulum mass [kg] (point mass, ball at tip)
    l_pole: float = 0.5          # pendulum half length = distance to COM [m]
    m_cart: float = 0.5          # cart mass [kg]
    gravity: float = 9.81        # [m/s^2]

    # --- actuation -------------------------------------------------------
    action_mode: ActionMode = "force"
    # 50 N on a 0.7 kg system leaves the controller real headroom: a stabilising
    # PD/LQR law uses a few newtons in steady state.  With 15 N the balance task
    # sits at ~97 % of saturation permanently, which makes learning very hard
    # (measured, see README "设计取舍").
    max_force: float = 50.0      # |F| <= max_force  [N]
    max_torque: float = 2.5      # |tau| <= max_torque [N*m] (pivot model)

    # --- integration / timing -------------------------------------------
    sim_dt: float = 0.002        # physics step [s]
    control_dt: float = 0.02     # agent step [s] -> 50 Hz decisions
    #: Hard horizon in agent steps.  ``None`` means "no clock": the episode then
    #: only ends when the *plant* fails (pole fell over, out of rail, divergence),
    #: which is what evaluation wants when the question is "how long can it hold
    #: the pole up?".  Training keeps a finite horizon so that episodes terminate
    #: even while the policy is still hopeless.
    max_episode_steps: int | None = 500  # 10 s at 50 Hz

    # --- task ------------------------------------------------------------
    init_mode: InitMode = "upright"
    init_angle_range: float = 0.05   # used by init_mode="upright" [rad]
    init_pos_range: float = 0.05     # used by init_mode="upright" [m]
    x_limit: float = 2.4             # rail half length [m]
    #: Curriculum on the *initial state distribution*.  For ``init_mode="random"``
    #: the pole angle is drawn uniformly from ``[-init_angle_limit, +init_angle_limit]``
    #: (``None`` = the full circle, ``[-pi, pi]``) and the rates from
    #: ``[-init_rate_limit, +init_rate_limit]`` (``None`` = the old fixed ranges).
    #: Widening these across training phases is the standard way to grow the task
    #: difficulty; it is exactly how Gymnasium's Pendulum samples its start state.
    init_angle_limit: float | None = None
    init_rate_limit: float | None = None
    #: Optional offset for the sampled angle, so a phase can start the pole near
    #: hanging (``init_angle_center=pi``) instead of near upright.
    init_angle_center: float = 0.0

    # --- reward shaping (potential-based, off by default) ----------------
    #: ``"none"`` keeps the historical reward bit-for-bit.  ``"energy"`` adds
    #: ``F(s, s') = gamma * Phi(s') - Phi(s)`` with
    #: ``Phi(s) = -shape_coef * (E_pend(s) - E*) ** 2``.
    #:
    #: This is *potential-based reward shaping* in the sense of Ng, Harada &
    #: Russell (1999).  Summed over an episode the term is
    #:
    #:     sum_t F_t = gamma * Phi(s_T) - Phi(s_0) - (1 - gamma) * sum_t Phi(s_t)
    #:
    #: The last piece is the only one that does not cancel, and it is
    #: *state-only*: at a fixed state it is the same number for every action, so
    #: ``argmax_a Q(s, a)`` is unchanged -- the shaping cannot make "fall over and
    #: pump" or "spin forever" a better plan than balancing.  (An event bonus for
    #: flipping the pole up has no such argument, and is farmable by falling down
    #: and flipping again.)  What the shaping buys is a dense gradient towards the
    #: pumping behaviour, which i.i.d. Gaussian exploration cannot find on its own
    #: (it satisfies ``E[a theta_dot cos theta] = 0``, so it shakes the pole
    #: instead of pumping it).
    #:
    #: ``E_pend`` is the pendulum's energy in the cart frame, measured from the
    #: hanging-at-rest state: it rises from 0 (hanging) to ``E* = 2 m g l``
    #: (upright at rest), see ``pendulum_energy``.  Two properties matter for a
    #: balance-warm-started curriculum: ``Phi`` peaks at the upright equilibrium
    #: (``gap = 0``), so ``Phi`` *and its gradient* vanish there and the shaping is
    #: silent exactly where balance lives; and pumping *past* ``E*`` turns the
    #: shaping negative, which damps the "spin the pole round and round" failure
    #: mode instead of rewarding it.
    shaping: ShapingMode = "none"
    shape_coef: float = 1.0
    #: Discount used by the shaping term only.  Keep it equal to the PPO gamma:
    #: the invariance guarantee is stated for the MDP actually being solved.
    shape_gamma: float = 0.99

    # --- reward & termination -------------------------------------------
    # r = A(theta) - w_theta*theta^2 - w_x*x^2 - w_v*x_dot^2
    #              - w_omega*theta_dot^2 - w_u*util^2
    #
    # A(theta) is the "keep it upright" term.  It is deliberately offset so that
    # a *hanging* pendulum scores exactly 0 (see `a_theta_offset`), otherwise
    # the gravity term rewards merely not falling over: a cart that lets the pole
    # swing or spin forever collects about the same average reward as one that
    # balances it, and the gradient picks the easier behaviour.  Offsetting makes
    # every hanging/horizontal state worth nothing and only the upright region
    # positive.  `a_theta_power=2` sharpens that peak further.
    a_theta_power: int = 2
    w_theta: float = 3.0
    w_x: float = 0.1
    w_v: float = 0.01
    w_omega: float = 0.1
    w_u: float = 0.001
    terminate_on_limit: bool = False   # True -> terminal state, False -> truncate
    #: If set, an episode *ends* (terminated, not truncated) as soon as
    #: ``|theta| > terminate_angle`` — the "pole fell over" rule.  It is off by
    #: default because it is fundamentally incompatible with swing-up: going from
    #: hanging (theta = pi) to upright (theta = 0) *must* sweep through
    #: ``|theta| = pi/2``, so any angle threshold ends a swing-up episode on its
    #: very first steps and the behaviour can never be learned.  Enable it for
    #: balance-style tasks (``upright`` / ``random`` starts) where falling over
    #: really is a failure, and leave it off whenever ``hanging`` is involved.
    terminate_angle: float | None = None
    # "solved" bookkeeping: |theta| < success_angle for success_steps in a row,
    # while the cart stays inside success_x_range.
    success_angle: float = 0.12        # ~7 deg
    success_steps: int = 100
    success_x_range: float = 1.5

    def __post_init__(self) -> None:
        if self.model not in (MODEL_CART, MODEL_PIVOT):
            raise ValueError(f"unknown model: {self.model!r}")
        if self.sim_dt <= 0 or self.control_dt <= 0:
            raise ValueError("sim_dt / control_dt must be positive")
        ratio = self.control_dt / self.sim_dt
        self.frame_skip = max(1, int(round(ratio)))
        if not np.isclose(ratio, self.frame_skip, atol=1e-6):
            raise ValueError(
                f"control_dt ({self.control_dt}) must be an integer multiple of sim_dt ({self.sim_dt})"
            )
        # Offset such that A(theta) == 0 for a hanging pole (theta = +-pi) while
        # A(0) == 1 when upright.  With cos^p: A(0) = 1 + offset and
        # A(pi) = (-1)^p + offset, so offset = 1 for odd p (A(pi) = 0) and
        # offset = 0 for even p (A(pi) = 1 - 1 = 0).
        self.a_theta_offset = float(self.a_theta_power % 2)
        self.shape_energy_target = 2.0 * self.m_pole * self.gravity * self.l_pole
        if self.shaping not in ("none", "energy"):
            raise ValueError(f"unknown shaping mode: {self.shaping!r}")
        if self.shape_coef < 0:
            raise ValueError("shape_coef must be non-negative")

    #: target pendulum energy of the shaping potential, ``E* = 2 m g l`` [J]
    #: (0 = hanging at rest, ``E*`` = upright at rest).
    shape_energy_target: float = field(default=1.0, init=False)

    #: normalisation constant of the upright reward term (set in __post_init__)
    a_theta_offset: float = field(default=1.0, init=False)

    #: number of physics sub-steps per agent action (filled in __post_init__)
    frame_skip: int = field(default=5, init=False)

    @property
    def action_limit(self) -> float:
        return self.max_force if self.model == MODEL_CART else self.max_torque

    @property
    def obs_dim(self) -> int:
        # [x, x_dot, cos(theta), sin(theta), theta_dot, theta]
        return 5 if self.model == MODEL_PIVOT else 6

    @property
    def action_dim(self) -> int:
        return 1


class InvertedPendulumEnv(gym.Env):
    """Single inverted pendulum with continuous torque/force control."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(self, cfg: EnvConfig | None = None, render_mode: str | None = None, **kwargs: Any):
        if cfg is None:
            cfg = EnvConfig(**kwargs)
        elif kwargs:
            raise ValueError("pass either cfg or keyword overrides, not both")
        self.cfg = cfg
        self.render_mode = render_mode

        limit = cfg.action_limit
        self.action_space = spaces.Box(low=-limit, high=limit, shape=(1,), dtype=np.float32)
        # Observations are already scaled to O(1); the agent additionally keeps
        # running statistics, see `pendulum_rl/agents/ppo.py`.
        high = np.array([np.inf] * cfg.obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self._rng = np.random.default_rng()
        self.state = np.zeros(2)      # [x, theta]  (x == 0 for the pivot model)
        self.state_dot = np.zeros(2)  # [x_dot, theta_dot]
        self._steps = 0
        self._elapsed = 0.0
        self._success_streak = 0
        self._max_success_streak = 0
        self._shape_prev = 0.0
        self._viewer = None

    # ------------------------------------------------------------------ core
    @property
    def dt(self) -> float:
        return self.cfg.control_dt

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # NOTE: only re-create the generator when a seed is actually given.
        # Vector rollouts call `reset()` mid-training; reseeding there would
        # replay the very same initial state in every episode.
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        cfg = self.cfg
        mode = cfg.init_mode
        if options and "init_mode" in options:
            mode = options["init_mode"]

        if mode == "upright":
            x = self._rng.uniform(-cfg.init_pos_range, cfg.init_pos_range)
            theta = self._rng.uniform(-cfg.init_angle_range, cfg.init_angle_range)
            x_dot = theta_dot = 0.0
        elif mode == "hanging":
            x = self._rng.uniform(-0.2, 0.2)
            theta = np.pi + self._rng.uniform(-0.2, 0.2)
            x_dot = 0.0
            theta_dot = self._rng.uniform(-0.2, 0.2)
        elif mode == "random":
            if cfg.init_angle_limit is not None:
                # curriculum form: a bounded window around init_angle_center
                theta = cfg.init_angle_center + self._rng.uniform(
                    -cfg.init_angle_limit, cfg.init_angle_limit
                )
                rate = cfg.init_rate_limit if cfg.init_rate_limit is not None else 0.5
                x = self._rng.uniform(-1.0, 1.0)
                x_dot = self._rng.uniform(-rate, rate)
                theta_dot = self._rng.uniform(-rate, rate)
            else:
                x = self._rng.uniform(-1.0, 1.0)
                theta = self._rng.uniform(-np.pi, np.pi)
                x_dot = self._rng.uniform(-0.5, 0.5)
                theta_dot = self._rng.uniform(-0.5, 0.5)
        else:  # pragma: no cover - guarded by dataclass typing
            raise ValueError(f"unknown init mode: {mode!r}")

        if cfg.model == MODEL_PIVOT:
            x = x_dot = 0.0
        self.state = np.array([x, theta], dtype=np.float64)
        self.state_dot = np.array([x_dot, theta_dot], dtype=np.float64)
        self._steps = 0
        self._elapsed = 0.0
        self._success_streak = 0
        self._max_success_streak = 0
        self._shape_prev = self._shape_potential()
        return self._obs(), self._info()

    def step(self, action):
        cfg = self.cfg
        u = float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -cfg.action_limit, cfg.action_limit))

        for _ in range(cfg.frame_skip):
            self._integrate(u)

        self._steps += 1
        self._elapsed += cfg.control_dt

        theta = float(wrap_angle(self.state[1]))
        theta_dot = float(self.state_dot[1])
        x = float(self.state[0])
        x_dot = float(self.state_dot[0])

        # --- reward (per control step, angle wrapped to [-pi, pi)) --------
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
        reward = float(reward)

        # --- failure / truncation ----------------------------------------
        out_of_rail = cfg.model == MODEL_CART and abs(x) > cfg.x_limit
        diverged = (not np.isfinite(self.state).all()) or (not np.isfinite(self.state_dot).all()) or (
            abs(theta_dot) > 1e3
        )
        # "the pole fell over": a genuine failure, so the episode is *terminated*
        # (value bootstrapping stops) rather than truncated.
        fell_over = cfg.terminate_angle is not None and abs(theta) > cfg.terminate_angle
        failed = out_of_rail or diverged or fell_over
        terminated = bool(failed and (fell_over or cfg.terminate_on_limit))
        truncated = bool(failed) and not terminated
        if cfg.max_episode_steps is not None and self._steps >= cfg.max_episode_steps:
            truncated = True
        if truncated or terminated:
            reward -= 0.0  # no extra penalty; the drop in reward is signal enough

        # --- potential-based shaping (no-op unless cfg.shaping == "energy") ----
        # F(s, s') = gamma * Phi(s') - Phi(s).  A terminal state has no successor,
        # so the standard convention Phi(s_terminal) = 0 applies to *that* term
        # only; Phi(s) still enters as the state we actually left.
        if cfg.shaping == "energy":
            shape_next = 0.0 if terminated else self._shape_potential()
            reward += cfg.shape_gamma * shape_next - self._shape_prev
            self._shape_prev = shape_next

        # --- success bookkeeping -----------------------------------------
        balanced = (
            abs(theta) < cfg.success_angle
            and abs(x) < cfg.success_x_range
            and abs(theta_dot) < 1.5
        )
        self._success_streak = self._success_streak + 1 if balanced else 0
        self._max_success_streak = max(self._max_success_streak, self._success_streak)

        info = self._info()
        info["out_of_rail"] = bool(out_of_rail)
        info["diverged"] = bool(diverged)
        info["fell_over"] = bool(fell_over)
        if terminated or truncated:
            info["is_success"] = bool(self._max_success_streak >= cfg.success_steps)
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------- dynamics
    def _integrate(self, u: float) -> None:
        """One RK4 step of the (stiff, nonlinear) plant."""
        cfg = self.cfg
        dt = cfg.sim_dt
        s = self.state
        sd = self.state_dot

        def deriv(state: np.ndarray, state_dot: np.ndarray) -> np.ndarray:
            return self._accel(state, state_dot, u)

        k1x, k1v = sd, deriv(s, sd)
        k2x, k2v = sd + 0.5 * dt * k1v, deriv(s + 0.5 * dt * k1x, sd + 0.5 * dt * k1v)
        k3x, k3v = sd + 0.5 * dt * k2v, deriv(s + 0.5 * dt * k2x, sd + 0.5 * dt * k2v)
        k4x, k4v = sd + dt * k3v, deriv(s + dt * k3x, sd + dt * k3v)

        self.state = s + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        self.state_dot = sd + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

        if cfg.model == MODEL_PIVOT:
            self.state[0] = 0.0
            self.state_dot[0] = 0.0

    def _accel(self, state: np.ndarray, state_dot: np.ndarray, u: float) -> np.ndarray:
        """Return [x_ddot, theta_ddot] for the current state and input."""
        cfg = self.cfg
        theta = state[1]
        theta_dot = state_dot[1]
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        m, l, M, g = cfg.m_pole, cfg.l_pole, cfg.m_cart, cfg.gravity

        if cfg.model == MODEL_PIVOT:
            theta_ddot = (g * sin_t + u / (m * l)) / l
            return np.array([0.0, theta_ddot])

        # Solve the 2x2 system for [x_ddot, theta_ddot].
        if cfg.action_mode == "force":
            f = u
        else:
            # Input is a desired cart acceleration; convert to the force that
            # the pendulum reaction requires.
            f = u * (M + m * (1.0 - 0.75 * cos_t**2))
        a11, a12 = M + m, m * l * cos_t
        a21, a22 = m * l * cos_t, m * l**2
        b1 = f + m * l * sin_t * theta_dot**2
        b2 = m * g * l * sin_t
        det = a11 * a22 - a12 * a21
        if abs(det) < 1e-12:  # only reachable with cos(theta) == +-1 and m >> M
            det = np.sign(det) * 1e-12
        x_ddot = (b1 * a22 - a12 * b2) / det
        theta_ddot = (a11 * b2 - b1 * a21) / det
        return np.array([x_ddot, theta_ddot])

    # ------------------------------------------------------------------ obs
    def _obs(self) -> np.ndarray:
        cfg = self.cfg
        theta = float(wrap_angle(self.state[1]))
        theta_dot = float(self.state_dot[1])
        x = float(self.state[0])
        x_dot = float(self.state_dot[0])
        if cfg.model == MODEL_PIVOT:
            obs = [np.cos(theta), np.sin(theta), _scale(theta_dot, 8.0), _scale(theta, np.pi)]
        else:
            obs = [
                _scale(x, cfg.x_limit),
                _scale(x_dot, 10.0),
                np.cos(theta),
                np.sin(theta),
                _scale(theta_dot, 8.0),
                _scale(theta, np.pi),
            ]
        return np.asarray(obs, dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        theta = float(wrap_angle(self.state[1]))
        return {
            "x": float(self.state[0]),
            "theta": theta,
            "x_dot": float(self.state_dot[0]),
            "theta_dot": float(self.state_dot[1]),
            "balanced_streak": self._success_streak,
        }

    # --------------------------------------------------------------- extras
    def energy(self) -> float:
        """Total mechanical energy of the *whole* system, with the hanging
        state as the zero of potential energy (0 = upright and at rest).

        For the cart model the cart's own kinetic energy is included because
        the cart is a real degree of freedom: with no external force its
        horizontal momentum makes the pendulum's energy oscillate as the cart
        recoils, so the pendulum-only energy is *not* the conserved quantity.
        """
        cfg = self.cfg
        theta = self.state[1]
        theta_dot = self.state_dot[1]
        if cfg.model == MODEL_CART:
            x_dot = self.state_dot[0]
            # |v_pole|^2 = x_dot^2 + l^2 theta_dot^2 + 2 l x_dot theta_dot cos(theta)
            pole_ke = 0.5 * cfg.m_pole * (
                x_dot**2 + (cfg.l_pole * theta_dot) ** 2 + 2.0 * cfg.l_pole * x_dot * theta_dot * np.cos(theta)
            )
            cart_ke = 0.5 * cfg.m_cart * x_dot**2
            kinetic = pole_ke + cart_ke
        else:
            kinetic = 0.5 * cfg.m_pole * (cfg.l_pole * theta_dot) ** 2
        potential = cfg.m_pole * cfg.gravity * cfg.l_pole * (np.cos(theta) + 1.0)
        return float(kinetic + potential)

    def pendulum_energy(self) -> float:
        """Energy of the pendulum alone, seen from the cart frame.

        This is the quantity the classical swing-up law pumps, measured from the
        upright-at-rest state: ``m g l (cos(theta) + 1) + 0.5 m (l theta_dot)^2``
        is ``E* = 2 m g l`` when the pole is upright and at rest and 0 when it
        hangs at rest.  The classical energy controller drives the *difference*
        ``E_pend - E*`` (and hence the shape of the ``energy`` shaping potential)
        to zero; it is the quantity the cart frame can actually pump.
        """
        return float(self._pendulum_energy_value())

    def _pendulum_energy_value(self):
        """``pendulum_energy`` without the float() cast, so subclasses (and the
        batched twin in ``pendulum_rl/batched_env.py``) can reuse the formula."""
        cfg = self.cfg
        theta = self.state[1]
        theta_dot = self.state_dot[1]
        kinetic = 0.5 * cfg.m_pole * (cfg.l_pole * theta_dot) ** 2
        potential = cfg.m_pole * cfg.gravity * cfg.l_pole * (np.cos(theta) + 1.0)
        return kinetic + potential

    def _shape_potential(self) -> float:
        """``Phi(s) = -shape_coef * (E_pend(s) - E*) ** 2`` (0 at upright rest)."""
        gap = self._pendulum_energy_value() - self.cfg.shape_energy_target
        return -self.cfg.shape_coef * gap**2

    def render(self):  # pragma: no cover - exercised only in demos
        from .rendering import render_frame

        return render_frame(self, mode=self.render_mode or "rgb_array")

    def close(self) -> None:
        if self._viewer is not None:  # pragma: no cover
            self._viewer.close()
            self._viewer = None


def _scale(value: float, half_range: float) -> float:
    """Scale to roughly [-1, 1] and clip to keep observations bounded."""
    return float(np.clip(value / half_range, -1.0, 1.0))


def make_env(cfg: EnvConfig | None = None, render_mode: str | None = None, **kwargs: Any) -> InvertedPendulumEnv:
    """Factory that keeps environment construction in one place."""
    return InvertedPendulumEnv(cfg=cfg, render_mode=render_mode, **kwargs)


def register_envs() -> None:
    """Register the envs with Gymnasium so `gymnasium.make` also works."""
    for env_id, model in (("InvertedPendulumCart-v0", MODEL_CART), ("InvertedPendulumPivot-v0", MODEL_PIVOT)):
        if env_id in gym.registry:
            continue
        gym.register(
            id=env_id,
            entry_point="pendulum_rl.envs.inverted_pendulum:InvertedPendulumEnv",
            kwargs={"cfg": EnvConfig(model=model)},
            max_episode_steps=None,
        )
