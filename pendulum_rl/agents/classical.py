"""Classical controllers, used as a physics sanity check and RL baseline.

The swing-up controller is the textbook energy-shaping law by Åström & Furuta
(2000): pump energy into the pendulum until it reaches the homoclinic orbit,
then hand over to a stabilising state-feedback controller inside a capture
region.  Seeing this work proves the simulated plant and its sign conventions
are right *before* any RL training is attempted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..envs.inverted_pendulum import EnvConfig, wrap_angle


class LQRController:
    """Linear-quadratic regulator via the continuous-time Riccati recursion.

    For the cart-pole we linearise around the upright equilibrium, which gives
    the classic double integrator / inverted pendulum A, B pair (note the sign:
    gravity destabilises the angle).
    """

    def __init__(self, cfg: EnvConfig, q_diag: tuple[float, ...] | None = None, r: float = 100.0):
        self.cfg = cfg
        self.A, self.B = self._linearise()
        n = self.A.shape[0]
        # Defaults below were found by direct search over the simulated plant
        # (see README): they keep the pole within ~2.6 deg while using only
        # ~0.05 N on a 50 N actuator.  Aggressive weights (e.g. q_theta=100,
        # r=1) also "work" but sit pinned at the force limit, which is a poor
        # reference for comparing against a learned policy.
        q_diag = q_diag or ((10.0, 1.0, 100.0, 5.0) if cfg.model == "cart" else (100.0, 10.0))
        self.Q = np.diag(np.asarray(q_diag, dtype=np.float64))
        self.R = np.array([[float(r)]])
        self.K = self._solve_care()

    def _linearise(self) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        m, l, M, g = cfg.m_pole, cfg.l_pole, cfg.m_cart, cfg.gravity
        if cfg.model == "pivot":
            A = np.array([[0.0, 1.0], [g / l, 0.0]])
            B = np.array([[0.0], [1.0 / (m * l**2)]])
            return A, B
        det = (M + m) * m * l**2 - (m * l) ** 2  # = M m l^2
        # Linearised about (x=0, theta=0, x_dot=0, theta_dot=0), state order
        # [x, x_dot, theta, theta_dot]:
        #   x''     = -(m^2 l^2 g / det) theta + ((M+m) l^2 / det) F
        #   theta'' =  (M+m) m g l / det theta - (m l / det) F
        A = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -m**2 * l**2 * g / det, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, (M + m) * m * g * l / det, 0.0],
            ]
        )
        B = np.array([[0.0], [(M + m) * l**2 / det], [0.0], [-m * l / det]])
        return A, B

    def _solve_care(self) -> np.ndarray:
        """Solve the continuous-time algebraic Riccati equation (CARE).

        Uses the Hamiltonian eigenvector method, which needs nothing but
        ``numpy.linalg.eig`` (no SciPy) and, unlike iterating the Riccati
        difference equation, cannot diverge for a stabilisable plant::

            H = [[A, -B R^-1 B^T], [-Q, -A^T]]

        The stable invariant subspace of ``H`` supplies
        ``P = U21 @ inv(U11)``, and ``K = R^-1 B^T P``.

        ``A`` is rescaled by its own norm first because the cart-pole has
        entries of very different magnitudes (``g/l`` vs. unit coupling),
        which would otherwise wreck the eigenvector conditioning.
        """
        A, B, Q, R = self.A, self.B, self.Q, self.R
        scale = float(np.linalg.norm(A, ord="fro")) or 1.0
        A_s = A / scale
        # P scales with A, so scaling A by s scales P by s -> X_s = P / s.
        H = np.block(
            [
                [A_s, -B @ np.linalg.solve(R, B.T)],
                [-Q, -A_s.T],
            ]
        )
        eigvals, eigvecs = np.linalg.eig(H)
        stable = eigvals.real < 0.0
        if stable.sum() != A.shape[0]:  # pragma: no cover - guard for exotic configs
            order = np.argsort(eigvals.real)
            stable = np.zeros_like(eigvals, dtype=bool)
            stable[order[: A.shape[0]]] = True
        U = eigvecs[:, stable]
        U11, U21 = U[: A.shape[0], :], U[A.shape[0] :, :]
        X_scaled = np.real(U21 @ np.linalg.pinv(U11))
        X_scaled = 0.5 * (X_scaled + X_scaled.T)  # symmetrise away round-off
        P = X_scaled * scale
        return np.linalg.solve(R, B.T @ P)

    def control(self, state: np.ndarray) -> float:
        """Feedback law ``u = -K x`` (unclipped, physical units)."""
        x = np.asarray(state, dtype=np.float64).reshape(-1)
        return float(np.asarray(self.K @ x).reshape(-1)[0]) * -1.0

    def __call__(self, obs_state: np.ndarray) -> float:
        """`obs_state` is [x, x_dot, theta, theta_dot] in physical units (clipped)."""
        return float(np.clip(self.control(obs_state), -self.cfg.action_limit, self.cfg.action_limit))


@dataclass
class BaggedController:
    """Energy-based swing-up + LQR stabilisation with automatic hand-over.

    Energy control law.  Two identities were measured numerically (constant
    command, no feedback, so `dE` over one step is unambiguous):

    *   pivot model: ``dE/dt = +tau theta_dot`` hence ``tau = -k (E - E*) theta_dot``
        gives ``dE/dt = -k (E - E*)^2 theta_dot^2 <= 0`` for the energy *error*:
        the energy can only move towards the target.
    *   cart model: with ``a`` the cart acceleration,
        ``dE_pend/dt = -m l a theta_dot cos(theta)`` hence
        ``a = +k (E - E*) theta_dot cos(theta)``.

    In both cases the command opposes ``(E - E*) * theta_dot``.  On the cart
    model the acceleration command still has to be converted to a force via
    ``m_eff(theta) = M + m (1 - 3/4 cos^2 theta)``.
    """

    cfg: EnvConfig
    k_energy: float = 1.0
    cart_gain: float = 0.3          # keeps the rail from being exhausted
    # Hand-over region.  It has to be *wide*: the energy law arrives at the
    # target energy with the pole already spinning through the top at several
    # rad/s, so a narrow window is almost never entered and the pole simply
    # keeps coasting.  The LQR also acts as the brake that kills that rate.
    capture_angle: float = 1.5      # rad, |theta| below which LQR takes over
    capture_rate: float = 10.0      # rad/s
    lqr_q_diag: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        self.lqr = LQRController(self.cfg, q_diag=self._q())

    def _q(self) -> tuple[float, ...]:
        if self.lqr_q_diag is not None:
            return self.lqr_q_diag
        # [x, x_dot, theta, theta_dot]: prefer pole angle accuracy, but weight
        # cart position enough to bring it back to the middle of the rail.
        return (1.0, 1.0, 100.0, 10.0) if self.cfg.model == "cart" else (100.0, 10.0)

    def _cart_effective_mass(self, theta: float) -> float:
        """``F -> a`` conversion at the current pole angle."""
        cfg = self.cfg
        return cfg.m_cart + cfg.m_pole * (1.0 - 0.75 * np.cos(theta) ** 2)

    # ------------------------------------------------------------------ control
    def __call__(self, env) -> float:
        cfg = self.cfg
        theta = float(wrap_angle(env.state[1]))
        theta_dot = float(env.state_dot[1])
        x = float(env.state[0])
        theta_ddot_sign = np.cos(theta)

        e_target = 2.0 * cfg.m_pole * cfg.gravity * cfg.l_pole
        e_error = env.pendulum_energy() - e_target

        # --- acceleration command that pumps energy (or damps it) ----------
        # dE_pend/dt = -m l a theta_dot cos(theta)  =>  a = +k e theta_dot cos(theta)
        a = self.k_energy * e_error * theta_dot * theta_ddot_sign
        if cfg.model == "pivot":
            # dE/dt = tau * theta_dot  =>  tau = -k e theta_dot
            u = -self.k_energy * e_error * theta_dot
            if abs(theta) < self.capture_angle and abs(theta_dot) < self.capture_rate:
                u = self.lqr(np.array([theta, theta_dot]))
            return float(np.clip(u, -cfg.action_limit, cfg.action_limit))

        a += -self.cart_gain * x - self.cart_gain * float(env.state_dot[0])
        u = a * self._cart_effective_mass(theta)  # F = m_eff * a

        # --- hand over to the LQR when the pole is inside the capture region
        balanced = (
            abs(theta) < self.capture_angle
            and abs(theta_dot) < self.capture_rate
            and abs(x) < 0.8 * cfg.x_limit
        )
        if balanced:
            u = self.lqr(np.array([x, float(env.state_dot[0]), theta, theta_dot]))
        return float(np.clip(u, -cfg.action_limit, cfg.action_limit))


#: Default PD gains per model, plus the cart position/velocity terms used on the
#: balance task (verified to hold the pole for a full 10 s episode over 20 seeds).
DEFAULT_PD_GAINS = {
    "cart": (200.0, 25.0),
    "pivot": (60.0, 8.0),
}
DEFAULT_CART_POSITION_GAINS = (0.5, 0.6)


class PDController:
    """Simple PD on the pendulum angle with optional cart-position feedback."""

    def __init__(
        self,
        cfg: EnvConfig,
        kp: float | None = None,
        kd: float | None = None,
        kx: float | None = None,
        kv: float | None = None,
    ):
        default_kp, default_kd = DEFAULT_PD_GAINS[cfg.model]
        default_kx, default_kv = DEFAULT_CART_POSITION_GAINS if cfg.model == "cart" else (0.0, 0.0)
        self.cfg = cfg
        self.kp = default_kp if kp is None else kp
        self.kd = default_kd if kd is None else kd
        self.kx = default_kx if kx is None else kx
        self.kv = default_kv if kv is None else kv

    def __call__(self, env) -> float:
        state = self.control_state(env)
        return float(np.clip(float(self.K @ state), -self.cfg.action_limit, self.cfg.action_limit))

    def control_state(self, env) -> np.ndarray:
        theta = float(wrap_angle(env.state[1]))
        theta_dot = float(env.state_dot[1])
        if self.cfg.model == "cart":
            return np.array([float(env.state[0]), float(env.state_dot[0]), theta, theta_dot])
        return np.array([theta, theta_dot])

    @property
    def K(self) -> np.ndarray:
        """Feedback row vector, shaped like the LQR gain for easy comparison."""
        if self.cfg.model == "cart":
            return np.array([-self.kx, -self.kv, self.kp, self.kd])
        return np.array([self.kp, self.kd])
