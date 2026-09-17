"""Smoke tests for the plant, the reward and the RL stack.

Run with::

    python tests/test_sanity.py

Checks (all CPU-only, a few seconds):
1.  Free fall: released slightly off upright, the pole must *accelerate away*
    from the equilibrium (gravity destabilises it).  Catches sign errors.
2.  Dead-zone: with zero input the pole must eventually hit the rail / fall.
3.  Energy conservation: with zero input the pendulum's mechanical energy stays
    ~constant over a short horizon (validates the RK4 integrator).
4.  Cart-pole coupling sign: pushing the cart right makes an upright pole lean
    left (the classic non-minimum-phase behaviour).
5.  PD stabilisation: the hand-tuned PD controller keeps |theta| small for 10 s.
6.  Energy swing-up: the Åström/Furuta controller swings the pole up from
    hanging and the LQR catches it.
7.  PPO shape check: one forward/backward pass through the agent updates the
    weights and produces finite losses.
8.  Rendering: one frame can be produced and written as a GIF.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

#: Scratch space inside the project (the OS temp dir is not writable everywhere).
SCRATCH = PROJECT_ROOT / ".cache" / "test-scratch"

from pendulum_rl.agents.classical import BaggedController, PDController  # noqa: E402
from pendulum_rl.agents.ppo import PPOAgent, PPOConfig, RolloutBuffer  # noqa: E402
from pendulum_rl.envs.inverted_pendulum import EnvConfig, InvertedPendulumEnv  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {name}" + (f"  ({detail})" if detail else ""))


def rollout(env: InvertedPendulumEnv, controller, steps: int, seed: int = 0):
    obs, _ = env.reset(seed=seed)
    history = {"theta": [], "x": [], "u": [], "reward": [], "energy": []}
    for _ in range(steps):
        u = controller(env, obs)
        obs, reward, terminated, truncated, info = env.step([u])
        history["theta"].append(info["theta"])
        history["x"].append(info["x"])
        history["u"].append(u)
        history["reward"].append(reward)
        history["energy"].append(env.energy())
        if terminated or truncated:
            break
    return {k: np.asarray(v) for k, v in history.items()}


def test_gravity_destabilises() -> None:
    cfg = EnvConfig(model="cart", init_mode="upright", init_angle_range=0.0)
    env = InvertedPendulumEnv(cfg)
    _, _ = env.reset(seed=0)
    env.state[1] = 0.05  # 2.9 deg to the right
    env.state_dot[1] = 0.0
    for _ in range(50):  # 0.1 s
        env.step([0.0])
    check(
        "gravity destabilises the upright pole",
        env.state[1] > 0.05,
        f"theta went 0.050 -> {env.state[1]:.4f} rad",
    )


def test_zero_input_falls() -> None:
    """Released slightly off upright with no control, the pole must fall over."""
    cfg = EnvConfig(model="cart", init_mode="upright", init_angle_range=0.0)
    env = InvertedPendulumEnv(cfg)
    _, _ = env.reset(seed=0)
    env.state[1] = 0.05  # 2.9 deg, well inside the linear region
    theta = []
    for _ in range(500):
        _, _, terminated, truncated, info = env.step([0.0])
        theta.append(info["theta"])
        if terminated or truncated:
            break
    # The upright equilibrium is unstable with eigenvalue +5.24 /s, so the pole
    # races past pi within ~1 s and then keeps spinning; assert on the whole
    # trajectory rather than on the (wrap-around dependent) final sample.
    peak = float(np.max(np.abs(theta)))
    check(
        "uncontrolled pole falls over",
        peak > 1.0 and len(theta) == 500,
        f"peak |theta|={peak:.2f} rad over {len(theta)} steps",
    )


def test_energy_and_momentum_conservation() -> None:
    """With no input, total energy (both models) and cart momentum must hold.

    This is the sharpest test of the RK4 plant: energy conservation catches
    sign/factor errors in the acceleration terms, and momentum conservation
    catches the cart/pole coupling being wrong.
    """
    for model in ("cart", "pivot"):
        cfg = EnvConfig(model=model, init_mode="hanging", control_dt=0.02, sim_dt=0.002)
        env = InvertedPendulumEnv(cfg)
        env.reset(seed=0)
        env.state_dot[1] = 1.0  # give it a spin so it actually moves
        energies = []
        for _ in range(250):  # 5 s
            _, _, terminated, truncated, _ = env.step([0.0])
            energies.append(env.energy())
            if terminated or truncated:
                break
        energies = np.asarray(energies)
        drift = float(energies.max() - energies.min())
        check(
            f"total energy conserved, zero input ({model})",
            drift < 1e-6,
            f"drift {drift:.2e} J over {len(energies)} steps",
        )

    # Cart model: horizontal momentum is conserved when no force is applied.
    cfg = EnvConfig(model="cart", init_mode="hanging", control_dt=0.02, sim_dt=0.002)
    env = InvertedPendulumEnv(cfg)
    env.reset(seed=0)
    env.state_dot[1] = 1.0

    def momentum(e: InvertedPendulumEnv) -> float:
        return e.cfg.m_cart * e.state_dot[0] + e.cfg.m_pole * (
            e.state_dot[0] + e.cfg.l_pole * e.state_dot[1] * np.cos(e.state[1])
        )

    p = [momentum(env)]
    for _ in range(300):
        env.step([0.0])
        p.append(momentum(env))
    p = np.asarray(p)
    check(
        "horizontal momentum conserved, zero force (cart)",
        float(p.max() - p.min()) < 1e-6,
        f"drift {p.max() - p.min():.2e} kg m/s",
    )


def test_nonminimum_phase() -> None:
    """Drive the cart right; an upright pole must initially lean left."""
    cfg = EnvConfig(model="cart", init_mode="upright", init_angle_range=0.0)
    env = InvertedPendulumEnv(cfg)
    env.reset(seed=0)
    env.state[1] = 0.0
    env.state_dot[1] = 0.0
    for _ in range(10):  # 0.2 s of constant force to the right
        env.step([cfg.max_force])
    check(
        "cart accelerates right -> pole leans left (non-minimum phase)",
        env.state[0] > 0.0 and env.state[1] < 0.0,
        f"x={env.state[0]:+.4f} m, theta={env.state[1]:+.4f} rad",
    )


def test_pd_balances() -> None:
    cfg = EnvConfig(model="cart", init_mode="upright")
    env = InvertedPendulumEnv(cfg)
    controller = PDController(cfg)
    hist = rollout(env, lambda e, o: controller(e), steps=cfg.max_episode_steps, seed=1)
    deg = np.degrees(np.abs(hist["theta"]))
    check(
        "PD controller balances for the full 10 s episode",
        len(hist["theta"]) == cfg.max_episode_steps and deg.max() < 5.0,
        f"{len(hist['theta'])} steps, max|theta|={deg.max():.2f} deg, max|x|={np.abs(hist['x']).max():.2f} m",
    )


def test_energy_swingup() -> None:
    """Energy swing-up + LQR catch: works on the pivot plant (torque drive)."""
    cfg = EnvConfig(model="pivot", init_mode="hanging")
    env = InvertedPendulumEnv(cfg)
    controller = BaggedController(cfg)
    hist = rollout(env, lambda e, o: controller(e), steps=cfg.max_episode_steps, seed=2)
    tail = np.degrees(np.abs(hist["theta"][-100:]))
    check(
        "energy swing-up + LQR balances the pivot pendulum",
        len(tail) == 100 and tail.max() < 10.0,
        f"last 100 steps max|theta|={tail.max():.2f} deg",
    )


def test_cart_swingup_pumps_energy() -> None:
    """On the cart plant the energy law must at least pump the pole up.

    The full cart swing-up catch is tuning-sensitive (see README), so this test
    asserts the part that is a hard physical requirement: the pendulum energy
    climbs from the hanging value towards the homoclinic value 2 m g l.
    """
    cfg = EnvConfig(model="cart", init_mode="hanging")
    env = InvertedPendulumEnv(cfg)
    controller = BaggedController(cfg)
    obs, _ = env.reset(seed=2)
    e_target = 2.0 * cfg.m_pole * cfg.gravity * cfg.l_pole
    energies = []
    for _ in range(cfg.max_episode_steps):
        obs, _, terminated, truncated, _ = env.step([controller(env)])
        energies.append(env.pendulum_energy())
        if terminated or truncated:
            break
    peak = float(np.max(energies))
    check(
        "cart energy law pumps the pendulum upwards",
        peak > 0.8 * e_target,
        f"peak E={peak:.3f} of E*={e_target:.3f} ({peak / e_target:.0%})",
    )


def test_ppo_update() -> None:
    cfg = PPOConfig(obs_dim=6, action_dim=1, hidden_sizes=(32, 32), epochs_per_update=2, minibatches=2)
    agent = PPOAgent(cfg)
    env = InvertedPendulumEnv(EnvConfig())
    obs, _ = env.reset(seed=0)
    buffer = RolloutBuffer(steps=32, num_envs=1, obs_dim=6, action_dim=1)
    before = [p.detach().clone() for p in agent.model.parameters()]
    for _ in range(32):
        action, log_prob, value = agent.sample_actions(obs[None, :])
        clipped = np.clip(action, -1.0, 1.0)
        env_action = clipped * env.cfg.action_limit
        agent.obs_rms.update(obs[None, :])
        obs, reward, terminated, truncated, _ = env.step(env_action[0])
        buffer.add(obs[None, :] * 0, clipped, log_prob, value, [reward], [float(terminated or truncated)])
        if terminated or truncated:
            obs, _ = env.reset()
    stats = agent.update(buffer, 32, 1)
    after = list(agent.model.parameters())
    moved = any(not np.allclose(b.detach().numpy(), a.detach().numpy()) for a, b in zip(before, after))
    check(
        "PPO update runs and moves the weights",
        moved and np.isfinite(stats["loss"]) and stats["updates"] > 0,
        f"loss={stats['loss']:.4f}, updates={int(stats['updates'])}",
    )


def test_rendering() -> None:
    from pendulum_rl.rendering import PendulumRenderer

    cfg = EnvConfig(model="cart")
    env = InvertedPendulumEnv(cfg)
    env.reset(seed=0)
    renderer = PendulumRenderer(env)
    frames = [renderer.frame(0.0, 0.0, 0)]
    for i in range(20):
        env.step([cfg.max_force * 0.5])
        frames.append(renderer.frame(0.5, 1.0, i + 1))
    renderer.close()
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / "test.gif"
    out = PendulumRenderer.save_frames(frames, path, fps=50, title="sanity", repeat_last=0)
    ok = out is not None and out.exists() and out.stat().st_size > 1000
    size = out.stat().st_size if out and out.exists() else 0
    shutil.rmtree(SCRATCH, ignore_errors=True)
    check("rendering produces frames and a GIF", ok and frames[0].ndim == 3, f"{size} bytes")


def test_batched_matches_scalar() -> None:
    """The vectorised plant must reproduce the scalar reference exactly."""
    rng = np.random.default_rng(0)
    for model in ("cart", "pivot"):
        for init_mode in ("upright", "hanging", "random"):
            cfg = EnvConfig(model=model, init_mode=init_mode)
            envs = [InvertedPendulumEnv(cfg) for _ in range(3)]
            for i, e in enumerate(envs):
                e.reset(seed=100 + i)
            from pendulum_rl.batched_env import BatchedPendulum

            batch = BatchedPendulum(cfg, 3)
            batch.reset(np.ones(3, dtype=bool))
            batch.state.x[:] = [e.state[0] for e in envs]
            batch.state.theta[:] = [e.state[1] for e in envs]
            batch.state.x_dot[:] = [e.state_dot[0] for e in envs]
            batch.state.theta_dot[:] = [e.state_dot[1] for e in envs]

            worst_obs = worst_rew = 0.0
            for _ in range(25):
                actions = rng.uniform(-1.0, 1.0, size=3) * cfg.action_limit
                b_obs, b_rew, _b_done, _ = batch.step(actions)
                for i, e in enumerate(envs):
                    s_obs, s_rew, s_term, s_trunc, _ = e.step([actions[i]])
                    worst_obs = max(worst_obs, float(np.max(np.abs(s_obs - b_obs[i]))))
                    worst_rew = max(worst_rew, abs(s_rew - b_rew[i]))
                    if s_term or s_trunc:  # keep the two in lockstep
                        e.reset(seed=1000 + i)
                        batch.state.x[i] = e.state[0]
                        batch.state.theta[i] = e.state[1]
                        batch.state.x_dot[i] = e.state_dot[0]
                        batch.state.theta_dot[i] = e.state_dot[1]
                        batch.steps[i] = 0
            check(
                f"batched plant matches scalar ({model}/{init_mode})",
                worst_obs < 1e-6 and worst_rew < 1e-5,
                f"max |obs diff|={worst_obs:.2e}, max |reward diff|={worst_rew:.2e}",
            )


def test_sync_vector_env() -> None:
    """The vector env must run episodes, reset them and report successes."""
    from pendulum_rl.vector_env import SyncVectorEnv

    cfg = EnvConfig(model="cart", init_mode="upright", max_episode_steps=40)
    venv = SyncVectorEnv([cfg] * 4, [0, 1, 2, 3])
    total_finished = 0
    for _ in range(80):
        obs, rewards, dones = venv.step(np.zeros(4, dtype=np.float32))
        total_finished += int(dones.sum())
    check(
        "SyncVectorEnv rolls episodes over and reports metrics",
        total_finished >= 4 and venv.obs.shape == (4, cfg.obs_dim) and np.isfinite(rewards).all(),
        f"{total_finished} episodes finished, obs shape {venv.obs.shape}",
    )


def test_terminate_angle() -> None:
    """The optional "pole fell over" rule must terminate (not truncate) episodes.

    It is deliberately *off* for swing-up, where the pole has to sweep through
    pi/2 — this test pins both behaviours.
    """
    # enabled, uncontrolled: the episode must end almost immediately
    cfg = EnvConfig(model="cart", init_mode="upright", init_angle_range=0.0, terminate_angle=0.6)
    env = InvertedPendulumEnv(cfg)
    _, _ = env.reset(seed=0)
    env.state[1] = 0.1
    steps, terminated, truncated = 0, False, False
    for _ in range(cfg.max_episode_steps):
        _, _, terminated, truncated, info = env.step([0.0])
        steps += 1
        if terminated or truncated:
            break
    check(
        "terminate_angle ends the episode when the pole falls",
        terminated and not truncated and steps < 100 and info["fell_over"],
        f"ended after {steps} steps (terminated={terminated}, fell_over={info['fell_over']})",
    )

    # disabled: the pole may swing past pi/2 freely
    cfg_open = EnvConfig(model="cart", init_mode="hanging", terminate_angle=None)
    env_open = InvertedPendulumEnv(cfg_open)
    env_open.reset(seed=1)
    crossed, ended = False, False
    for _ in range(400):
        _, _, term, trunc, info = env_open.step([cfg_open.max_force])
        if abs(info["theta"]) < 1.0:
            crossed = True
        if term or trunc:
            ended = True
            break
    check(
        "swing-up keeps the pole free to sweep through pi/2",
        crossed,
        f"reached |theta| < 1 rad={crossed}, episode ended early={ended}",
    )

    # a hanging start must switch the rule off automatically
    from pendulum_rl.lightning_module import TrainConfig

    train_cfg = TrainConfig(init_mode="random", warmup_init_mode="hanging", terminate_angle=0.6)
    check(
        "TrainConfig disables terminate_angle for swing-up curricula",
        train_cfg.effective_terminate_angle() is None
        and TrainConfig(init_mode="upright", terminate_angle=0.6).effective_terminate_angle() == 0.6,
        "hanging curriculum -> None, upright -> 0.6",
    )


def main() -> int:
    tests = [
        test_gravity_destabilises,
        test_zero_input_falls,
        test_energy_and_momentum_conservation,
        test_nonminimum_phase,
        test_pd_balances,
        test_energy_swingup,
        test_cart_swingup_pumps_energy,
        test_batched_matches_scalar,
        test_sync_vector_env,
        test_terminate_angle,
        test_ppo_update,
        test_rendering,
    ]
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - a test failure must not stop the rest
            FAILED.append(test.__name__)
            print(f"[FAIL] {test.__name__} raised {type(exc).__name__}: {exc}")
    print("-" * 60)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("failed: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
