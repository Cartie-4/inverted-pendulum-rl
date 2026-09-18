"""Imitation warm start: clone the classical swing-up controller into the PPO policy.

    python scripts/imitation_warmstart.py \
        --base outputs/checkpoints/p5_swing_fixed/best.ckpt \
        --out  outputs/checkpoints/bc_warmstart.ckpt \
        --episodes 300 --min-abs-theta-deg 60

Why
---
Four pure-RL attempts failed to discover energy pumping.  The reason is physical
rather than a tuning problem: the pumping law is *phase coherent* (``a = k (E - E*)
theta_dot cos theta``), and i.i.d. Gaussian exploration satisfies
``E[a theta_dot cos theta] = 0`` -- it injects no energy on average, it only shakes.
So the missing skill is demonstrated instead of explored.

State-selective cloning with a self-distillation anchor
------------------------------------------------------
The teacher (``BaggedController``: energy swing-up + LQR catch) is strong exactly
where the base policy is hopeless -- the pumping regime, roughly ``|theta| > 60 deg``
-- and weak near upright, where its narrow LQR catch is far worse than the base
policy's learned balance (99% at +-45 deg).  So per state the target is

    |theta| >  --min-abs-theta-deg   ->  the teacher's action
    |theta| <= --min-abs-theta-deg   ->  the base policy's own action   (anchor)

The anchor matters because the network is a *shared trunk*: training only on
large-angle samples still moves the weights, and therefore the output everywhere.
Feeding the base policy's own actions on the near-upright states keeps the balance
behaviour in the loss instead of hoping it survives.

What is not touched: the reward function, the MDP, the configuration, the critic and
``log_std`` (only the actor's mean head is trained).  The saved checkpoint keeps every
field of the base checkpoint and replaces only ``agent.model``, so
``scripts/train.py --init-from`` loads it as an ordinary warm start.

Measured outcome (2026-09-18): **negative, do not use as-is**
------------------------------------------------------------
Run: base ``p5_swing_fixed/best.ckpt``, 300 episodes, 150 epochs, lr 3e-4,
``--min-abs-theta-deg 60``; 200 head-less episodes per band, seed 900.

    band        base      after cloning      delta
    upright     100.0%    100.0%              0
    +-15 deg    100.0%    100.0%              0
    +-45 deg     99.5%     94.5%           -5.0   <- balance damaged
    +-90 deg     62.0%     56.5%           -5.5   (no terminate angle)
    +-180 deg    31.5%     33.5%           +2.0   (within noise)

In the hard 45-90 deg band the clone produced *fewer* successes (11 vs 19) and many
more "reached upright but failed to hold it" episodes (40 vs 22), i.e. it traded
near-upright stability for pumping without earning the skill back.  The shared trunk
drifts even with the self-distillation anchor (that anchor *pulls*, it does not
*lock*), and an offline clone never experiences the consequence of pumping, so the
reward signal plays no part in learning the skill.

The experiment that follows from this failure is to move the teacher inside the
training loop -- a teacher-regularised PPO auxiliary loss ``lambda *
MSE(mu(s), a_teacher(s))`` restricted to ``|theta| > 60 deg`` and annealed to zero --
or, better, to shape the reward itself with a potential that cannot change the
optimal policy (see ``pendulum_rl/envs`` energy shaping).  Keep this script for the
record and for re-running the clone once the base policy is stronger.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pendulum_rl.agents.classical import BaggedController  # noqa: E402
from pendulum_rl.envs.inverted_pendulum import InvertedPendulumEnv, wrap_angle  # noqa: E402
from pendulum_rl.lightning_module import env_config_from_checkpoint, load_policy  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True, help="checkpoint to start from")
    p.add_argument("--out", type=Path, required=True, help="checkpoint to write")
    p.add_argument("--episodes", type=int, default=300, help="teacher rollouts to collect")
    p.add_argument("--max-seconds", type=float, default=10.0, help="horizon per rollout")
    p.add_argument("--init-angle-limit", type=float, default=180.0, help="degrees, collection distribution")
    p.add_argument("--min-abs-theta-deg", type=float, default=60.0,
                   help="above this |theta| the teacher's action is the target, below it the base policy's own")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args(argv)


def collect(agent, env, teacher, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll the teacher out and record (obs, target, is_teacher_target) per step."""
    obs_list: list[np.ndarray] = []
    target_list: list[float] = []
    teacher_list: list[bool] = []
    horizon = int(round(args.max_seconds / env.cfg.control_dt))
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        for _ in range(horizon):
            force = float(teacher(env))
            teacher_action = float(np.clip(force / env.cfg.action_limit, -1.0, 1.0))
            action = agent.sample_actions(np.asarray(obs, dtype=np.float32)[None, :], deterministic=True)[0]
            policy_action = float(np.asarray(action).reshape(-1)[0])
            theta_deg = abs(float(np.degrees(wrap_angle(env.state[1]))))
            use_teacher = theta_deg > args.min_abs_theta_deg
            obs_list.append(np.asarray(obs, dtype=np.float32))
            target_list.append(teacher_action if use_teacher else policy_action)
            teacher_list.append(use_teacher)
            obs, _, terminated, truncated, _ = env.step([force])
            if terminated or truncated:
                break
    return (
        np.stack(obs_list),
        np.asarray(target_list, dtype=np.float32),
        np.asarray(teacher_list, dtype=bool),
    )


def train_actor(agent, obs: np.ndarray, targets: np.ndarray, args: argparse.Namespace) -> float:
    """Supervised fit of the actor's mean head; critic and log_std untouched."""
    device = agent.device
    obs_t = torch.as_tensor(agent.obs_rms.normalize(obs), dtype=torch.float32, device=device)
    y_t = torch.as_tensor(targets, dtype=torch.float32, device=device).unsqueeze(-1)
    optimiser = torch.optim.Adam(agent.model.actor.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    total = len(obs_t)
    final = float("nan")
    for epoch in range(args.epochs):
        order = rng.permutation(total)
        running = 0.0
        for start in range(0, total, args.batch):
            idx = order[start : start + args.batch]
            loss = torch.nn.functional.mse_loss(agent.model.actor(obs_t[idx]), y_t[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += float(loss) * len(idx)
        final = running / total
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(f"  bc epoch {epoch:3d}   mse {final:.5f}", flush=True)
    return final


def save_checkpoint(base: Path, out: Path, agent) -> None:
    checkpoint = torch.load(base, map_location="cpu", weights_only=False)
    checkpoint["agent"]["model"] = {k: v.detach().cpu() for k, v in agent.model.state_dict().items()}
    checkpoint["imitation_warmstart"] = {"base": str(base), "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"base      : {args.base}")
    agent, cfg, raw = load_policy(str(args.base), device=args.device)
    env_cfg = env_config_from_checkpoint(
        raw,
        init_mode="random",
        init_angle_limit=float(np.radians(args.init_angle_limit)),
        init_rate_limit=0.5,
        terminate_angle=None,
        max_episode_steps=int(round(args.max_seconds / 0.02)),
    )
    env = InvertedPendulumEnv(env_cfg)
    teacher = BaggedController(env_cfg)

    print(f"collecting: {args.episodes} teacher rollouts, init +-{args.init_angle_limit:.0f} deg, "
          f"|theta| > {args.min_abs_theta_deg:.0f} deg gets the teacher's action")
    t0 = time.perf_counter()
    obs, targets, is_teacher = collect(agent, env, teacher, args)
    print(f"  collected {len(obs):,} samples in {time.perf_counter() - t0:.1f} s "
          f"({is_teacher.sum():,} teacher / {(~is_teacher).sum():,} anchor)")

    before = float(np.mean(np.abs(agent.sample_actions(obs, deterministic=True)[0][:, 0] - targets)))
    final_mse = train_actor(agent, obs, targets, args)
    after = float(np.mean(np.abs(agent.sample_actions(obs, deterministic=True)[0][:, 0] - targets)))
    teacher_part = float(np.mean(np.abs(
        agent.sample_actions(obs[is_teacher], deterministic=True)[0][:, 0] - targets[is_teacher]
    )))
    print(f"  mean |policy - target| : {before:.4f} -> {after:.4f}   (teacher region alone: {teacher_part:.4f})")
    print(f"  final mse             : {final_mse:.5f}")

    save_checkpoint(args.base, args.out, agent)
    print(f"written   : {args.out}")
    print("next      : evaluate it against the anti-regression ledger (upright / +-15 / +-45) before any fine-tuning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
