#!/usr/bin/env python
"""Deploy a trained policy -- single model, or two models chosen per episode.

Two supported configurations, both starting from the same question ("how well
does this do on the start distribution I actually care about?"):

    # single model: S2 is the best all-angle policy on its own
    python scripts/policy_switch.py --checkpoint checkpoints/s2_full_best.ckpt

    # two models: S2 at the ends of the angle range, S8 in the middle, chosen
    # once per episode from |theta_0| (77.0% band mean vs 59.7% / 58.3% alone)
    python scripts/policy_switch.py \
        --checkpoint checkpoints/s2_full_best.ckpt \
        --middle outputs/checkpoints/s8_scratch34b/best.ckpt \
        --inner 72 --outer 135

    # the same thing, live, on the real plant
    python scripts/policy_switch.py --checkpoint checkpoints/s2_full_best.ckpt \
        --middle outputs/checkpoints/s8_scratch34b/best.ckpt --render

Why a threshold on the *initial* angle and not on the current state: |theta_0| is
measurable once at t=0 on a real system, and choosing once per episode means
every episode is a pure rollout of one policy -- a mid-episode switch would
break a swing-up that is already under way (the pole crosses the moderate-tilt
range on its way up from hanging, which is exactly where the two policies
disagree about what to do).

The band weights are the *distribution* weights: with `--init-mode random
--init-angle-limit 180` every angle is equally likely, so the 45-degree-wide
bands carry three times the weight of the 15-degree ones.  A plain average over
bands would silently over-weight the narrow bands.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Re-use evaluate.py rather than re-implementing the rollout: a deployment script
# that measures something subtly different from the evaluation tool is worse than
# no script at all.
_spec = importlib.util.spec_from_file_location("evaluate_mod", PROJECT_ROOT / "scripts" / "evaluate.py")
evaluate = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(evaluate)

#: default ladder: wide enough to show where each policy falls over, narrow
#: enough that a "band" is a meaningful description of the start state.
DEFAULT_BANDS = (0.0, 45.0, 60.0, 75.0, 90.0, 135.0, 180.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one policy, or two policies selected per episode by the start angle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="the policy used outside the middle interval (S2 in our runs)")
    p.add_argument("--middle", type=Path, default=None,
                   help="optional second policy for the middle interval; omit for single-model mode")
    p.add_argument("--inner", type=float, default=72.0,
                   help="|theta_0| in DEGREES at or below which --checkpoint is used")
    p.add_argument("--outer", type=float, default=135.0,
                   help="|theta_0| in DEGREES above which --checkpoint is used again")
    p.add_argument("--bands", type=float, nargs="+", default=list(DEFAULT_BANDS),
                   help="band edges in DEGREES; band i is (edges[i], edges[i+1]]")
    p.add_argument("--episodes", type=int, default=100, help="episodes per band")
    p.add_argument("--seed", type=int, default=51500, help="seed block base (bands do not share seeds)")
    p.add_argument("--x-limit", type=float, default=3.4,
                   help="rail half-length [m] for the evaluation; the policies here were trained at 3.4")
    p.add_argument("--obs-x-scale", type=float, default=2.4,
                   help="observation normalisation for x; pinned to the rail the policies were trained on")
    p.add_argument("--rate-limit", type=float, default=0.5, help="initial |theta_dot|, |x_dot| bound")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--render", action="store_true",
                   help="live window on the real plant instead of the head-less band sweep")
    p.add_argument("--out", type=Path, default=None, help="write the report as JSON")
    return p.parse_args(argv)


def load_policy_checkpoint(path: Path, device: str):
    from pendulum_rl.lightning_module import load_policy

    _, _, raw = load_policy(str(path), device=device)
    return raw


def make_env(raw: dict, lo: float, hi: float, args: argparse.Namespace):
    from pendulum_rl.envs.inverted_pendulum import InvertedPendulumEnv
    from pendulum_rl.lightning_module import env_config_from_checkpoint

    cfg = env_config_from_checkpoint(
        raw,
        init_mode="random",
        init_angle_center=float(np.radians(0.5 * (lo + hi))),
        init_angle_limit=float(np.radians(0.5 * (hi - lo))),
        init_rate_limit=float(args.rate_limit),
        terminate_angle=None,
        max_episode_steps=500,
        x_limit=float(args.x_limit),
        obs_x_scale=float(args.obs_x_scale),
    )
    return InvertedPendulumEnv(cfg)


def make_controller(checkpoint: Path, env, device: str):
    args = evaluate.parse_args([
        "--checkpoint", str(checkpoint), "--init-mode", "random",
        "--init-angle-limit", "1", "--terminate-angle", "0",
        "--no-render", "--no-video", "--no-plot",
    ])
    return evaluate.build_controller(args, env)[1]


def rollout(env, controller, seed: int) -> tuple[bool, float]:
    obs, _ = env.reset(seed=seed)
    info: dict = {}
    terminated = truncated = False
    ret = 0.0
    for _ in range(int(env.cfg.max_episode_steps or 500)):
        obs, reward, terminated, truncated, info = env.step([controller(env, obs)])
        ret += reward
        if terminated or truncated:
            break
    return bool(info.get("is_success", False)), ret


def run_live(args: argparse.Namespace) -> int:
    """Real-time window, switching per episode exactly as the sweep does."""
    from pendulum_rl.utils import ensure_dirs

    env = make_env(load_policy_checkpoint(args.checkpoint, args.device), 0.0, args.bands[-1], args)
    args.init_angle_limit = args.bands[-1]
    args.init_angle_center = 0.0
    ctrls = {"main": make_controller(args.checkpoint, env, args.device)}
    if args.middle is not None:
        ctrls["middle"] = make_controller(args.middle, env, args.device)

    ensure_dirs()
    summary = evaluate._run_live(env, _SwitchingController(ctrls, args), env.cfg, args)
    print("summary: %s" % json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2, default=str))
    return 0


class _SwitchingController:
    """Pick the arm from the episode's start angle; hold it for that episode.

    The arm is re-read from the plant when the step counter resets, i.e. exactly
    once per episode.  Re-reading it every step would switch mid-manoeuvre: a
    swing-up launched from hanging passes through the moderate-tilt interval on
    its way up, which is precisely where the two policies disagree.
    """

    def __init__(self, ctrls: dict, args: argparse.Namespace):
        self.ctrls = ctrls
        self.args = args
        self.active = ctrls["main"]
        self._last_step = -1

    def __call__(self, env, obs=None):
        step = int(env._steps)
        if step <= self._last_step:  # a reset happened: choose again for this episode
            theta0 = abs(float(np.degrees(env.state[1])))
            if "middle" in self.ctrls and self.args.inner < theta0 <= self.args.outer:
                self.active = self.ctrls["middle"]
            else:
                self.active = self.ctrls["main"]
        self._last_step = step
        return self.active(env)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.render:
        return run_live(args)

    arms = {"main": args.checkpoint}
    if args.middle is not None:
        arms["middle"] = args.middle
    raws = {k: load_policy_checkpoint(v, args.device) for k, v in arms.items()}

    edges = list(args.bands)
    bands = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    print("rail +-%.1f m, obs x-scale %.2f, %d episodes/band" % (args.x_limit, args.obs_x_scale, args.episodes))
    if args.middle is None:
        print("mode: single model (%s)\n" % args.checkpoint.name)
    else:
        print("mode: two models -- |theta0|<=%.0f or >%.0f -> %s ; in between -> %s\n"
              % (args.inner, args.outer, args.checkpoint.name, args.middle.name))

    header = "%-11s %14s" % ("band", "main")
    if args.middle is not None:
        header += " %14s %14s" % ("middle", "rule")
    print(header)
    print("-" * len(header))

    widths = {b: (b[1] - b[0]) for b in bands}
    W = sum(widths.values())
    weighted = {k: 0.0 for k in arms}
    weighted["rule"] = 0.0
    per_band: dict[str, dict[str, float]] = {}

    for i, (lo, hi) in enumerate(bands):
        seeds = [args.seed + i * args.episodes + ep for ep in range(args.episodes)]
        envs = {k: make_env(raw, lo, hi, args) for k, raw in raws.items()}
        ctrls = {k: make_controller(arms[k], envs[k], args.device) for k in raws}

        cells = {}
        for k in arms:
            wins = sum(rollout(envs[k], ctrls[k], s)[0] for s in seeds)
            cells[k] = wins / args.episodes
            weighted[k] += widths[(lo, hi)] * cells[k]

        if args.middle is None:
            cells["rule"] = cells["main"]
        else:
            wins = 0
            for s in seeds:
                envs["main"].reset(seed=s)
                theta0 = abs(float(np.degrees(envs["main"].state[1])))
                key = "middle" if args.inner < theta0 <= args.outer else "main"
                wins += int(rollout(envs[key], ctrls[key], s)[0])
            cells["rule"] = wins / args.episodes
        weighted["rule"] += widths[(lo, hi)] * cells["rule"]

        per_band["%g-%g" % (lo, hi)] = cells
        row = "%-11s %13.1f%%" % ("(%g,%g]" % (lo, hi), 100 * cells["main"])
        if args.middle is not None:
            row += " %13.1f%% %13.1f%%" % (100 * cells["middle"], 100 * cells["rule"])
        print(row)

    n = len(bands)
    print("-" * len(header))
    row = "%-11s %13.1f%%" % ("band mean", 100 * sum(c["main"] for c in per_band.values()) / n)
    if args.middle is not None:
        row += " %13.1f%% %13.1f%%" % (100 * sum(c["middle"] for c in per_band.values()) / n,
                                       100 * sum(c["rule"] for c in per_band.values()) / n)
    print(row)
    row = "%-11s %13.1f%%" % ("angle-weighted", 100 * weighted["main"] / W)
    if args.middle is not None:
        row += " %13.1f%% %13.1f%%" % (100 * weighted["middle"] / W, 100 * weighted["rule"] / W)
    print(row)
    print("\n(angle-weighted = the distribution the policy actually meets: every start angle is")
    print(" equally likely, so a 45-degree band carries three times the weight of a 15-degree one.)")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "main": str(args.checkpoint), "middle": None if args.middle is None else str(args.middle),
            "inner": args.inner, "outer": args.outer, "episodes_per_band": args.episodes,
            "x_limit": args.x_limit, "obs_x_scale": args.obs_x_scale, "per_band": per_band,
            "angle_weighted": {k: v / W for k, v in weighted.items()},
        }, indent=2), encoding="utf-8")
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
