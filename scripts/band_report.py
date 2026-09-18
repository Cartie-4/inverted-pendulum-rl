#!/usr/bin/env python
"""Per-band capability ladder: success rate as a function of the *starting* angle.

Why this exists
---------------
A single "success rate on +-90 deg starts" number hides the thing a curriculum
needs to know.  The useful question is *where the cliff is*: at which starting
tilt does the policy stop being able to recover?  This script evaluates a policy
on a ladder of narrow bands of the initial angle and prints the ladder, so a new
stage can be compared with the previous one band by band instead of by one
aggregate that both stages can score the same on.

The bands are half-open, ``(lo, hi]``, and are carved out of ``(0, 180]``.  The
default ladder follows the curriculum's own stage limits (15 / 45 / 90 / 135 /
180 deg) with extra resolution around 45-60 deg, which is where the balance-only
policies were measured to fall off (P4: 40% at 45-50, 11% at 50-55, 0/77 above
55 deg).

Comparing two policies honestly
-------------------------------
``--base`` evaluates a second checkpoint on the *same episode seeds*, and the
report includes an exact McNemar test on the discordant pairs plus a
Newcombe/Wilson interval for the difference.  Two runs at 200 episodes per band
can differ by 3-4 points from seed noise alone, so an unpaired comparison is not
evidence of anything; the paired test is.  Seed blocks are distinct per band
(``--seed`` is the block base) so the same band is comparable across policies
while different bands do not share episodes.

Examples
--------
    # one policy, the default ladder
    python scripts/band_report.py --checkpoint outputs/checkpoints/p5_swing_fixed/best.ckpt

    # two policies, paired, larger sample, saved report
    python scripts/band_report.py \
        --checkpoint outputs/checkpoints/p5_shaped/best.ckpt \
        --base       outputs/checkpoints/p5_swing_fixed/best.ckpt \
        --episodes 200 --out outputs/logs/band/p5_shaped.json

    # swing-up bands only, no termination angle (the pole must sweep through pi/2)
    python scripts/band_report.py --checkpoint CKPT --bands 0 45 90 135 180 --terminate 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Evaluation is re-used rather than re-implemented: a band report that measures
# something subtly different from `evaluate.py` would be worse than no report.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("evaluate_mod", PROJECT_ROOT / "scripts" / "evaluate.py")
evaluate = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(evaluate)

#: default ladder: the curriculum's stage limits, with 60 deg inserted because the
#: balance-only policies were measured to die between 55 and 60 deg.
DEFAULT_BANDS = (0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Success rate per initial-angle band (the capability ladder).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True, help="policy to measure")
    p.add_argument("--base", type=Path, default=None,
                   help="optional second policy, evaluated paired on the same seeds")
    p.add_argument("--bands", type=float, nargs="+", default=list(DEFAULT_BANDS),
                   help="band edges in DEGREES; band i is (edges[i], edges[i+1]]")
    p.add_argument("--episodes", type=int, default=200, help="episodes per band per policy")
    p.add_argument("--seed", type=int, default=900,
                   help="seed-block base; band k uses seeds [seed + k*episodes, ...)")
    p.add_argument("--terminate", type=float, default=0.0,
                   help="terminate angle in DEGREES; 0 = off, which is required once a "
                        "band reaches past pi/2 (see README 3.1)")
    p.add_argument("--rate-limit", type=float, default=0.5, help="initial |theta_dot|, |x_dot| bound")
    p.add_argument("--shaping", choices=["none", "energy"], default=None,
                   help="override the reward shaping (default: whatever the checkpoint recorded)")
    p.add_argument("--run-eval", action="store_true",
                   help="call scripts/evaluate.py per band and store its JSON instead of "
                        "reading pre-computed per-band JSON via --from-dir")
    p.add_argument("--from-dir", type=Path, default=None,
                   help="read band JSONs named <band>.json from this directory (no rollouts)")
    p.add_argument("--out", type=Path, default=None, help="write the report as JSON here")
    p.add_argument("--device", type=str, default="cpu", help="inference device")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- stats
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (better than normal at the edges)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(centre - half), float(centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # sum of the lower tail of Binomial(n, 1/2), doubled
    from math import comb

    tail = sum(comb(n, i) for i in range(k + 1)) / 2.0**n
    return float(min(1.0, 2.0 * tail))


def newcombe_diff(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Newcombe's hybrid score interval for p1 - p2 (independent samples)."""
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (float(lo), float(hi))


# ------------------------------------------------------------------------ rollouts
def run_band(
    checkpoint: Path,
    lo: float,
    hi: float,
    episodes: int,
    seed: int,
    terminate: float,
    rate_limit: float,
    shaping: str | None,
    device: str,
) -> dict:
    """Evaluate one half-open band (lo, hi] by sampling the angle uniformly inside it."""
    center = 0.5 * (lo + hi)
    half_width = 0.5 * (hi - lo)
    argv = [
        "--checkpoint", str(checkpoint),
        "--init-mode", "random",
        "--init-angle-center", f"{center:.6f}",
        "--init-angle-limit", f"{half_width:.6f}",
        "--init-rate-limit", f"{rate_limit}",
        "--terminate-angle", f"{terminate}",
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--no-render", "--no-video", "--no-plot",
    ]
    if shaping is not None:
        argv += ["--shaping", shaping]
    args = evaluate.parse_args(argv)
    args.device = device  # evaluate.py has no --device; load_policy takes it directly
    return _run_inline(args)


def _run_inline(args: argparse.Namespace) -> dict:
    """Head-less batch rollout, mirroring evaluate.py's own batch branch.

    It re-uses evaluate.py's argument parsing, its checkpoint loading, its
    controller factory and its ending classifier, so a band report cannot drift
    away from what ``evaluate.py`` would have printed for the same settings.
    """
    from pendulum_rl.envs.inverted_pendulum import InvertedPendulumEnv
    from pendulum_rl.lightning_module import env_config_from_checkpoint, load_policy

    _, _, raw = load_policy(args.checkpoint, device=args.device)
    overrides = {
        "init_mode": "random",
        "init_angle_center": float(np.radians(args.init_angle_center)),
        "init_angle_limit": float(np.radians(args.init_angle_limit)),
        "init_rate_limit": float(args.init_rate_limit),
        "terminate_angle": (float(np.radians(args.terminate_angle)) if args.terminate_angle > 0 else None),
    }
    if getattr(args, "shaping", None) is not None:
        overrides["shaping"] = args.shaping
    env_cfg = env_config_from_checkpoint(raw, **overrides)
    if env_cfg.max_episode_steps is None:
        env_cfg = evaluate.replace(env_cfg, max_episode_steps=500)
    env = InvertedPendulumEnv(env_cfg)
    _, controller = evaluate.build_controller(args, env)

    successes: list[bool] = []
    endings: list[str] = []
    thetas: list[float] = []
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        info: dict = {}
        terminated = truncated = False
        for _ in range(int(env_cfg.max_episode_steps)):
            action = controller(env, obs)
            obs, _reward, terminated, truncated, info = env.step([action])
            if terminated or truncated:
                break
        successes.append(bool(info.get("is_success", False)))
        endings.append(evaluate.ending_reason(info, terminated, truncated))
        thetas.append(abs(float(info.get("theta", 0.0))))
    env.close()
    return {
        "checkpoint": str(args.checkpoint),
        "episodes": args.episodes,
        "seed": args.seed,
        "success_rate": float(np.mean(successes)),
        "successes": successes,
        "failures": endings,
    }


# --------------------------------------------------------------------------- report
def band_edges(edges: list[float]) -> list[tuple[float, float]]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            raise ValueError(f"band edges must increase: {lo} -> {hi}")
        out.append((float(lo), float(hi)))
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bands = band_edges(args.bands)
    arms = [args.checkpoint] + ([args.base] if args.base else [])

    results: dict[str, dict] = {}
    for arm_index, ckpt in enumerate(arms):
        name = f"arm{arm_index}"
        results[name] = {"checkpoint": str(ckpt), "bands": {}}
        for band_index, (lo, hi) in enumerate(bands):
            seed = args.seed + band_index * args.episodes
            if args.from_dir is not None:
                path = args.from_dir / f"{int(lo)}_{int(hi)}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                summary = {
                    "success_rate": payload["success_rate"],
                    "episodes": payload.get("episodes", args.episodes),
                    "successes": [e["success"] for e in payload.get("episodes", [])],
                    "failures": payload.get("failures", []),
                }
            else:
                summary = run_band(
                    ckpt, lo, hi, args.episodes, seed, args.terminate,
                    args.rate_limit, args.shaping, args.device,
                )
            results[name]["bands"][f"{lo:g}-{hi:g}"] = summary
            print(
                "  %-28s %-12s success %6.1f%%  (%d eps, seeds %d..%d)"
                % (Path(ckpt).parent.name, f"({lo:g},{hi:g}]",
                   100 * summary["success_rate"], summary["episodes"], seed, seed + summary["episodes"] - 1)
            )

    # ---- ladder table
    print()
    header = f"{'band (deg)':>12} | " + " | ".join(
        f"{Path(results[a]['checkpoint']).parent.name:>22}" for a in results
    )
    print(header)
    print("-" * len(header))
    for key in results["arm0"]["bands"]:
        cells = []
        for arm in results:
            s = results[arm]["bands"][key]
            k = int(round(s["success_rate"] * s["episodes"]))
            lo_ci, hi_ci = wilson(k, s["episodes"])
            cells.append(f"{100 * s['success_rate']:6.1f}% [{100 * lo_ci:4.1f},{100 * hi_ci:5.1f}]")
        print(f"{key:>12} | " + " | ".join(f"{c:>22}" for c in cells))

    if len(results) == 2:
        print()
        print("paired comparison (exact McNemar on discordant episodes):")
        print(f"{'band (deg)':>12} | {'both ok':>7} | {'only A':>6} | {'only B':>6} | {'both bad':>8} | {'p':>7} | 95% CI of diff")
        print("-" * 92)
        for key in results["arm0"]["bands"]:
            a = results["arm0"]["bands"][key]["successes"]
            b = results["arm1"]["bands"][key]["successes"]
            n = min(len(a), len(b))
            both_ok = sum(1 for i in range(n) if a[i] and b[i])
            only_a = sum(1 for i in range(n) if a[i] and not b[i])
            only_b = sum(1 for i in range(n) if b[i] and not a[i])
            both_bad = sum(1 for i in range(n) if not a[i] and not b[i])
            p = mcnemar_exact(only_a, only_b)
            lo_ci, hi_ci = newcombe_diff(sum(a[:n]), n, sum(b[:n]), n)
            print(
                f"{key:>12} | {both_ok:>7} | {only_a:>6} | {only_b:>6} | {both_bad:>8} | {p:>7.4f} | "
                f"[{100 * lo_ci:+5.1f}, {100 * hi_ci:+5.1f}] pts"
            )

    results["bands"] = [f"{lo:g}-{hi:g}" for lo, hi in bands]
    results["episodes_per_band"] = args.episodes
    results["seed_base"] = args.seed
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
