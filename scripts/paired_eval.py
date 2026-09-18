"""Paired (same initial states) comparison of two checkpoints or two evaluations.

The evaluator is deterministic: ``evaluate.py`` resets the plant with
``seed + episode`` and acts deterministically, so the *same* seeds always produce
the *same* initial states.  Two checkpoints can therefore be compared on matched
states, which is what makes an exact McNemar test on individual states possible
instead of comparing two independent proportions -- the latter needs hundreds of
episodes to resolve a 5-point difference, the former resolves it with six
one-directional discordant states (p = 0.031).

Two ways to use it
------------------
Compare evaluations that already exist::

    python scripts/paired_eval.py --baseline-json base.json --candidate-json cand.json

Or let it drive ``evaluate.py`` first (one call per seed, both checkpoints)::

    python scripts/paired_eval.py \
        --baseline-ckpt outputs/checkpoints/p1_rule/best.ckpt \
        --candidate-ckpt outputs/checkpoints/p2/best.ckpt \
        --seeds 900-909 --episodes 20 \
        --init-mode random --init-angle-limit 15 --terminate-angle 23.5

Verdict rules (see CURRICULUM_PLAN.pdf section 5)
-------------------------------------------------
1. paired no-regression : the candidate must not be significantly worse
                          (exact McNemar, alpha = 0.05), and its failure count
                          may not exceed the baseline's by more than
                          ``--tolerance`` episodes;
2. paired improvement    : reported as significant when p < alpha in the
                          candidate's favour, which is the operative criterion
                          when the baseline already meets the absolute target.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from math import comb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATE = PROJECT_ROOT / "scripts" / "evaluate.py"

#: Ending reasons that describe a physical failure of the plant.
FAILURE_ENDINGS = ("fell over", "out of rail", "diverged")


def parse_seeds(text: str) -> list[int]:
    """Accept ``900-909``, ``900,901`` or a mix, and return an ordered unique list."""
    seeds: list[int] = []
    for chunk in text.replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk:
            low, high = chunk.split("-", 1)
            seeds.extend(range(int(low), int(high) + 1))
        else:
            seeds.append(int(chunk))
    return list(dict.fromkeys(seeds))


def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(comb(n, i) * p**i * (1.0 - p) ** (n - i) for i in range(0, min(k, n) + 1))


def mcnemar_exact(fixed: int, broken: int) -> float:
    """Two-sided exact McNemar p-value over the discordant pairs only."""
    discordant = fixed + broken
    if discordant == 0:
        return 1.0
    return min(1.0, 2.0 * binom_cdf(min(fixed, broken), discordant, 0.5))


def load_episodes(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    if not episodes:
        raise SystemExit(
            f"{path} has no per-episode records.  Re-run evaluate.py with --json-out "
            "on a head-less batch (no --render)."
        )
    return episodes


def summarize(episodes: list[dict]) -> dict:
    endings: dict[str, int] = {}
    for episode in episodes:
        endings[episode["ending"]] = endings.get(episode["ending"], 0) + 1
    failures = sum(count for reason, count in endings.items() if reason in FAILURE_ENDINGS)
    total = len(episodes)
    return {
        "episodes": total,
        "successes": sum(1 for episode in episodes if episode["success"]),
        "success_rate": sum(1 for episode in episodes if episode["success"]) / total if total else 0.0,
        "failures": failures,
        "endings": endings,
    }


def check_seed_blocks(seeds: list[int], episodes: int) -> None:
    """Refuse overlapping episode-seed blocks -- they are pseudo-replication.

    ``evaluate.py`` resets with ``seed + episode``, so a run started at S covers
    ``S .. S + episodes - 1``.  Ten runs of 20 episodes from 900..909 therefore
    cover only episode seeds 900..928: 200 paired keys over 29 distinct states,
    which would inflate McNemar's significance by counting the same state many
    times.  Either use one seed with many episodes, or space the seeds apart.
    """
    blocks = sorted((seed, seed + episodes - 1) for seed in seeds)
    for (a_low, a_high), (b_low, b_high) in zip(blocks, blocks[1:]):
        if b_low <= a_high:
            raise SystemExit(
                f"seed blocks overlap: run {a_low} covers episode seeds {a_low}..{a_high}, run "
                f"{b_low} covers {b_low}..{b_high}.  Use a single seed with --episodes N, or "
                f"space the seeds by at least {episodes}."
            )


def pair_key(episode: dict) -> tuple[int, int]:
    """Identify one episode across runs.

    ``seed`` alone is *not* unique: a run started with ``--seed S`` covers episode
    seeds ``S .. S + episodes - 1``, so two runs with nearby seeds overlap and
    keying on ``seed`` silently drops the overlapping states (measured: 21 paired
    states instead of 40).  The pair (run seed, episode index) is unique, and the
    run seed is recoverable because evaluate.py records ``seed = args.seed + episode``.
    """
    return (int(episode["seed"]) - int(episode["episode"]), int(episode["episode"]))


def compare(baseline: list[dict], candidate: list[dict], tolerance: int, alpha: float) -> dict:
    base_by_key = {pair_key(episode): episode for episode in baseline}
    cand_by_key = {pair_key(episode): episode for episode in candidate}
    shared = sorted(set(base_by_key) & set(cand_by_key))
    if not shared:
        raise SystemExit("the two evaluations share no seeds; pairing is impossible")

    # Guard against pseudo-replication: with overlapping seed blocks several paired
    # keys describe the *same* initial state, which would inflate the significance.
    distinct_states = {
        tuple(round(value, 9) for value in (base_by_key[key].get("init_state") or ()))
        for key in shared
    }
    if len(distinct_states) < len(shared):
        print(f"[paired-eval] WARNING: {len(shared)} paired keys cover only "
              f"{len(distinct_states)} distinct initial states (overlapping seed blocks?)")

    both_ok = both_fail = fixed = broken = 0
    state_mismatch = 0
    for key in shared:
        base, cand = base_by_key[key], cand_by_key[key]
        base_state, cand_state = base.get("init_state"), cand.get("init_state")
        if base_state and cand_state and len(base_state) == len(cand_state):
            if any(abs(a - b) > 1e-9 for a, b in zip(base_state, cand_state)):
                state_mismatch += 1
        if base["success"] and cand["success"]:
            both_ok += 1
        elif base["success"] and not cand["success"]:
            broken += 1
        elif not base["success"] and cand["success"]:
            fixed += 1
        else:
            both_fail += 1

    p_value = mcnemar_exact(fixed, broken)
    base_summary = summarize(baseline)
    cand_summary = summarize(candidate)

    if p_value < alpha and fixed > broken:
        verdict = "SIGNIFICANTLY BETTER"
    elif p_value < alpha and broken > fixed:
        verdict = "SIGNIFICANTLY WORSE"
    else:
        verdict = "NO MEASURABLE CHANGE"

    failure_delta = cand_summary["failures"] - base_summary["failures"]
    no_regression = verdict != "SIGNIFICANTLY WORSE" and failure_delta <= tolerance

    return {
        "paired_states": len(shared),
        "distinct_states": len(distinct_states),
        "both_ok": both_ok,
        "fixed": fixed,
        "broken": broken,
        "both_fail": both_fail,
        "p_value": p_value,
        "verdict": verdict,
        "failure_delta": failure_delta,
        "tolerance": tolerance,
        "no_regression": no_regression,
        "state_mismatch": state_mismatch,
        "baseline": base_summary,
        "candidate": cand_summary,
    }


def run_evaluations(args: argparse.Namespace, workdir: Path) -> tuple[Path, Path]:
    workdir.mkdir(parents=True, exist_ok=True)
    common: list[str] = ["--episodes", str(args.episodes), "--no-render", "--no-video", "--no-plot"]
    if args.init_mode:
        common += ["--init-mode", args.init_mode]
    if args.init_angle_limit is not None:
        common += ["--init-angle-limit", str(args.init_angle_limit)]
    if args.init_rate_limit is not None:
        common += ["--init-rate-limit", str(args.init_rate_limit)]
    if args.terminate_angle is not None:
        common += ["--terminate-angle", str(args.terminate_angle)]

    written: dict[str, Path] = {}
    for label, checkpoint in (("baseline", args.baseline_ckpt), ("candidate", args.candidate_ckpt)):
        merged: list[dict] = []
        for seed in args.seed_list:
            out_json = workdir / f"{label}_seed{seed}.json"
            command = [
                sys.executable, str(EVALUATE),
                "--checkpoint", str(checkpoint),
                "--seed", str(seed),
                "--json-out", str(out_json),
                *common,
            ]
            print(f"[paired-eval] {label} seed={seed} ...", flush=True)
            subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))
            merged.extend(load_episodes(out_json))
        merged_path = workdir / f"{label}.json"
        merged_path.write_text(json.dumps({"episodes": merged}, indent=2), encoding="utf-8")
        written[label] = merged_path
    return written["baseline"], written["candidate"]


def print_report(result: dict, alpha: float) -> None:
    base, cand = result["baseline"], result["candidate"]
    print("=" * 70)
    print(f"paired states     : {result['paired_states']} "
          f"({result['distinct_states']} distinct initial states)"
          + (f"   (!! {result['state_mismatch']} with mismatched initial states)"
             if result["state_mismatch"] else ""))
    print(f"baseline  success : {base['successes']}/{base['episodes']} "
          f"({base['success_rate']:.1%})   failures {base['failures']}")
    print(f"candidate success : {cand['successes']}/{cand['episodes']} "
          f"({cand['success_rate']:.1%})   failures {cand['failures']}"
          f"   ({result['failure_delta']:+d} vs baseline)")
    print("-" * 70)
    print(f"both success      : {result['both_ok']}")
    print(f"fixed             : {result['fixed']}   (baseline failed, candidate succeeded)")
    print(f"broken            : {result['broken']}   (baseline succeeded, candidate failed)")
    print(f"both failed       : {result['both_fail']}")
    print(f"exact McNemar p   : {result['p_value']:.4f}  (alpha = {alpha})")
    print("-" * 70)
    print("endings baseline  : " + ", ".join(f"{k} x{v}" for k, v in sorted(base["endings"].items())))
    print("endings candidate : " + ", ".join(f"{k} x{v}" for k, v in sorted(cand["endings"].items())))
    print("-" * 70)
    print(f"verdict           : {result['verdict']}")
    print(f"no-regression     : {'OK' if result['no_regression'] else 'VIOLATED'}"
          f"   (failure delta {result['failure_delta']:+d}, tolerance +{result['tolerance']})")
    print("=" * 70)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--baseline-json", type=Path, help="per-episode JSON from evaluate.py --json-out")
    source.add_argument("--baseline-ckpt", type=Path, help="run the baseline evaluation first")
    p.add_argument("--candidate-json", type=Path)
    p.add_argument("--candidate-ckpt", type=Path)
    p.add_argument("--seeds", type=str, default="900",
                   help="run seeds; episode seeds are seed..seed+episodes-1, so blocks must not "
                        "overlap.  The default is one 200-episode block (900..1099)")
    p.add_argument("--episodes", type=int, default=200, help="episodes per seed")
    p.add_argument("--init-mode", choices=["upright", "hanging", "random"], default=None)
    p.add_argument("--init-angle-limit", type=float, default=None, help="degrees")
    p.add_argument("--init-rate-limit", type=float, default=None)
    p.add_argument("--terminate-angle", type=float, default=None, help="degrees; 0 disables")
    p.add_argument("--tolerance", type=int, default=2,
                   help="allowed increase in failure count before no-regression fails")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--workdir", type=Path, default=None,
                   help="where the per-seed JSONs land (default outputs/logs/paired/<tag>)")
    p.add_argument("--json-out", type=Path, default=None, help="also write the comparison result")
    args = p.parse_args(argv)
    args.seed_list = parse_seeds(args.seeds)
    if args.baseline_ckpt is not None:
        if args.candidate_ckpt is None:
            p.error("--baseline-ckpt requires --candidate-ckpt")
    elif args.candidate_json is None:
        p.error("--baseline-json requires --candidate-json")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.baseline_ckpt is not None:
        check_seed_blocks(args.seed_list, args.episodes)
        workdir = args.workdir or (PROJECT_ROOT / "outputs" / "logs" / "paired" / args.candidate_ckpt.parent.name)
        baseline_json, candidate_json = run_evaluations(args, workdir)
    else:
        baseline_json, candidate_json = args.baseline_json, args.candidate_json

    result = compare(
        load_episodes(baseline_json),
        load_episodes(candidate_json),
        tolerance=args.tolerance,
        alpha=args.alpha,
    )
    print_report(result, args.alpha)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"comparison written: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
