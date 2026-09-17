"""Compare the RL policy against the classical controllers on the same seeds.

Prints a table (and optionally writes JSON) for:

* ``random``            – uniform random force (lower bound)
* ``zero``              – no control at all (the plant simply falls over)
* ``pd``                – PD on the pole angle (balance-only task)
* ``energy``            – Åström/Furuta energy swing-up + LQR catch
* ``ppo``               – the trained policy from ``--checkpoint``
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATE = PROJECT_ROOT / "scripts" / "evaluate.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--init-mode", choices=["upright", "hanging", "random"], default=None)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args(argv)


def run_eval(extra: list[str]) -> dict:
    cmd = [sys.executable, str(EVALUATE), "--no-video", *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"evaluation failed:\n{proc.stdout}\n{proc.stderr}")
    out = proc.stdout
    summary = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "mean return":
            summary["mean_return"] = float(value.split("+-")[0])
        elif key == "theta RMS":
            summary["theta_rms_deg"] = float(value.split()[0])
        elif key == "success rate":
            summary["success_rate"] = float(value.split()[0].rstrip("%")) / 100.0
        elif key == "mean length":
            summary["mean_length"] = float(value.split()[0])
        elif key == "mean |u| / u_max":
            summary["mean_action_frac"] = float(value.split()[0])
        elif key == "controller":
            summary["controller"] = value
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    common = ["--episodes", str(args.episodes), "--max-steps", str(args.max_steps), "--seed", str(args.seed)]
    if args.init_mode:
        common += ["--init-mode", args.init_mode]

    rows = []
    for baseline in ("zero", "random", "pd", "energy"):
        rows.append(run_eval(["--baseline", baseline, *common]))
    rows.append(run_eval(["--checkpoint", str(args.checkpoint), *common]))

    header = f"{'controller':<22}{'return':>10}{'theta RMS [deg]':>17}{'success':>10}{'ep length':>11}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.get('controller', '?'):<22}"
            f"{row.get('mean_return', float('nan')):>10.2f}"
            f"{row.get('theta_rms_deg', float('nan')):>17.2f}"
            f"{row.get('success_rate', float('nan')):>9.0%}"
            f"{row.get('mean_length', float('nan')):>11.1f}"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
