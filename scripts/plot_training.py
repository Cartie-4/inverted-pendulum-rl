"""Plot training curves from the metrics.jsonl written by scripts/train.py."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pendulum_rl.utils import LOG_DIR, OUTPUT_DIR, MetricsWriter, ensure_dirs  # noqa: E402

PANELS = [
    ("train/mean_episode_return", "episode return"),
    ("train/success_rate", "success rate"),
    ("train/mean_abs_theta_deg", "mean |theta| [deg]"),
    ("train/policy_std", "policy std (log-std exp)"),
    ("train/entropy", "policy entropy"),
    ("train/approx_kl", "approx KL"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="*", type=Path, help="run directories (default: every run in outputs/logs)")
    p.add_argument("--out", type=Path, default=OUTPUT_DIR / "training_curves.png")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_dirs()
    runs = args.runs or [d for d in sorted(LOG_DIR.glob("*")) if (d / "metrics.jsonl").exists()]
    runs = [r if (r / "metrics.jsonl").exists() else LOG_DIR / r for r in runs]
    if not runs:
        print("no metrics.jsonl found; train a model first")
        return 1

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(9, 13), sharex=True)
    for run in runs:
        records = [r for r in MetricsWriter.read(run / "metrics.jsonl") if not r.get("val")]
        if not records:
            continue
        steps = [r.get("env_steps", r["epoch"]) for r in records]
        for ax, (key, label) in zip(axes, PANELS):
            values = [(r["epoch"], r[key]) for r in records if key in r and r[key] == r[key]]
            if values:
                ax.plot([v[0] for v in values], [v[1] for v in values], lw=1.3, label=run.name)
            ax.set_ylabel(label, fontsize=9)
            ax.grid(alpha=0.3, lw=0.5)
        val_records = [r for r in MetricsWriter.read(run / "metrics.jsonl") if r.get("val")]
        if val_records:
            axes[0].plot(
                [r["epoch"] for r in val_records],
                [r["val/mean_return"] for r in val_records],
                "o--",
                ms=3,
                lw=1.0,
                alpha=0.8,
                label=f"{run.name} (deterministic eval)",
            )
        _ = steps
    axes[-1].set_xlabel("PPO iteration")
    axes[0].legend(fontsize=8)
    fig.suptitle("PPO inverted pendulum — training curves", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
