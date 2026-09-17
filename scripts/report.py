"""Build outputs/report.md from the metrics of every run in outputs/logs.

The report collects, per run: the configuration that produced it, the learning
curve summary, and (when a checkpoint exists) a table of PPO vs. the classical
controllers evaluated on identical seeds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pendulum_rl.utils import CKPT_DIR, LOG_DIR, OUTPUT_DIR, MetricsWriter, ensure_dirs  # noqa: E402


def load_run(run_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    config = {}
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
    records = MetricsWriter.read(run_dir / "metrics.jsonl")
    train = [r for r in records if not r.get("val")]
    val = [r for r in records if r.get("val")]
    return config, train, val


def best_checkpoint(run_name: str) -> Path | None:
    """Prefer ``best.ckpt`` (written by CheckpointEveryEpoch), fall back to last."""
    ckpt_dir = CKPT_DIR / run_name
    if not ckpt_dir.exists():
        return None
    for name in ("best.ckpt", "last.ckpt"):
        candidate = ckpt_dir / name
        if candidate.exists():
            return candidate
    versions = sorted(ckpt_dir.glob("last-v*.ckpt"))
    return versions[-1] if versions else None


def compare_table(checkpoint: Path, init_mode: str | None, episodes: int) -> str:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "compare.py"),
        "--checkpoint",
        str(checkpoint),
        "--episodes",
        str(episodes),
    ]
    if init_mode:
        cmd += ["--init-mode", init_mode]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return f"_(comparison failed: {proc.stderr.strip().splitlines()[-1] if proc.stderr else 'unknown'})_\n"
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return "```\n" + "\n".join(lines) + "\n```\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10, help="episodes per controller in the comparison")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "report.md")
    args = parser.parse_args(argv)

    ensure_dirs()
    runs = sorted(d for d in LOG_DIR.glob("*") if (d / "metrics.jsonl").exists())
    if not runs:
        print("no runs found under outputs/logs")
        return 1

    out: list[str] = ["# 训练与评估结果 (auto-generated)\n"]
    out.append("本文件由 `python scripts/report.py` 依据 `outputs/logs/*/metrics.jsonl` 自动生成。\n")

    for run_dir in runs:
        config, train, val = load_run(run_dir)
        if not train:
            continue
        name = run_dir.name
        out.append(f"\n## 运行 `{name}`\n")
        if config:
            out.append(
                "| 参数 | 值 |\n|---|---|\n"
                + "\n".join(
                    f"| `{k}` | `{v}` |"
                    for k, v in (
                        ("model", config.get("model")),
                        ("init_mode", config.get("init_mode")),
                        ("warmup_init_mode", config.get("warmup_init_mode")),
                        ("max_force", config.get("max_force")),
                        ("control_dt", config.get("control_dt")),
                        ("max_episode_steps", config.get("max_episode_steps")),
                        ("num_envs x rollout_steps", f"{config.get('num_envs')} x {config.get('rollout_steps')}"),
                        ("max_epochs", config.get("max_epochs")),
                        ("learning_rate", config.get("learning_rate")),
                        ("gamma", config.get("gamma")),
                        ("gae_lambda", config.get("gae_lambda")),
                        ("clip_range", config.get("clip_range")),
                        ("entropy_coef", config.get("entropy_coef")),
                        ("target_kl", config.get("target_kl")),
                        ("log_std_init", config.get("log_std_init")),
                        ("hidden_sizes", config.get("hidden_sizes")),
                        ("seed", config.get("seed")),
                    )
                )
                + "\n"
            )
        total_steps = int(train[-1].get("perf/env_steps", 0))
        first, last = train[0], train[-1]
        out.append(
            f"\n- 总环境步数: **{total_steps:,}**（{len(train)} 次 PPO 迭代）\n"
            f"- 训练回报: 首次 {first.get('train/mean_episode_return', float('nan')):.1f} "
            f"→ 末次 {last.get('train/mean_episode_return', float('nan')):.1f}\n"
            f"- 训练成功率: 首次 {first.get('train/success_rate', float('nan')):.0%} "
            f"→ 末次 {last.get('train/success_rate', float('nan')):.0%}\n"
            f"- 末次平均 |θ|: **{last.get('train/mean_abs_theta_deg', float('nan')):.2f} deg**\n"
            f"- 策略标准差: {first.get('train/policy_std', float('nan')):.3f} → "
            f"{last.get('train/policy_std', float('nan')):.3f}\n"
        )
        if val:
            best = max(val, key=lambda r: r.get("val/mean_return", float("-inf")))
            out.append(
                f"- 确定性评估最佳: 第 {best['epoch']} 次迭代，平均回报 **{best['val/mean_return']:.1f}**，"
                f"成功率 **{best.get('val/success_rate', 0):.0%}**\n"
                f"- 最后一次评估: 平均回报 {val[-1].get('val/mean_return', float('nan')):.1f}，"
                f"成功率 {val[-1].get('val/success_rate', float('nan')):.0%}\n"
            )

        ckpt = best_checkpoint(name)
        if ckpt is not None:
            init_mode = config.get("init_mode") if config else None
            out.append(f"\n最佳 checkpoint: `{ckpt.relative_to(PROJECT_ROOT)}`\n\n")
            out.append(f"### 与经典控制器对比（同为 {args.episodes} 个回合、同一种子）\n\n")
            out.append(compare_table(ckpt, init_mode, args.episodes))

    out.append("\n---\n\n训练曲线可用 `python scripts/plot_training.py` 生成，"
               "TensorBoard 日志位于 `outputs/logs/<run>/tb`。\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(out), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
