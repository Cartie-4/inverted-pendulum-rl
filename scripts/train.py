"""Train a PPO agent to control the single inverted pendulum.

Examples
--------
Balance only (starts near upright), the fast default run::

    python scripts/train.py --model cart --init-mode upright --max-epochs 200

Full swing-up + balance, with a curriculum (hanging first, then random starts)::

    python scripts/train.py --model cart --init-mode random \
        --warmup-init-mode hanging --curriculum-fraction 0.4 --max-epochs 600

Torque-driven pendulum with a fixed pivot (the "benchmark" variant)::

    python scripts/train.py --model pivot --init-mode random --warmup-init-mode hanging

Everything is logged to TensorBoard (``outputs/logs/<run>``) *and* to
``outputs/logs/<run>/metrics.jsonl`` so results are readable without TB.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Keep matplotlib's cache inside the project before anything pulls matplotlib in
# (the default %LOCALAPPDATA% location is not writable on locked-down accounts).
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import pytorch_lightning as pl  # noqa: E402
from pytorch_lightning.callbacks import LearningRateMonitor  # noqa: E402
from pytorch_lightning.loggers import TensorBoardLogger  # noqa: E402

from pendulum_rl.lightning_module import (  # noqa: E402
    CheckpointEveryEpoch,
    PPOLightningModule,
    RolloutDataModule,
    TrainConfig,
)
from pendulum_rl.utils import CKPT_DIR, LOG_DIR, ensure_dirs, human_time, set_seed  # noqa: E402


def build_config(args: argparse.Namespace) -> TrainConfig:
    run_name = args.run_name or f"ppo_{args.model}_{args.init_mode}"
    return TrainConfig(
        model=args.model,
        init_mode=args.init_mode,
        init_angle_limit=(None if args.init_angle_limit is None
                          else float(np.radians(args.init_angle_limit))),
        init_rate_limit=args.init_rate_limit,
        warmup_init_mode=args.warmup_init_mode,
        curriculum_fraction=args.curriculum_fraction,
        max_episode_steps=args.max_episode_steps,
        max_force=args.max_force,
        max_torque=args.max_torque,
        action_mode=args.action_mode,
        control_dt=args.control_dt,
        sim_dt=args.sim_dt,
        a_theta_power=args.a_theta_power,
        w_theta=args.w_theta,
        w_x=args.w_x,
        w_omega=args.w_omega,
        w_u=args.w_u,
        terminate_on_limit=args.terminate_on_limit,
        # The CLI flag is in DEGREES, like --init-angle-limit (and like evaluate.py).
        # EnvConfig compares this against the angle wrapped to [-pi, pi), so passing
        # the raw value made every documented threshold (11.5 / 23.5 / 53.5 / ...) a
        # silent no-op: 11.5 "rad" is ~659 deg and can never be exceeded.
        terminate_angle=(float(np.radians(args.terminate_angle))
                         if args.terminate_angle > 0 else None),
        shaping=args.shaping,
        shape_coef=args.shaping_coef,
        shape_gamma=args.shaping_gamma,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        entropy_coef=args.entropy_coef,
        entropy_coef_final=args.entropy_coef_final,
        epochs_per_update=args.epochs_per_update,
        minibatches=args.minibatches,
        target_kl=args.target_kl,
        hidden_sizes=tuple(args.hidden_sizes),
        log_std_init=args.log_std_init,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        max_epochs=args.max_epochs,
        val_num_envs=args.val_num_envs,
        val_max_episode_steps=args.val_max_episode_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        seed=args.seed,
        run_name=run_name,
        device=args.device,
        render=args.render,
        render_fps=args.render_fps,
        render_env=args.render_env,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    task = p.add_argument_group("task")
    task.add_argument("--model", choices=["cart", "pivot"], default="cart", help="plant model")
    task.add_argument(
        "--init-mode",
        choices=["upright", "hanging", "random"],
        default="upright",
        help="initial state distribution of the task",
    )
    task.add_argument(
        "--warmup-init-mode",
        choices=["upright", "hanging", "random"],
        default=None,
        help="curriculum: use this init mode for the first --curriculum-fraction of training",
    )
    task.add_argument(
        "--curriculum-fraction", type=float, default=0.4,
        help="fraction of training using --warmup-init-mode (curriculum over the init mode)",
    )
    task.add_argument(
        "--init-angle-limit",
        type=float,
        default=None,
        help="with --init-mode random: draw theta ~ U(-limit, +limit) in DEGREES instead of the "
             "full circle. This is the main curriculum knob for growing task difficulty; it is "
             "how Gymnasium's Pendulum samples its start state.",
    )
    task.add_argument(
        "--init-rate-limit",
        type=float,
        default=None,
        help="with --init-angle-limit: also draw the rates (x_dot, theta_dot) from "
             "U(-limit, +limit); default 0.5 rad/s.",
    )
    task.add_argument("--max-episode-steps", type=int, default=500, help="10 s at 50 Hz")
    task.add_argument("--control-dt", type=float, default=0.02, help="agent decision period [s]")
    task.add_argument("--sim-dt", type=float, default=0.002, help="RK4 physics step [s]")
    task.add_argument("--action-mode", choices=["force", "acceleration"], default="force")
    task.add_argument("--max-force", type=float, default=50.0)
    task.add_argument("--max-torque", type=float, default=2.5)
    task.add_argument("--a-theta-power", type=int, default=2, help="exponent of the upright reward term")
    task.add_argument("--w-theta", type=float, default=3.0, help="angle penalty weight")
    task.add_argument("--w-x", type=float, default=0.1, help="cart position penalty weight")
    task.add_argument("--w-omega", type=float, default=0.1, help="angular rate penalty weight")
    task.add_argument("--w-u", type=float, default=0.001, help="control effort penalty weight")
    task.add_argument(
        "--terminate-on-limit",
        action="store_true",
        help="treat rail limit / divergence as a terminal state instead of a truncation",
    )
    task.add_argument(
        "--terminate-angle",
        type=float,
        # 34.4 deg ~= the historical 0.6 rad default.  The value has to move with the
        # unit: leaving 0.6 here would have tightened the default from ~34 deg to
        # 0.6 deg once the flag started being converted from degrees.
        default=34.4,
        help="end an episode when |theta| exceeds this many DEGREES (pole fell over); "
             "0 disables. Automatically ignored for swing-up tasks (hanging starts), "
             "where the pole must be free to sweep through pi/2.",
    )
    task.add_argument(
        "--shaping",
        choices=["none", "energy"],
        default="none",
        help="potential-based reward shaping. 'energy' adds "
             "gamma*Phi(s')-Phi(s) with Phi = -coef*(E_pend - 2mgl)^2, which is "
             "provably policy-invariant (Ng et al. 1999) and silent at upright, but "
             "gives the pumping behaviour a dense gradient that i.i.d. Gaussian "
             "exploration cannot find. Default 'none' = unchanged reward.",
    )
    task.add_argument(
        "--shaping-coef",
        type=float,
        default=1.0,
        help="coefficient c of the energy shaping potential; the per-step shaping "
             "is O(c * (dE_pend per step) * (E_pend - E*)) ~ 0.5 at c=1 while "
             "pumping, i.e. the same order as the upright reward.",
    )
    task.add_argument(
        "--shaping-gamma",
        type=float,
        default=None,
        help="discount used by the shaping term only; default = --gamma, which is "
             "what the policy-invariance guarantee assumes.",
    )

    alg = p.add_argument_group("PPO")
    alg.add_argument("--learning-rate", type=float, default=3e-4)
    alg.add_argument("--gamma", type=float, default=0.99)
    alg.add_argument("--gae-lambda", type=float, default=0.95)
    alg.add_argument("--clip-range", type=float, default=0.2)
    alg.add_argument("--entropy-coef", type=float, default=1e-3)
    alg.add_argument("--entropy-coef-final", type=float, default=1e-4)
    alg.add_argument("--epochs-per-update", type=int, default=10)
    alg.add_argument("--minibatches", type=int, default=16)
    alg.add_argument("--target-kl", type=float, default=0.1)
    alg.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    alg.add_argument(
        "--log-std-init",
        type=float,
        default=-3.0,
        help="initial log std of the Gaussian policy (exp(-3) ~ 5%% of full action range)",
    )

    loop = p.add_argument_group("training loop")
    loop.add_argument("--num-envs", type=int, default=8)
    loop.add_argument("--rollout-steps", type=int, default=256, help="env steps per env per iteration")
    loop.add_argument("--max-epochs", type=int, default=300, help="number of PPO iterations")
    loop.add_argument("--val-num-envs", type=int, default=8)
    loop.add_argument("--val-max-episode-steps", type=int, default=500)
    loop.add_argument("--check-val-every-n-epoch", type=int, default=10)
    loop.add_argument("--seed", type=int, default=0)
    loop.add_argument("--run-name", type=str, default=None)
    loop.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | mps")
    loop.add_argument(
        "--render",
        action="store_true",
        default=True,
        help="open a live window and draw the pendulum in real time while training (default: on)",
    )
    loop.add_argument(
        "--no-render",
        dest="render",
        action="store_false",
        help="disable the live window (head-less / faster runs)",
    )
    loop.add_argument(
        "--render-fps",
        type=float,
        default=30.0,
        help="max live-view refresh rate; the canvas redraw costs ~1-2 ms, so keep it modest",
    )
    loop.add_argument("--render-env", type=int, default=0, help="which parallel env the live view follows")
    loop.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="warm-start the policy from another run's checkpoint (weights + obs normaliser)",
    )
    loop.add_argument("--fast-dev-run", action="store_true", help="1 epoch on a tiny rollout to smoke-test the code")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.fast_dev_run:
        args.max_epochs = 1
        args.rollout_steps = 32
        args.num_envs = 2
        args.epochs_per_update = 2
        args.minibatches = 2
        args.check_val_every_n_epoch = 1

    ensure_dirs()
    cfg = build_config(args)
    set_seed(cfg.seed)

    run_dir = LOG_DIR / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(cfg.to_dict(), fh, indent=2)

    train_cfgs = [cfg.env_config() for _ in range(cfg.num_envs)]
    val_cfgs = [cfg.env_config(for_eval=True) for _ in range(cfg.val_num_envs)]
    datamodule = RolloutDataModule(train_cfgs, val_cfgs, cfg.rollout_steps, seed=cfg.seed)

    module = PPOLightningModule(cfg, datamodule, metrics_path=metrics_path)
    if args.init_from is not None:
        module.load_pretrained_weights(args.init_from)
        print(f"warm start      : {args.init_from}")

    # Our own callback: ModelCheckpoint with a monitor silently wrote no files in
    # this project, so a killed run lost all its weights.  This one always keeps
    # last.ckpt and refreshes best.ckpt whenever the metric improves.
    checkpoint = CheckpointEveryEpoch(CKPT_DIR / cfg.run_name, monitor="val/mean_return", mode="max")
    logger = TensorBoardLogger(save_dir=str(LOG_DIR), name=cfg.run_name, version="tb")
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto" if cfg.device == "auto" else cfg.device,
        devices=1,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        deterministic=False,
        fast_dev_run=args.fast_dev_run,
    )

    print("=" * 78)
    print(f"PPO inverted pendulum | run '{cfg.run_name}'")
    print(f"  plant          : {cfg.model} ({cfg.action_mode}), |u| <= "
          f"{cfg.max_force if cfg.model == 'cart' else cfg.max_torque}")
    print(f"  task           : init={cfg.init_mode}"
          + (f" (curriculum: {cfg.warmup_init_mode} for the first {cfg.curriculum_fraction:.0%})"
             if cfg.warmup_init_mode else ""))
    print(f"  timing         : physics {cfg.sim_dt} s, agent {cfg.control_dt} s, "
          f"episode {cfg.max_episode_steps * cfg.control_dt:.1f} s")
    if cfg.shaping != "none":
        # Say it loudly: mean_return is no longer comparable with earlier runs,
        # because the shaping term is part of the return.  The gates are all
        # success rates, which the shaping cannot change.
        gamma = cfg.gamma if cfg.shape_gamma is None else cfg.shape_gamma
        print(f"  reward shaping : {cfg.shaping}, coef {cfg.shape_coef}, gamma {gamma} "
              f"-> 'mean return' is NOT comparable with unshaped runs")
    print(f"  per iteration  : {cfg.num_envs} envs x {cfg.rollout_steps} steps = {cfg.steps_per_epoch} transitions")
    print(f"  total budget   : {cfg.max_epochs} iterations = {cfg.max_epochs * cfg.steps_per_epoch:,} env steps")
    print(f"  outputs        : {run_dir}")
    print("=" * 78)

    started = time.perf_counter()
    trainer.fit(module, datamodule=datamodule)
    elapsed = time.perf_counter() - started

    best = checkpoint.best_path if checkpoint.best_path.exists() else Path("n/a")
    print("-" * 78)
    print(f"training finished in {human_time(elapsed)} ({elapsed / max(1, trainer.current_epoch + 1):.1f} s/iteration)")
    print(f"best checkpoint : {best} (val/mean_return={checkpoint.best:.2f} @ iteration {checkpoint.best_epoch + 1})")
    print(f"last checkpoint : {checkpoint.last_path}")
    print(f"metrics         : {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
