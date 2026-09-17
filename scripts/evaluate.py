"""Evaluate a trained PPO policy on the inverted pendulum and record a video.

Examples
--------
::

    # balance check + GIF from an upright start
    python scripts/evaluate.py --checkpoint outputs/checkpoints/ppo_cart_upright/last.ckpt

    # full swing-up: start hanging, watch the agent pump energy and catch the pole
    python scripts/evaluate.py --checkpoint <ckpt> --init-mode hanging --episodes 10

    # no checkpoint yet? look at the untrained policy or the classical baseline
    python scripts/evaluate.py --baseline random --init-mode hanging
    python scripts/evaluate.py --baseline energy --init-mode hanging
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

from pendulum_rl.agents.classical import BaggedController, PDController  # noqa: E402
from pendulum_rl.envs.inverted_pendulum import EnvConfig, InvertedPendulumEnv  # noqa: E402
from pendulum_rl.lightning_module import env_config_from_checkpoint  # noqa: E402
from pendulum_rl.rendering import render_episode  # noqa: E402
from pendulum_rl.utils import VIDEO_DIR, ensure_dirs  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--checkpoint", type=Path, help="Lightning .ckpt produced by scripts/train.py")
    src.add_argument(
        "--baseline",
        choices=["random", "pd", "energy", "zero"],
        help="controller to evaluate instead of a trained policy",
    )
    p.add_argument("--config-from", type=Path, help="checkpoint to copy the plant config from (with --baseline)")
    p.add_argument("--model", choices=["cart", "pivot"], default=None)
    p.add_argument("--init-mode", choices=["upright", "hanging", "random"], default=None)
    p.add_argument(
        "--init-angle-limit",
        type=float,
        default=None,
        help="with --init-mode random: draw theta ~ U(-limit, +limit) in DEGREES instead of the "
             "full circle. Use this to see how a policy does on a harder start distribution "
             "than it was trained on, e.g. --init-angle-limit 45.",
    )
    p.add_argument(
        "--init-rate-limit",
        type=float,
        default=None,
        help="with --init-angle-limit: also draw the rates from U(-limit, +limit); default 0.5.",
    )
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="hard cap on steps per episode; default None = no step cap at all. An episode "
             "then ends only when the plant actually fails (pole fell over / out of rail).",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="simulated-time budget per episode. Default: whatever the checkpoint recorded as its "
             "training horizon (500 steps = 10 s), so a run reports episodes of the same length "
             "the policy was trained on. Use 0 for truly unlimited (only sensible in --render "
             "mode, where you close the window).",
    )
    p.add_argument(
        "--wall-timeout",
        type=float,
        default=900.0,
        help="stop the whole evaluation after this many seconds of wall clock (0 = never). "
             "A backstop so a head-less run cannot hang forever if a policy simply never "
             "fails; ignored in --render mode, where you close the window yourself.",
    )
    p.add_argument(
        "--stop-on-failure",
        action="store_true",
        default=True,
        help="an episode ends as soon as the plant fails (default)",
    )
    p.add_argument(
        "--no-stop-on-failure",
        dest="stop_on_failure",
        action="store_false",
        help="keep simulating past a failure (pole down / cart off rail) for a fixed horizon",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--terminate-angle",
        type=float,
        default=None,
        help="end an episode once |theta| exceeds this many radians; 0 disables the rule. "
             "Required to be 0 (or unset with a hanging start) for swing-up.",
    )
    p.add_argument("--video", type=Path, default=None, help="output .gif/.mp4 path (default outputs/videos/<name>.gif)")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--plot", type=Path, default=None, help="trajectory plot output path")
    p.add_argument("--no-plot", action="store_true", help="skip the trajectory plot")
    p.add_argument("--video-episodes", type=int, default=1, help="how many episodes to render")
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--render", action="store_true", default=True, help="show the run in a live window (default: on)")
    p.add_argument("--no-render", dest="render", action="store_false", help="disable the live window")
    p.add_argument("--render-seconds", type=float, default=0.0,
                   help="with --render: stop after N seconds (0 = run until the window is closed)")
    p.add_argument("--json-out", type=Path, default=None, help="also dump the summary as JSON")
    return p.parse_args(argv)


def build_controller(args: argparse.Namespace, env: InvertedPendulumEnv):
    """Return (name, callable(env, obs) -> action, deterministic?) for the chosen source."""
    cfg = env.cfg
    if args.baseline is None and args.checkpoint is None:
        raise SystemExit("pass either --checkpoint or --baseline (use --baseline zero for a bare rollout)")
    if args.checkpoint is not None:
        from pendulum_rl.lightning_module import load_policy

        agent, _, _ = load_policy(args.checkpoint)
        limit = cfg.action_limit

        def policy(env, obs):  # noqa: ARG001 - signature shared with the baselines
            action, _, _ = agent.sample_actions(np.asarray([obs], dtype=np.float32), deterministic=True)
            return float(np.clip(action.reshape(-1)[0], -1.0, 1.0) * limit)

        return f"ppo:{Path(args.checkpoint).name}", policy

    if args.baseline == "random":
        rng = np.random.default_rng(args.seed)

        def policy(env, obs):  # noqa: ARG001
            return float(rng.uniform(-cfg.action_limit, cfg.action_limit))

        return "random", policy
    if args.baseline == "zero":

        def policy(env, obs):  # noqa: ARG001
            return 0.0

        return "zero", policy
    if args.baseline == "pd":
        controller = PDController(cfg)
        return "pd", lambda env, obs: controller(env)  # noqa: ARG005
    if args.baseline == "energy":
        controller = BaggedController(cfg)
        return "energy+pd", lambda env, obs: controller(env)  # noqa: ARG005
    raise SystemExit(f"unknown baseline: {args.baseline}")


def _run_live(env, controller, env_cfg, args) -> dict:
    """Drive the plant in a real-time window until stopped.

    The control loop is paced to wall-clock time (``control_dt`` per step) so the
    animation runs at true speed instead of as fast as the CPU allows, which
    would look like a blur.  Returns a small summary of what was shown.
    """
    import time
    from collections import deque

    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
    from pendulum_rl.live_view import LiveViewer

    viewer = LiveViewer(
        env_cfg,
        fps=min(args.fps, 60),
        title=f"inverted pendulum — {args.baseline or args.checkpoint.name if args.checkpoint else 'baseline'}",
        snapshot_path=None,
    )
    if viewer.mode == "off":
        print("live view unavailable (no display); falling back to a recorded video")
        return {"mode": "off"}

    print(f"[live-view] showing a real-time window ({env_cfg.control_dt * 1000:.0f} ms per step).")
    if env_cfg.max_episode_steps is None:
        print("            no time limit: the episode runs until the pole actually falls.")
    else:
        print(f"            episode cap = {env_cfg.max_episode_steps} steps "
              f"({env_cfg.max_episode_steps * env_cfg.control_dt:.0f} s, --max-steps)")
    print("            close the window to stop, or press Ctrl+C.")
    episodes_shown, returns, durations = 0, [], []
    failures: list[str] = []
    theta_window: deque[float] = deque(maxlen=100)
    history: dict[str, list[float]] = {"theta": [], "x": [], "u": [], "reward": []}
    theta_sq_all: list[float] = []
    action_abs_all: list[float] = []
    first_episode_done = False
    t0 = time.perf_counter()
    try:
        while True:
            obs, _ = env.reset(seed=args.seed + episodes_shown)
            ep_return, step = 0.0, 0
            while True:
                loop_start = time.perf_counter()
                action = controller(env, obs)
                obs, reward, terminated, truncated, info = env.step([action])
                ep_return += reward
                step += 1
                theta_sq_all.append(info["theta"] ** 2)
                action_abs_all.append(abs(action))
                if not first_episode_done:
                    history["theta"].append(info["theta"])
                    history["x"].append(info["x"])
                    history["u"].append(action)
                    history["reward"].append(reward)
                if viewer.should_draw():
                    theta_window.append(abs(info["theta"]))
                    held = step * env_cfg.control_dt
                    viewer.draw(
                        x=info["x"],
                        theta=info["theta"],
                        action=action,
                        reward=reward,
                        step=step,
                        episode_return=ep_return,
                        episode=episodes_shown,
                        iteration=0,
                        env_steps=0,
                        success_rate=(np.mean(durations) if durations else float("nan")),
                        # rolling |theta|, not the instantaneous value that is
                        # already shown on the line above it
                        mean_abs_theta_deg=float(np.degrees(np.mean(theta_window))),
                        hold_time_s=held,
                    )
                    viewer.draw_status(
                        f"episode {episodes_shown + 1}: holding for {held:6.1f} s"
                        + (f"   (previous episode: {durations[-1]:.1f} s)" if durations else ""),
                        "runs until the pole falls — close the window to stop",
                    )
                if viewer.mode == "off":  # window closed by the user
                    raise KeyboardInterrupt
                # --render-seconds is a wall-clock stop, checked *inside* the step
                # loop: a good policy never ends an episode on its own, so testing
                # it only between episodes would never fire.
                if args.render_seconds and time.perf_counter() - t0 >= args.render_seconds:
                    raise KeyboardInterrupt
                # pace to real time
                slack = env_cfg.control_dt - (time.perf_counter() - loop_start)
                if slack > 0:
                    time.sleep(slack)
                if (terminated or truncated) and args.stop_on_failure:
                    break
            returns.append(ep_return)
            durations.append(step * env_cfg.control_dt)
            episodes_shown += 1
            if terminated:
                failures.append("fell over" if info.get("fell_over") else (
                    "out of rail" if info.get("out_of_rail") else "diverged"))
            elif truncated:
                failures.append("time cap")
            else:
                failures.append("still standing")
            first_episode_done = True
            print(f"[live-view] episode {episodes_shown}: held {durations[-1]:.1f} s "
                  f"({step} steps), return {ep_return:.1f}, ending: {failures[-1]}")
            if args.render_seconds and time.perf_counter() - t0 >= args.render_seconds:
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        # Keep a partially observed episode: with --render-seconds or a manual
        # window close the run usually stops mid-episode, and dropping that hold
        # time would throw away the very number the user was watching.
        if step > 0 and not (terminated or truncated):
            returns.append(ep_return)
            durations.append(step * env_cfg.control_dt)
            episodes_shown += 1
            failures.append("stopped by user")
            print(f"[live-view] stopped mid-episode after {durations[-1]:.1f} s "
                  f"({step} steps), still standing")
    finally:
        mode_at_exit = viewer.mode
        viewer.close()
    # NOTE: capture the mode *before* close() — closing sets it to "off", which
    # used to make the caller think the live run had failed and fall back to the
    # head-less batch statistics.
    summary = {
        "mode": mode_at_exit,
        "episodes": episodes_shown,
        "returns": returns,
        "durations": durations,
        "failures": failures,
        "history": history,
        "theta_rms_deg": float(np.degrees(np.sqrt(np.mean(theta_sq_all)))) if theta_sq_all else float("nan"),
        "mean_abs_action_frac": (float(np.mean(action_abs_all) / env_cfg.action_limit)
                                 if action_abs_all else float("nan")),
        "success_rate": 1.0 if failures and all(f not in ("fell over", "out of rail", "diverged") for f in failures) else 0.0,
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "mean_hold_s": float(np.mean(durations)) if durations else float("nan"),
        "max_hold_s": float(np.max(durations)) if durations else float("nan"),
        "frames": viewer.frames,
        "seconds": time.perf_counter() - t0,
    }
    print(
        f"[live-view] {episodes_shown} episodes, {viewer.frames:,} frames, "
        f"{summary['seconds']:.1f} s wall clock; mean hold "
        f"{summary['mean_hold_s']:.1f} s (longest {summary['max_hold_s']:.1f} s)"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_dirs()

    meta: dict = {}
    if args.checkpoint is not None:
        import torch

        meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    elif args.config_from is not None:
        import torch

        meta = torch.load(args.config_from, map_location="cpu", weights_only=False)

    raw_env_cfg: dict = meta.get("env_config", {}) if meta else {}
    model = args.model or raw_env_cfg.get("model", "cart")
    init_mode = args.init_mode or raw_env_cfg.get("init_mode", "upright")
    # Rebuild the *exact* plant the policy was trained on, then override only
    # what the command line asked for.  Hand-rolling this dict used to silently
    # fall back to the class defaults (e.g. max_force 15 instead of 50), which
    # quietly evaluates the policy on a different system than it learned.
    if meta:
        env_cfg = env_config_from_checkpoint(
            meta,
            init_mode=init_mode,
            model=model,
            terminate_on_limit=False,
        )
    else:
        env_cfg = EnvConfig(
            model=model,
            init_mode=init_mode,
            terminate_on_limit=False,
        )
    # Command line wins over whatever the checkpoint recorded, and a swing-up
    # evaluation must not terminate on the angle at all.  NOTE: the flag is in
    # DEGREES (like --init-angle-limit); forgetting the conversion silently made
    # the threshold ~3400 deg, i.e. it never fired and no episode ever "fell".
    if args.terminate_angle is not None:
        env_cfg = replace(
            env_cfg,
            terminate_angle=(float(np.radians(args.terminate_angle))
                             if args.terminate_angle > 0 else None),
        )
    if args.init_angle_limit is not None:
        env_cfg = replace(
            env_cfg,
            init_mode="random",
            init_angle_limit=float(np.radians(args.init_angle_limit)),
            init_rate_limit=(args.init_rate_limit if args.init_rate_limit is not None else 0.5),
        )
    elif args.init_rate_limit is not None:
        env_cfg = replace(env_cfg, init_rate_limit=args.init_rate_limit)
    if init_mode == "hanging" and env_cfg.terminate_angle is not None:
        print("[warn] hanging start with a terminate angle: the episode ends as soon as the "
              "pole leaves the upright window, so swing-up cannot be observed "
              "(pass --terminate-angle 0 to disable)")
    # Resolve the per-episode horizon and apply it unconditionally: an explicit
    # --max-steps wins, otherwise --max-seconds, otherwise the horizon the
    # checkpoint recorded.  Defaulting to the *training* horizon matters: a much
    # longer budget folds several attempts into one "episode" and makes the
    # headline numbers (mean hold, "never fell") look far better than reality.
    # NOTE: this must overwrite whatever the checkpoint recorded.  Doing it behind
    # a `if args.max_steps != env_cfg.max_episode_steps` guard silently kept a
    # stored ``None`` (both sides were None) and the run never ended.
    if args.max_steps is not None:
        horizon = args.max_steps
    elif args.max_seconds is not None and args.max_seconds > 0:
        horizon = int(round(args.max_seconds / env_cfg.control_dt))
    elif args.max_seconds == 0:
        horizon = None
    else:
        horizon = env_cfg.max_episode_steps  # checkpoint default (training horizon)
    env_cfg = replace(env_cfg, max_episode_steps=horizon)
    env_cfg = replace(env_cfg, max_episode_steps=horizon)
    env = InvertedPendulumEnv(env_cfg)
    name, controller = build_controller(args, env)

    # ------------------------------------------------- live window (optional)
    # When rendering, the window *is* the evaluation: it already collects hold
    # times, so running a head-less batch first would just do the same work twice
    # (and an uncapped batch can take a very long time).
    live_status = None
    if args.render:
        live_status = _run_live(env, controller, env_cfg, args)
        if live_status.get("mode") == "off":
            print("[live-view] falling back to a head-less batch evaluation")

    use_live_stats = bool(live_status and live_status.get("mode") != "off")
    if use_live_stats:
        returns = list(live_status["returns"])
        lengths = [int(round(d / env_cfg.control_dt)) for d in live_status["durations"]]
        failures = list(live_status["failures"])
        angle_rms = [np.radians(live_status["theta_rms_deg"])]
        actions = [0.0]
        energies = [live_status["mean_abs_action_frac"]]
        successes = [live_status["success_rate"]]
        first_actions = []
        history = live_status["history"]
        if history.get("theta") and "success_angle" not in history:
            history["success_angle"] = env_cfg.success_angle
            history["x_limit"] = env_cfg.x_limit
    else:
        # ------------------------------------------------------------ rollouts
        # The episode ends when the *plant* fails, not on the training clock, so
        # the headline number is "how long did it stay up".  The horizon is only a
        # safety valve for head-less batch runs.
        step_cap = env_cfg.max_episode_steps if env_cfg.max_episode_steps is not None else 10**9
        returns, lengths, successes, angle_rms, actions, energies = [], [], [], [], [], []
        failures: list[str] = []
        first_actions: list[float] = []
        history: dict[str, list[float]] = {"theta": [], "x": [], "u": [], "reward": []}
        wall_start = time.perf_counter()
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            episode_return = 0.0
            theta_sq = []
            ep_actions = []
            info: dict = {}
            terminated = truncated = False
            wall_stop = False
            for _ in range(step_cap):
                action = controller(env, obs)
                obs, reward, terminated, truncated, info = env.step([action])
                episode_return += reward
                theta_sq.append(info["theta"] ** 2)
                ep_actions.append(action)
                if episode == 0:
                    history["theta"].append(info["theta"])
                    history["x"].append(info["x"])
                    history["u"].append(action)
                    history["reward"].append(reward)
                if args.stop_on_failure and (terminated or truncated):
                    break
                # Wall-clock backstop, checked *inside* the step loop: with no step
                # cap and a policy that never falls, checking only between episodes
                # would never fire.  It is deliberately not reported as a plant
                # failure, and success is judged on the env's own streak counter.
                if args.wall_timeout and time.perf_counter() - wall_start > args.wall_timeout:
                    print(f"[warn] --wall-timeout {args.wall_timeout:.0f}s reached at step "
                          f"{len(theta_sq)} of episode {episode + 1}; stopping (never fails)")
                    wall_stop = True
                    break
            returns.append(episode_return)
            lengths.append(len(theta_sq))
            # success = the env's own criterion, which is simply whether it ever
            # held |theta| < success_angle continuously for success_steps
            successes.append(bool(info.get("is_success", False)) or bool(info.get("balanced_streak", 0) >= env_cfg.success_steps))
            angle_rms.append(float(np.sqrt(np.mean(theta_sq))))
            actions.append(float(np.mean(np.abs(ep_actions))))
            energies.append(float(np.mean([abs(a) for a in ep_actions]) / env_cfg.action_limit))
            if terminated:
                failures.append("fell over" if info.get("fell_over") else ("out of rail" if info.get("out_of_rail") else "diverged"))
            elif truncated:
                failures.append("time cap")
            elif wall_stop:
                failures.append("wall-clock stop")
            else:
                failures.append("still standing")
            if episode == 0:
                first_actions = ep_actions
            if wall_stop:
                break
        if len(lengths) < args.episodes:
            args.episodes = len(lengths)
        history["success_angle"] = env_cfg.success_angle
        history["x_limit"] = env_cfg.x_limit
    uncapped = env_cfg.max_episode_steps is None
    survived = [length * env_cfg.control_dt for length in lengths]
    # An episode counts as "did not fall" when it ended for a non-physical reason:
    # hitting the step/seconds budget, the wall-clock backstop, or the user
    # stopping the run.  Everything else (fell over / out of rail / diverged) is
    # a real failure.  A bare "not in FELL" test would wrongly count 'time cap'
    # episodes as clean, which made "never fell" read 20/20 while the mean hold
    # time showed that episodes were in fact ending early.
    NOT_A_FALL = {"time cap", "wall-clock stop", "stopped by user", "still standing"}
    still_up = sum(1 for f in failures if f in NOT_A_FALL)
    n_fell = len(failures) - still_up

    summary = {
        "controller": name,
        "model": model,
        "init_mode": init_mode,
        "episodes": args.episodes,
        "uncapped": uncapped,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "mean_survival_s": float(np.mean(survived)),
        "max_survival_s": float(np.max(survived)),
        "success_rate": float(np.mean(successes)),
        "episodes_still_standing": still_up,
        "failures": failures,
        "theta_rms_deg": float(np.degrees(np.mean(angle_rms))),
        "mean_abs_action_frac": float(np.mean(energies)),
    }

    # ------------------------------------------------------------------ video
    # With a live window the run is *shown* rather than recorded: rendering a GIF
    # from the same episode would double the work for no extra information.
    video_path = None
    if args.no_video or (args.render and not args.video) or not first_actions:
        video_path = None
    else:
        video_path = args.video or (VIDEO_DIR / f"{name.replace(':', '_')}_{init_mode}.gif")
        title = f"{name} | {model} | init={init_mode} | mean return {summary['mean_return']:.1f}"
        # Same seed as episode 0, otherwise the replayed actions are applied to
        # a different initial state and the video shows the wrong trajectory.
        video_path, _ = render_episode(env, first_actions, video_path, fps=args.fps, title=title, seed=args.seed)

    # ------------------------------------------------- time-series plot (ep 0)
    plot_path = None
    if not args.no_plot and history.get("theta"):
        from pendulum_rl.rendering import plot_episode

        plot_path = args.plot or (VIDEO_DIR / f"{name.replace(':', '_')}_{init_mode}_trajectory.png")
        plot_episode(
            history,
            plot_path,
            title=f"{name} | {model} | init={init_mode} | episode 0: held {survived[0]:.1f} s, return {returns[0]:.1f}",
            control_dt=env_cfg.control_dt,
            action_limit=env_cfg.action_limit,
        )

    # ----------------------------------------------------------------- report
    print("=" * 70)
    print(f"controller        : {name}")
    print(f"plant             : {model} ({env_cfg.action_mode}), init={init_mode}, seed={args.seed}")
    n_eps = len(lengths)
    if env_cfg.max_episode_steps is None:
        print(f"episodes          : {n_eps}, no time limit (each runs until the plant fails)")
    else:
        print(f"episodes          : {n_eps} x up to {env_cfg.max_episode_steps} steps "
              f"({env_cfg.max_episode_steps * env_cfg.control_dt:.1f} s each; "
              f"{'--max-steps' if args.max_steps is not None else '--max-seconds budget'})")
    print(f"mean hold time    : {summary['mean_survival_s']:8.2f} s   "
          f"(longest {summary['max_survival_s']:.2f} s, {summary['mean_length']:.0f} steps)")
    ends = set(summary["failures"])
    n_fell_ep = len(summary["failures"]) - still_up
    horizon_s = (env_cfg.max_episode_steps or 0) * env_cfg.control_dt
    if ends == {"time cap"}:
        print(f"episode endings   : all {n_eps} episodes ran out the {horizon_s:.0f} s budget "
              f"while still standing - none fell")
    elif ends == {"wall-clock stop"}:
        print(f"episode endings   : --wall-timeout {args.wall_timeout:.0f}s stopped the run, "
              f"nothing fell")
    elif ends == {"still standing"}:
        print("episode endings   : no time limit and no failure - nothing fell")
    elif ends == {"stopped by user"}:
        print(f"episode endings   : stopped by the user after {summary['mean_survival_s']:.1f} s, "
              f"still standing")
    else:
        counts: dict[str, int] = {}
        for reason in summary["failures"]:
            counts[reason] = counts.get(reason, 0) + 1
        print("episode endings   : " + ", ".join(f"{r} x{c}" for r, c in counts.items()))
    print(f"episodes that fell: {n_fell_ep}/{n_eps}"
          + (f"   (mean hold {summary['mean_survival_s']:.1f} s, longest "
             f"{summary['max_survival_s']:.1f} s)" if n_eps else ""))
    print(f"mean return       : {summary['mean_return']:8.2f} +- {summary['std_return']:.2f}")
    print(f"theta RMS         : {summary['theta_rms_deg']:8.2f} deg")
    print(f"mean |u| / u_max  : {summary['mean_abs_action_frac']:8.3f}")
    print(f"success rate      : {summary['success_rate']:8.1%}  "
          f"(|theta| < {env_cfg.success_angle} rad for {env_cfg.success_steps} steps)")
    if video_path:
        print(f"video             : {video_path}")
    if plot_path:
        print(f"trajectory plot   : {plot_path}")
    if live_status:
        print(f"live view         : {live_status['episodes']} episodes, "
              f"{live_status['frames']:,} frames, {live_status['seconds']:.1f} s wall clock, "
              f"mean hold {live_status['mean_hold_s']:.1f} s")
    print("=" * 70)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        # drop the bulky per-step trajectory and the per-episode arrays' history
        payload = {k: v for k, v in summary.items() if k != "history"}
        if live_status:
            payload["live"] = {
                k: v for k, v in live_status.items() if k not in ("history", "returns", "durations")
            }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
