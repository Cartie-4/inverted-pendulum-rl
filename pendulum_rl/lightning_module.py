"""PPO training driven by PyTorch Lightning.

Design notes
------------
*   PPO is an on-policy algorithm, so the "dataset" of an epoch *is* a fresh
    rollout collected with the current policy.  We therefore use Lightning's
    ``LightningDataModule`` to define one iteration worth of experience
    (``rollout_steps * num_envs`` transitions, regenerated every epoch via
    ``reload_dataloaders_every_n_epochs=1``) and ``LightningModule.training_step``
    to consume it.
*   Lightning 2.x runs ``training_step`` under ``torch.no_grad()`` when
    ``automatic_optimization = False``, so the gradient-enabled section is
    wrapped in ``torch.enable_grad()`` and the optimiser is stepped by hand:
    the actual PPO update (multiple epochs x minibatches) lives inside
    :class:`~pendulum_rl.agents.ppo.PPOAgent`.
*   Environments are stepped in-process with plain Python loops through a
    lightweight vector wrapper.  The plant is cheap (RK4 on a 2-DOF system), so
    numpy vectorisation buys little while hurting readability.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch_lightning import Callback, LightningDataModule, LightningModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from .agents.ppo import PPOAgent, PPOConfig, RolloutBuffer
from .envs.inverted_pendulum import EnvConfig
from .utils import OUTPUT_DIR, MetricsWriter, init_fields
from .vector_env import SyncVectorEnv


# ----------------------------------------------------------------- data module
class RolloutDataModule(LightningDataModule):
    """Owns the live environments; one epoch == one fresh PPO rollout."""

    def __init__(self, train_cfgs: list[EnvConfig], val_cfgs: list[EnvConfig], rollout_steps: int, seed: int = 0):
        super().__init__()
        self.train_cfgs = train_cfgs
        self.val_cfgs = val_cfgs
        self.rollout_steps = rollout_steps
        self.seed = seed
        self.train_envs = SyncVectorEnv(train_cfgs, [seed + i for i in range(len(train_cfgs))])
        self.val_envs = SyncVectorEnv(val_cfgs, [seed + 10_000 + i for i in range(len(val_cfgs))])

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002 - required by the API
        return None

    def train_dataloader(self):
        # Lightning just needs the length; the actual data is the live rollout.
        return list(range(1))

    def val_dataloader(self):
        return list(range(1))


# ------------------------------------------------------------------------ module
@dataclass
class TrainConfig:
    """Everything a training run needs, so a run is reproducible from one object."""

    # environment / task
    model: str = "cart"
    init_mode: str = "upright"
    #: initial-state curriculum (see EnvConfig.init_angle_limit)
    init_angle_limit: float | None = None
    init_rate_limit: float | None = None
    init_angle_center: float = 0.0
    #: Optional gentler initial |theta_dot|, |x_dot| for the warm-up window.  The
    #: angle band and the starting rates are two different difficulties and the
    #: band alone does not make the interesting behaviour visible: from 75 deg at
    #: rest the pole must be *let go* to gain any energy at all, whereas a random
    #: +-0.5 rad/s initial rate lets the policy get away with fighting for the
    #: catch most of the time.  Starting slow and ramping the rate up keeps the
    #: "commit to the fall" decision in the gradient.  See init_rate_limit_at.
    warmup_init_rate_limit: float | None = None
    max_episode_steps: int | None = 500
    max_force: float = 50.0
    max_torque: float = 2.5
    action_mode: str = "force"
    control_dt: float = 0.02
    sim_dt: float = 0.002
    a_theta_power: int = 2
    w_theta: float = 3.0
    w_x: float = 0.1
    w_omega: float = 0.1
    w_u: float = 0.001
    terminate_on_limit: bool = False
    #: End the episode when the pole falls past this angle [rad].  ``None`` means
    #: "never end because of the angle", which is required for swing-up (the pole
    #: has to sweep through pi/2).  See EnvConfig.terminate_angle.
    terminate_angle: float | None = None
    # curriculum: switch init mode from `warmup_init_mode` to `init_mode` after
    # `curriculum_fraction` of the run (e.g. learn swing-up, then random starts).
    warmup_init_mode: str | None = None
    curriculum_fraction: float = 0.4

    # reward shaping: potential-based, off by default so the historical reward is
    # reproduced bit-for-bit.  See EnvConfig.shaping for why this cannot change
    # the optimal policy.
    shaping: str = "none"
    shape_coef: float = 1.0
    #: Discount of the shaping term.  Set from ``gamma`` unless overridden, so the
    #: invariance statement holds for the MDP PPO is actually solving.
    shape_gamma: float | None = None

    # algorithm
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 1e-3
    entropy_coef_final: float = 1e-4
    epochs_per_update: int = 10
    minibatches: int = 16
    target_kl: float = 0.1
    hidden_sizes: tuple[int, ...] = (256, 256)
    log_std_init: float = -3.0

    # loops
    num_envs: int = 8
    rollout_steps: int = 256
    max_epochs: int = 300
    val_num_envs: int = 8
    val_max_episode_steps: int = 500
    check_val_every_n_epoch: int = 10
    seed: int = 0
    run_name: str = "ppo_cart"
    device: str = "auto"

    # live visualisation (see pendulum_rl/live_view.py)
    render: bool = False
    render_fps: float = 50.0
    render_env: int = 0

    #: derived, filled by :meth:`ppo_config` / :meth:`env_config`
    steps_per_epoch: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.steps_per_epoch = self.num_envs * self.rollout_steps

    def env_config(self, init_mode: str | None = None, for_eval: bool = False) -> EnvConfig:
        return EnvConfig(
            model=self.model,
            init_mode=init_mode or self.init_mode,
            init_angle_limit=self.init_angle_limit,
            init_rate_limit=self.init_rate_limit,
            init_angle_center=self.init_angle_center,
            max_episode_steps=self.val_max_episode_steps if for_eval else self.max_episode_steps,
            max_force=self.max_force,
            max_torque=self.max_torque,
            action_mode=self.action_mode,
            control_dt=self.control_dt,
            sim_dt=self.sim_dt,
            a_theta_power=self.a_theta_power,
            w_theta=self.w_theta,
            w_x=self.w_x,
            w_omega=self.w_omega,
            w_u=self.w_u,
            terminate_on_limit=False if for_eval else self.terminate_on_limit,
            # Never apply the "fell over" rule while the pole has to swing up.
            terminate_angle=self.effective_terminate_angle(for_eval=for_eval),
            shaping=self.shaping,
            shape_coef=self.shape_coef,
            shape_gamma=self.gamma if self.shape_gamma is None else self.shape_gamma,
        )

    def effective_terminate_angle(self, for_eval: bool = False) -> float | None:
        """Resolve the termination angle, refusing it for swing-up tasks.

        Training on ``hanging``/``random`` needs the pole to be *allowed* to fall
        (it starts fallen), so the rule is dropped whenever a hanging start is
        part of the task definition; for pure balance tasks it applies to both
        training and evaluation, which is what makes "how long did it stay up"
        the thing the metric measures.
        """
        del for_eval  # kept for signature stability / future per-phase overrides
        if self.terminate_angle is None:
            return None
        swing_up = self.init_mode == "hanging" or self.warmup_init_mode == "hanging"
        return None if swing_up else self.terminate_angle

    def init_mode_at(self, progress: float) -> str:
        if self.warmup_init_mode and progress < self.curriculum_fraction:
            return self.warmup_init_mode
        return self.init_mode

    def init_rate_limit_at(self, progress: float) -> float | None:
        """Initial |theta_dot| / |x_dot| bound for this point in the run.

        Shares the warm-up window with :meth:`init_mode_at`, so a single
        ``--curriculum-fraction`` describes both ramps and the two cannot drift
        apart.  ``None`` means "keep whatever the env config has".
        """
        if self.warmup_init_rate_limit is not None and progress < self.curriculum_fraction:
            return float(self.warmup_init_rate_limit)
        return self.init_rate_limit

    def ppo_config(self, obs_dim: int, action_dim: int) -> PPOConfig:
        return PPOConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            entropy_coef=self.entropy_coef,
            entropy_coef_final=self.entropy_coef_final,
            epochs_per_update=self.epochs_per_update,
            minibatches=self.minibatches,
            target_kl=self.target_kl,
            hidden_sizes=tuple(self.hidden_sizes),
            log_std_init=self.log_std_init,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CheckpointEveryEpoch(Callback):
    """Write ``last.ckpt`` every epoch and ``best.ckpt`` on the best metric.

    ``ModelCheckpoint`` with ``monitor=...`` silently produced *nothing* in this
    project (its ``best_model_path`` stayed ``None`` and no files appeared
    besides an occasional ``last.ckpt``), which meant a long run that was
    interrupted lost all of its weights.  This callback makes checkpointing
    explicit and boring:

    * ``last.ckpt``  — overwritten every epoch, so an interrupted run is never a
      total loss;
    * ``best.ckpt``  — only when the monitored metric improves.
    """

    def __init__(self, dirpath, monitor: str = "val/mean_return", mode: str = "max"):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.greater_is_better = mode == "max"
        self.best = float("-inf") if self.greater_is_better else float("inf")
        self.best_epoch = -1

    @property
    def best_path(self) -> Path:
        return self.dirpath / "best.ckpt"

    @property
    def last_path(self) -> Path:
        return self.dirpath / "last.ckpt"

    def _score(self, trainer, pl_module) -> float | None:
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return None
        try:
            score = float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None
        return score if np.isfinite(score) else None

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(self.last_path)
        score = self._score(trainer, pl_module)
        if score is None:
            return
        improved = score > self.best if self.greater_is_better else score < self.best
        if improved:
            self.best = score
            self.best_epoch = int(trainer.current_epoch)
            trainer.save_checkpoint(self.best_path)
            print(f"[checkpoint] new best {self.monitor}={score:.2f} at iteration {self.best_epoch + 1}")

    def on_train_end(self, trainer, pl_module) -> None:
        # guarantee a checkpoint even when the loop ran without validation
        self.dirpath.mkdir(parents=True, exist_ok=True)
        if not self.last_path.exists():
            trainer.save_checkpoint(self.last_path)


class PPOLightningModule(LightningModule):
    """Lightning wrapper around :class:`PPOAgent`."""

    def __init__(self, cfg: TrainConfig, datamodule: RolloutDataModule, metrics_path=None):
        super().__init__()
        self.cfg = cfg
        self.dm = datamodule
        self.save_hyperparameters(cfg.to_dict(), ignore=["datamodule"])
        self.automatic_optimization = False

        train_envs = datamodule.train_envs
        env_cfg = train_envs.cfg
        self.obs_dim = env_cfg.obs_dim
        self.action_dim = env_cfg.action_dim
        self.action_limit = float(env_cfg.action_limit)

        self.agent = PPOAgent(cfg.ppo_config(self.obs_dim, self.action_dim), device=cfg.device)
        self.buffer = RolloutBuffer(cfg.rollout_steps, cfg.num_envs, self.obs_dim, self.action_dim)
        self.metrics = MetricsWriter(metrics_path) if metrics_path else None

        self._obs = train_envs.obs
        self._ep_return_sum = 0.0
        self._ep_count = 0
        self._epoch_start = time.perf_counter()
        self._total_env_steps = 0
        self._last_success_rate = float("nan")
        self._last_mean_abs_theta = float("nan")
        # rolling window of |theta| used by the live HUD: an instantaneous
        # per-environment value is meaningless next to the displayed angle, so the
        # HUD shows "how steady has it been lately" instead.
        self._theta_window: deque[float] = deque(maxlen=100)
        self._theta_window_batch: deque[float] = deque(maxlen=100)

        # --- live viewer -------------------------------------------------
        self.viewer = None
        if cfg.render:
            from .live_view import LiveViewer

            self.viewer = LiveViewer(
                env_cfg,
                fps=cfg.render_fps,
                title=f"inverted pendulum — {cfg.run_name} ({cfg.model}, init={cfg.init_mode})",
                snapshot_path=(OUTPUT_DIR / "live" / f"{cfg.run_name}.png"),
            )
        self._view_ep_return = 0.0
        self._view_episode = 0
        self._view_step = 0

    # ------------------------------------------------------------- optimiser
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.agent.optimizer

    def on_train_epoch_start(self) -> None:
        self._epoch_start = time.perf_counter()
        progress = self.trainer.current_epoch / max(1, self.cfg.max_epochs)
        # Entropy annealing + curriculum on the initial state distribution.
        self.agent.cfg.lr_lambda(progress)
        mode = self.cfg.init_mode_at(progress)
        # BOTH vector envs follow the curriculum.  Updating only the training envs
        # (which is what this did) left validation pinned on the warm-up
        # distribution for the whole run, so `val/mean_return` -- the metric that
        # picks `best.ckpt` -- kept scoring "swing up from hanging" while the
        # policy was actually being trained on the post-switch distribution.  A
        # run could then finish with a best checkpoint chosen by a task it is no
        # longer being asked to solve.
        rate = self.cfg.init_rate_limit_at(progress)
        for envs in (self.dm.train_envs, self.dm.val_envs):
            envs.set_init_mode(mode)
            if rate is not None:
                envs.set_init_rate_limit(rate)

    # ------------------------------------------------------------ train step
    def training_step(self, batch, batch_idx):  # noqa: ARG002 - batch is a placeholder
        cfg = self.cfg
        envs = self.dm.train_envs
        rollout_start = time.perf_counter()

        # ---- 1. collect one iteration worth of experience with the policy ---
        plant = envs.plant
        view_i = self.cfg.render_env if self.viewer is not None else None
        for t in range(cfg.rollout_steps):
            actions, log_probs, values = self.agent.sample_actions(self._obs)
            clipped = np.clip(actions, -1.0, 1.0).astype(np.float32)
            env_actions = (clipped * self.action_limit).astype(np.float32)
            next_obs, rewards, dones = envs.step(env_actions)
            self.agent.obs_rms.update(self._obs)
            # GAE takes the *termination* mask, not `done` (a truncated episode must keep
            # bootstrapping), and the bootstrap value comes from the state the transition
            # actually reached -- `terminal_obs`, the observation captured before the
            # vector env reset the finished envs.  Using the post-reset observation would
            # value a rail exit as "a fresh episode starts" and the policy would seek the
            # rail.
            self.buffer.add(
                self._obs, clipped, log_probs, values, rewards,
                np.asarray(envs.last_info["terminated"], dtype=np.float32),
                self.agent.value(envs.last_info["terminal_obs"]),
            )
            self._obs = next_obs

            # --- live view: follow one environment, drawn with the *pre-step*
            # state and reset across episode boundaries so the window shows a
            # continuous episode rather than frame-skipped jumps.
            if view_i is not None:
                self._view_ep_return += float(rewards[view_i])
                self._view_step += 1
                if self.viewer.should_draw():
                    # rolling means over the last 100 steps (2 s of sim time):
                    # the followed env alone, and the whole parallel batch
                    self._theta_window.append(abs(float(plant.state.theta[view_i])))
                    self._theta_window_batch.append(float(np.mean(np.abs(plant.state.theta))))
                    self.viewer.draw(
                        x=float(plant.state.x[view_i]),
                        theta=float(plant.state.theta[view_i]),
                        action=float(np.asarray(env_actions[view_i]).reshape(-1)[0]),
                        reward=float(rewards[view_i]),
                        step=self._view_step,
                        episode_return=self._view_ep_return,
                        episode=self._view_episode,
                        iteration=self.trainer.current_epoch + 1,
                        env_steps=self._total_env_steps + t * cfg.num_envs,
                        success_rate=self._last_success_rate,
                        mean_abs_theta_deg=float(np.degrees(np.mean(self._theta_window))),
                        batch_mean_abs_theta_deg=float(np.degrees(np.mean(self._theta_window_batch))),
                        rollout_steps=cfg.rollout_steps,
                    )
                if dones[view_i]:
                    self._view_episode += 1
                    self._view_ep_return = 0.0
                    self._view_step = 0
        self._total_env_steps += cfg.steps_per_epoch
        rollout_time = time.perf_counter() - rollout_start

        # ---- 2. PPO update -------------------------------------------------
        # The window is refreshed (with a status line) while the gradients are
        # computed, otherwise Tk stops repainting and the run looks frozen.
        if self.viewer is not None:
            self.viewer.draw_status(
                f"iteration {self.trainer.current_epoch + 1}/{cfg.max_epochs}: PPO update "
                f"({cfg.epochs_per_update} epochs x {cfg.minibatches} minibatches)",
                f"rollout collected: {cfg.steps_per_epoch:,} transitions",
            )
        update_start = time.perf_counter()
        with torch.enable_grad():
            stats = self.agent.update(self.buffer, cfg.rollout_steps, cfg.num_envs)
        update_time = time.perf_counter() - update_start
        self.buffer.ptr = 0

        # ---- 3. logging ----------------------------------------------------
        returned = envs.finished_returns
        if returned:
            self._ep_return_sum += float(np.sum(returned))
            self._ep_count += len(returned)
        mean_return = self._ep_return_sum / max(1, self._ep_count) if self._ep_count else float("nan")
        mean_length = float(np.mean(envs.finished_lengths)) if envs.finished_lengths else float("nan")
        success_rate = float(np.mean(envs.finished_success)) if envs.finished_success else float("nan")
        if envs.finished_returns:
            self._ep_return_sum = 0.0
            self._ep_count = 0
            envs.finished_returns.clear()
            envs.finished_lengths.clear()
            envs.finished_success.clear()
        # remembered for the live view's HUD
        self._last_success_rate = success_rate
        self._last_mean_abs_theta = float(np.degrees(envs.mean_abs_theta))

        logs = {
            "train/mean_episode_return": mean_return,
            "train/mean_episode_length": mean_length,
            "train/success_rate": success_rate,
            "train/mean_abs_theta_deg": float(np.degrees(envs.mean_abs_theta)),
            "train/policy_std": stats["std"],
            "train/entropy": stats["entropy"],
            "train/approx_kl": stats["approx_kl"],
            "train/clip_fraction": stats["clip_fraction"],
            "train/value_loss": stats["value_loss"],
            "train/loss": stats["loss"],
            "train/entropy_coef": self.agent.cfg.entropy_coef,
            "perf/rollout_s": rollout_time,
            "perf/update_s": update_time,
            "perf/env_steps": float(self._total_env_steps),
        }
        for key, value in logs.items():
            self.log(
                key,
                value,
                prog_bar=key in ("train/mean_episode_return", "train/success_rate"),
                on_step=False,
                on_epoch=True,
                # The "batch" here is a placeholder (the real data is the live
                # rollout), so tell Lightning how to weight the epoch average.
                batch_size=cfg.num_envs,
            )
        if self.metrics:
            self.metrics.write({"epoch": int(self.trainer.current_epoch), "env_steps": self._total_env_steps, **logs})

        loss = torch.tensor(stats["loss"], requires_grad=True)
        return loss

    # ----------------------------------------------------------------- eval
    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        """Deterministic (mean-action) evaluation over complete episodes.

        The validation envs never truncate on the rail limit, so every episode
        runs the full ``val_max_episode_steps`` horizon and the mean return is
        comparable across iterations.
        """
        envs = self.dm.val_envs
        cfg = envs.cfg
        envs.reset()
        obs = envs.obs
        returns = np.zeros(envs.num_envs)
        lengths = np.zeros(envs.num_envs, dtype=int)
        max_streak = np.zeros(envs.num_envs)
        #: Whether an env has already contributed its first episode to the metrics.
        #: The vector env rolls straight into a new episode on ``done``, so without
        #: this mask the partial rollover episode at the horizon would be counted too.
        scored = np.zeros(envs.num_envs, dtype=bool)
        episode_returns: list[float] = []
        episode_success: list[bool] = []
        episode_lengths: list[int] = []

        for _ in range(self.cfg.val_max_episode_steps):
            actions, _, _ = self.agent.sample_actions(obs, deterministic=True)
            env_actions = np.clip(actions, -1.0, 1.0) * self.action_limit
            obs, rewards, dones = envs.step(env_actions)
            returns += rewards
            lengths += 1
            # keep the window alive through the (blocking) validation pass
            if self.viewer is not None and self.viewer.should_draw():
                self.viewer.draw_status(
                    f"iteration {self.trainer.current_epoch + 1}: evaluating "
                    f"({int(lengths.max())}/{self.cfg.val_max_episode_steps} steps)",
                )
            # `balanced_streak` comes from the step that was just taken; it is
            # zeroed for envs that restarted, so read it from the info first.
            max_streak = np.maximum(max_streak, envs.last_info["balanced_streak"])
            for i in range(envs.num_envs):
                if dones[i]:
                    if not scored[i]:
                        # Score the env's FIRST episode however short it is.  The old
                        # `lengths[i] >= cfg.success_steps` guard was meant to skip
                        # rollover episodes, but it also discarded every fast failure
                        # and turned both validation metrics into survivorship
                        # statistics: measured on a +-90 deg validation set, 98.9%
                        # reported against 45.5% when every episode counts (108 of 200
                        # episodes died inside 100 steps, all of them failures, and all
                        # of them dropped).  CheckpointEveryEpoch monitors
                        # val/mean_return, so that bias also picked best.ckpt.
                        episode_returns.append(float(returns[i]))
                        episode_success.append(bool(max_streak[i] >= cfg.success_steps))
                        episode_lengths.append(int(lengths[i]))
                        scored[i] = True
                    returns[i] = 0.0
                    lengths[i] = 0
                    max_streak[i] = 0.0

        # Episodes still in flight at the horizon: score them on the return
        # actually collected.  Without this a short/early-terminating policy
        # produces an empty list and the metric becomes NaN, which silently
        # disables the checkpoint monitor.  Envs whose first episode was already
        # scored are excluded: they are mid-rollover and would be double counted.
        in_flight = (lengths > 0) & ~scored
        scored_returns = list(episode_returns) + [float(returns[i]) for i in np.flatnonzero(in_flight)]
        scored_success = list(episode_success) + [
            bool(max_streak[i] >= cfg.success_steps) for i in np.flatnonzero(in_flight)
        ]

        mean_return = float(np.mean(scored_returns))
        success_rate = float(np.mean(scored_success))
        balanced = float(np.mean(max_streak >= cfg.success_steps))
        log_kwargs = {"on_step": False, "on_epoch": True, "batch_size": envs.num_envs}
        self.log("val/mean_return", mean_return, prog_bar=True, **log_kwargs)
        self.log("val/success_rate", success_rate, prog_bar=True, **log_kwargs)
        self.log("val/balanced_env_fraction", balanced, **log_kwargs)
        self.log("val/mean_episode_length", float(np.mean(episode_lengths)) if episode_lengths else float(np.mean(lengths)), **log_kwargs)
        if self.metrics:
            self.metrics.write(
                {
                    "epoch": int(self.trainer.current_epoch),
                    "val": True,
                    "val/mean_return": mean_return,
                    "val/success_rate": success_rate,
                    "val/completed_episodes": len(episode_returns),
                }
            )
        return mean_return

    # ------------------------------------------------------------ checkpoints
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["agent"] = self.agent.state_dict()
        checkpoint["obs_dim"] = self.obs_dim
        checkpoint["action_dim"] = self.action_dim
        checkpoint["action_limit"] = self.action_limit
        checkpoint["train_config"] = self.cfg.to_dict()
        checkpoint["env_config"] = asdict(self.cfg.env_config())

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        agent_state = checkpoint.get("agent")
        if agent_state is not None:
            self.agent.load_state_dict(agent_state)

    def load_pretrained_weights(self, path, strict_model: bool = False) -> None:
        """Warm-start the policy from another run's checkpoint.

        Only the network weights and the observation-normaliser statistics are
        transferred; the optimiser state and the new run's hyper-parameters are
        kept.  This lets a policy that already balances be fine-tuned into one
        that also recovers from hanging / arbitrary states, instead of relearning
        stabilisation from scratch.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        agent_state = state.get("agent")
        if agent_state is None:
            raise ValueError(f"{path} has no 'agent' state to warm-start from")
        try:
            self.agent.model.load_state_dict(agent_state["model"], strict=strict_model)
        except RuntimeError as exc:
            raise ValueError(
                f"warm-start checkpoint is incompatible with this run's architecture: {exc}"
            ) from exc
        if "obs_rms" in agent_state:
            self.agent.obs_rms.load_state_dict(agent_state["obs_rms"])

    def freeze_normalizer(self) -> None:
        self.agent.freeze_normalizer()

    def on_train_end(self) -> None:
        if self.viewer is not None:
            # keep the frame count available for the caller's summary
            self._live_frames = getattr(self.viewer, "frames", 0)
            print(f"[live-view] {self.viewer.frames:,} frames drawn "
                  f"({self.viewer.mode} backend)")
            self.viewer.close()
            self.viewer = None


def env_config_from_checkpoint(ckpt: dict[str, Any], **overrides) -> EnvConfig:
    """Rebuild the plant config stored in a checkpoint.

    Derived fields (``frame_skip``, ``a_theta_offset``) are dropped: they are
    recomputed by ``EnvConfig.__post_init__`` and are not constructor arguments.
    """
    raw = dict(ckpt.get("env_config", {}))
    raw.update(overrides)
    return EnvConfig(**init_fields(EnvConfig, raw))


def load_policy(path, device: str = "cpu"):
    """Rebuild a trained policy from a Lightning checkpoint (inference only)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    train_cfg = dict(ckpt.get("train_config", {}))
    if isinstance(train_cfg.get("hidden_sizes"), list):
        train_cfg["hidden_sizes"] = tuple(train_cfg["hidden_sizes"])
    obs_dim = int(ckpt.get("obs_dim", 6))
    action_dim = int(ckpt.get("action_dim", 1))
    cfg = TrainConfig(**init_fields(TrainConfig, train_cfg))
    ppo_cfg = cfg.ppo_config(obs_dim, action_dim)
    agent = PPOAgent(ppo_cfg, device=device)
    agent.load_state_dict(ckpt["agent"])
    agent.model.eval()
    agent.freeze_normalizer()
    return agent, cfg, ckpt
