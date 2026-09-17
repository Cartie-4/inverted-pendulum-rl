"""Matplotlib based rendering for the inverted pendulum.

Rendering through matplotlib (Agg backend) keeps the project free of GUI
dependencies while still producing PNG frames and GIF/MP4 animations on any
machine, including head-less CI boxes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .envs.inverted_pendulum import InvertedPendulumEnv

FIG_DPI = 100
#: The rail is ~2.4 m long while the pole is only 0.5 m, so rendering the whole
#: rail makes the mechanism nearly invisible.  The main view therefore shows a
#: window around the cart, with the full rail drawn as a guide; a small thumbnail
#: shows the true scale.
FIG_SIZE = (4.4, 3.0)
CART_WINDOW = 1.6  # half-width of the zoomed view [m]


class PendulumRenderer:
    """Draws one frame per `env.step` and can stitch them into an animation."""

    def __init__(self, env: "InvertedPendulumEnv", x_range: float | None = None):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        self._plt = plt
        self.env = env
        cfg = env.cfg
        self.x_range = float(x_range if x_range is not None else cfg.x_limit)
        self.fig, self.ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
        self._setup_axes()
        self._build_artists()

    def _setup_axes(self) -> None:
        cfg = self.env.cfg
        ax = self.ax
        pole_len = cfg.l_pole
        # Zoomed window: enough to see the pole tip swing all the way around.
        ax.set_xlim(-CART_WINDOW, CART_WINDOW)
        ax.set_ylim(-pole_len * 1.35, pole_len * 1.45)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#f4f6f8")
        for spine in ax.spines.values():
            spine.set_visible(False)
        # ground / rail plus world-anchored ticks (they move as the camera follows)
        ax.plot([-CART_WINDOW * 4, CART_WINDOW * 4], [-0.045, -0.045], color="#b0b8c1", lw=2.0, zorder=0)
        self._ticks = []
        for k in range(-8, 9):
            world_x = k * 0.5
            major = k % 2 == 0
            (line,) = ax.plot(
                [world_x, world_x],
                [-0.045, -0.045 + (0.05 if major else 0.03)],
                color="#9fb0bd" if major else "#dde3e8",
                lw=1.6 if major else 1.0,
                zorder=1,
            )
            self._ticks.append((line, world_x))
        self._camera_x = None

    def _build_artists(self) -> None:
        cfg = self.env.cfg
        l = cfg.l_pole
        cart_half = 0.09
        cart_h = 0.06
        self.cart_half, self.cart_h = cart_half, cart_h
        self.cart = self._plt.Rectangle(
            (-cart_half, -cart_h), 2 * cart_half, 2 * cart_h, fc="#3d7ea6", ec="#22415a", zorder=3
        )
        self.ax.add_patch(self.cart)
        (self.pole,) = self.ax.plot([], [], color="#d1495b", lw=4.0, solid_capstyle="round", zorder=4)
        (self.bob,) = self.ax.plot([], [], "o", color="#d1495b", ms=7, zorder=5)
        (self.pivot,) = self.ax.plot([], [], "o", color="#22415a", ms=3.5, zorder=6)
        # upright reference line
        self.ax.plot([0, 0], [0, l * 1.15], color="#c9d1d9", lw=1.0, ls="--", zorder=1)
        self.title = self.ax.set_title("", fontsize=9, color="#333333", pad=3)
        # metrics panel moved *below* the axes so it can never cover the drawing
        self.text = self.fig.text(
            0.02, 0.015, "", fontsize=7.5, color="#333333", va="bottom", ha="left", family="monospace"
        )
        self.fig.tight_layout(rect=(0, 0.16, 1, 1))
        _ = self.x_range

    # ------------------------------------------------------------------ api
    def frame(self, action: float = 0.0, reward: float = 0.0, step: int = 0) -> np.ndarray:
        cfg = self.env.cfg
        x = float(self.env.state[0])
        theta = float(self.env.state[1])
        l = cfg.l_pole
        if cfg.model == "pivot":
            x = 0.0
        # Follow the cart with the camera (see LiveViewer for the same reasoning):
        # otherwise a drifting-but-stable cart walks out of the frame.
        self._set_camera(x if cfg.model == "cart" else 0.0)
        tip_x = x + l * np.sin(theta)
        tip_y = l * np.cos(theta)

        self.cart.set_xy((x - self.cart_half, -self.cart_h))
        self.pole.set_data([x, tip_x], [0.0, tip_y])
        self.bob.set_data([tip_x], [tip_y])
        self.pivot.set_data([x], [0.0])
        if cfg.model == "pivot":
            self.cart.set_visible(False)
        self.title.set_text(f"t = {step * cfg.control_dt:5.2f} s")
        theta_wrapped = np.arctan2(np.sin(theta), np.cos(theta))
        self.text.set_text(
            f"theta = {np.degrees(theta_wrapped):+6.1f} deg    x = {x:+5.2f} m\n"
            f"u     = {action:+7.2f} N"
            f" ({(action / cfg.action_limit) if cfg.action_limit else 0:+5.1%} of limit)"
            f"    r = {reward:+6.2f}"
        )
        return self.buffer()

    def _set_camera(self, cart_x: float) -> None:
        """Keep the cart centred and move the rail ticks instead."""
        if getattr(self, "_camera_x", None) == cart_x:
            return
        self._camera_x = cart_x
        half = CART_WINDOW
        self.ax.set_xlim(cart_x - half, cart_x + half)
        for line, world_x in getattr(self, "_ticks", []):
            line.set_xdata([world_x, world_x])

    def buffer(self) -> np.ndarray:
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        return buf[:, :, :3].copy()

    def close(self) -> None:
        self._plt.close(self.fig)

    # --------------------------------------------------------------- videos
    @staticmethod
    def save_frames(
        frames: Sequence[np.ndarray],
        path: Path,
        fps: int = 50,
        title: str | None = None,
        repeat_last: int = 10,
    ) -> Path | None:
        """Write frames to GIF (always) or MP4 (when imageio-ffmpeg is present)."""
        import imageio.v2 as imageio

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = list(frames)
        if not frames:
            return None
        if repeat_last:
            frames = frames + [frames[-1]] * repeat_last
        if title:
            frames = [_annotate(f, title) for f in frames]
        suffix = path.suffix.lower()
        if suffix == ".gif":
            imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)
            return path
        try:
            imageio.mimsave(path, frames, fps=fps, quality=7)
            return path
        except Exception:  # noqa: BLE001 - fall back to GIF when ffmpeg is missing
            gif = path.with_suffix(".gif")
            imageio.mimsave(gif, frames, duration=1.0 / fps, loop=0)
            return gif


def render_episode(
    env: "InvertedPendulumEnv",
    actions: Iterable[float],
    path: Path,
    fps: int = 50,
    title: str | None = None,
    max_frames: int | None = None,
    seed: int | None = None,
) -> tuple[Path | None, list[np.ndarray]]:
    """Roll out pre-computed actions, render each step, and write a video file.

    ``seed`` must be the same seed that produced ``actions``: without it the
    episode restarts from a *different* random initial state and the rendered
    video shows a trajectory that does not correspond to the recorded actions.
    """
    renderer = PendulumRenderer(env)
    frames: list[np.ndarray] = []
    try:
        env.reset(seed=seed)
        frames.append(renderer.frame(0.0, 0.0, 0))
        for i, action in enumerate(actions, start=1):
            if max_frames is not None and len(frames) >= max_frames:
                break
            _, reward, terminated, truncated, _ = env.step([action])
            frames.append(renderer.frame(action, reward, i))
            if terminated or truncated:
                break
    finally:
        renderer.close()
    return PendulumRenderer.save_frames(frames, path, fps=fps, title=title), frames


def _annotate(frame: np.ndarray, title: str) -> np.ndarray:
    """Burn a caption bar into a frame without needing a font file."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(frame.shape[1] / FIG_DPI, frame.shape[0] / FIG_DPI), dpi=FIG_DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(frame)
    ax.axis("off")
    ax.text(
        0.5,
        0.972,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#111111",
        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5),
    )
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return out


def plot_episode(
    history: dict[str, list[float]],
    path: Path,
    title: str = "",
    control_dt: float = 0.02,
    action_limit: float = 1.0,
) -> Path:
    """Time-series figure for one episode.

    A balancing policy produces a nearly static animation, so the informative
    artefact is the trajectory: pole angle, cart position, control input and the
    per-step reward.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    t = np.arange(len(history["theta"])) * control_dt
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)

    axes[0].plot(t, np.degrees(history["theta"]), color="#d1495b", lw=1.4)
    axes[0].axhline(0, color="#888", lw=0.8, ls="--")
    for sign in (-1, 1):  # success band
        axes[0].axhline(sign * np.degrees(history.get("success_angle", 0.12)), color="#2a9d8f", lw=0.9, ls=":")
    axes[0].set_ylabel("pole angle [deg]")
    axes[0].set_title(title, fontsize=11)

    axes[1].plot(t, history["x"], color="#3d7ea6", lw=1.4, label="cart position x [m]")
    if "x_limit" in history:
        for sign in (-1, 1):
            axes[1].axhline(sign * history["x_limit"], color="#e07a5f", lw=0.9, ls="--")
    axes[1].set_ylabel("cart x [m]")

    axes[2].plot(t, np.asarray(history["u"]) / max(action_limit, 1e-9), color="#6a4c93", lw=1.2)
    axes[2].fill_between(t, -1, 1, color="#eee", zorder=0)
    axes[2].set_ylabel("control  u / u_max")
    axes[2].set_xlabel("time [s]")

    for ax in axes:
        ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
