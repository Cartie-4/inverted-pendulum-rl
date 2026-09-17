"""Live (real-time) rendering window for the inverted pendulum.

Tkinter is used because it ships with CPython and its canvas API is fast enough
to redraw a cart-pole at the control rate (50 Hz) — the whole scene is a
handful of rectangles and lines, so there is no need to depend on pygame or
OpenCV, and no frame-buffer copy is required (unlike the matplotlib path used
for GIF/MP4 export).

The viewer is *optional* and degrades gracefully:

* ``LiveViewer`` opens a real window (``prefer_window=True``) when a display is
  available;
* otherwise it falls back to writing a small PNG periodically so a head-less run
  still leaves something to look at;
* if nothing is available it becomes a no-op, and training continues unchanged.

It is deliberately cheap: the caller skips frames (``should_draw``) so the
window can never throttle the learner, and all drawing happens on the caller's
thread, which keeps Tk happy on every platform.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

#: palette shared with the export renderer
BG = "#f4f6f8"
RAIL = "#b0b8c1"
CART = "#3d7ea6"
CART_EDGE = "#22415a"
POLE = "#d1495b"
GHOST = "#c9d1d9"
TEXT = "#22303c"
GOOD = "#2a9d8f"
BAD = "#e07a5f"


def _fit(value: float, lo: float, hi: float, p0: float, p1: float) -> float:
    """Map ``value`` from [lo, hi] onto pixel range [p0, p1]."""
    if hi <= lo:
        return 0.5 * (p0 + p1)
    t = (value - lo) / (hi - lo)
    return p0 + t * (p1 - p0)


class LiveViewer:
    """Non-blocking Tkinter window that draws the plant in real time."""

    def __init__(
        self,
        cfg: Any,
        width: int = 480,
        height: int = 320,
        fps: float = 50.0,
        title: str = "inverted pendulum — live",
        prefer_window: bool = True,
        snapshot_path: Path | None = None,
    ):
        self.cfg = cfg
        self.width = width
        self.height = height
        self.min_interval = 1.0 / max(fps, 1.0)
        self.title_text = title
        self.snapshot_path = Path(snapshot_path) if snapshot_path else None
        self.mode = "off"
        self.frames = 0
        self._last_draw = 0.0
        self._last_snapshot = 0.0
        self._closed = False
        self._bad_frames = 0
        self.cfg_half_window = max(0.9, min(cfg.x_limit, 1.6)) if cfg.model == "cart" else 1.2

        if prefer_window:
            try:
                self._init_window()
            except Exception as exc:  # noqa: BLE001 - fall back rather than fail a run
                print(f"[live-view] window unavailable ({type(exc).__name__}: {exc}); using snapshots")
                self.mode = "off"
        self._init_snapshot_fallback()

    # ------------------------------------------------------------- window init
    def _init_window(self) -> None:
        import tkinter as tk

        self._tk = tk
        self.root = tk.Tk()
        self.root.title(self.title_text)
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(
            self.root, width=self.width, height=self.height, bg=BG, highlightthickness=0
        )
        self.canvas.pack()
        # Close box -> stop drawing instead of killing the training process.
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.update_idletasks()
        self.root.update()
        self.mode = "window"

    def _init_snapshot_fallback(self) -> None:
        if self.mode == "window":
            return
        if self.snapshot_path is not None:
            self.mode = "png"
            print(f"[live-view] no display; writing live snapshots to {self.snapshot_path}")

    # ------------------------------------------------------------------ control
    def should_draw(self) -> bool:
        if self.mode == "off" or self._closed:
            return False
        now = time.perf_counter()
        interval = self.min_interval if self.mode == "window" else max(self.min_interval, 1.0)
        if now - self._last_draw < interval:
            return False
        self._last_draw = now
        return True

    def close(self) -> None:
        self._closed = True
        if self.mode == "window":
            try:
                self.root.destroy()
            except Exception:  # noqa: BLE001 - window may already be gone
                pass
            self.mode = "off"

    def draw_status(self, text: str, note: str = "") -> None:
        """Keep the window responsive and show what the trainer is busy with.

        Lightning performs the PPO update and the validation pass *without*
        calling :meth:`draw`, so without this the Tk window stops repainting and
        the run looks frozen on the last frame of the rollout.
        """
        if self.mode != "window" or self._closed:
            return
        try:
            self.canvas.delete("status")
            self.canvas.create_text(
                self.width / 2,
                self.height - 44,
                text=text,
                fill=TEXT,
                font=("Consolas", 9, "bold"),
                tags="status",
            )
            if note:
                self.canvas.create_text(
                    self.width / 2,
                    self.height - 32,
                    text=note,
                    fill="#6b7683",
                    font=("Consolas", 8),
                    tags="status",
                )
            self.root.update_idletasks()
            self.root.update()
            self._bad_frames = 0
        except Exception as exc:  # noqa: BLE001 - window closed by the user
            self._bad_frames += 1
            if self._bad_frames >= 3:
                print(f"[live-view] disabling live window ({type(exc).__name__}: {exc})")
                self.close()

    def set_iteration(self, *_: Any) -> None:
        """Compatibility shim: the iteration number is passed on each draw."""

    # -------------------------------------------------------------- rendering
    def draw(
        self,
        *,
        x: float,
        theta: float,
        action: float,
        reward: float,
        step: int,
        episode_return: float = 0.0,
        episode: int = 0,
        iteration: int = 0,
        env_steps: int = 0,
        success_rate: float = float("nan"),
        mean_abs_theta_deg: float = float("nan"),
        batch_mean_abs_theta_deg: float = float("nan"),
        rollout_steps: int = 0,
        hold_time_s: float = float("nan"),
    ) -> None:
        if self.mode == "off" or self._closed:
            return
        if self.mode == "window":
            try:
                self._draw_canvas(
                    x, theta, action, reward, step, episode_return, episode, iteration,
                    env_steps, success_rate, mean_abs_theta_deg, rollout_steps,
                    batch_mean_abs_theta_deg, hold_time_s,
                )
                self.root.update_idletasks()
                self.root.update()
                self.frames += 1
                self._bad_frames = 0
            except Exception as exc:  # noqa: BLE001 - e.g. window closed by the user
                self._bad_frames += 1
                if self._bad_frames >= 3:
                    print(f"[live-view] disabling live window ({type(exc).__name__}: {exc})")
                    self.close()
        else:  # PNG fallback
            self._draw_png(x, theta, action, reward, step, episode_return, iteration, success_rate)
            self.frames += 1

    # ---------------------------------------------------------------- tk canvas
    def _draw_canvas(
        self, x, theta, action, reward, step, episode_return, episode, iteration,
        env_steps, success_rate, mean_abs_theta_deg, rollout_steps=0,
        batch_mean_abs_theta_deg=float("nan"), hold_time_s=float("nan"),
    ) -> None:
        c = self.canvas
        c.delete("all")
        cfg = self.cfg
        W, H = self.width, self.height
        # --- layout -----------------------------------------------------
        scene_top, scene_bottom = 42, H - 66
        ground = scene_bottom - 28
        scale = (scene_bottom - scene_top) / (2.0 * cfg.l_pole * 1.25)  # px per metre
        cart_w, cart_h = 0.20 * scale, 0.10 * scale
        # The camera follows the cart: a balancing controller often lets the cart
        # drift a metre or more, and a fixed view would push the whole mechanism
        # off screen exactly when it is working.  Drift stays visible through the
        # rail ticks, which are drawn at fixed world positions.
        cam = x if cfg.model == "cart" else 0.0
        cx = W / 2

        # --- rail -------------------------------------------------------
        c.create_line(0, ground, W, ground, fill=RAIL, width=3)
        if cfg.model == "cart":
            for k in range(-6, 7):
                mx = k * 0.5
                px = _fit(mx - cam, -self.cfg_half_window, self.cfg_half_window, 0, W)
                if 0 < px < W:
                    off_rail = abs(mx) > cfg.x_limit
                    major = k % 2 == 0
                    col = BAD if off_rail else ("#9fb0bd" if major else "#dde3e8")
                    c.create_line(px, ground - 6, px, ground + 6, fill=col, width=2 if major else 1)
                    if major:
                        c.create_text(px, ground + 16, text=f"{mx:+.1f}", fill="#8a97a3", font=("Consolas", 7))
            # upright reference
            c.create_line(cx, ground, cx, ground - cfg.l_pole * 1.15 * scale, fill=GHOST, dash=(4, 3))

        # --- pole -------------------------------------------------------
        tip_x = cx + cfg.l_pole * np.sin(theta) * scale
        tip_y = ground - cfg.l_pole * np.cos(theta) * scale
        c.create_line(cx, ground, tip_x, tip_y, fill=POLE, width=6, capstyle="round")
        r = 6
        c.create_oval(tip_x - r, tip_y - r, tip_x + r, tip_y + r, fill=POLE, outline="")

        # --- cart -------------------------------------------------------
        if cfg.model == "cart":
            c.create_rectangle(
                cx - cart_w / 2, ground - cart_h, cx + cart_w / 2, ground, fill=CART, outline=CART_EDGE, width=2
            )
        else:
            c.create_oval(cx - 5, ground - 5, cx + 5, ground + 5, fill=CART_EDGE, outline="")

        # --- text -------------------------------------------------------
        deg = np.degrees(np.arctan2(np.sin(theta), np.cos(theta)))
        limit = cfg.action_limit
        frac = action / limit if limit else 0.0
        c.create_text(10, 8, anchor="nw", fill=TEXT, font=("Consolas", 10, "bold"),
                      text=f"iteration {iteration}   env steps {env_steps:,}")
        c.create_text(10, 24, anchor="nw", fill=TEXT, font=("Consolas", 9),
                      text=f"episode {episode}   t = {step * cfg.control_dt:5.2f} s   return {episode_return:+7.1f}")
        # progress inside the current rollout, so a long iteration is visibly moving
        if rollout_steps:
            frac = min(1.0, step / float(rollout_steps))
            c.create_rectangle(W - 120, 10, W - 10, 22, outline=RAIL)
            c.create_rectangle(W - 120, 10, W - 120 + frac * 110, 22, fill=CART, outline="")
            c.create_text(W - 65, 30, text=f"rollout {frac:4.0%}", fill=TEXT, font=("Consolas", 8))

        status = "BALANCED" if abs(deg) < np.degrees(cfg.success_angle) else ""
        c.create_text(W / 2, 8, anchor="n", fill=GOOD if status else TEXT,
                      font=("Consolas", 11, "bold"), text=status)

        y = H - 58
        c.create_text(10, y, anchor="nw", fill=TEXT, font=("Consolas", 10),
                      text=f"theta {deg:+7.2f} deg      x {x:+6.3f} m      u {action:+7.2f} N ({frac:+6.1%})")
        # Two clearly-labelled averages, because an instantaneous per-env value
        # next to the displayed angle is just the same number twice.
        batch_txt = f"{batch_mean_abs_theta_deg:5.2f} deg" if batch_mean_abs_theta_deg == batch_mean_abs_theta_deg else "  n/a"
        c.create_text(10, y + 16, anchor="nw", fill=TEXT, font=("Consolas", 10),
                      text=f"reward {reward:+6.3f}   |theta| avg (this env, last 100 steps) "
                           f"{mean_abs_theta_deg:5.2f} deg   all envs now {batch_txt}   success {success_rate:5.0%}")
        if hold_time_s == hold_time_s:  # only meaningful for a live evaluation run
            c.create_text(10, y + 32, anchor="nw", fill=TEXT, font=("Consolas", 10),
                          text=f"holding for {hold_time_s:6.1f} s  (no time limit — runs until the pole falls)")

        # --- action bar -------------------------------------------------
        bar_y = H - 22
        c.create_rectangle(10, bar_y - 8, W - 10, bar_y + 8, outline=RAIL)
        mid = (10 + W - 10) / 2
        c.create_line(mid, bar_y - 10, mid, bar_y + 10, fill=RAIL)
        half = (W - 20) / 2
        end = mid + float(np.clip(frac, -1, 1)) * half
        c.create_rectangle(min(mid, end), bar_y - 6, max(mid, end), bar_y + 6, fill=CART, outline="")

    # -------------------------------------------------------------- png fallback
    def _draw_png(self, x, theta, action, reward, step, episode_return, iteration, success_rate) -> None:
        from PIL import Image, ImageDraw

        cfg = self.cfg
        img = Image.new("RGB", (self.width, self.height), BG)
        d = ImageDraw.Draw(img)
        ground = self.height - 40
        scale = (ground - 40) / (2.0 * cfg.l_pole * 1.25)
        cx = self.width / 2 + (x * scale if cfg.model == "cart" else 0.0)
        d.line((0, ground, self.width, ground), fill=RAIL, width=3)
        tip = (cx + cfg.l_pole * np.sin(theta) * scale, ground - cfg.l_pole * np.cos(theta) * scale)
        d.line((cx, ground, tip[0], tip[1]), fill=POLE, width=5)
        d.ellipse((tip[0] - 5, tip[1] - 5, tip[0] + 5, tip[1] + 5), fill=POLE)
        if cfg.model == "cart":
            w, h = 0.20 * scale, 0.10 * scale
            d.rectangle((cx - w / 2, ground - h, cx + w / 2, ground), fill=CART, outline=CART_EDGE)
        deg = np.degrees(np.arctan2(np.sin(theta), np.cos(theta)))
        d.text((10, 10), f"iter {iteration}  ep {step * cfg.control_dt:5.2f}s  return {episode_return:+.1f}",
               fill=TEXT)
        d.text((10, 26), f"theta {deg:+7.2f} deg   x {x:+6.3f} m   u {action:+7.2f} N   r {reward:+6.3f}",
               fill=TEXT)
        d.text((10, 42), f"success {success_rate:.0%}", fill=GOOD)
        if self.snapshot_path is not None:
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(self.snapshot_path)

    # ----------------------------------------------------------------- context
    def __enter__(self) -> "LiveViewer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
