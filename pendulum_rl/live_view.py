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

Camera
------
``camera="fixed"`` (default) pins the view to the rail: the world does not move,
the cart does.  A follow camera (``camera="follow"``) keeps the cart centred and
slides the rail underneath instead, which is what a narrow window forced before.
The camera is resolved by :func:`compute_camera`, a pure function, so the
transform is unit-testable without a display.

Controls
--------
The window carries a button row -- ``Next episode``, ``Replay episode``,
``Speed 1x`` and ``Quit`` -- and the driving loop polls a tiny API once per
step, because Tk callbacks run inside ``root.update()`` on the caller's thread
and must never touch the physics themselves:

    if viewer.take_skip():   ...end this episode, count it as skipped...
    if viewer.take_replay(): ...end this episode, re-run the same seed...
    dt = control_dt / viewer.speed

Buttons only set those flags; everything else happens in the loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: palette shared with the export renderer
BG = "#f4f6f8"
PANEL = "#ffffff"
RAIL = "#b0b8c1"
CART = "#3d7ea6"
CART_EDGE = "#22415a"
POLE = "#d1495b"
GHOST = "#c9d1d9"
TEXT = "#22303c"
MUTED = "#6b7683"
GOOD = "#2a9d8f"
BAD = "#e07a5f"
DANGER = "#f7e3df"   # wash beyond the rail
TARGET = "#e6f2ef"   # wash inside the success |x| window

#: speeds offered by the speed button, in real time per simulated second
SPEEDS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_SPEED = 1.0


def _fit(value: float, lo: float, hi: float, p0: float, p1: float) -> float:
    """Map ``value`` from [lo, hi] onto pixel range [p0, p1]."""
    if hi <= lo:
        return 0.5 * (p0 + p1)
    t = (value - lo) / (hi - lo)
    return p0 + t * (p1 - p0)


@dataclass(frozen=True)
class Camera:
    """Resolved view transform: world metres -> a centred pixel window."""

    centre: float          # world x at the centre of the view [m]
    half_window: float     # half-width of the view [m]


def compute_camera(x: float, cfg: Any, mode: str = "fixed") -> Camera:
    """Where to look.

    ``fixed`` always shows ``x = 0`` at the centre of the canvas, so the rail and
    the tick marks are stationary and the cart visibly moves along them --
    including out to the rail limits, which is where the interesting failures
    happen and which the old 1.6 m window pushed off screen.  ``follow`` keeps
    the cart centred (the historical behaviour); the rail then slides instead.
    """
    limit = float(getattr(cfg, "x_limit", 2.4)) or 2.4
    if mode == "follow":
        if getattr(cfg, "model", "cart") != "cart":
            return Camera(0.0, limit)
        return Camera(float(x), max(0.9, min(limit, 1.6)))
    # fixed: the whole rail plus a margin, so the cart never leaves the frame
    return Camera(0.0, max(1.2, 1.1 * limit))


def world_to_px(world_x: float, cam: Camera, width: float) -> float:
    """Horizontal screen pixel for a world x under ``cam`` (pure, testable)."""
    return _fit(world_x, cam.centre - cam.half_window, cam.centre + cam.half_window, 0.0, width)


class LiveViewer:
    """Non-blocking Tkinter window that draws the plant in real time."""

    def __init__(
        self,
        cfg: Any,
        width: int = 640,
        height: int = 360,
        fps: float = 50.0,
        title: str = "inverted pendulum — live",
        prefer_window: bool = True,
        snapshot_path: Path | None = None,
        camera: str = "fixed",
    ):
        self.cfg = cfg
        self.width = width
        self.height = height
        self.min_interval = 1.0 / max(fps, 1.0)
        self.title_text = title
        self.snapshot_path = Path(snapshot_path) if snapshot_path else None
        self.camera_mode = camera if camera in ("fixed", "follow") else "fixed"
        self.mode = "off"
        self.frames = 0
        self._last_draw = 0.0
        self._last_snapshot = 0.0
        self._closed = False
        self._bad_frames = 0
        self.cfg_half_window = max(0.9, min(cfg.x_limit, 1.6)) if cfg.model == "cart" else 1.2
        # control state, written by Tk callbacks and read by the driving loop
        self._skip = False
        self._replay = False
        self.speed = DEFAULT_SPEED
        self.status_line = ""
        self.controls_enabled = True
        self.buttons: dict[str, Any] = {}

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
        from tkinter import font as tkfont

        self._tk = tk
        self.root = tk.Tk()
        self.root.title(self.title_text)
        self.root.resizable(False, False)
        self.root.configure(bg=BG)
        # Scale all fonts with the canvas: the defaults are tiny on a high-DPI
        # display and the coordinates below assume a ~640 px wide window.
        unit = max(1.0, self.width / 640.0)
        self._font_title = tkfont.Font(family="Segoe UI", size=int(round(11 * unit)), weight="bold")
        self._font_body = tkfont.Font(family="Segoe UI", size=int(round(9 * unit)))
        self._font_mono = tkfont.Font(family="Consolas", size=int(round(9 * unit)))
        self._font_mono_bold = tkfont.Font(family="Consolas", size=int(round(10 * unit)), weight="bold")
        self._font_button = tkfont.Font(family="Segoe UI", size=int(round(9 * unit)))

        self.canvas = tk.Canvas(
            self.root, width=self.width, height=self.height, bg=PANEL, highlightthickness=1,
            highlightbackground=RAIL,
        )
        self.canvas.pack(padx=10, pady=(10, 6))
        self._build_controls(tk)
        self._build_status(tk)
        # Close box -> stop drawing instead of killing the training process.
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.update_idletasks()
        # Freeze the layout: a window that is allowed to be resized re-runs the
        # geometry manager whenever any label's requested size changes, and the
        # numbers on screen change every frame.  Pinning min == max makes the
        # window size a constant of the program rather than a function of the
        # last frame's text.
        self._lock_geometry()
        self.root.update()
        self.mode = "window"

    def _lock_geometry(self) -> None:
        try:
            w, h = self.root.winfo_reqwidth(), self.root.winfo_reqheight()
            self.root.minsize(w, h)
            self.root.maxsize(w, h)
            self.root.geometry("%dx%d" % (w, h))
        except Exception:  # noqa: BLE001 - cosmetic only, never worth failing a run
            pass

    def _build_controls(self, tk) -> None:
        if not self.controls_enabled:
            return
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", padx=10, pady=(0, 4))
        specs = (
            ("next", "Next episode", self._on_next, "end this episode now and start the next one"),
            ("replay", "Replay episode", self._on_replay, "re-run this episode from the same start state"),
            ("speed", None, self._on_speed, "cycle playback speed"),
            ("quit", "Quit", self.close, "stop and close the window"),
        )
        for key, label, command, tip in specs:
            text = label if label is not None else "Speed %gx" % self.speed
            button = tk.Button(
                bar, text=text, command=command, font=self._font_button,
                relief="flat", bg="#e8edf2", activebackground="#d7e0e8", fg=TEXT,
                padx=10, pady=3, cursor="hand2", borderwidth=0,
            )
            if key == "speed":
                # "Speed 0.25x" is wide, "Speed 4x" is narrow: without a pinned
                # width the button (and the row it sits in) resizes on every
                # click, and every other button shifts with it.
                button.configure(width=12)
            button.pack(side="left", padx=(0, 6))
            try:
                self._tooltip(button, tip)
            except Exception:  # noqa: BLE001 - a tooltip is never worth failing over
                pass
            self.buttons[key] = button
        # The live counter the loops update through note_episode().  Its width is
        # pinned: a label that grows with its text makes Tk re-run the pack
        # geometry every time the text changes, which resizes the whole window
        # and shows up as a flicker.
        self._progress = tk.Label(bar, text="episode —", font=self._font_body, bg=BG, fg=MUTED,
                                  anchor="e", width=12)
        self._progress.pack(side="right")

    def _tooltip(self, widget, text: str) -> None:
        tip = {"win": None}

        def show(_event=None):
            if tip["win"] is not None:
                return
            win = self._tk.Toplevel(widget)
            win.wm_overrideredirect(True)
            win.wm_geometry("+%d+%d" % (widget.winfo_rootx() + 12, widget.winfo_rooty() + 26))
            self._tk.Label(win, text=text, font=self._font_body, bg="#22303c", fg="#ffffff",
                           padx=6, pady=3).pack()
            tip["win"] = win

        def hide(_event=None):
            if tip["win"] is not None:
                tip["win"].destroy()
                tip["win"] = None

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    def _build_status(self, tk) -> None:
        panel = tk.Frame(self.root, bg=BG)
        panel.pack(fill="x", padx=10, pady=(0, 10))
        # Fixed-width, left/right anchored labels: every number on screen changes
        # as the cart moves, and any label allowed to resize with its content
        # drags the pack geometry (and the window) around with it.
        self._left = tk.Label(panel, text="", font=self._font_mono, bg=BG, fg=TEXT,
                              justify="left", anchor="w", width=64)
        self._left.pack(side="left")
        self._right = tk.Label(panel, text="", font=self._font_mono, bg=BG, fg=TEXT,
                               justify="right", anchor="e", width=26)
        self._right.pack(side="right")

    def _init_snapshot_fallback(self) -> None:
        if self.mode == "window":
            return
        if self.snapshot_path is not None:
            self.mode = "png"
            print(f"[live-view] no display; writing live snapshots to {self.snapshot_path}")

    # ------------------------------------------------------------- button hooks
    def _on_next(self) -> None:
        self._skip = True
        self._replay = False

    def _on_replay(self) -> None:
        self._replay = True
        self._skip = False

    def _on_speed(self) -> None:
        try:
            index = SPEEDS.index(self.speed)
        except ValueError:
            index = SPEEDS.index(DEFAULT_SPEED)
        self.speed = SPEEDS[(index + 1) % len(SPEEDS)]
        if "speed" in self.buttons:
            self.buttons["speed"].configure(text="Speed %gx" % self.speed)

    # ------------------------------------------------------- control API (poll)
    def take_skip(self) -> bool:
        """True once per click on 'Next episode'."""
        if self._skip:
            self._skip = False
            return True
        return False

    def take_replay(self) -> bool:
        """True once per click on 'Replay episode'."""
        if self._replay:
            self._replay = False
            return True
        return False

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

    def note_episode(self, episode: int, total: int | None = None) -> None:
        """Update the 'episode —' counter shown in the button row."""
        if self.mode != "window":
            return
        text = "episode %d" % episode if total is None else "episode %d/%d" % (episode, total)
        try:
            self._progress.configure(text=text)
        except Exception:  # noqa: BLE001 - window may already be gone
            pass

    def draw_status(self, text: str, note: str = "") -> None:
        """Keep the window responsive and show what the trainer is busy with.

        Lightning performs the PPO update and the validation pass *without*
        calling :meth:`draw`, so without this the Tk window stops repainting and
        the run looks frozen on the last frame of the rollout.  In window mode
        the status line lives in the bottom panel, which costs one label update
        instead of a full canvas redraw.
        """
        self.status_line = text
        if self.mode != "window" or self._closed:
            return
        try:
            self._left.configure(text=text)
            self._right.configure(text=note)
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
        cam = compute_camera(x, cfg, self.camera_mode)

        # --- layout: horizontal scale is uniform (metres per pixel), so the
        #    mechanism keeps its real proportions; the ground sits low enough to
        #    leave the pole its full swing even on a long rail.
        scale = 0.88 * W / (2.0 * cam.half_window)
        ground = H - 62
        pole_px = cfg.l_pole * scale
        cart_w, cart_h = 0.24 * scale, 0.11 * scale

        def px(world_x: float) -> float:
            return world_to_px(world_x, cam, W)

        cart_px = px(x if cfg.model == "cart" else 0.0)

        # --- ground plane with the success window and the rail limits --------
        if cfg.model == "cart":
            x_lo, x_hi = cam.centre - cam.half_window, cam.centre + cam.half_window
            if x_lo < -cfg.x_limit:
                c.create_rectangle(px(x_lo), ground - 3, px(-cfg.x_limit), ground + 3,
                                   fill=DANGER, outline="")
            if x_hi > cfg.x_limit:
                c.create_rectangle(px(cfg.x_limit), ground - 3, px(x_hi), ground + 3,
                                   fill=DANGER, outline="")
            lo, hi = max(x_lo, -cfg.success_x_range), min(x_hi, cfg.success_x_range)
            if hi > lo:
                c.create_rectangle(px(lo), ground - 2, px(hi), ground + 2, fill=TARGET, outline="")
        c.create_line(0, ground, W, ground, fill=RAIL, width=2)

        if cfg.model == "cart":
            step_m = 0.5
            k = int(np.floor((cam.centre - cam.half_window) / step_m))
            while k * step_m <= cam.centre + cam.half_window:
                mx = k * step_m
                tick_x = px(mx)
                off_rail = abs(mx) > cfg.x_limit
                major = abs(mx % 1.0) < 1e-9
                col = BAD if off_rail else (RAIL if major else "#e3e8ee")
                c.create_line(tick_x, ground, tick_x, ground + (7 if major else 4), fill=col,
                              width=2 if major else 1)
                if major:
                    c.create_text(tick_x, ground + 17, text="%+.0f" % mx, fill=MUTED, font=self._font_body)
                k += 1
            # the rail ends themselves, so "how much room is left" is visible
            for edge in (-cfg.x_limit, cfg.x_limit):
                if abs(edge - cam.centre) <= cam.half_window:
                    c.create_line(px(edge), ground - 26, px(edge), ground + 8, fill=BAD, width=2, dash=(3, 2))
            # upright reference at the world origin
            if abs(cam.centre) <= cam.half_window:
                c.create_line(px(0.0), ground, px(0.0), ground - 1.15 * pole_px,
                              fill=GHOST, dash=(4, 3))
            # target-position band edges
            for edge in (-cfg.success_x_range, cfg.success_x_range):
                if abs(edge - cam.centre) <= cam.half_window:
                    c.create_line(px(edge), ground - 6, px(edge), ground + 6, fill=GOOD, width=1)

        # --- pole -----------------------------------------------------------
        tip_x = cart_px + cfg.l_pole * np.sin(theta) * scale
        tip_y = ground - cfg.l_pole * np.cos(theta) * scale
        c.create_line(cart_px, ground - cart_h * 0.5, tip_x, tip_y, fill=POLE, width=5, capstyle="round")
        r = 5
        c.create_oval(tip_x - r, tip_y - r, tip_x + r, tip_y + r, fill=POLE, outline="")

        # --- cart -----------------------------------------------------------
        if cfg.model == "cart":
            c.create_rectangle(cart_px - cart_w / 2, ground - cart_h, cart_px + cart_w / 2, ground,
                               fill=CART, outline=CART_EDGE, width=2)
        else:
            c.create_oval(cart_px - 5, ground - 5, cart_px + 5, ground + 5, fill=CART_EDGE, outline="")

        # --- status pill ----------------------------------------------------
        deg = np.degrees(np.arctan2(np.sin(theta), np.cos(theta)))
        if cfg.model == "cart" and abs(x) > cfg.x_limit:
            label, colour = "OUT OF RAIL", BAD
        elif abs(deg) < np.degrees(cfg.success_angle):
            label, colour = "BALANCED", GOOD
        else:
            label, colour = "RECOVERING", TEXT
        self._pill(c, W - 12, 12, label, colour)

        # --- action bar -----------------------------------------------------
        limit = cfg.action_limit
        frac = action / limit if limit else 0.0
        bar_y = H - 20
        c.create_rectangle(12, bar_y - 7, W - 12, bar_y + 7, outline=RAIL, fill="#fbfcfd")
        mid = W / 2
        c.create_line(mid, bar_y - 10, mid, bar_y + 10, fill=RAIL)
        end = mid + float(np.clip(frac, -1, 1)) * ((W - 24) / 2)
        c.create_rectangle(min(mid, end), bar_y - 5, max(mid, end), bar_y + 5,
                           fill=CART if abs(frac) < 0.98 else BAD, outline="")
        c.create_text(12, bar_y - 15, anchor="w", text="force", fill=MUTED, font=self._font_body)

        # --- text: the two panels carry the numbers, the canvas keeps the scene.
        #     Every field is formatted to a constant width: these labels are
        #     fixed-width now, but stable columns also stop the digits from
        #     jittering sideways as the values change.
        left = "theta %+7.2f deg    x %+6.3f m    u %+7.2f N (%+6.1f%% of limit)" % (
            deg, x, action, 100 * frac)
        right = "t %5.2f s    return %+7.1f" % (step * cfg.control_dt, episode_return)
        batch_txt = ("%5.2f deg" % batch_mean_abs_theta_deg
                     if batch_mean_abs_theta_deg == batch_mean_abs_theta_deg else "n/a     ")
        left2 = "|theta| avg (100 steps) %5.2f deg    all envs %s    reward %+6.3f" % (
            mean_abs_theta_deg, batch_txt, reward)
        right2 = "success %5.1f%%    frames %8s" % (
            100 * success_rate if success_rate == success_rate else float("nan"), f"{self.frames:,}")
        self._left.configure(text=left + "\n" + left2)
        self._right.configure(text=right + "\n" + right2)

    def _pill(self, canvas, right_x: float, top_y: float, text: str, colour: str) -> None:
        """Small rounded status badge in the top-right corner."""
        pad = 8
        width = pad * 2 + 7 * len(text)
        x1, y1 = right_x - width, top_y
        x2, y2 = right_x, top_y + 20
        canvas.create_rectangle(x1, y1, x2, y2, fill=colour, outline="")
        canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=text, fill="#ffffff", font=self._font_title)

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
