"""Small shared helpers: seeding, observation normalisation, paths."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
VIDEO_DIR = OUTPUT_DIR / "videos"


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs or (OUTPUT_DIR, CKPT_DIR, LOG_DIR, VIDEO_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # torch is always installed for the RL parts
        pass


@dataclass
class RunningMeanStd:
    """Welford/parallel running mean & variance with a frozen-eval mode.

    Used to keep the policy's input well conditioned while the agent drives the
    plant all over its state space.  ``freeze`` is flipped on for inference so a
    deployment run cannot drift with the incoming observations.
    """

    shape: tuple[int, ...] = ()
    clip: float = 10.0
    epsilon: float = 1e-8
    freeze: bool = False

    mean: np.ndarray = field(init=False)
    var: np.ndarray = field(init=False)
    count: float = field(default=1e-4, init=False)

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        if self.freeze:
            return
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.shape):
            x = x.reshape(1, -1) if self.shape else x.reshape(1)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray, update: bool = False) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if update:
            self.update(x)
        out = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(out, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state["var"], dtype=np.float64).copy()
        self.count = float(state["count"])


class MetricsWriter:
    """Append scalars to a JSONL file so runs stay inspectable without TensorBoard."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def read(path: Path) -> list[dict[str, Any]]:
        path = Path(path)
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


def human_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def init_fields(dataclass_type, values: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields a dataclass actually accepts in ``__init__``.

    Derived fields (``field(init=False)``, e.g. ``frame_skip`` or
    ``steps_per_epoch``) are recomputed by ``__post_init__`` and must be dropped
    when a config is round-tripped through JSON, otherwise rebuilding it fails
    with "unexpected keyword argument".
    """
    return {k: v for k, v in values.items() if k in dataclass_type.__dataclass_fields__ and dataclass_type.__dataclass_fields__[k].init}

