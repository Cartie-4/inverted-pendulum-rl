"""Package entry point.

Two environment fixes happen before anything else is imported:

1.  Matplotlib insists on a cache directory.  ``%LOCALAPPDATA%`` is not always
    writable (locked-down Windows accounts, sandboxes), and a failed cache
    write spams warnings and slows every import, so we redirect it to a folder
    inside the project unless the user already chose one.
2.  ``pytorch_lightning`` is imported as the top-level module; the unified
    ``lightning`` namespace package is *not* assumed to be installed (the
    ``pytorch-lightning`` distribution alone is enough).
"""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not os.environ.get("MPLCONFIGDIR"):
    _cache = _PROJECT_ROOT / ".cache" / "matplotlib"
    try:
        _cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_cache)
    except OSError:  # pragma: no cover - read-only checkout
        pass

from .agents.ppo import PPOAgent, PPOConfig
from .envs.inverted_pendulum import EnvConfig, InvertedPendulumEnv
from .lightning_module import PPOLightningModule, RolloutDataModule, TrainConfig, load_policy
from .vector_env import SyncVectorEnv

__version__ = "0.1.0"

__all__ = [
    "EnvConfig",
    "InvertedPendulumEnv",
    "PPOAgent",
    "PPOConfig",
    "PPOLightningModule",
    "RolloutDataModule",
    "SyncVectorEnv",
    "TrainConfig",
    "load_policy",
    "__version__",
]
