"""Shared defaults for every entry point in `scripts/`.

Keeping the task/algo presets here means `train.py`, `evaluate.py` and any
future experiment agree on what "the balance task" or "the swing-up task" is.
"""

from __future__ import annotations

TASKS = {
    # name: (init_mode, warmup_init_mode, curriculum_fraction, comment)
    "balance": ("upright", None, 0.0, "start near upright, stay upright for 10 s"),
    "swingup": ("hanging", None, 0.0, "start hanging, swing up and balance"),
    "swingup-robust": ("random", "hanging", 0.4, "curriculum: hanging, then random states"),
    "robust": ("random", None, 0.0, "random starts over the whole state space"),
}

CHECKPOINT_MONITOR = "val/mean_return"
