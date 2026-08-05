"""Global reproducibility controls for optional numerical backends."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
