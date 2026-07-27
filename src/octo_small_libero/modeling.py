from __future__ import annotations

from pathlib import Path
from typing import Any


def load_pytorch_model(
    checkpoint_path: str | Path,
    *,
    device: Any = "cpu",
) -> tuple[Any, Any]:
    """Load the self-contained PyTorch Octo-small model and tokenizer."""
    from .torch_model import OctoSmallPolicy

    return OctoSmallPolicy.from_pretrained(checkpoint_path, device=device)
