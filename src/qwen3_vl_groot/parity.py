from __future__ import annotations

from typing import Any, Mapping

import torch


def compare_tensors(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "close": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "elements": 0,
            "max_abs_difference": None,
            "max_relative_difference": None,
        }
    reference_float = reference.detach().float().cpu()
    candidate_float = candidate.detach().float().cpu()
    difference = (reference_float - candidate_float).abs()
    denominator = reference_float.abs().clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "close": bool(
            torch.allclose(
                reference_float,
                candidate_float,
                rtol=rtol,
                atol=atol,
                equal_nan=False,
            )
        ),
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "elements": reference.numel(),
        "max_abs_difference": float(difference.max()) if difference.numel() else 0.0,
        "max_relative_difference": (
            float((difference / denominator).max()) if difference.numel() else 0.0
        ),
    }


def compare_named_tensors(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    reference_names = set(reference)
    candidate_names = set(candidate)
    missing = sorted(reference_names - candidate_names)
    unexpected = sorted(candidate_names - reference_names)
    mismatched = []
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for name in sorted(reference_names & candidate_names):
        result = compare_tensors(
            reference[name],
            candidate[name],
            rtol=rtol,
            atol=atol,
        )
        if not result["close"]:
            mismatched.append(name)
        maximum_absolute = max(
            maximum_absolute,
            float(result["max_abs_difference"] or 0.0),
        )
        maximum_relative = max(
            maximum_relative,
            float(result["max_relative_difference"] or 0.0),
        )
    return {
        "close": not (missing or unexpected or mismatched),
        "parameters_compared": len(reference_names & candidate_names),
        "missing_parameters": missing,
        "unexpected_parameters": unexpected,
        "mismatched_parameters": mismatched,
        "max_abs_difference": maximum_absolute,
        "max_relative_difference": maximum_relative,
    }
