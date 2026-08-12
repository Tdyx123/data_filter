#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qwen3_vl_groot.benchmarking import (  # noqa: E402
    combine_phase_metrics,
    is_meaningfully_faster,
    summarize_metrics,
)


TRAIN_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "train_libero_qwen3_vl_4b_groot_all_tasks_4x4090.sh"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    micro_batch_size: int
    gradient_accumulation_steps: int
    context_forward: str
    compile_action_head: bool = False
    episode_cache_size: int = 2


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _phase_schedule(phase: str, steps: int) -> tuple[int, int, int]:
    if phase == "head_only":
        return steps, steps, min(10, steps)
    if phase == "lora_active":
        return 0, steps, steps
    raise ValueError(f"Unknown benchmark phase: {phase}")


def _command(
    candidate: Candidate,
    *,
    phase: str,
    output_dir: Path,
    gpu_ids: str,
    steps: int,
    skip_memory_probe: bool,
) -> list[str]:
    freeze_steps, cycle_steps, active_steps = _phase_schedule(phase, steps)
    command = [
        "bash",
        str(TRAIN_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--gpu-count",
        "4",
        "--gpu-ids",
        gpu_ids,
        "--micro-batch-size",
        str(candidate.micro_batch_size),
        "--gradient-accumulation-steps",
        str(candidate.gradient_accumulation_steps),
        "--max-steps",
        str(steps),
        "--lora-freeze-steps",
        str(freeze_steps),
        "--lora-cycle-steps",
        str(cycle_steps),
        "--lora-active-steps",
        str(active_steps),
        "--qwen-context-forward",
        candidate.context_forward,
        "--no-compile-qwen-backbone",
        (
            "--compile-action-head"
            if candidate.compile_action_head
            else "--no-compile-action-head"
        ),
        "--episode-cache-size",
        str(candidate.episode_cache_size),
    ]
    if skip_memory_probe:
        command.append("--skip-memory-probe")
    return command


def _run_candidate(
    candidate: Candidate,
    *,
    output_root: Path,
    gpu_ids: str,
    steps: int,
    warmup_steps: int,
) -> dict[str, Any]:
    phases: dict[str, Any] = {}
    for phase_index, phase in enumerate(("head_only", "lora_active")):
        output_dir = output_root / candidate.name / phase
        command = _command(
            candidate,
            phase=phase,
            output_dir=output_dir,
            gpu_ids=gpu_ids,
            steps=steps,
            skip_memory_probe=phase_index > 0,
        )
        print("Launching:", " ".join(command), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        metrics = summarize_metrics(
            output_dir / "metrics.jsonl",
            warmup_steps=warmup_steps,
            expected_steps=steps,
            effective_batch_size=64,
        )
        phases[phase] = {
            "output_dir": str(output_dir),
            "command": command,
            **metrics.as_dict(),
        }
    head = phases["head_only"]
    active = phases["lora_active"]
    weighted_step_seconds = combine_phase_metrics(
        head_only_step_seconds=head["step_seconds"],
        lora_active_step_seconds=active["step_seconds"],
    )
    return {
        "candidate": asdict(candidate),
        "phases": phases,
        "weighted_step_seconds": weighted_step_seconds,
        "weighted_samples_per_second": 64 / weighted_step_seconds,
        "weighted_data_wait_fraction": combine_phase_metrics(
            head_only_step_seconds=head["data_wait_fraction"],
            lora_active_step_seconds=active["data_wait_fraction"],
        ),
        "memory_probe": json.loads(
            (output_root / candidate.name / "head_only" / "preflight.json").read_text(
                encoding="utf-8"
            )
        )["memory_probe"],
    }


def _try_candidate(
    candidate: Candidate,
    *,
    report: dict[str, Any],
    report_path: Path,
    output_root: Path,
    gpu_ids: str,
    steps: int,
    warmup_steps: int,
) -> dict[str, Any] | None:
    try:
        result = _run_candidate(
            candidate,
            output_root=output_root,
            gpu_ids=gpu_ids,
            steps=steps,
            warmup_steps=warmup_steps,
        )
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError) as error:
        report["failures"][candidate.name] = f"{type(error).__name__}: {error}"
        _atomic_json(report_path, report)
        return None
    report["results"][candidate.name] = result
    _atomic_json(report_path, report)
    return result


def _accepted(reference: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return is_meaningfully_faster(
        reference_step_seconds=float(reference["weighted_step_seconds"]),
        candidate_step_seconds=float(candidate["weighted_step_seconds"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark strict-semantics Qwen LIBERO cyclic-training accelerations."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    gpu_ids = [piece.strip() for piece in arguments.gpu_ids.split(",")]
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4 or any(
        not piece.isdigit() for piece in gpu_ids
    ):
        raise SystemExit("--gpu-ids must contain four distinct non-negative integers")
    if arguments.steps <= 0 or not 0 <= arguments.warmup_steps < arguments.steps:
        raise SystemExit("require 0 <= --warmup-steps < --steps")
    output_root = arguments.output_root.expanduser().resolve()
    if output_root.exists():
        raise SystemExit(f"benchmark output root must be new: {output_root}")
    output_root.mkdir(parents=True)
    report_path = output_root / "benchmark_summary.json"
    report: dict[str, Any] = {
        "protocol": {
            "optimizer_steps": arguments.steps,
            "discard_steps": arguments.warmup_steps,
            "gpu_ids": [int(piece) for piece in gpu_ids],
            "effective_batch_size": 64,
            "head_only_weight": 0.9,
            "lora_active_weight": 0.1,
            "minimum_incremental_speedup": 0.05,
            "reserved_memory_limit_gib": 22.0,
        },
        "results": {},
        "failures": {},
        "decisions": [],
    }
    _atomic_json(report_path, report)
    common = {
        "output_root": output_root,
        "gpu_ids": ",".join(gpu_ids),
        "steps": arguments.steps,
        "warmup_steps": arguments.warmup_steps,
        "report": report,
        "report_path": report_path,
    }

    baseline_spec = Candidate("baseline_causal_mbs1", 1, 16, "causal_lm")
    baseline = _try_candidate(baseline_spec, **common)
    if baseline is None:
        raise SystemExit("baseline benchmark failed; see benchmark_summary.json")

    direct_spec = Candidate("backbone_mbs1", 1, 16, "backbone")
    direct = _try_candidate(direct_spec, **common)
    current_spec = baseline_spec
    current = baseline
    if direct is not None and direct["weighted_step_seconds"] < baseline["weighted_step_seconds"]:
        current_spec, current = direct_spec, direct
        report["decisions"].append("accepted direct Qwen backbone context forward")
    else:
        report["decisions"].append("kept causal-LM context forward")

    for micro_batch_size, accumulation in ((2, 8), (4, 4)):
        candidate_spec = Candidate(
            f"{current_spec.context_forward}_mbs{micro_batch_size}",
            micro_batch_size,
            accumulation,
            current_spec.context_forward,
        )
        candidate = _try_candidate(candidate_spec, **common)
        if candidate is not None and _accepted(current, candidate):
            current_spec, current = candidate_spec, candidate
            report["decisions"].append(
                f"accepted micro-batch {micro_batch_size} / accumulation {accumulation}"
            )
        else:
            report["decisions"].append(
                f"rejected micro-batch {micro_batch_size}: below 5% incremental gain or failed"
            )

    compiled_spec = Candidate(
        f"{current_spec.context_forward}_mbs{current_spec.micro_batch_size}_head_compile",
        current_spec.micro_batch_size,
        current_spec.gradient_accumulation_steps,
        current_spec.context_forward,
        compile_action_head=True,
    )
    compiled = _try_candidate(compiled_spec, **common)
    if compiled is not None and _accepted(current, compiled):
        current_spec, current = compiled_spec, compiled
        report["decisions"].append("accepted action-head torch.compile")
    else:
        report["decisions"].append("rejected action-head torch.compile: below 5% gain or failed")

    if float(current["weighted_data_wait_fraction"]) > 0.10:
        for cache_size in (16, 32):
            cache_spec = Candidate(
                f"{current_spec.name}_cache{cache_size}",
                current_spec.micro_batch_size,
                current_spec.gradient_accumulation_steps,
                current_spec.context_forward,
                current_spec.compile_action_head,
                cache_size,
            )
            cache_result = _try_candidate(cache_spec, **common)
            if cache_result is not None and _accepted(current, cache_result):
                current_spec, current = cache_spec, cache_result
                report["decisions"].append(f"accepted episode LRU {cache_size}")
            else:
                report["decisions"].append(
                    f"rejected episode LRU {cache_size}: below 5% gain or failed"
                )
    else:
        report["decisions"].append("skipped LRU sweep because data wait was at most 10%")

    baseline_seconds = float(baseline["weighted_step_seconds"])
    winner_seconds = float(current["weighted_step_seconds"])
    report["winner"] = {
        **asdict(current_spec),
        "weighted_step_seconds": winner_seconds,
        "weighted_samples_per_second": 64 / winner_seconds,
        "speedup_fraction": 1.0 - winner_seconds / baseline_seconds,
        "target_step_seconds": 4.6,
        "target_25_percent_met": winner_seconds <= baseline_seconds * 0.75,
    }
    _atomic_json(report_path, report)
    print(json.dumps(report["winner"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
