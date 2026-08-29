import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "benchmark_qwen_cyclic_acceleration.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen_benchmark_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_command_preserves_effective_batch_and_phase_schedule(tmp_path):
    module = _load_script()
    candidate = module.Candidate(
        "backbone_mbs4",
        micro_batch_size=4,
        gradient_accumulation_steps=4,
        compile_action_head=True,
        episode_cache_size=16,
    )

    command = module._command(
        candidate,
        phase="lora_active",
        output_dir=tmp_path / "run",
        gpu_ids="0,1,2,3",
        steps=100,
        skip_memory_probe=True,
    )

    def value(option):
        return command[command.index(option) + 1]

    assert value("--micro-batch-size") == "4"
    assert value("--gradient-accumulation-steps") == "4"
    assert 4 * int(value("--micro-batch-size")) * int(
        value("--gradient-accumulation-steps")
    ) == 64
    assert value("--lora-freeze-steps") == "0"
    assert value("--lora-cycle-steps") == "100"
    assert value("--lora-active-steps") == "100"
    assert "--qwen-context-forward" not in command
    assert value("--episode-cache-size") == "16"
    assert "--compile-action-head" in command
    assert "--no-compile-qwen-backbone" in command
    assert "--skip-memory-probe" in command
