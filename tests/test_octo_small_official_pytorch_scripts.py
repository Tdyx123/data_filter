import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONVERT = PROJECT_ROOT / "scripts" / "convert_octo_small_official_to_pytorch.sh"
EVALUATE = PROJECT_ROOT / "scripts" / "evaluate_simpler_octo_small_official_pytorch.sh"


def test_conversion_launcher_is_independent_and_defaults_to_official_output():
    text = CONVERT.read_text(encoding="utf-8")

    assert "octo_small_official_pytorch.convert_checkpoint" in text
    assert "octo-small-pytorch-official" in text
    assert "octo-small-pytorch\n" not in text


def test_evaluation_launcher_help_exposes_official_defaults_without_base_model():
    result = subprocess.run(
        ["bash", str(EVALUATE), "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "/data/dwb/models/octo-small-pytorch-official" in result.stdout
    assert "outputs/octo_small_official_pytorch_simpler_eval" in result.stdout
    assert "--base-model" not in result.stdout
    assert "first_action" in result.stdout


def test_evaluation_launcher_rejects_missing_checkpoint_with_conversion_hint(tmp_path):
    missing = tmp_path / "missing-checkpoint"
    result = subprocess.run(
        ["bash", str(EVALUATE), "--checkpoint", str(missing), "--preflight-only"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "convert_octo_small_official_to_pytorch.sh" in result.stderr
