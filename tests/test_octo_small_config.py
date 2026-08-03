import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from octo_small_libero.checkpoint import (
    inspect_flax_octo_checkpoint,
)
from octo_small_libero.config import (
    ConfigError,
    apply_overrides,
    load_config,
    resolved_paths,
    validate_config,
)
from octo_small_libero.convert_checkpoint import (
    _ensure_sentencepiece_model,
    build_parser as build_conversion_parser,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_octo_config_is_independent_and_points_to_local_checkpoint():
    config = load_config(PROJECT_ROOT / "configs" / "octo_small_libero_4x4090.yaml")
    assert config["paths"]["model"] == "/data/dwb/models/octo-small-pytorch"
    assert config["paths"]["lerobot"] == "/data/dwb/datasets/LIBERO_lerobot"
    assert config["paths"]["output"] == "outputs/octo_small_libero_4gpu"
    assert "statistics" not in config["paths"]
    assert config["model"]["pretrained_step"] == 270000
    assert config["data"]["action_horizon"] == 8
    assert config["data"]["sample_weights"] == [1.0, 1.0]
    assert config["data"]["target_dataset"] == "libero10_5"
    assert config["data"]["target_task_index"] is None
    assert config["data"]["target_all_tasks"] is False
    assert config["data"]["prior_selection"] == {
        "scores": "outputs/tdus/libero90/chunk/scores.csv",
        "top_percent": None,
        "prefiltered": False,
    }
    assert config["model"]["required_observation_tokenizers"] == ["primary", "wrist"]
    assert config["train"]["gpu_ids"] == [0, 1, 2, 3]
    assert config["train"]["micro_batch_size_per_gpu"] == 8
    assert config["train"]["gradient_accumulation_steps"] == 4
    assert config["train"]["precision"] == "bf16"
    assert config["train"]["max_steps"] == 10_000
    assert config["train"]["learning_rate"]["warmup_steps"] == 400
    assert config["train"]["learning_rate"]["decay_steps"] == 10_000
    assert config["train"]["save_every_steps"] == 1_000
    assert "lora" not in config["model"]
    assert "deepspeed_stage" not in config["train"]


def test_octo_conversion_defaults_to_local_t5_artifact():
    arguments = build_conversion_parser().parse_args([])

    assert arguments.t5_source == "/data/dwb/models/t5-base"


def test_octo_conversion_copies_missing_sentencepiece_without_overwriting(tmp_path):
    source = tmp_path / "source.model"
    source.write_bytes(b"source sentencepiece")
    tokenizer = SimpleNamespace(vocab_file=str(source))

    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    copied = _ensure_sentencepiece_model(tokenizer, missing_root)
    assert copied == missing_root / "spiece.model"
    assert copied.read_bytes() == source.read_bytes()

    existing_root = tmp_path / "existing"
    existing_root.mkdir()
    existing = existing_root / "spiece.model"
    existing.write_bytes(b"existing sentencepiece")
    retained = _ensure_sentencepiece_model(tokenizer, existing_root)
    assert retained == existing
    assert retained.read_bytes() == b"existing sentencepiece"


def test_octo_conversion_rejects_missing_sentencepiece_source(tmp_path):
    tokenizer = SimpleNamespace(vocab_file=str(tmp_path / "missing.model"))

    with pytest.raises(RuntimeError, match="source is missing or empty"):
        _ensure_sentencepiece_model(tokenizer, tmp_path)


def test_octo_config_lerobot_cli_override_and_derived_prior_statistics(tmp_path):
    config = load_config(PROJECT_ROOT / "configs" / "octo_small_libero_4x4090.yaml")
    config = apply_overrides(config, lerobot_path=str(tmp_path / "lerobot"))
    paths = resolved_paths(config)

    assert paths["lerobot"] == (tmp_path / "lerobot").resolve()
    assert paths["prior_dataset"] == paths["lerobot"] / "libero90"
    assert paths["target_dataset"] == (paths["lerobot"] / config["data"]["target_dataset"])
    assert paths["statistics"] == (paths["prior_dataset"] / "meta" / "stats.json")
    legacy_format = "rl" + "ds"
    assert legacy_format not in config["paths"]


def test_octo_cli_exposes_only_lerobot_data_override():
    from octo_small_libero.cli import build_parser

    option_strings = {
        option for action in build_parser()._actions for option in action.option_strings
    }
    assert "--lerobot-path" in option_strings
    assert "--max-steps" in option_strings
    assert "--smoke-test" in option_strings
    assert "--prior-top-percent" in option_strings
    assert "--prior-scores" in option_strings
    assert "--prior-prefiltered-scores" in option_strings
    assert "--sample-weights" in option_strings
    assert "--task-index" in option_strings
    assert "--all-tasks" in option_strings
    legacy_format = "rl" + "ds"
    assert f"--{legacy_format}-path" not in option_strings
    assert "--statistics-path" not in option_strings


def test_octo_max_steps_override_keeps_learning_rate_schedule_valid():
    config = load_config(PROJECT_ROOT / "configs" / "octo_small_libero_4x4090.yaml")
    config = apply_overrides(config, max_steps=12_000)

    assert config["train"]["max_steps"] == 12_000
    assert config["train"]["learning_rate"]["decay_steps"] == 12_000


def test_octo_sample_weights_override_requires_exact_local_batch_counts():
    config = load_config(PROJECT_ROOT / "configs" / "octo_small_libero_4x4090.yaml")

    updated = apply_overrides(config, sample_weights=[3.0, 1.0])
    assert updated["data"]["sample_weights"] == [3.0, 1.0]

    invalid = load_config(PROJECT_ROOT / "configs" / "octo_small_libero_4x4090.yaml")
    invalid["data"]["sample_weights"] = [3.0, 2.0]
    with pytest.raises(ConfigError, match="positive whole-number counts"):
        validate_config(invalid)

    invalid["data"]["sample_weights"] = [float("nan"), 1.0]
    with pytest.raises(ConfigError, match="positive finite numbers"):
        validate_config(invalid)


def test_octo_prior_percent_override_is_validated_and_isolates_output():
    config = load_config(PROJECT_ROOT / "configs" / "octo_small_libero_4x4090.yaml")
    updated = apply_overrides(config, prior_top_percent=12.5)

    assert updated["data"]["prior_selection"]["top_percent"] == 12.5
    assert updated["paths"]["output"].endswith("_top12p5pct")

    explicit = apply_overrides(
        config,
        prior_top_percent=10,
        output_dir="outputs/explicit",
    )
    assert explicit["paths"]["output"] == "outputs/explicit"

    with pytest.raises(ConfigError, match="in \\(0, 100\\]"):
        apply_overrides(config, prior_top_percent=0)


def test_octo_prefiltered_scores_override_enables_all_rows_without_percent():
    config = load_config(PROJECT_ROOT / "configs" / "octo_small_libero_4x4090.yaml")

    updated = apply_overrides(
        config,
        prior_prefiltered_scores="/data/sqcn/filter/top10pct/scores.csv",
    )

    assert updated["data"]["prior_selection"] == {
        "scores": "/data/sqcn/filter/top10pct/scores.csv",
        "top_percent": None,
        "prefiltered": True,
    }


def test_octo_config_rejects_prefiltered_scores_with_top_percent():
    config = load_config(PROJECT_ROOT / "configs" / "octo_small_libero_4x4090.yaml")
    config["data"]["prior_selection"].update(
        {"prefiltered": True, "top_percent": 10}
    )

    with pytest.raises(ConfigError, match="prefiltered.*top_percent"):
        validate_config(config)


def test_octo_cli_rejects_ranked_and_prefiltered_selection_together():
    from octo_small_libero.cli import parse_arguments

    repeated = parse_arguments(
        [
            "--all-tasks",
            "--prior-prefiltered-scores",
            "/data/first.csv",
            "--prior-prefiltered-scores",
            "/data/second.csv",
        ]
    )
    assert repeated.prior_prefiltered_scores == "/data/second.csv"

    with pytest.raises(SystemExit):
        parse_arguments(
            [
                "--all-tasks",
                "--prior-prefiltered-scores",
                "/data/sqcn.csv",
                "--prior-top-percent",
                "10",
            ]
        )
    with pytest.raises(SystemExit):
        parse_arguments(
            [
                "--all-tasks",
                "--prior-prefiltered-scores",
                "/data/sqcn.csv",
                "--prior-scores",
                "/data/tdus.csv",
            ]
        )


def test_octo_task_index_override_is_required_by_cli_and_isolates_output():
    from octo_small_libero.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--task-index", "-1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--task-index", "10"])

    arguments = parser.parse_args(["--task-index", "5"])
    assert arguments.task_index == 5
    assert arguments.all_tasks is False
    all_arguments = parser.parse_args(["--all-tasks"])
    assert all_arguments.task_index is None
    assert all_arguments.all_tasks is True
    weighted_arguments = parser.parse_args(
        ["--all-tasks", "--sample-weights", "3", "1"]
    )
    assert weighted_arguments.sample_weights == [3.0, 1.0]
    with pytest.raises(SystemExit):
        parser.parse_args(["--task-index", "5", "--all-tasks"])

    config = load_config(PROJECT_ROOT / "configs" / "octo_small_libero_4x4090.yaml")
    task_only = apply_overrides(config, target_task_index=5)
    assert task_only["paths"]["output"].endswith("_task-5")

    task_and_prior = apply_overrides(
        config,
        target_task_index=5,
        prior_top_percent=10,
    )
    assert task_and_prior["paths"]["output"].endswith("_task-5_top10pct")

    explicit = apply_overrides(
        config,
        target_task_index=5,
        output_dir="outputs/explicit-task",
    )
    assert explicit["paths"]["output"] == "outputs/explicit-task"


def test_octo_all_tasks_override_isolates_output():
    config = load_config(PROJECT_ROOT / "configs" / "octo_small_libero_4x4090.yaml")

    all_tasks = apply_overrides(config, target_all_tasks=True)
    assert all_tasks["data"]["target_task_index"] is None
    assert all_tasks["data"]["target_all_tasks"] is True
    assert all_tasks["paths"]["output"].endswith("_all-tasks")

    config_with_task = apply_overrides(config, target_task_index=5)
    all_tasks_from_task_config = apply_overrides(
        config_with_task,
        target_all_tasks=True,
    )
    assert all_tasks_from_task_config["data"]["target_task_index"] is None
    assert all_tasks_from_task_config["data"]["target_all_tasks"] is True

    all_tasks_and_prior = apply_overrides(
        config,
        target_all_tasks=True,
        prior_top_percent=10,
    )
    assert all_tasks_and_prior["paths"]["output"].endswith(
        "_all-tasks_top10pct"
    )


def test_octo_config_rejects_invalid_target_task_index():
    config = load_config(PROJECT_ROOT / "configs" / "octo_small_libero_4x4090.yaml")
    for value in (-1, 10, True):
        config["data"]["target_task_index"] = value
        with pytest.raises(ConfigError, match="target_task_index"):
            validate_config(config)


def test_octo_config_rejects_conflicting_or_invalid_all_tasks_selection():
    config = load_config(PROJECT_ROOT / "configs" / "octo_small_libero_4x4090.yaml")
    config["data"]["target_all_tasks"] = "yes"
    with pytest.raises(ConfigError, match="target_all_tasks must be a bool"):
        validate_config(config)

    config = load_config(PROJECT_ROOT / "configs" / "octo_small_libero_4x4090.yaml")
    config["data"]["target_task_index"] = 5
    config["data"]["target_all_tasks"] = True
    with pytest.raises(ConfigError, match="cannot both be enabled"):
        validate_config(config)


def test_octo_config_rejects_removing_wrist_camera():
    config = load_config(PROJECT_ROOT / "configs" / "octo_small_libero_4x4090.yaml")
    config["model"]["required_observation_tokenizers"] = ["primary"]
    with pytest.raises(ConfigError, match="retain both primary and wrist"):
        validate_config(config)


def test_local_flax_octo_small_source_checkpoint_contract():
    path = Path("/data/dwb/models/octo-small")
    if not path.is_dir():
        pytest.skip("Local Octo-small checkpoint is not mounted")
    report = inspect_flax_octo_checkpoint(path, step=270000)
    assert report["transformer_layers"] == 12
    assert report["pretrained_action_horizon"] == 4
    assert {"primary", "wrist"}.issubset(report["observation_tokenizers"])


def test_octo_pytorch_runtime_has_no_legacy_framework_imports():
    runtime = PROJECT_ROOT / "src" / "octo_small_libero"
    conversion_only = {"convert_checkpoint.py"}
    banned = {"tensorflow", "dlimp", "jax", "flax", "optax", "orbax"}
    found = []
    for path in runtime.glob("*.py"):
        if path.name in conversion_only:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = {node.module.split(".", 1)[0]}
            else:
                continue
            for name in sorted(roots & banned):
                found.append(f"{path.name}:{node.lineno}:{name}")
    assert not found

    requirements = (
        PROJECT_ROOT / "requirements-octo-pytorch.txt"
    ).read_text(encoding="utf-8").lower()
    for package in banned:
        assert package not in requirements
