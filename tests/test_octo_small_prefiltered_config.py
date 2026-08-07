from pathlib import Path

import pytest

from octo_small_libero.cli import build_parser, parse_arguments
from octo_small_libero.config import (
    ConfigError,
    apply_overrides,
    load_config,
    resolved_paths,
    validate_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "octo_small_libero_4x4090.yaml"


def test_cli_exposes_only_the_unified_prefiltered_prior_option():
    option_strings = {
        option for action in build_parser()._actions for option in action.option_strings
    }

    assert "--prior-prefiltered-scores" in option_strings
    assert "--prior-top-percent" not in option_strings
    assert "--prior-scores" not in option_strings
    assert "--prior-relcore-manifest" not in option_strings
    assert "--prior-quality-filter-scores" not in option_strings


@pytest.mark.parametrize(
    "removed_arguments",
    [
        ["--prior-top-percent", "10"],
        ["--prior-scores", "/data/tdus.csv"],
        ["--prior-relcore-manifest", "/data/selected.jsonl"],
        ["--prior-quality-filter-scores", "/data/quality.csv"],
    ],
)
def test_cli_rejects_removed_prior_options(removed_arguments):
    with pytest.raises(SystemExit):
        parse_arguments(
            [
                "--all-tasks",
                "--output-dir",
                "outputs/test",
                *removed_arguments,
            ]
        )


def test_target_only_rejects_the_unified_prefiltered_prior_option():
    with pytest.raises(SystemExit):
        parse_arguments(
            [
                "--all-tasks",
                "--output-dir",
                "outputs/test",
                "--target-only",
                "--prior-prefiltered-scores",
                "/data/selected.csv",
            ]
        )


def test_config_rejects_target_only_with_prefiltered_prior():
    config = load_config(CONFIG_PATH)
    config["data"]["target_only"] = True
    config["data"]["prior_selection"]["prefiltered_scores"] = "/data/selected.csv"

    with pytest.raises(ConfigError, match="target_only.*prefiltered_scores"):
        validate_config(config)


def test_prefiltered_override_uses_the_single_config_key_and_resolved_path():
    config = load_config(CONFIG_PATH)

    assert config["data"]["prior_selection"] == {"prefiltered_scores": None}

    updated = apply_overrides(
        config,
        prior_prefiltered_scores="/data/selected.csv",
        output_dir="outputs/test",
    )

    assert updated["data"]["prior_selection"] == {
        "prefiltered_scores": "/data/selected.csv"
    }
    assert resolved_paths(updated)["prior_prefiltered_scores"] == Path(
        "/data/selected.csv"
    )
