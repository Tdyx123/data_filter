from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from cocore import cli as cocore_cli
from cocore_bridge_v2 import cli as bridge_cli
from cocore_bridge_v2.config import build_config


def _args(module, command, output):
    args = [command, "--output-dir", str(output)]
    if module is bridge_cli:
        args += ["--relation", "sequence", "--relation-weight", "1"]
    return args


@pytest.mark.parametrize("module", [cocore_cli, bridge_cli])
@pytest.mark.parametrize("command", ["build-graph", "select", "run", "validate"])
@pytest.mark.parametrize("value", [None, "1", "20"])
@pytest.mark.parametrize("metric", ["support", "support_old"])
def test_support_k_reaches_pipeline_config(module, command, value, metric, tmp_path, monkeypatch):
    received = {}
    output = tmp_path / "output"
    output.mkdir()
    args = _args(module, command, output) + ["--reliability-metrics", metric]
    if module is cocore_cli:
        config = yaml.safe_load(Path(cocore_cli.__file__).with_name("config_libero90.yaml").read_text())
        config["quality"]["knn"] = 7
        config_path = output / "resolved_config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        # Explicit support-k on validate must load the saved configuration itself.
        if command != "validate" or value is None:
            args += ["--config", str(config_path)]
    else:
        monkeypatch.setattr(module, "validate_bridge_dataset", lambda _: None)
    if value is not None:
        args += ["--support-k", value]

    def capture(config, **kwargs):
        received.update(config)
        if command == "build-graph":
            return output, None, None, SimpleNamespace(sample_ids=[]), "fingerprint"
        return output

    def validate(output_dir, *, config):
        received.update(config)
        return {"status": "valid"}

    monkeypatch.setattr(module, "graph_stage", capture)
    monkeypatch.setattr(module, "select_stage", capture)
    monkeypatch.setattr(module, "run_pipeline", capture)
    monkeypatch.setattr(module, "validate_output", validate)
    module.main(args)
    expected = int(value) if value is not None else (7 if module is cocore_cli else 10)
    assert received["reliability_metrics"] == [metric]
    assert received["quality"]["knn"] == expected
    assert received["graph"]["knn"] == 32


@pytest.mark.parametrize("module", [cocore_cli, bridge_cli])
@pytest.mark.parametrize("command", ["build-graph", "select", "run", "validate"])
@pytest.mark.parametrize("value", ["0", "-1", "1.5", "abc"])
def test_support_k_rejects_invalid_values(module, command, value, tmp_path):
    with pytest.raises(SystemExit) as error:
        module.build_parser().parse_args(_args(module, command, tmp_path) + ["--support-k", value])
    assert error.value.code == 2


@pytest.mark.parametrize("module", [cocore_cli, bridge_cli])
@pytest.mark.parametrize("command", ["scan", "encode"])
def test_support_k_is_only_available_for_graph_commands(module, command, tmp_path):
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(_args(module, command, tmp_path) + ["--support-k", "20"])


def test_bridge_build_config_support_k_override():
    config = build_config(relation="sequence", relation_weight=1, support_k=20)
    assert config["quality"]["knn"] == 20
    assert config["graph"]["knn"] == 32


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "20"])
def test_bridge_build_config_rejects_invalid_support_k(value):
    with pytest.raises(ValueError, match="support_k"):
        build_config(relation="sequence", relation_weight=1, support_k=value)


@pytest.mark.parametrize("module", [cocore_cli, bridge_cli])
@pytest.mark.parametrize("command", ["build-graph", "select", "run", "validate"])
def test_support_modes_are_rejected_together_before_pipeline(module, command, tmp_path):
    args = _args(module, command, tmp_path) + [
        "--reliability-metrics", "support", "support_old",
    ]
    with pytest.raises(SystemExit, match="mutually exclusive"):
        module.main(args)


def test_bridge_config_support_old_and_mutual_exclusion():
    config = build_config(relation="sequence", relation_weight=1,
                          reliability_metrics=["progress", "support_old"], support_k=7)
    assert config["reliability_metrics"] == ["support_old", "progress"]
    assert config["quality"]["knn"] == 7
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_config(relation="sequence", relation_weight=1,
                     reliability_metrics=["support_old", "support"])
