from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bridge_info() -> dict[str, object]:
    video_feature = {
        "dtype": "video",
        "shape": [256, 256, 3],
        "names": ["height", "width", "rgb"],
        "info": {
            "video.fps": 5.0,
            "video.height": 256,
            "video.width": 256,
            "video.channels": 3,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }
    features = {
        f"observation.images.image_{index}": copy.deepcopy(video_feature) for index in range(4)
    }
    features.update(
        {
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": {
                    "motors": [
                        "x",
                        "y",
                        "z",
                        "roll",
                        "pitch",
                        "yaw",
                        "pad",
                        "gripper",
                    ]
                },
            },
            "action": {"dtype": "float32", "shape": [7]},
        }
    )
    return {
        "codebase_version": "v2.0",
        "robot_type": "widowx",
        "fps": 5,
        "features": features,
    }


def _write_bridge_info(root: Path, payload: dict[str, object]) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_av1_video(path: Path, frames: np.ndarray) -> None:
    av = pytest.importorskip("av")
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("av1", rate=5)
    except av.error.ValueError:
        container.close()
        pytest.skip("The PyAV build has no AV1 encoder")
    stream.width = int(frames.shape[2])
    stream.height = int(frames.shape[1])
    stream.pix_fmt = "yuv420p"
    for array in frames:
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _write_synthetic_bridge_dataset(
    root: Path,
    *,
    stop_first_valid_episode: bool = False,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    tasks = [
        {"task_index": 0, "task": ""},
        {"task_index": 1, "task": "move object one"},
        {"task_index": 2, "task": "move object two"},
    ]
    episodes = [
        {"episode_index": 0, "tasks": [""], "length": 45},
        {"episode_index": 1, "tasks": ["move object one"], "length": 605},
        {"episode_index": 2, "tasks": ["move object two"], "length": 605},
    ]
    _write_jsonl(meta / "tasks.jsonl", tasks)
    _write_jsonl(meta / "episodes.jsonl", episodes)

    info = _bridge_info()
    info.update(
        {
            "total_episodes": 3,
            "total_frames": 1255,
            "total_tasks": 3,
            "total_videos": 12,
            "total_chunks": 1,
            "chunks_size": 1000,
            "splits": {"train": "0:3"},
            "data_path": ("data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"),
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            ),
        }
    )
    for index in range(4):
        feature = info["features"][f"observation.images.image_{index}"]
        feature["shape"] = [32, 32, 3]
        feature["info"]["video.height"] = 32
        feature["info"]["video.width"] = 32
    info["features"].update(
        {
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        }
    )
    _write_bridge_info(root, info)

    for episode_id in (1, 2):
        steps = np.arange(605, dtype=np.float32)
        states = np.zeros((605, 8), dtype=np.float32)
        if not (stop_first_valid_episode and episode_id == 1):
            states[:, 0] = steps * 0.02
            states[:, 3] = steps * 0.07
        actions = np.zeros((605, 7), dtype=np.float32)
        actions[:, 0] = 0.01
        table = pa.table(
            {
                "observation.state": pa.array(
                    states.tolist(), type=pa.list_(pa.float32(), list_size=8)
                ),
                "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), list_size=7)),
                "timestamp": pa.array(steps / 5.0, type=pa.float32()),
                "frame_index": pa.array(np.arange(605), type=pa.int64()),
                "episode_index": pa.array([episode_id] * 605, type=pa.int64()),
                "task_index": pa.array([episode_id] * 605, type=pa.int64()),
            }
        )
        pq.write_table(table, data / f"episode_{episode_id:06d}.parquet")
        pixel_values = np.mod(np.arange(605) + episode_id * 37, 255).astype(np.uint8)
        frames = np.broadcast_to(pixel_values[:, None, None, None], (605, 32, 32, 3)).copy()
        _write_av1_video(
            root
            / "videos"
            / "chunk-000"
            / "observation.images.image_0"
            / f"episode_{episode_id:06d}.mp4",
            frames,
        )


def test_module_entrypoint_exposes_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cocore_bridge_v2", "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "BridgeData V2" in result.stdout


def test_package_exposes_only_version() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import cocore_bridge_v2 as package; "
                "print(package.__version__); print(package.__all__)"
            ),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["0.12.0", "['__version__']"]


def test_bridge_config_fixes_dataset_and_cocore_contract(tmp_path: Path) -> None:
    from cocore.config import to_relcore_config
    from cocore_bridge_v2.config import build_config

    dataset_path = tmp_path / "mounted-bridge"
    config = build_config(
        relation="sequence",
        relation_weight=1.5,
        selection_ratio=0.2,
        dataset_path=dataset_path,
        max_episodes=100,
    )

    assert config["dataset"] == {
        "type": "lerobot",
        "name": "bridge_orig_1.0.0",
        "path": str(dataset_path),
        "use_images": True,
        "empty_task_policy": "exclude",
        "feature_keys": {
            "action": "action",
            "timestamp": "timestamp",
            "frame_index": "frame_index",
            "episode_index": "episode_index",
            "vector_observations": ["observation.state"],
            "image_observations": ["observation.images.image_0"],
        },
    }
    assert "clip" not in config
    assert "relation" not in config
    assert "normalization" not in config
    assert config["encoding"] == {
        "visual_dim": 128,
        "pca_fit_max_samples": None,
        "quantile_low": 0.01,
        "quantile_high": 0.99,
        "epsilon": 1.0e-8,
    }
    assert config["prototypes"]["method"] == "motion_primitives"
    assert config["prototypes"]["profile"] == "bridge_v2"
    assert config["prototypes"]["use_stop_bucket"] is True
    assert set(config["prototypes"]) == {
        "method",
        "profile",
        "batch_size",
        "max_iter",
        "tol",
        "num_threads",
        "use_stop_bucket",
    }
    assert config["prototypes"]["num_threads"] == 4
    assert config["reliability_metrics"] == [
        "support",
        "progress",
        "action_variation",
        "visual_action_consistency",
    ]
    assert config["objective"] == {"relation": "sequence", "relation_weight": 1.5}
    assert config["selection"] == {"ratio": 0.2, "budget": None}
    assert config["runtime"]["max_episodes"] == 100
    assert config["output"]["directory"] == ("outputs/cocore_bridge_v2/bridge_orig_1.0.0")
    translated = to_relcore_config(config)
    assert "num_threads" not in translated["prototypes"]
    assert "profile" not in translated["prototypes"]
    assert "tol" not in translated["prototypes"]
    assert translated["clip"] == {"length": 7, "stride": 7}
    assert translated["selection"]["quota_mode"] == "none"
    assert translated["selection"]["minimum_per_task"] == 0


def test_bridge_config_accepts_support_only_reliability(tmp_path: Path) -> None:
    from cocore_bridge_v2.config import build_config

    config = build_config(
        relation="sequence",
        relation_weight=1.0,
        dataset_path=tmp_path / "bridge",
        reliability_metrics=("support",),
    )

    assert config["reliability_metrics"] == ["support"]


def test_bridge_config_rejects_removed_selection_method(tmp_path: Path) -> None:
    from cocore_bridge_v2.config import build_config

    with pytest.raises(TypeError, match="selection_method"):
        build_config(
            relation="sequence",
            relation_weight=1.0,
            selection_method="random_multibranch",  # type: ignore[call-arg]
            dataset_path=tmp_path / "bridge",
        )


def test_bridge_config_can_disable_stop_bucket(tmp_path: Path) -> None:
    from cocore_bridge_v2.config import build_config

    config = build_config(
        relation="sequence",
        relation_weight=1.0,
        dataset_path=tmp_path / "bridge",
        use_stop_bucket=False,
    )

    assert config["prototypes"]["use_stop_bucket"] is False


@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_bridge_config_rejects_non_boolean_stop_bucket(value: object) -> None:
    from cocore_bridge_v2.config import build_config

    with pytest.raises(ValueError, match="prototypes.use_stop_bucket"):
        build_config(
            relation="sequence",
            relation_weight=1.0,
            use_stop_bucket=value,  # type: ignore[arg-type]
        )


def test_preflight_accepts_fixed_bridge_v2_schema(tmp_path: Path) -> None:
    from cocore_bridge_v2.preflight import validate_bridge_dataset

    root = tmp_path / "bridge"
    _write_bridge_info(root, _bridge_info())

    info = validate_bridge_dataset(root)

    assert info["codebase_version"] == "v2.0"
    assert info["features"]["observation.images.image_0"]["dtype"] == "video"


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("codebase_version", "v3.0", "LeRobot v2.0"),
        ("robot_type", "panda", "widowx"),
        ("fps", 10, "5 Hz"),
    ],
)
def test_preflight_rejects_wrong_dataset_identity(
    tmp_path: Path,
    key: str,
    value: object,
    message: str,
) -> None:
    from cocore_bridge_v2.preflight import BridgeDatasetError, validate_bridge_dataset

    payload = _bridge_info()
    payload[key] = value
    root = tmp_path / "bridge"
    _write_bridge_info(root, payload)

    with pytest.raises(BridgeDatasetError, match=message):
        validate_bridge_dataset(root)


@pytest.mark.parametrize(
    ("feature_key", "replacement", "message"),
    [
        (
            "observation.state",
            {
                "dtype": "float32",
                "shape": [7],
                "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
            },
            "observation.state",
        ),
        ("action", {"dtype": "float32", "shape": [8]}, "action"),
        (
            "observation.images.image_0",
            {"dtype": "image", "shape": [256, 256, 3]},
            "RGB video",
        ),
    ],
)
def test_preflight_rejects_wrong_feature_schema(
    tmp_path: Path,
    feature_key: str,
    replacement: dict[str, object],
    message: str,
) -> None:
    from cocore_bridge_v2.preflight import BridgeDatasetError, validate_bridge_dataset

    payload = _bridge_info()
    payload["features"][feature_key] = replacement
    root = tmp_path / "bridge"
    _write_bridge_info(root, payload)

    with pytest.raises(BridgeDatasetError, match=message):
        validate_bridge_dataset(root)


def test_preflight_rejects_wrong_state_axis_names(tmp_path: Path) -> None:
    from cocore_bridge_v2.preflight import BridgeDatasetError, validate_bridge_dataset

    payload = _bridge_info()
    payload["features"]["observation.state"]["names"]["motors"][-1] = "jaw"
    root = tmp_path / "bridge"
    _write_bridge_info(root, payload)

    with pytest.raises(BridgeDatasetError, match="axis names"):
        validate_bridge_dataset(root)


def test_preflight_normalizes_malformed_video_fps_to_bridge_error(tmp_path: Path) -> None:
    from cocore_bridge_v2.preflight import BridgeDatasetError, validate_bridge_dataset

    payload = _bridge_info()
    payload["features"]["observation.images.image_0"]["info"]["video.fps"] = "fast"
    root = tmp_path / "bridge"
    _write_bridge_info(root, payload)

    with pytest.raises(BridgeDatasetError, match="RGB video"):
        validate_bridge_dataset(root)


@pytest.mark.parametrize(
    "command",
    ["scan", "encode", "build-graph", "select", "run", "validate"],
)
def test_every_command_requires_relation_and_weight(command: str) -> None:
    from cocore_bridge_v2 import cli

    prefix = [command]
    if command == "validate":
        prefix += ["--output-dir", "result"]
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(prefix)
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(prefix + ["--relation", "sequence"])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(prefix + ["--relation-weight", "1"])


@pytest.mark.parametrize(
    "command",
    ["scan", "encode", "build-graph", "select", "run", "validate"],
)
def test_every_command_accepts_explicit_relation_and_weight(command: str) -> None:
    from cocore_bridge_v2 import cli

    arguments = [command]
    if command == "validate":
        arguments += ["--output-dir", "result"]
    arguments += [
        "--relation",
        "cooccurrence",
        "--relation-weight",
        "2.5",
        "--max-episodes",
        "7",
    ]

    parsed = cli.build_parser().parse_args(arguments)

    assert parsed.relation == "cooccurrence"
    assert parsed.relation_weight == 2.5
    assert parsed.max_episodes == 7
    if command in {"select", "run", "validate"}:
        assert parsed.selection_ratio == 0.10


@pytest.mark.parametrize("command", ["select", "run", "validate"])
def test_bridge_selection_commands_reject_removed_selection_method(command: str) -> None:
    from cocore_bridge_v2 import cli

    arguments = [
        command,
        "--relation",
        "sequence",
        "--relation-weight",
        "1",
        "--selection-method",
        "random_multibranch",
    ]
    if command == "validate":
        arguments += ["--output-dir", "result"]

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(arguments)


@pytest.mark.parametrize("command", ["build-graph", "select", "run", "validate"])
def test_graph_commands_accept_stop_bucket_disable_flag(command: str) -> None:
    from cocore_bridge_v2 import cli

    arguments = [
        command,
        "--relation",
        "sequence",
        "--relation-weight",
        "1",
        "--no-use-stop-bucket",
    ]
    if command == "validate":
        arguments += ["--output-dir", "result"]

    parsed = cli.build_parser().parse_args(arguments)

    assert parsed.no_use_stop_bucket is True


@pytest.mark.parametrize("command", ["build-graph", "select", "run", "validate"])
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["support"], ["support"]),
        (
            ["visual_action_consistency", "action_variation", "support"],
            ["visual_action_consistency", "action_variation", "support"],
        ),
    ],
)
def test_graph_commands_accept_reliability_metrics(
    command: str,
    values: list[str],
    expected: list[str],
) -> None:
    from cocore_bridge_v2 import cli

    arguments = [
        command,
        "--relation",
        "sequence",
        "--relation-weight",
        "1",
        "--reliability-metrics",
        *values,
    ]
    if command == "validate":
        arguments += ["--output-dir", "result"]

    parsed = cli.build_parser().parse_args(arguments)

    assert parsed.reliability_metrics == expected


@pytest.mark.parametrize("command", ["scan", "encode"])
def test_non_graph_commands_reject_stop_bucket_disable_flag(command: str) -> None:
    from cocore_bridge_v2 import cli

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                command,
                "--relation",
                "sequence",
                "--relation-weight",
                "1",
                "--no-use-stop-bucket",
            ]
        )


@pytest.mark.parametrize("command", ["scan", "encode"])
def test_non_graph_commands_reject_reliability_metrics(command: str) -> None:
    from cocore_bridge_v2 import cli

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                command,
                "--relation",
                "sequence",
                "--relation-weight",
                "1",
                "--reliability-metrics",
                "support",
            ]
        )


@pytest.mark.parametrize("value", ["none", "support,progress", "smoothness"])
def test_graph_commands_reject_unsupported_reliability_metrics(value: str) -> None:
    from cocore_bridge_v2 import cli

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                "run",
                "--relation",
                "sequence",
                "--relation-weight",
                "1",
                "--reliability-metrics",
                value,
            ]
        )


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1"])
def test_cli_rejects_invalid_relation_weight(value: str) -> None:
    from cocore_bridge_v2 import cli

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["scan", "--relation", "sequence", "--relation-weight", value]
        )


@pytest.mark.parametrize("value", ["0", "-1", "1.01", "nan", "inf"])
def test_cli_rejects_invalid_selection_ratio(value: str) -> None:
    from cocore_bridge_v2 import cli

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                "run",
                "--relation",
                "sequence",
                "--relation-weight",
                "1",
                "--selection-ratio",
                value,
            ]
        )


@pytest.mark.parametrize("command", ["scan", "validate"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_non_positive_max_episodes(command: str, value: str) -> None:
    from cocore_bridge_v2 import cli

    arguments = [command]
    if command == "validate":
        arguments += ["--output-dir", "result"]
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            arguments
            + [
                "--relation",
                "sequence",
                "--relation-weight",
                "1",
                "--max-episodes",
                value,
            ]
        )


def test_scan_cli_preflights_and_delegates_resolved_bridge_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cocore_bridge_v2 import cli

    dataset = tmp_path / "bridge"
    output = tmp_path / "output"
    _write_bridge_info(dataset, _bridge_info())
    received: dict[str, object] = {}

    def fake_scan_stage(config, **kwargs):
        received["config"] = config
        received["kwargs"] = kwargs
        return output, None, [object(), object()], "scan-fingerprint"

    monkeypatch.setattr(cli, "scan_stage", fake_scan_stage)

    cli.main(
        [
            "scan",
            "--relation",
            "sequence",
            "--relation-weight",
            "1.5",
            "--dataset-path",
            str(dataset),
            "--output-dir",
            str(output),
            "--max-episodes",
            "100",
            "--force",
        ]
    )

    config = received["config"]
    assert config["dataset"]["path"] == str(dataset)
    assert config["objective"] == {"relation": "sequence", "relation_weight": 1.5}
    assert config["runtime"]["max_episodes"] == 100
    assert received["kwargs"] == {"output_dir": str(output), "force": True}
    assert capsys.readouterr().out.strip() == f"cocore_output={output} clips=2"


@pytest.mark.parametrize(
    ("command", "target", "result", "expected"),
    [
        (
            "encode",
            "encode_stage",
            lambda root: (root, None, SimpleNamespace(clips=(1, 2, 3))),
            "cocore_output={root} clips=3",
        ),
        (
            "build-graph",
            "graph_stage",
            lambda root: (root, None, None, SimpleNamespace(sample_ids=(1, 2)), "fingerprint"),
            "cocore_output={root}/graph-18-motion-hard-nearest-pca nodes=2",
        ),
        (
            "select",
            "select_stage",
            lambda root: root / "select-sequence-w1-top20pct",
            "cocore_output={root}/select-sequence-w1-top20pct",
        ),
        (
            "run",
            "run_pipeline",
            lambda root: root / "select-sequence-w1-top20pct",
            "cocore_output={root}/select-sequence-w1-top20pct",
        ),
    ],
)
def test_execution_commands_delegate_to_matching_cocore_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    target: str,
    result,
    expected: str,
) -> None:
    from cocore_bridge_v2 import cli

    dataset = tmp_path / "bridge"
    output = tmp_path / "output"
    _write_bridge_info(dataset, _bridge_info())
    received: dict[str, object] = {}

    def fake_stage(config, **kwargs):
        received["config"] = config
        received["kwargs"] = kwargs
        return result(output)

    monkeypatch.setattr(cli, target, fake_stage)
    arguments = [
        command,
        "--relation",
        "sequence",
        "--relation-weight",
        "1",
        "--dataset-path",
        str(dataset),
        "--output-dir",
        str(output),
    ]
    if command in {"select", "run"}:
        arguments += ["--selection-ratio", "0.2"]

    cli.main(arguments)

    expected_ratio = 0.2 if command in {"select", "run"} else 0.1
    assert received["config"]["selection"]["ratio"] == expected_ratio
    assert received["kwargs"] == {"output_dir": str(output), "force": False}
    assert capsys.readouterr().out.strip() == expected.format(root=output)


def test_run_cli_disables_stop_bucket_in_delegated_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cocore_bridge_v2 import cli

    dataset = tmp_path / "bridge"
    output = tmp_path / "output"
    _write_bridge_info(dataset, _bridge_info())
    received: dict[str, object] = {}

    def fake_run_pipeline(config, **kwargs):
        received["config"] = config
        return output / "select-sequence-w1-top10pct"

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    cli.main(
        [
            "run",
            "--relation",
            "sequence",
            "--relation-weight",
            "1",
            "--dataset-path",
            str(dataset),
            "--output-dir",
            str(output),
            "--no-use-stop-bucket",
        ]
    )

    assert received["config"]["prototypes"]["use_stop_bucket"] is False
    capsys.readouterr()


def test_run_cli_passes_support_only_to_delegated_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cocore_bridge_v2 import cli

    dataset = tmp_path / "bridge"
    output = tmp_path / "output"
    _write_bridge_info(dataset, _bridge_info())
    received: dict[str, object] = {}

    def fake_run_pipeline(config, **kwargs):
        received["config"] = config
        return output / "select-sequence-w1-top10pct"

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    cli.main(
        [
            "run",
            "--relation",
            "sequence",
            "--relation-weight",
            "1",
            "--dataset-path",
            str(dataset),
            "--output-dir",
            str(output),
            "--reliability-metrics",
            "support",
        ]
    )

    assert received["config"]["reliability_metrics"] == ["support"]
    capsys.readouterr()


def test_run_cli_canonicalizes_reliability_metric_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cocore_bridge_v2 import cli

    dataset = tmp_path / "bridge"
    output = tmp_path / "output"
    _write_bridge_info(dataset, _bridge_info())
    received: dict[str, object] = {}

    def fake_run_pipeline(config, **kwargs):
        received["config"] = config
        return output / "select-sequence-w1-top10pct"

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    cli.main(
        [
            "run",
            "--relation",
            "sequence",
            "--relation-weight",
            "1",
            "--dataset-path",
            str(dataset),
            "--output-dir",
            str(output),
            "--reliability-metrics",
            "action_variation",
            "support",
        ]
    )

    assert received["config"]["reliability_metrics"] == [
        "support",
        "action_variation",
    ]
    capsys.readouterr()


def test_run_cli_rejects_duplicate_reliability_metrics() -> None:
    from cocore_bridge_v2 import cli

    with pytest.raises(SystemExit, match="cannot contain duplicates"):
        cli.main(
            [
                "run",
                "--relation",
                "sequence",
                "--relation-weight",
                "1",
                "--reliability-metrics",
                "support",
                "support",
            ]
        )


def test_validate_cli_passes_custom_dataset_path_without_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cocore_bridge_v2 import cli

    dataset_path = tmp_path / "custom-bridge"
    result_path = tmp_path / "select-sequence-w1-top20pct"
    received: dict[str, object] = {}

    def fake_validate_output(output_dir, *, config):
        received["output_dir"] = output_dir
        received["config"] = config
        return {"status": "valid", "selected_clips": 7}

    def fail_preflight(_):
        raise AssertionError("validate must not run preflight")

    monkeypatch.setattr(cli, "validate_output", fake_validate_output)
    monkeypatch.setattr(cli, "validate_bridge_dataset", fail_preflight)

    cli.main(
        [
            "validate",
            "--output-dir",
            str(result_path),
            "--dataset-path",
            str(dataset_path),
            "--relation",
            "sequence",
            "--relation-weight",
            "1",
            "--selection-ratio",
            "0.2",
            "--max-episodes",
            "11",
            "--no-use-stop-bucket",
        ]
    )

    assert received["output_dir"] == str(result_path)
    assert received["config"]["dataset"]["path"] == str(dataset_path)
    assert received["config"]["selection"]["ratio"] == 0.2
    assert received["config"]["runtime"]["max_episodes"] == 11
    assert received["config"]["prototypes"]["use_stop_bucket"] is False
    assert json.loads(capsys.readouterr().out) == {
        "selected_clips": 7,
        "status": "valid",
    }


def test_synthetic_bridge_dataset_runs_cocore_with_only_image_zero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import pyarrow.parquet as pq

    from cocore.pipeline import run_pipeline, validate_output
    from cocore_bridge_v2.config import build_config
    from cocore_bridge_v2.preflight import validate_bridge_dataset

    class DummyVisualEncoder:
        output_dim = 3

        def encode(self, images: np.ndarray) -> np.ndarray:
            values = images[:, 0, 0, 0].astype(np.float32)
            return np.stack([values + 1.0, values + 2.0, values + 4.0], axis=1)

    dataset = tmp_path / "bridge"
    output = tmp_path / "output"
    _write_synthetic_bridge_dataset(dataset)
    validate_bridge_dataset(dataset)
    config = build_config(
        relation="sequence",
        relation_weight=1.0,
        selection_ratio=0.5,
        dataset_path=dataset,
    )
    config["visual"]["encoder"] = "dummy"
    config["runtime"]["num_workers"] = 0
    config["quality"]["knn"] = 2
    config["graph"]["knn"] = 2
    config["prototypes"]["max_iter"] = 2
    config["selection"]["budget"] = 10

    result = run_pipeline(config, output_dir=output, visual_encoder=DummyVisualEncoder())

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cocore_timing step=scan " in captured.err
    assert "cocore_timing step=select " in captured.err

    scan_manifest = json.loads((output / "scan" / "manifest.json").read_text())
    assert scan_manifest["dataset_summary"] == {
        "source_episodes": 3,
        "indexed_episodes": 2,
        "retained_episodes": 2,
        "excluded_episodes": 1,
        "excluded_empty_task_episodes": 1,
    }
    assert scan_manifest["clips"] == 174
    assert scan_manifest["clip_length"] == 7
    scanned_episode_ids = pq.read_table(output / "scan" / "episodes.parquet")[
        "episode_id"
    ].to_pylist()
    assert scanned_episode_ids == [1, 2]
    scanned_clips = pq.read_table(output / "scan" / "clips.parquet").to_pylist()
    assert {row["length"] for row in scanned_clips} == {7}
    encode_manifest = json.loads((output / "encode" / "manifest.json").read_text())
    assert encode_manifest["clips"] == 174
    assert encode_manifest["clip_length"] == 7
    assert encode_manifest["clip_anchors"] == [0, 3, 6]
    assert encode_manifest["visual_half_windows"] == [[0, 4], [3, 7]]
    assert encode_manifest["visual_half_encoding"] == "l2_normalized_four_frame_mean"
    assert np.load(output / "encode" / "state_sequences.npy").shape == (174, 7, 8)
    selected = [
        json.loads(line) for line in (result / "selected_manifest.jsonl").read_text().splitlines()
    ]
    all_rows = pq.read_table(result / "all_clips.parquet").to_pylist()
    assert len(selected) == 10
    assert all("roll positive" in row["primary_action_label"] for row in all_rows)
    assert all(
        any("roll positive" in label for label in row["prototype_action_labels"])
        for row in all_rows
    )
    select_manifest = json.loads((result / "manifest.json").read_text())
    run_manifest = json.loads((result / "run_manifest.json").read_text())
    graph_root = output / "graph-18-motion-hard-nearest-pca"
    graph_manifest = json.loads((graph_root / "manifest.json").read_text())
    catalog = json.loads((graph_root / "prototype_catalog.json").read_text())
    assert select_manifest["producer"] == "cocore"
    assert select_manifest["cocore_version"] == "0.19.0"
    assert select_manifest["prototype_profile"] == "bridge_v2"
    assert run_manifest["producer"] == "cocore"
    assert run_manifest["cocore_version"] == "0.19.0"
    assert run_manifest["stage_directories"]["graph"] == "graph-18-motion-hard-nearest-pca"
    assert run_manifest["prototype_schema_version"] == 10
    assert run_manifest["prototype_profile"] == "bridge_v2"
    assert run_manifest["clip_length"] == 7
    assert run_manifest["clip_anchors"] == [0, 3, 6]
    assert run_manifest["visual_half_windows"] == [[0, 4], [3, 7]]
    assert run_manifest["visual_half_encoding"] == "l2_normalized_four_frame_mean"
    assert run_manifest["trajectory_window_length"] == 4
    assert run_manifest["trajectory_horizon"] == 3
    assert run_manifest["trajectory_window_max_gap"] == 2
    assert run_manifest["trajectory_window_policy"] == ("full_coverage_max_gap_2_tail_rebalanced")
    assert graph_manifest["prototype_profile"] == "bridge_v2"
    assert graph_manifest["prototype_visual_normalization"] == (
        "l2_normalized_four_frame_mean_after_projection"
    )
    assert graph_manifest["trajectory_window_length"] == 4
    assert graph_manifest["trajectory_horizon"] == 3
    assert graph_manifest["trajectory_window_max_gap"] == 2
    assert graph_manifest["trajectory_window_policy"] == ("full_coverage_max_gap_2_tail_rebalanced")
    assert graph_manifest["motion_primitive"]["roll_labels"] == {
        "positive": "roll positive",
        "negative": "roll negative",
    }
    assert catalog["schema_version"] == 10
    assert catalog["profile"] == "bridge_v2"
    assert catalog["constants"]["primitive_thresholds"] == {
        "translation": 0.03,
        "roll": 0.18,
        "tilt": 0.18,
        "rotation": 0.24,
        "gripper": 0.2,
    }
    assert catalog["constants"]["min_action_frequency"] == 0.005
    assert catalog["constants"]["roll_labels"] == {
        "positive": "roll positive",
        "negative": "roll negative",
    }
    assert catalog["constants"]["cyclic_axes"] == [3, 5]
    assert catalog["total_raw_actions"] == 604
    assert catalog["constants"]["trajectory_window_length"] == 4
    assert catalog["constants"]["trajectory_window_max_gap"] == 2
    assert catalog["constants"]["trajectory_window_policy"] == (
        "full_coverage_max_gap_2_tail_rebalanced"
    )
    assert catalog["constants"]["visual_half_windows"] == [[0, 4], [3, 7]]
    assert catalog["constants"]["visual_half_encoding"] == (
        "l2_normalized_mean_of_four_projected_frames"
    )
    assert any(
        leaf["action_label"] == "move forward, roll positive" for leaf in catalog["leaf_prototypes"]
    )
    assert validate_output(result, config=config) == {
        "status": "valid",
        "selected_clips": 10,
    }

    legacy_run_manifest = copy.deepcopy(run_manifest)
    legacy_run_manifest["motion_primitive"]["min_action_frequency"] = 0.001
    (result / "run_manifest.json").write_text(json.dumps(legacy_run_manifest))
    with pytest.raises(ValueError, match="motion primitive contract"):
        validate_output(result, config=config)
    (result / "run_manifest.json").write_text(json.dumps(run_manifest))

    legacy_catalog = copy.deepcopy(catalog)
    legacy_catalog["constants"]["primitive_thresholds"].update(
        {"roll": 0.12, "tilt": 0.12, "rotation": 0.18}
    )
    (graph_root / "prototype_catalog.json").write_text(json.dumps(legacy_catalog))
    with pytest.raises(ValueError, match="prototype catalog schema"):
        validate_output(result, config=config)
    (graph_root / "prototype_catalog.json").write_text(json.dumps(catalog))

    legacy_temporal_manifest = copy.deepcopy(run_manifest)
    legacy_temporal_manifest["trajectory_window_max_gap"] = 3
    legacy_temporal_manifest["trajectory_window_policy"] = "full_coverage_max_gap_3_tail_rebalanced"
    (result / "run_manifest.json").write_text(json.dumps(legacy_temporal_manifest))
    with pytest.raises(ValueError, match="temporal geometry"):
        validate_output(result, config=config)
    (result / "run_manifest.json").write_text(json.dumps(run_manifest))

    legacy_temporal_catalog = copy.deepcopy(catalog)
    legacy_temporal_catalog["constants"]["trajectory_window_max_gap"] = 3
    legacy_temporal_catalog["constants"]["trajectory_window_policy"] = (
        "full_coverage_max_gap_3_tail_rebalanced"
    )
    (graph_root / "prototype_catalog.json").write_text(json.dumps(legacy_temporal_catalog))
    with pytest.raises(ValueError, match="prototype catalog schema"):
        validate_output(result, config=config)
    (graph_root / "prototype_catalog.json").write_text(json.dumps(catalog))

    run_manifest["clip_length"] = 15
    run_manifest["clip_anchors"] = [0, 7, 14]
    run_manifest["visual_half_windows"] = [[0, 8], [7, 15]]
    run_manifest["visual_half_encoding"] = "l2_normalized_eight_frame_mean"
    run_manifest["trajectory_window_length"] = 8
    run_manifest["trajectory_horizon"] = 7
    (result / "run_manifest.json").write_text(json.dumps(run_manifest))
    with pytest.raises(ValueError, match="temporal geometry"):
        validate_output(result, config=config)


def test_bridge_validate_replays_the_same_max_episode_subset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cocore.pipeline import run_pipeline
    from cocore_bridge_v2 import cli
    from cocore_bridge_v2.config import build_config

    class DummyVisualEncoder:
        output_dim = 3

        def encode(self, images: np.ndarray) -> np.ndarray:
            values = images[:, 0, 0, 0].astype(np.float32)
            return np.stack([values + 1.0, values + 2.0, values + 4.0], axis=1)

    dataset = tmp_path / "bridge-subset"
    output = tmp_path / "output-subset"
    _write_synthetic_bridge_dataset(dataset, stop_first_valid_episode=True)
    config = build_config(
        relation="sequence",
        relation_weight=1.0,
        selection_ratio=0.5,
        dataset_path=dataset,
        max_episodes=1,
    )
    config["visual"]["encoder"] = "dummy"
    config["runtime"]["num_workers"] = 0
    config["quality"]["knn"] = 2
    config["graph"]["knn"] = 2
    config["prototypes"]["max_iter"] = 2

    original_build_config = cli.build_config

    def build_test_config(**kwargs):
        replay_config = original_build_config(**kwargs)
        replay_config["prototypes"]["max_iter"] = 2
        return replay_config

    monkeypatch.setattr(cli, "build_config", build_test_config)

    result = run_pipeline(config, output_dir=output, visual_encoder=DummyVisualEncoder())

    cli.main(
        [
            "validate",
            "--output-dir",
            str(result),
            "--dataset-path",
            str(dataset),
            "--relation",
            "sequence",
            "--relation-weight",
            "1",
            "--selection-ratio",
            "0.5",
            "--max-episodes",
            "1",
        ]
    )

    assert json.loads(capsys.readouterr().out) == {
        "selected_clips": 44,
        "status": "valid",
    }


def test_bridge_cli_forwards_explicit_dwell_thresholds(tmp_path, monkeypatch):
    from cocore_bridge_v2 import cli

    received = {}

    def validate(path, *, config):
        received.update(config)
        return {"valid": True}

    monkeypatch.setattr(cli, "validate_output", validate)
    cli.main(
        [
            "validate",
            "--relation",
            "sequence",
            "--relation-weight",
            "1",
            "--output-dir",
            str(tmp_path),
            "--reliability-metrics",
            "non_dwell",
            "--dwell-position-speed-threshold",
            ".1",
            "--dwell-angular-speed-threshold",
            ".2",
            "--dwell-gripper-mode",
            "binary",
        ]
    )
    assert received["reliability_metrics"] == ["non_dwell"]
    assert received["dwell"] == {
        "position_speed_threshold": 0.1,
        "angular_speed_threshold": 0.2,
        "gripper_mode": "binary",
    }


def test_bridge_config_keeps_default_metrics_with_dwell_diagnostics():
    from cocore_bridge_v2.config import build_config

    config = build_config(
        relation="sequence",
        relation_weight=1,
        dwell={
            "position_speed_threshold": 0.1,
            "angular_speed_threshold": 0.2,
            "gripper_mode": "binary",
        },
    )
    assert config["reliability_metrics"] == [
        "support",
        "progress",
        "action_variation",
        "visual_action_consistency",
    ]
