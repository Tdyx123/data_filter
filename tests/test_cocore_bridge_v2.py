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
        {"episode_index": 1, "tasks": ["move object one"], "length": 45},
        {"episode_index": 2, "tasks": ["move object two"], "length": 45},
    ]
    _write_jsonl(meta / "tasks.jsonl", tasks)
    _write_jsonl(meta / "episodes.jsonl", episodes)

    info = _bridge_info()
    info.update(
        {
            "total_episodes": 3,
            "total_frames": 135,
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
        steps = np.arange(45, dtype=np.float32)
        states = np.zeros((45, 8), dtype=np.float32)
        if not (stop_first_valid_episode and episode_id == 1):
            states[:, 0] = steps * 0.01
            states[:, 7] = np.clip(steps / 44.0, 0.0, 1.0)
        actions = np.zeros((45, 7), dtype=np.float32)
        actions[:, 0] = 0.01
        actions[:, 6] = states[:, 7]
        table = pa.table(
            {
                "observation.state": pa.array(
                    states.tolist(), type=pa.list_(pa.float32(), list_size=8)
                ),
                "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), list_size=7)),
                "timestamp": pa.array(steps / 5.0, type=pa.float32()),
                "frame_index": pa.array(np.arange(45), type=pa.int64()),
                "episode_index": pa.array([episode_id] * 45, type=pa.int64()),
                "task_index": pa.array([episode_id] * 45, type=pa.int64()),
            }
        )
        pq.write_table(table, data / f"episode_{episode_id:06d}.parquet")
        pixel_values = np.mod(np.arange(45) + episode_id * 37, 255).astype(np.uint8)
        frames = np.broadcast_to(pixel_values[:, None, None, None], (45, 32, 32, 3)).copy()
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
    assert result.stdout.splitlines() == ["0.1.0", "['__version__']"]


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
    assert set(config["prototypes"]) == {"method", "batch_size", "max_iter"}
    assert config["objective"] == {"relation": "sequence", "relation_weight": 1.5}
    assert config["selection"]["ratio"] == 0.2
    assert config["selection"]["budget"] is None
    assert config["runtime"]["max_episodes"] == 100
    assert config["output"]["directory"] == ("outputs/cocore_bridge_v2/bridge_orig_1.0.0")
    translated = to_relcore_config(config)
    assert translated["selection"]["quota_mode"] == "none"
    assert translated["selection"]["minimum_per_task"] == 0


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
            "cocore_output={root}/graph-13-motion-softmax nodes=2",
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
        ]
    )

    assert received["output_dir"] == str(result_path)
    assert received["config"]["dataset"]["path"] == str(dataset_path)
    assert received["config"]["selection"]["ratio"] == 0.2
    assert received["config"]["runtime"]["max_episodes"] == 11
    assert json.loads(capsys.readouterr().out) == {
        "selected_clips": 7,
        "status": "valid",
    }


def test_synthetic_bridge_dataset_runs_cocore_with_only_image_zero(tmp_path: Path) -> None:
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
    config["selection"]["budget"] = 6

    result = run_pipeline(config, output_dir=output, visual_encoder=DummyVisualEncoder())

    scan_manifest = json.loads((output / "scan" / "manifest.json").read_text())
    assert scan_manifest["dataset_summary"] == {
        "source_episodes": 3,
        "indexed_episodes": 2,
        "retained_episodes": 2,
        "excluded_episodes": 1,
        "excluded_empty_task_episodes": 1,
    }
    assert scan_manifest["clips"] == 6
    scanned_episode_ids = pq.read_table(output / "scan" / "episodes.parquet")[
        "episode_id"
    ].to_pylist()
    assert scanned_episode_ids == [1, 2]
    selected = [
        json.loads(line) for line in (result / "selected_manifest.jsonl").read_text().splitlines()
    ]
    assert len(selected) == 6
    select_manifest = json.loads((result / "manifest.json").read_text())
    run_manifest = json.loads((result / "run_manifest.json").read_text())
    assert select_manifest["producer"] == "cocore"
    assert select_manifest["cocore_version"] == "0.8.0"
    assert run_manifest["producer"] == "cocore"
    assert run_manifest["cocore_version"] == "0.8.0"
    assert run_manifest["stage_directories"]["graph"] == "graph-13-motion-softmax"
    assert validate_output(result, config=config) == {
        "status": "valid",
        "selected_clips": 6,
    }


def test_bridge_validate_replays_the_same_max_episode_subset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
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
        "selected_clips": 2,
        "status": "valid",
    }
