import argparse
import os

import pytest

from qwen3_vl_groot.cli import build_parser, configure_visible_gpus, parse_gpu_ids


def test_parse_gpu_ids():
    assert parse_gpu_ids("2,3, 6,7") == [2, 3, 6, 7]
    with pytest.raises(argparse.ArgumentTypeError, match="duplicates"):
        parse_gpu_ids("0,1,1,2")
    with pytest.raises(argparse.ArgumentTypeError, match="non-negative"):
        parse_gpu_ids("0,1,-2,3")


def test_configure_visible_gpus(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    config = {"train": {"gpu_count": 4, "gpu_ids": [2, 3, 6, 7]}}
    assert configure_visible_gpus(config) == "2,3,6,7"
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3,6,7"


def test_launch_cli_rejects_removed_resume_option():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "launch",
                "--config",
                "configs/bridge_4x4090.yaml",
                "--resume",
                "latest",
            ]
        )
