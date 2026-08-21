from types import SimpleNamespace

import pytest


EXPECTED_PACKAGES = {
    "torch": "2.10.0+cu128",
    "transformers": "5.2.0",
    "numpy": "2.2.0",
    "diffusers": "0.38.0",
    "accelerate": "1.13.0",
}


class _Cuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def is_bf16_supported():
        return True


def test_runtime_contract_accepts_only_the_fixed_pyenv_versions():
    from starvla_bridge.runtime import validate_model_runtime

    versions = validate_model_runtime(
        device="cuda:0",
        python_version=(3, 12, 12),
        package_versions=EXPECTED_PACKAGES,
        torch_module=SimpleNamespace(cuda=_Cuda()),
    )

    assert versions == {"python": "3.12.12", **EXPECTED_PACKAGES}


@pytest.mark.parametrize(
    "python_version,packages,message",
    [
        ((3, 11, 9), EXPECTED_PACKAGES, "Python 3.12.12"),
        (
            (3, 12, 12),
            {**EXPECTED_PACKAGES, "transformers": "4.57.6"},
            "transformers==5.2.0",
        ),
        (
            (3, 12, 12),
            {key: value for key, value in EXPECTED_PACKAGES.items() if key != "diffusers"},
            "diffusers==0.38.0",
        ),
    ],
)
def test_runtime_contract_rejects_wrong_or_missing_versions(
    python_version, packages, message
):
    from starvla_bridge.runtime import StarVLARuntimeError, validate_model_runtime

    with pytest.raises(StarVLARuntimeError, match=message):
        validate_model_runtime(
            device="cuda:0",
            python_version=python_version,
            package_versions=packages,
            torch_module=SimpleNamespace(cuda=_Cuda()),
        )


@pytest.mark.parametrize(
    "device,cuda,message",
    [
        ("cpu", _Cuda(), "requires a CUDA device"),
        (
            "cuda:0",
            SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: True),
            "CUDA device is unavailable",
        ),
        (
            "cuda:0",
            SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: False),
            "does not support BF16",
        ),
    ],
)
def test_runtime_contract_requires_cuda_and_bf16(device, cuda, message):
    from starvla_bridge.runtime import StarVLARuntimeError, validate_model_runtime

    with pytest.raises(StarVLARuntimeError, match=message):
        validate_model_runtime(
            device=device,
            python_version=(3, 12, 12),
            package_versions=EXPECTED_PACKAGES,
            torch_module=SimpleNamespace(cuda=cuda),
        )


def test_server_validates_runtime_before_loading_checkpoint(monkeypatch, tmp_path):
    from starvla_bridge import server

    events = []
    spec = SimpleNamespace(
        model_dir=tmp_path / "model",
        checkpoint_path=tmp_path / "model.pt",
        base_model=tmp_path / "base",
        action_horizon=16,
        action_dim=7,
    )
    predicted_images = []
    loaded = SimpleNamespace(
        checkpoint_report=SimpleNamespace(
            tensor_count=962,
            parameter_bytes=9_976_489_486,
            dtypes=("torch.bfloat16",),
        ),
        predict_actions=lambda image, instruction: predicted_images.append(
            (image.copy(), instruction)
        )
        or __import__("numpy").zeros((1, 16, 7), dtype="float32"),
    )

    monkeypatch.setattr(
        server,
        "validate_model_runtime",
        lambda **kwargs: events.append(("runtime", kwargs))
        or {"python": "3.12.12", **EXPECTED_PACKAGES},
    )
    monkeypatch.setattr(server, "load_model_spec", lambda *args, **kwargs: spec)
    monkeypatch.setattr(
        server,
        "load_starvla_policy",
        lambda *args, **kwargs: events.append(("load", kwargs)) or loaded,
    )
    monkeypatch.setattr(
        server,
        "serve_policy",
        lambda **kwargs: events.append(("serve", kwargs)),
    )

    status = server.main(
        [
            "--socket",
            str(tmp_path / "policy.sock"),
            "--auth-key-hex",
            "001122",
            "--model-dir",
            str(tmp_path / "model"),
            "--base-model",
            str(tmp_path / "base"),
            "--device",
            "cuda:3",
        ]
    )

    assert status == 0
    assert [event[0] for event in events] == ["runtime", "load", "serve"]
    assert events[0][1] == {"device": "cuda:3"}
    assert events[1][1] == {"device": "cuda:3"}
    assert events[-1][1]["metadata"]["device"] == "cuda:3"
    assert events[-1][1]["metadata"]["runtime"]["diffusers"] == "0.38.0"
    assert predicted_images[0][0].shape == (224, 224, 3)
    assert predicted_images[0][0].dtype.name == "uint8"
    assert not predicted_images[0][0].any()
    assert events[-1][1]["metadata"]["startup_preflight"] == {
        "action_shape": [1, 16, 7],
        "finite": True,
    }


def test_server_startup_preflight_rejects_invalid_action_chunk():
    import numpy as np

    from starvla_bridge.server import StarVLAServerError, run_startup_preflight

    policy = SimpleNamespace(
        predict_actions=lambda image, instruction: np.full(
            (1, 16, 7), np.nan, dtype=np.float32
        )
    )

    with pytest.raises(StarVLAServerError, match="finite"):
        run_startup_preflight(policy)
