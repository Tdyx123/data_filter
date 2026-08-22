from types import SimpleNamespace


def test_oft_evaluator_uses_oft_routes_and_shared_remote_protocol(tmp_path, monkeypatch):
    from qwen3_vl_groot import evaluate_simpler as shared
    from qwen_vl_oft import evaluate_simpler

    captured = {}
    client = SimpleNamespace(shutdown=lambda: captured.setdefault("shutdown", True))
    policy = SimpleNamespace(
        checkpoint_report={"requested_path": "/models/step-00019000"},
        model_device="cuda:2",
        protocol_metadata=lambda: {
            "inference_strategy": "deterministic-causal-query-oft",
            "native_action_chunk_size": 8,
        },
    )
    monkeypatch.setattr(shared, "validate_simpler_source", lambda path: {})
    monkeypatch.setattr(shared, "QwenIPCClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(shared, "QwenRemotePolicy", lambda value: policy)

    def evaluate(settings, **kwargs):
        captured["settings"] = settings
        captured["kwargs"] = kwargs
        return {"status": "complete"}

    monkeypatch.setattr(shared, "evaluate_simpler_policy", evaluate)

    status = evaluate_simpler.main(
        [
            "--socket",
            "/tmp/qwen-oft.sock",
            "--auth-key-hex",
            "abcd",
            "--tasks",
            "spoon",
            "--output-dir",
            str(tmp_path / "results"),
            "--smoke-test",
        ]
    )

    assert status == 0
    assert captured["kwargs"]["route"] == "qwen-vl-oft-simpler-widowx-eval"
    assert captured["kwargs"]["protocol_metadata"] == {
        "inference_strategy": "deterministic-causal-query-oft",
        "native_action_chunk_size": 8,
    }
    assert captured["settings"].device == "remote-pyenv:cuda:2"
    assert captured["shutdown"] is True


def test_oft_evaluator_failure_report_uses_oft_route(tmp_path, monkeypatch):
    from qwen3_vl_groot import evaluate_simpler as shared
    from qwen_vl_oft import evaluate_simpler

    monkeypatch.setattr(
        shared,
        "validate_simpler_source",
        lambda path: (_ for _ in ()).throw(shared.SimplerEvaluationError("bad source")),
    )
    output_dir = tmp_path / "output"

    status = evaluate_simpler.main(
        [
            "--socket",
            "/tmp/qwen-oft.sock",
            "--auth-key-hex",
            "abcd",
            "--output-dir",
            str(output_dir),
        ]
    )

    import json

    failure = json.loads((output_dir / "failure.json").read_text(encoding="utf-8"))
    assert status == 2
    assert failure["route"] == "qwen-vl-oft-simpler-widowx-eval"
