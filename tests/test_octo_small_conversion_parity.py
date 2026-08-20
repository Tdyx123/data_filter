import os
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.slow
@pytest.mark.skip(reason="Long-running checkpoint conversion parity test")
def test_converted_checkpoint_matches_flax_reference_modules():
    torch = pytest.importorskip("torch")
    orbax = pytest.importorskip("orbax.checkpoint")
    pytest.importorskip("flax")
    octo_source = os.environ.get("OCTO_SOURCE_DIR")
    pytorch_model = os.environ.get("OCTO_PYTORCH_MODEL")
    source_model = Path(os.environ.get("OCTO_FLAX_MODEL", "/data/dwb/models/octo-small"))
    if not octo_source or not Path(octo_source).is_dir():
        pytest.skip("Set OCTO_SOURCE_DIR to the pinned official Octo source")
    if not pytorch_model or not Path(pytorch_model).is_dir():
        pytest.skip("Set OCTO_PYTORCH_MODEL to a converted model artifact")
    if not source_model.is_dir():
        pytest.skip("The source Octo-small checkpoint is not mounted")
    sys.path.insert(0, octo_source)

    from octo.model.components.diffusion import create_diffusion_model
    from octo.model.components.transformer import Transformer
    from octo.model.components.vit_encoders import SmallStem16
    from transformers import FlaxT5EncoderModel, T5Config

    from octo_small_libero.convert_checkpoint import _set_tensor
    from octo_small_libero.torch_model import (
        DiffusionActionHead,
        OctoSmallConfig,
        OctoSmallPolicy,
    )

    params = orbax.PyTreeCheckpointer().restore(
        str(source_model / "270000" / "default")
    )
    model, _ = OctoSmallPolicy.from_pretrained(pytorch_model)
    model.eval()

    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(1, 64, 64, 6), dtype=np.uint8)
    vision_params = params["octo_transformer"][
        "observation_tokenizers_primary"
    ]["SmallStem16_0"]
    flax_vision = np.asarray(
        SmallStem16().apply({"params": vision_params}, image)
    )
    normalized = torch.from_numpy(image.astype(np.float32) / 127.5 - 1).permute(
        0, 3, 1, 2
    )
    with torch.no_grad():
        pytorch_vision = (
            model.primary_encoder(normalized).reshape(1, 4, 4, 512).numpy()
        )
    np.testing.assert_allclose(flax_vision, pytorch_vision, atol=2e-4, rtol=2e-4)

    transformer_input = rng.normal(size=(1, 10, 384)).astype(np.float32)
    transformer_mask = np.ones((1, 1, 10, 10), dtype=bool)
    transformer_params = params["octo_transformer"]["BlockTransformer_0"][
        "Transformer_0"
    ]
    flax_transformer = np.asarray(
        Transformer(
            num_layers=12,
            mlp_dim=1536,
            num_attention_heads=6,
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
            add_position_embedding=False,
        ).apply(
            {"params": transformer_params},
            transformer_input,
            transformer_mask,
            train=False,
        )
    )
    with torch.no_grad():
        pytorch_transformer = model.transformer(
            torch.from_numpy(transformer_input),
            torch.from_numpy(transformer_mask[:, 0]),
        ).numpy()
    np.testing.assert_allclose(
        flax_transformer, pytorch_transformer, atol=2e-4, rtol=2e-4
    )

    text_config = T5Config.from_pretrained(
        Path(pytorch_model) / "text_encoder", local_files_only=True
    )
    input_ids = np.asarray([[3, 17, 42, 1] + [0] * 12], dtype=np.int32)
    text_mask = (input_ids != 0).astype(np.int32)
    text_params = params["octo_transformer"]["task_tokenizers_language"]["hf_model"]
    flax_text = np.asarray(
        FlaxT5EncoderModel(text_config, _do_init=False).module.apply(
            {"params": text_params},
            input_ids=input_ids,
            attention_mask=text_mask,
            deterministic=True,
        )[0]
    )
    with torch.no_grad():
        pytorch_text = model.text_encoder(
            input_ids=torch.from_numpy(input_ids).long(),
            attention_mask=torch.from_numpy(text_mask).long(),
        ).last_hidden_state.numpy()
    np.testing.assert_allclose(flax_text, pytorch_text, atol=3e-4, rtol=3e-4)

    flat_params = {}

    def flatten(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                flatten(child, f"{prefix}/{key}" if prefix else key)
        else:
            flat_params[prefix] = value

    flatten(params)
    head = DiffusionActionHead(OctoSmallConfig(action_horizon=4))
    state = head.state_dict()
    base = "heads_action/diffusion_model"
    mappings = [
        ("time_features.weight", f"{base}/time_preprocess/kernel", "identity"),
        ("time_linear1.weight", f"{base}/cond_encoder/Dense_0/kernel", "linear"),
        ("time_linear1.bias", f"{base}/cond_encoder/Dense_0/bias", "identity"),
        ("time_linear2.weight", f"{base}/cond_encoder/Dense_1/kernel", "linear"),
        ("time_linear2.bias", f"{base}/cond_encoder/Dense_1/bias", "identity"),
        ("reverse_input.weight", f"{base}/reverse_network/Dense_0/kernel", "linear"),
        ("reverse_input.bias", f"{base}/reverse_network/Dense_0/bias", "identity"),
        ("reverse_output.weight", f"{base}/reverse_network/Dense_1/kernel", "linear"),
        ("reverse_output.bias", f"{base}/reverse_network/Dense_1/bias", "identity"),
    ]
    for block in range(3):
        target = f"reverse_blocks.{block}"
        source = f"{base}/reverse_network/MLPResNetBlock_{block}"
        mappings.extend(
            [
                (f"{target}.norm.weight", f"{source}/LayerNorm_0/scale", "identity"),
                (f"{target}.norm.bias", f"{source}/LayerNorm_0/bias", "identity"),
                (f"{target}.linear1.weight", f"{source}/Dense_0/kernel", "linear"),
                (f"{target}.linear1.bias", f"{source}/Dense_0/bias", "identity"),
                (f"{target}.linear2.weight", f"{source}/Dense_1/kernel", "linear"),
                (f"{target}.linear2.bias", f"{source}/Dense_1/bias", "identity"),
            ]
        )
    for target, source, transform in mappings:
        _set_tensor(state, flat_params, target, source, transform=transform)
    head.load_state_dict(state)
    head.eval()
    observation = rng.normal(size=(1, 384)).astype(np.float32)
    actions = rng.normal(size=(1, 28)).astype(np.float32)
    time = np.asarray([[7]], dtype=np.float32)
    flax_diffusion = np.asarray(
        create_diffusion_model(
            28,
            time_dim=32,
            num_blocks=3,
            dropout_rate=0.1,
            hidden_dim=256,
            use_layer_norm=True,
        ).apply(
            {"params": params["heads_action"]["diffusion_model"]},
            observation,
            actions,
            time,
            train=False,
        )
    )
    with torch.no_grad():
        pytorch_diffusion = head.score(
            torch.from_numpy(observation),
            torch.from_numpy(actions),
            torch.from_numpy(time),
        ).numpy()
    np.testing.assert_allclose(
        flax_diffusion, pytorch_diffusion, atol=2e-4, rtol=2e-4
    )
