"""Opt-in layer parity against the pinned official Flax implementation.

Run after conversion with OCTO_SOURCE_DIR pointing at commit 653c54ac... and
OCTO_OFFICIAL_PYTORCH_MODEL pointing at the converted artifact.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.slow
def test_official_flax_and_pytorch_fixed_input_layer_parity():
    torch = pytest.importorskip("torch")
    orbax = pytest.importorskip("orbax.checkpoint")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    source_checkout = os.environ.get("OCTO_SOURCE_DIR")
    if not source_checkout or not Path(source_checkout).is_dir():
        pytest.skip("Set OCTO_SOURCE_DIR to official Octo commit 653c54ac...")
    source_commit = subprocess.run(
        ["git", "-C", source_checkout, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert source_commit == "653c54acde686fde619855f2eac0dd6edad7116b"
    source_model = Path(os.environ.get("OCTO_FLAX_MODEL", "/data/dwb/models/octo-small"))
    pytorch_model = Path(
        os.environ.get(
            "OCTO_OFFICIAL_PYTORCH_MODEL",
            "/data/dwb/models/octo-small-pytorch-official",
        )
    )
    if not source_model.is_dir() or not pytorch_model.is_dir():
        pytest.skip("Source and converted official Octo-small artifacts are required")
    sys.path.insert(0, source_checkout)

    from octo.model.components.diffusion import create_diffusion_model
    from octo.model.components.transformer import Transformer
    from octo.model.components.vit_encoders import SmallStem16
    from transformers import FlaxT5EncoderModel, T5Config

    from octo_small_official_pytorch.model import build_block_causal_attention_mask
    from octo_small_official_pytorch.policy import load_official_policy

    params = orbax.PyTreeCheckpointer().restore(str(source_model / "270000" / "default"))
    _report, policy = load_official_policy(pytorch_model, device="cpu", precision="fp32")
    model = policy.model
    rng = np.random.default_rng(7)

    raw_images = rng.integers(0, 256, size=(1, 2, 256, 256, 3), dtype=np.uint8)
    flax_images = np.concatenate([raw_images, np.zeros_like(raw_images)], axis=-1).reshape(
        2, 256, 256, 6
    )
    vision_params = params["octo_transformer"]["observation_tokenizers_primary"]["SmallStem16_0"]
    flax_vision = np.asarray(SmallStem16().apply({"params": vision_params}, flax_images)).reshape(
        1, 2, 256, 512
    )
    torch_images = torch.from_numpy(raw_images.astype(np.float32) / 127.5 - 1).permute(
        0, 1, 4, 2, 3
    )
    with torch.no_grad():
        torch_vision = model.primary_encoder(model.with_zero_goal(torch_images)).reshape(
            1, 2, 256, 512
        )
    np.testing.assert_allclose(flax_vision, torch_vision.numpy(), atol=2e-4, rtol=2e-4)

    input_ids = np.asarray([[3, 17, 42, 1] + [0] * 12], dtype=np.int32)
    text_mask = (input_ids != 0).astype(np.int32)
    text_config = T5Config.from_pretrained(pytorch_model / "text_encoder", local_files_only=True)
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
        torch_text = model.text_encoder(
            input_ids=torch.from_numpy(input_ids).long(),
            attention_mask=torch.from_numpy(text_mask).long(),
        ).last_hidden_state.numpy()
    np.testing.assert_allclose(flax_text, torch_text, atol=3e-4, rtol=3e-4)

    transformer_params = params["octo_transformer"]
    language = (
        jnp.matmul(
            flax_text,
            transformer_params["task_language_projection"]["kernel"],
        )
        + transformer_params["task_language_projection"]["bias"]
        + transformer_params["task_language_pos_embedding"]
    )
    observations = (
        jnp.matmul(
            flax_vision,
            transformer_params["obs_primary_projection"]["kernel"],
        )
        + transformer_params["obs_primary_projection"]["bias"]
        + transformer_params["obs_primary_pos_embedding"][:, :2]
    )
    readout = jnp.broadcast_to(
        transformer_params["readout_action_pos_embedding"][:, :2],
        (1, 2, 1, 384),
    )
    timestep = jnp.concatenate([observations, readout], axis=2).reshape(1, 2 * 257, 384)
    sequence = jnp.concatenate([language, timestep], axis=1)
    torch_mask = build_block_causal_attention_mask(
        language_mask=torch.ones(1, 16, dtype=torch.bool),
        timestep_pad_mask=torch.ones(1, 2, dtype=torch.bool),
        observation_tokens_per_timestep=256,
    )
    flax_encoded = np.asarray(
        Transformer(
            num_layers=12,
            mlp_dim=1536,
            num_attention_heads=6,
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
            add_position_embedding=False,
        ).apply(
            {"params": transformer_params["BlockTransformer_0"]["Transformer_0"]},
            sequence,
            np.asarray(torch_mask)[:, None],
            train=False,
        )
    )[:, -1]
    batch = {
        "image_primary": torch_images,
        "timestep_pad_mask": torch.ones(1, 2, dtype=torch.bool),
        "language_input_ids": torch.from_numpy(input_ids).long(),
        "language_attention_mask": torch.from_numpy(text_mask).bool(),
    }
    with torch.no_grad():
        torch_encoded = model.encode_observation(batch).numpy()
    np.testing.assert_allclose(flax_encoded, torch_encoded, atol=3e-4, rtol=3e-4)

    observation = rng.normal(size=(1, 384)).astype(np.float32)
    noisy_actions = rng.normal(size=(1, 28)).astype(np.float32)
    time = np.asarray([[7]], dtype=np.float32)
    flax_score = np.asarray(
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
            noisy_actions,
            time,
            train=False,
        )
    )
    with torch.no_grad():
        torch_score = model.action_head.score(
            torch.from_numpy(observation),
            torch.from_numpy(noisy_actions),
            torch.from_numpy(time),
        ).numpy()
    np.testing.assert_allclose(flax_score, torch_score, atol=2e-4, rtol=2e-4)
