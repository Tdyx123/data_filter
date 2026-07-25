import pytest

torch = pytest.importorskip("torch")

from qwen3_vl_groot.flow import (  # noqa: E402
    FlowMatchingActionHead,
    euler_denoise,
    masked_velocity_mse,
    sample_flow_batch,
)


def tiny_head():
    return FlowMatchingActionHead(
        state_dim=8,
        action_dim=7,
        horizon=8,
        context_dim=16,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2,
        dropout=0.0,
        gradient_checkpointing=True,
    )


def test_flow_matching_target_and_masked_loss():
    actions = torch.ones(2, 8, 7)
    generator = torch.Generator().manual_seed(7)
    noisy, timestep, velocity = sample_flow_batch(actions, generator=generator)
    assert noisy.shape == actions.shape
    assert velocity.shape == actions.shape
    assert ((0 <= timestep) & (timestep <= 0.999)).all()

    prediction = velocity.clone()
    prediction[:, -1] += 10
    mask = torch.ones(2, 8)
    mask[:, -1] = 0
    assert masked_velocity_mse(prediction, velocity, mask).item() == 0.0


def test_tiny_two_step_training_checkpoint_and_inference(tmp_path):
    torch.manual_seed(3)
    head = tiny_head()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1.0e-3)
    state = torch.randn(2, 8)
    context = torch.randn(2, 5, 16)
    context_mask = torch.ones(2, 5, dtype=torch.bool)
    actions = torch.randn(2, 8, 7)
    action_mask = torch.ones(2, 8)

    losses = []
    for _ in range(2):
        noisy, timestep, velocity = sample_flow_batch(actions)
        prediction = head(noisy, state, timestep, context, context_mask)
        loss = masked_velocity_mse(prediction, velocity, action_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert all(torch.isfinite(torch.tensor(losses)))

    path = tmp_path / "tiny_checkpoint.pt"
    torch.save({"model": head.state_dict(), "optimizer": optimizer.state_dict()}, path)
    restored = tiny_head()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1.0e-3)
    payload = torch.load(path, weights_only=False)
    restored.load_state_dict(payload["model"])
    restored_optimizer.load_state_dict(payload["optimizer"])
    for left, right in zip(head.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(left, right)

    restored.eval()
    prediction = euler_denoise(
        restored,
        state=state[:1],
        context=context[:1],
        context_attention_mask=context_mask[:1],
        steps=4,
    )
    assert prediction.shape == (1, 8, 7)
    assert torch.isfinite(prediction).all()
