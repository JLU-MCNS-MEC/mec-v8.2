"""CPU-only smoke test for the ChatGPT -> GitHub -> server workflow.

No datasets, checkpoints, or CUDA are required. This test checks that the five
V8.2 MEC model architectures can be constructed and perform a forward pass with
the documented input shapes.
"""

import gc
import torch

from config import AUX, ENCODER, GBS_MODEL_CONFIGS, NETWORK, NUM_ACTIONS
from gbs_model_v3 import build_gbs_model


def main():
    torch.manual_seed(20261017)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cpu")

    current_tokens = torch.randn(1, ENCODER.num_tokens, ENCODER.token_dim)
    goal_tokens = torch.randn(1, AUX.goal_token_count, ENCODER.token_dim)
    heatmap = torch.rand(
        1, AUX.spatial_heatmap_channels,
        AUX.spatial_heatmap_grid, AUX.spatial_heatmap_grid
    )
    network_state = torch.rand(1, NETWORK.network_state_dim)

    print("V8.2 MEC workflow smoke test (CPU only)")
    print(f"Current tokens: {tuple(current_tokens.shape)}")
    print(f"Goal tokens:    {tuple(goal_tokens.shape)}")
    print(f"Heatmap:        {tuple(heatmap.shape)}")
    print(f"Network state:  {tuple(network_state.shape)}")

    for name in GBS_MODEL_CONFIGS:
        model = build_gbs_model(name, token_dim=ENCODER.token_dim).to(device)
        model.eval()
        with torch.inference_mode():
            log_probs = model(current_tokens, goal_tokens, heatmap, network_state)
            decision = model.predict_action(
                current_tokens, goal_tokens, heatmap, network_state
            )

        assert tuple(log_probs.shape) == (1, NUM_ACTIONS), (
            f"{name}: unexpected output shape {tuple(log_probs.shape)}"
        )
        assert torch.isfinite(log_probs).all().item(), (
            f"{name}: output contains NaN or Inf"
        )
        probs = log_probs.exp()
        assert torch.allclose(
            probs.sum(dim=-1), torch.ones(1), atol=1e-4
        ), f"{name}: action probabilities do not sum to one"

        action = int(decision["action"].item())
        arrival_p = float(decision["arrival_probability"].item())
        assert 0 <= action < NUM_ACTIONS, f"{name}: invalid action {action}"
        assert 0.0 <= arrival_p <= 1.0, f"{name}: invalid arrival probability"

        print(
            f"[PASS] {name}: output={tuple(log_probs.shape)}, "
            f"action={action}, arrival_p={arrival_p:.4f}, "
            f"params={sum(p.numel() for p in model.parameters()):,}"
        )
        del model, log_probs, decision, probs
        gc.collect()

    print("WORKFLOW_SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
