"""V8.2 interface/shape sanity check. No token dataset or checkpoint is required."""
import torch

from config import AUX, ENCODER, GBS_MODEL_CONFIGS, NETWORK
from gbs_model_v3 import build_gbs_model


def main():
    print("=" * 88)
    print("V8.2 SANITY CHECK")
    print("=" * 88)
    print(
        f"Current RGB tokens : [{ENCODER.num_tokens}, {ENCODER.token_dim}]"
    )
    print(
        f"Goal RGB tokens    : [{AUX.goal_token_count}, {ENCODER.token_dim}]"
    )
    print(
        f"Spatial heatmap    : "
        f"[{AUX.spatial_heatmap_channels}, "
        f"{AUX.spatial_heatmap_grid}, "
        f"{AUX.spatial_heatmap_grid}]"
    )
    print(f"Network state      : {NETWORK.network_state_dim}D")
    print("Removed            : position / start / history / depth / capability")

    b = 2
    current = torch.randn(
        b, ENCODER.num_tokens, ENCODER.token_dim
    )
    goal = torch.randn(
        b, AUX.goal_token_count, ENCODER.token_dim
    )
    heatmap = torch.randn(
        b,
        AUX.spatial_heatmap_channels,
        AUX.spatial_heatmap_grid,
        AUX.spatial_heatmap_grid,
    )
    network_state = torch.rand(
        b, NETWORK.network_state_dim
    )

    print("\nMEC models")
    for name in GBS_MODEL_CONFIGS:
        model = build_gbs_model(name)
        with torch.no_grad():
            comp = model.forward_components(
                current, goal, heatmap, network_state
            )
            y = model(current, goal, heatmap, network_state)
            probs = torch.softmax(y, dim=-1)
        params = sum(p.numel() for p in model.parameters())
        print(
            f"  {name:>13s}: params={params:,} | "
            f"output={tuple(y.shape)} | "
            f"move={tuple(comp['move_logits'].shape)} | "
            f"arrival={tuple(comp['arrival_logit'].shape)} | "
            f"prob-sum={float(probs[0].sum()):.6f}"
        )

    print("\nPASS: all five heterogeneous MECs accept the V8.2 input interface.")


if __name__ == "__main__":
    main()
