"""V8代码/维度/Depth/参数量快速检查。无需token dataset或checkpoint。"""

import torch

from config import AUX, ENCODER, GBS_MODEL_CONFIGS
from env import ToyUAVEnv
from gbs_model_v3 import build_gbs_model
from token_data import depth_map_to_tokens


def main():
    print("Encoder tokens:", ENCODER.num_tokens, "x", ENCODER.token_dim)
    env = ToyUAVEnv()
    env.reset(seed=2026)
    depth = env.render_depth()
    depth_tokens = depth_map_to_tokens(depth)
    print("Start-goal distance:", f"{env.distance_to_goal:.2f} m")
    print("Initial goal projection:", env.goal_projection())
    print(
        "Depth:", depth.shape,
        "valid pixels=", int((depth > 0).sum()),
        "patch tokens=", tuple(depth_tokens.shape),
    )

    b = 2
    current = torch.randn(b, ENCODER.num_tokens, ENCODER.token_dim)
    start = torch.randn(b, AUX.binary_size, AUX.binary_size)
    goal = torch.randn(b, AUX.goal_token_count, ENCODER.token_dim)
    history = torch.randn(
        b, AUX.history_length, AUX.history_tokens_per_frame, ENCODER.token_dim
    )
    valid = torch.ones(b, AUX.history_length, dtype=torch.long)
    d = torch.rand(b, ENCODER.num_tokens, AUX.depth_token_dim)

    print("\nMEC models")
    for name in GBS_MODEL_CONFIGS:
        model = build_gbs_model(name)
        with torch.no_grad():
            comp = model.forward_components(current, start, goal, history, valid, d)
            y = model(current, start, goal, history, valid, d)
            probs = torch.softmax(y, dim=-1)
        params = sum(p.numel() for p in model.parameters())
        print(
            f"  {name:>13s}: params={params:,} | output={tuple(y.shape)} | "
            f"move={tuple(comp['move_logits'].shape)} | arrival={tuple(comp['arrival_logit'].shape)} | "
            f"prob-sum={float(probs[0].sum()):.6f}"
        )


if __name__ == "__main__":
    main()
