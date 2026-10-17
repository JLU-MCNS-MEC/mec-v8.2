"""检查重新训练后的shared_encoder是否真正具备Goal-Current匹配能力。"""

import numpy as np
import torch
import torch.nn.functional as F

from config import AUX, ENCODER
from encoder import VisualTokenEncoder
from env import ToyUAVEnv
from train_shared_encoder import goal_foreground_pool, images_to_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLES = 64
SEED_START = 610000


def main():
    encoder = VisualTokenEncoder().to(DEVICE)
    encoder.load_state_dict(
        torch.load("shared_encoder.pt", map_location=DEVICE, weights_only=True)
    )
    encoder.eval()

    currents = []
    goals = []
    masks = []
    for i in range(SAMPLES):
        env = ToyUAVEnv()
        env.reset(seed=SEED_START + i, render_observation=False)

        # 沿oracle随机推进到一处中间/近端状态，使尺度和位置更丰富。
        states = []
        for _ in range(env.max_steps):
            a = env.oracle_action()
            states.append((env.x, env.y, env.yaw, env.step_count))
            if a == 3:
                break
            env.step(a, render_observation=False)
        idx = int(env.rng.integers(0, len(states)))
        env.x, env.y, env.yaw, env.step_count = states[idx]

        currents.append(env.render())
        scale = AUX.goal_reference_object_pixels[i % len(AUX.goal_reference_object_pixels)]
        goals.append(env.get_goal_reference_scale_rgb(scale))
        masks.append(env.goal_token_soft_mask(ENCODER.grid_size))

    current_x = images_to_tensor(np.stack(currents), DEVICE)
    goal_x = images_to_tensor(np.stack(goals), DEVICE)
    mask = torch.from_numpy(np.stack(masks)).float().to(DEVICE)

    with torch.no_grad():
        current_tokens = encoder(current_x)
        goal_tokens_full = encoder(goal_x)
        goal_feat = goal_foreground_pool(goal_tokens_full, np.stack(goals))
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        current_target = (current_tokens * mask.unsqueeze(-1)).sum(dim=1) / denom

    g = F.normalize(goal_feat, dim=-1)
    c = F.normalize(current_target, dim=-1)
    sim = g @ c.t()
    target = torch.arange(SAMPLES, device=DEVICE)
    top1 = float((sim.argmax(dim=1) == target).float().mean().item())
    pos = float(sim.diag().mean().item())
    wrong = float(sim[torch.arange(SAMPLES, device=DEVICE), (target + 1) % SAMPLES].mean().item())

    print("=" * 76)
    print("Shared encoder Goal-Current matching diagnostic")
    print("=" * 76)
    print(f"Samples                 : {SAMPLES}")
    print(f"Correct-pair cosine sim : {pos:.4f}")
    print(f"Wrong-pair cosine sim   : {wrong:.4f}")
    print(f"Retrieval Top-1         : {100*top1:.2f}%")
    print("期望：correct similarity明显高于wrong，Top-1明显高于随机1/N。")


if __name__ == "__main__":
    main()
