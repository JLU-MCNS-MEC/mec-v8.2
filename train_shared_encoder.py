"""
V7 shared visual encoder重新预训练。

相比旧版“Current RGB -> Action”的单任务Behavior Cloning，本脚本融合两个目标：

1) Goal-conditioned action imitation
   Current RGB + Goal RGB -> F/L/R/Stop
   让编码器保留旧版对导航动作/Stop视觉线索的敏感性；

2) Goal-Current correspondence loss
   用simulator仅在训练阶段提供的Goal投影soft mask，从Current token中池化真实目标区域，
   与Goal-reference token做对比学习(InfoNCE)。
   这使shared encoder真正学习跨图像目标匹配，而不是假设“同一个encoder自然就能匹配”。

注意：Goal投影soft mask只用于shared encoder预训练，绝不会写入MEC数据集或在线推理输入。
训练完成后只保存encoder权重为shared_encoder.pt。
"""

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AUX, ENCODER, ENV, NUM_ACTIONS, TRAIN
from encoder import VisualTokenEncoder
from env import ACTION_NAMES, ToyUAVEnv


SEED = 2026
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_NAMES_LIST = [ACTION_NAMES[i] for i in range(NUM_ACTIONS)]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def images_to_tensor(images, device):
    arr = np.asarray(images, dtype=np.uint8)
    x = torch.from_numpy(arr).float().permute(0, 3, 1, 2) / 255.0
    return x.to(device)


def goal_foreground_pool(tokens, goal_images):
    """根据Goal reference的均匀背景自动估计前景mask，并池化成[B,D]。"""
    if tokens.ndim != 3:
        raise ValueError(tokens.shape)
    x = torch.from_numpy(np.asarray(goal_images, dtype=np.uint8)).float() / 255.0
    bg = x[:, 0:1, 0:1, :]
    diff = (x - bg).abs().amax(dim=-1)
    fg = (diff > (8.0 / 255.0)).float().unsqueeze(1)
    fg = F.interpolate(
        fg,
        size=(ENCODER.grid_size, ENCODER.grid_size),
        mode="area",
    ).squeeze(1)
    w = fg.reshape(fg.shape[0], -1).to(tokens.device)
    denom = w.sum(dim=1, keepdim=True)
    empty = denom.squeeze(1) < 1e-6
    if bool(empty.any()):
        w[empty] = 0.0
        center = (ENCODER.grid_size // 2) * ENCODER.grid_size + ENCODER.grid_size // 2
        w[empty, center] = 1.0
        denom = w.sum(dim=1, keepdim=True)
    return (tokens * w.unsqueeze(-1)).sum(dim=1) / denom


def current_target_pool(tokens, target_soft_masks):
    """用训练阶段simulator提供的soft mask池化Current中的真实Goal区域。"""
    w = target_soft_masks.to(tokens.device, dtype=tokens.dtype)
    denom = w.sum(dim=1, keepdim=True)
    valid = denom.squeeze(1) > 1e-6
    safe = denom.clamp_min(1e-6)
    pooled = (tokens * w.unsqueeze(-1)).sum(dim=1) / safe
    return pooled, valid


class GoalConditionedPretrainModel(nn.Module):
    """只用于shared encoder预训练；最终MEC不会加载此head。"""

    def __init__(self):
        super().__init__()
        self.encoder = VisualTokenEncoder()
        d = ENCODER.token_dim
        self.current_pos = nn.Parameter(torch.zeros(1, ENCODER.num_tokens, d))
        nn.init.trunc_normal_(self.current_pos, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=ENCODER.num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d)
        self.action_head = nn.Sequential(
            nn.Linear(3 * d, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, NUM_ACTIONS),
        )

    def forward(self, current_images, goal_images_np):
        current_tokens = self.encoder(current_images)
        goal_images = images_to_tensor(goal_images_np, current_images.device)
        goal_full_tokens = self.encoder(goal_images)
        goal_token = goal_foreground_pool(goal_full_tokens, goal_images_np)

        q = goal_token.unsqueeze(1)
        kv = current_tokens + self.current_pos
        matched, _ = self.cross_attn(q, kv, kv, need_weights=False)
        matched = self.norm(q + matched).squeeze(1)
        current_global = current_tokens.mean(dim=1)
        logits = self.action_head(torch.cat([matched, goal_token, current_global], dim=-1))
        return logits, current_tokens, goal_token


def collect_oracle_states(env):
    """无渲染地收集一条oracle轨迹状态，减少预训练数据生成开销。"""
    states = []
    for _ in range(env.max_steps):
        action = env.oracle_action()
        states.append((env.x, env.y, env.yaw, env.step_count, int(action)))
        if action == 3:
            break
        _, _, done = env.step(action, render_observation=False)
        if done:
            break
    return states


def sample_one(seed, desired_action):
    """尽量采到指定动作类别，使encoder预训练不会忽略Stop。"""
    for attempt in range(40):
        env = ToyUAVEnv()
        env.reset(seed=seed + attempt * 100003, render_observation=False)
        states = collect_oracle_states(env)
        candidates = [s for s in states if s[-1] == desired_action]
        if not candidates:
            continue

        state = candidates[int(env.rng.integers(0, len(candidates)))]
        env.x, env.y, env.yaw, env.step_count, action = state
        current = env.render()
        scale = AUX.goal_reference_object_pixels[
            int(env.rng.integers(0, len(AUX.goal_reference_object_pixels)))
        ]
        goal = env.get_goal_reference_scale_rgb(scale)
        target_mask = env.goal_token_soft_mask(ENCODER.grid_size)
        return current, goal, target_mask, int(action)

    raise RuntimeError(f"Unable to sample action={desired_action} from seed={seed}")


def sample_batch(seed_base, batch_size):
    currents = []
    goals = []
    masks = []
    labels = []
    # 周期性均衡四类动作，保留旧版“动作感知encoder”的优点，同时确保Stop被充分看到。
    for i in range(batch_size):
        desired = i % NUM_ACTIONS
        current, goal, mask, label = sample_one(seed_base + i * 7919, desired)
        currents.append(current)
        goals.append(goal)
        masks.append(mask)
        labels.append(label)
    return (
        np.stack(currents, axis=0),
        np.stack(goals, axis=0),
        torch.from_numpy(np.stack(masks, axis=0)).float(),
        torch.tensor(labels, dtype=torch.long),
    )


def correspondence_loss(goal_token, target_token, valid, temperature):
    idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if idx.numel() < 2:
        return goal_token.sum() * 0.0, 0.0
    g = F.normalize(goal_token[idx], dim=-1)
    c = F.normalize(target_token[idx], dim=-1)
    sim = g @ c.t() / float(temperature)
    target = torch.arange(sim.shape[0], device=sim.device)
    loss = 0.5 * (F.cross_entropy(sim, target) + F.cross_entropy(sim.t(), target))
    retrieval = 0.5 * (
        (sim.argmax(dim=1) == target).float().mean()
        + (sim.argmax(dim=0) == target).float().mean()
    )
    return loss, float(retrieval.item())


@torch.no_grad()
def validate(model, seed_base):
    model.eval()
    action_correct = 0
    total = 0
    stop_tp = 0
    stop_total = 0
    match_acc = []
    losses = []

    for b in range(TRAIN.encoder_validation_batches):
        cur_np, goal_np, masks, labels = sample_batch(
            seed_base + b * 100000,
            TRAIN.encoder_pretrain_batch_size,
        )
        cur = images_to_tensor(cur_np, DEVICE)
        masks = masks.to(DEVICE)
        labels = labels.to(DEVICE)
        logits, current_tokens, goal_token = model(cur, goal_np)
        target_token, valid = current_target_pool(current_tokens, masks)
        m_loss, m_acc = correspondence_loss(
            goal_token,
            target_token,
            valid,
            TRAIN.encoder_match_temperature,
        )
        a_loss = F.cross_entropy(logits, labels)
        loss = a_loss + TRAIN.encoder_match_loss_weight * m_loss
        losses.append(float(loss.item()))
        match_acc.append(m_acc)

        pred = logits.argmax(dim=-1)
        action_correct += int((pred == labels).sum())
        total += labels.numel()
        stop_mask = labels == 3
        stop_total += int(stop_mask.sum())
        stop_tp += int(((pred == 3) & stop_mask).sum())

    model.train()
    return {
        "loss": float(np.mean(losses)),
        "action_acc": action_correct / max(total, 1),
        "stop_recall": stop_tp / max(stop_total, 1),
        "match_top1": float(np.mean(match_acc)),
    }


def main():
    set_seed(SEED)
    print("Device:", DEVICE)
    print(
        "V7 shared encoder pretraining | "
        f"updates={TRAIN.encoder_pretrain_updates}, "
        f"batch={TRAIN.encoder_pretrain_batch_size}, "
        f"match_weight={TRAIN.encoder_match_loss_weight}"
    )
    print(
        "Environment: "
        f"alt={ENV.uav_altitude_m:.0f}m, pitch={ENV.camera_pitch_deg:.0f}deg, "
        f"VFOV={ENV.camera_vfov_deg:.0f}deg; Goal uses episode-specific visual identity."
    )

    model = GoalConditionedPretrainModel().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN.encoder_pretrain_lr,
        weight_decay=TRAIN.encoder_pretrain_weight_decay,
    )
    use_amp = bool(TRAIN.use_amp and DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    running = Counter()

    for update in range(1, TRAIN.encoder_pretrain_updates + 1):
        cur_np, goal_np, masks, labels = sample_batch(
            SEED * 100000 + update * 1000,
            TRAIN.encoder_pretrain_batch_size,
        )
        cur = images_to_tensor(cur_np, DEVICE)
        masks = masks.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, current_tokens, goal_token = model(cur, goal_np)
            target_token, valid = current_target_pool(current_tokens, masks)
            match_loss, match_acc = correspondence_loss(
                goal_token,
                target_token,
                valid,
                TRAIN.encoder_match_temperature,
            )
            action_loss = F.cross_entropy(logits, labels)
            loss = action_loss + TRAIN.encoder_match_loss_weight * match_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pred = logits.argmax(dim=-1)
        running["n"] += labels.numel()
        running["correct"] += int((pred == labels).sum())
        running["stop_n"] += int((labels == 3).sum())
        running["stop_tp"] += int(((pred == 3) & (labels == 3)).sum())
        running["loss_sum"] += float(loss.item())
        running["updates"] += 1

        if update % 50 == 0:
            print(
                f"Update {update:04d}/{TRAIN.encoder_pretrain_updates} | "
                f"Loss={running['loss_sum']/max(running['updates'],1):.4f} | "
                f"ActionAcc={100*running['correct']/max(running['n'],1):6.2f}% | "
                f"StopRecall={100*running['stop_tp']/max(running['stop_n'],1):6.2f}% | "
                f"BatchMatchTop1={100*match_acc:6.2f}%"
            )
            running = Counter()

        if update % TRAIN.encoder_validate_every == 0:
            metrics = validate(model, seed_base=900000000 + update * 1000)
            print(
                "  VAL | "
                f"Loss={metrics['loss']:.4f} | "
                f"ActionAcc={100*metrics['action_acc']:.2f}% | "
                f"StopRecall={100*metrics['stop_recall']:.2f}% | "
                f"MatchTop1={100*metrics['match_top1']:.2f}%"
            )
            if metrics["loss"] < best_val:
                best_val = metrics["loss"]
                torch.save(model.encoder.state_dict(), "shared_encoder.pt")
                print("  Saved best encoder -> shared_encoder.pt")

    # 如果极端情况下validate没有触发，也保证有输出。
    if not torch.cuda.is_available() and TRAIN.encoder_pretrain_updates < TRAIN.encoder_validate_every:
        torch.save(model.encoder.state_dict(), "shared_encoder.pt")

    print("Finished. Use this NEW shared_encoder.pt to regenerate all token datasets.")


if __name__ == "__main__":
    main()
