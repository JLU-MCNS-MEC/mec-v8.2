"""
V8.2 MEC navigation models: redesigned input modalities.

MEC inputs:
  - Current RGB tokens: [B, 576, 384]
  - Goal RGB tokens:    [B, 3, 384]
  - Spatial relation heatmap: [B, 3, 8, 8]
  - Global network state: [B, 5]

Removed from the MEC input:
  - UAV absolute/current position
  - Start mask
  - visual history
  - depth
  - MEC model capability / model type / compute capacity / quality prior
  - success reward hint

The spatial heatmap is not an oracle map. It is generated from frozen
Current/Goal RGB token cosine similarity, one channel per Goal scale (14/28/56 px).
The network state is a low-dimensional global observable derived from the same
20-MEC deployment and ignores model capability.

Movement and Arrival remain separate:
  movement head -> Forward/Left/Right
  binary Arrival head -> Stop gate

No simulator distance is read by the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    ARRIVAL_STOP_THRESHOLD,
    AUX,
    ENCODER,
    GBS_MODEL_CONFIGS,
    NETWORK,
    NUM_ACTIONS,
    NUM_MOVE_ACTIONS,
)


def goal_current_similarity(current_tokens, goal_tokens):
    """Return [B, S_goal, N_current] cosine similarity maps."""
    if current_tokens.ndim != 3 or goal_tokens.ndim != 3:
        raise ValueError("current_tokens and goal_tokens must be rank-3 tensors")
    current = F.normalize(current_tokens.float(), dim=-1)
    goal = F.normalize(goal_tokens.float(), dim=-1)
    return torch.einsum("bnd,bsd->bsn", current, goal)


class HeatmapNetworkContextEncoder(nn.Module):
    """
    Shared small context encoder.

    Heatmap path preserves the 8x8 spatial arrangement instead of global
    average pooling. The same encoded context is used by movement and arrival.
    """

    def __init__(
        self,
        heatmap_feature_dim=AUX.heatmap_feature_dim,
        network_feature_dim=AUX.network_feature_dim,
    ):
        super().__init__()
        self.heatmap_net = nn.Sequential(
            nn.Conv2d(AUX.spatial_heatmap_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(16 * AUX.spatial_heatmap_grid * AUX.spatial_heatmap_grid,
                      heatmap_feature_dim),
            nn.GELU(),
        )
        self.network_net = nn.Sequential(
            nn.Linear(NETWORK.network_state_dim, network_feature_dim),
            nn.GELU(),
            nn.Linear(network_feature_dim, network_feature_dim),
            nn.GELU(),
        )

    def forward(self, heatmap, network_state):
        if heatmap.ndim != 4:
            raise ValueError(f"heatmap must be [B,3,8,8], got {tuple(heatmap.shape)}")
        if network_state.ndim != 2:
            raise ValueError(
                f"network_state must be [B,5], got {tuple(network_state.shape)}"
            )
        heat_feat = self.heatmap_net(heatmap.float())
        net_feat = self.network_net(network_state.float())
        return heat_feat, net_feat


class ArrivalReasoningHead(nn.Module):
    """
    Binary Arrival/Stop head.

    It uses only observable Current/Goal matching plus the explicit heatmap and
    network context. The true simulator distance is never passed here.
    """

    def __init__(self, token_dim=ENCODER.token_dim):
        super().__init__()
        hidden = int(AUX.arrival_hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(8 + AUX.heatmap_feature_dim + AUX.network_feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def extract_features(
        self,
        current_tokens,
        goal_tokens,
        heatmap_feature,
        network_feature,
    ):
        sim = goal_current_similarity(current_tokens, goal_tokens)
        # For each current patch, choose the strongest match among the three
        # Goal scales. This produces an explicit target-localization score.
        patch_best = sim.max(dim=1).values
        b, n = patch_best.shape

        temperature = float(AUX.arrival_match_temperature)
        attention = torch.softmax(patch_best / max(temperature, 1e-4), dim=-1)

        g = int(ENCODER.grid_size)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, g, device=current_tokens.device),
            torch.linspace(-1.0, 1.0, g, device=current_tokens.device),
            indexing="ij",
        )
        xx = xx.reshape(1, n)
        yy = yy.reshape(1, n)

        max_match = patch_best.max(dim=-1).values
        topk = min(10, n)
        topk_mean = torch.topk(patch_best, k=topk, dim=-1).values.mean(dim=-1)
        mean_match = patch_best.mean(dim=-1)
        entropy = -(attention * (attention.clamp_min(1e-8)).log()).sum(dim=-1)
        weighted_x = (attention * xx).sum(dim=-1)
        weighted_y = (attention * yy).sum(dim=-1)
        best_idx = patch_best.argmax(dim=-1)
        best_x = xx.expand(b, -1).gather(1, best_idx[:, None]).squeeze(1)
        best_y = yy.expand(b, -1).gather(1, best_idx[:, None]).squeeze(1)

        match_features = torch.stack(
            [
                max_match,
                topk_mean,
                mean_match,
                entropy,
                weighted_x,
                weighted_y,
                best_x,
                best_y,
            ],
            dim=-1,
        )

        return torch.cat(
            [match_features, heatmap_feature, network_feature],
            dim=-1,
        )

    def forward(
        self,
        current_tokens,
        goal_tokens,
        heatmap_feature,
        network_feature,
    ):
        features = self.extract_features(
            current_tokens, goal_tokens, heatmap_feature, network_feature
        )
        logit = self.net(features).squeeze(-1)
        return logit, features


def combine_move_and_arrival(move_logits, arrival_logit):
    """
    Convert 3-way movement + binary arrival into normalized 4-action
    log-probabilities:
      P(Stop) = sigmoid(arrival)
      P(move_i) = (1-P(Stop)) * softmax(move_logits)_i
    """
    log_move_gate = F.logsigmoid(-arrival_logit).unsqueeze(-1)
    log_stop = F.logsigmoid(arrival_logit).unsqueeze(-1)
    move_log_prob = F.log_softmax(move_logits, dim=-1) + log_move_gate
    return torch.cat([move_log_prob, log_stop], dim=-1)


def arrival_gated_decision(
    move_logits,
    arrival_logit,
    stop_threshold=ARRIVAL_STOP_THRESHOLD,
):
    threshold = float(stop_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"stop_threshold must be in (0,1), got {threshold}")

    arrival_probability = torch.sigmoid(arrival_logit)
    move_probabilities = torch.softmax(move_logits, dim=-1)
    move_action = move_probabilities.argmax(dim=-1)
    stop_action = torch.full_like(move_action, NUM_MOVE_ACTIONS)
    action = torch.where(arrival_probability >= threshold, stop_action, move_action)

    joint_move = (1.0 - arrival_probability).unsqueeze(-1) * move_probabilities
    joint_probabilities = torch.cat(
        [joint_move, arrival_probability.unsqueeze(-1)], dim=-1
    )
    return {
        "action": action,
        "joint_probabilities": joint_probabilities,
        "arrival_probability": arrival_probability,
        "move_probabilities": move_probabilities,
        "stop_threshold": threshold,
    }


class ArrivalAwareModelBase:
    def _init_v82_context(self):
        self.context_encoder = HeatmapNetworkContextEncoder()
        self.arrival_head = ArrivalReasoningHead()

    def _resolve_inputs(self, current_tokens, goal_tokens, heatmap, network_state):
        if current_tokens is None or goal_tokens is None:
            raise ValueError("Current RGB tokens and Goal tokens are required")
        if heatmap is None or network_state is None:
            raise ValueError("Spatial heatmap and network_state are required")
        return (
            current_tokens.float(),
            goal_tokens.float(),
            heatmap.float(),
            network_state.float(),
        )

    def predict_action(
        self,
        current_tokens,
        goal_tokens,
        heatmap,
        network_state,
        stop_threshold=ARRIVAL_STOP_THRESHOLD,
        return_attention=False,
    ):
        components = self.forward_components(
            current_tokens, goal_tokens, heatmap, network_state,
            return_attention=return_attention,
        )
        decision = arrival_gated_decision(
            components["move_logits"],
            components["arrival_logit"],
            stop_threshold=stop_threshold,
        )
        decision["move_logits"] = components["move_logits"]
        decision["arrival_logit"] = components["arrival_logit"]
        decision["arrival_features"] = components["arrival_features"]
        decision["attention"] = components.get("attention", {})
        return decision


class MeanPoolMLP(ArrivalAwareModelBase, nn.Module):
    """Small: lightweight late-fusion movement reasoning."""

    def __init__(self, token_dim, hidden_dim, num_actions=NUM_ACTIONS):
        super().__init__()
        self._init_v82_context()
        fusion_dim = 2 * token_dim + AUX.heatmap_feature_dim + AUX.network_feature_dim
        self.move_classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, NUM_MOVE_ACTIONS),
        )

    def forward_components(
        self,
        current_tokens,
        goal_tokens,
        heatmap,
        network_state,
        return_attention=False,
    ):
        current_tokens, goal_tokens, heatmap, network_state = self._resolve_inputs(
            current_tokens, goal_tokens, heatmap, network_state
        )
        heat_feat, net_feat = self.context_encoder(heatmap, network_state)
        current = current_tokens.mean(dim=1)
        goal = goal_tokens.mean(dim=1)
        move_logits = self.move_classifier(
            torch.cat([current, goal, heat_feat, net_feat], dim=-1)
        )
        arrival_logit, arrival_features = self.arrival_head(
            current_tokens, goal_tokens, heat_feat, net_feat
        )
        return {
            "move_logits": move_logits,
            "arrival_logit": arrival_logit,
            "arrival_features": arrival_features,
            "attention": {},
        }

    def forward(
        self,
        current_tokens,
        goal_tokens,
        heatmap,
        network_state,
        return_attention=False,
    ):
        c = self.forward_components(
            current_tokens, goal_tokens, heatmap, network_state,
            return_attention=return_attention,
        )
        logits = combine_move_and_arrival(c["move_logits"], c["arrival_logit"])
        if return_attention:
            return logits, {
                "arrival_probability": torch.sigmoid(c["arrival_logit"]),
                "arrival_features": c["arrival_features"],
            }
        return logits


class GoalCrossAttentionMEC(ArrivalAwareModelBase, nn.Module):
    """Medium: Goal -> Current cross-attention plus explicit context."""

    def __init__(self, token_dim, hidden_dim, num_heads, num_actions=NUM_ACTIONS):
        super().__init__()
        self._init_v82_context()

        self.current_pos = nn.Parameter(
            torch.zeros(1, ENCODER.num_tokens, token_dim)
        )
        self.goal_scale = nn.Parameter(
            torch.zeros(1, AUX.goal_token_count, token_dim)
        )
        self.no_match = nn.Parameter(torch.zeros(1, 1, token_dim))

        self.goal_current_attn = nn.MultiheadAttention(
            token_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.goal_current_norm = nn.LayerNorm(token_dim)
        self.goal_current_ffn = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, token_dim),
        )
        self.goal_current_ffn_norm = nn.LayerNorm(token_dim)

        fusion_dim = 3 * token_dim + AUX.heatmap_feature_dim + AUX.network_feature_dim
        self.move_classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, NUM_MOVE_ACTIONS),
        )

    def forward_components(
        self,
        current_tokens,
        goal_tokens,
        heatmap,
        network_state,
        return_attention=False,
    ):
        current_tokens, goal_tokens, heatmap, network_state = self._resolve_inputs(
            current_tokens, goal_tokens, heatmap, network_state
        )
        b = current_tokens.size(0)
        current = current_tokens + self.current_pos
        goal = goal_tokens + self.goal_scale

        kv_current = torch.cat([current, self.no_match.expand(b, -1, -1)], dim=1)
        matched, attn_gc = self.goal_current_attn(
            query=goal,
            key=kv_current,
            value=kv_current,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        matched = self.goal_current_norm(goal + matched)
        matched = self.goal_current_ffn_norm(
            matched + self.goal_current_ffn(matched)
        )

        heat_feat, net_feat = self.context_encoder(heatmap, network_state)
        move_logits = self.move_classifier(
            torch.cat(
                [
                    current_tokens.mean(dim=1),
                    goal_tokens.mean(dim=1),
                    matched.mean(dim=1),
                    heat_feat,
                    net_feat,
                ],
                dim=-1,
            )
        )

        arrival_logit, arrival_features = self.arrival_head(
            current_tokens, goal_tokens, heat_feat, net_feat
        )
        return {
            "move_logits": move_logits,
            "arrival_logit": arrival_logit,
            "arrival_features": arrival_features,
            "attention": {"goal_to_current": attn_gc},
        }

    def forward(
        self,
        current_tokens,
        goal_tokens,
        heatmap,
        network_state,
        return_attention=False,
    ):
        c = self.forward_components(
            current_tokens, goal_tokens, heatmap, network_state,
            return_attention=return_attention,
        )
        logits = combine_move_and_arrival(c["move_logits"], c["arrival_logit"])
        if return_attention:
            out = dict(c["attention"])
            out["arrival_probability"] = torch.sigmoid(c["arrival_logit"])
            out["arrival_features"] = c["arrival_features"]
            return logits, out
        return logits


class ExplicitReasoningTransformer(ArrivalAwareModelBase, nn.Module):
    """Transformer-S/M/L: explicit multimodal reasoning with preserved spatial tokens."""

    def __init__(
        self,
        token_dim,
        hidden_dim,
        num_layers,
        num_heads,
        num_actions=NUM_ACTIONS,
        ff_mult=4,
    ):
        super().__init__()
        self._init_v82_context()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.visual_proj = nn.Linear(token_dim, hidden_dim)
        self.current_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.goal_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.heat_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.net_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.current_pos = nn.Parameter(
            torch.zeros(1, ENCODER.num_tokens, hidden_dim)
        )
        self.goal_pos = nn.Parameter(
            torch.zeros(1, AUX.goal_token_count, hidden_dim)
        )

        self.goal_current_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.goal_current_norm = nn.LayerNorm(hidden_dim)
        self.goal_current_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )
        self.goal_current_ffn_norm = nn.LayerNorm(hidden_dim)

        self.heat_proj = nn.Linear(AUX.heatmap_feature_dim, hidden_dim)
        self.net_proj = nn.Linear(AUX.network_feature_dim, hidden_dim)

        self.fusion_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * ff_mult,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.move_head = nn.Linear(hidden_dim, NUM_MOVE_ACTIONS)

    def forward_components(
        self,
        current_tokens,
        goal_tokens,
        heatmap,
        network_state,
        return_attention=False,
    ):
        current_tokens, goal_tokens, heatmap, network_state = self._resolve_inputs(
            current_tokens, goal_tokens, heatmap, network_state
        )
        b = current_tokens.size(0)

        current = self.visual_proj(current_tokens) + self.current_pos
        goal = self.visual_proj(goal_tokens) + self.goal_pos

        kv_current = torch.cat(
            [current, torch.zeros(b, 1, current.size(-1), device=current.device)],
            dim=1,
        )
        matched, attn_gc = self.goal_current_attn(
            query=goal,
            key=kv_current,
            value=kv_current,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        matched = self.goal_current_norm(goal + matched)
        matched = self.goal_current_ffn_norm(
            matched + self.goal_current_ffn(matched)
        )

        heat_feat, net_feat = self.context_encoder(heatmap, network_state)
        heat_token = self.heat_proj(heat_feat).unsqueeze(1)
        net_token = self.net_proj(net_feat).unsqueeze(1)

        # Keep the full current spatial token sequence. The heatmap/network
        # context is appended as two explicit modality tokens.
        cls = self.cls_token.expand(b, -1, -1)
        current = current + self.current_type
        matched = matched + self.goal_type
        heat_token = heat_token + self.heat_type
        net_token = net_token + self.net_type

        fusion = torch.cat([cls, current, matched, heat_token, net_token], dim=1)
        encoded = self.fusion_encoder(fusion)
        cls_feat = self.final_norm(encoded[:, 0])
        move_logits = self.move_head(cls_feat)

        arrival_logit, arrival_features = self.arrival_head(
            current_tokens, goal_tokens, heat_feat, net_feat
        )

        return {
            "move_logits": move_logits,
            "arrival_logit": arrival_logit,
            "arrival_features": arrival_features,
            "attention": {"goal_to_current": attn_gc},
        }

    def forward(
        self,
        current_tokens,
        goal_tokens,
        heatmap,
        network_state,
        return_attention=False,
    ):
        c = self.forward_components(
            current_tokens, goal_tokens, heatmap, network_state,
            return_attention=return_attention,
        )
        logits = combine_move_and_arrival(c["move_logits"], c["arrival_logit"])
        if return_attention:
            out = dict(c["attention"])
            out["arrival_probability"] = torch.sigmoid(c["arrival_logit"])
            out["arrival_features"] = c["arrival_features"]
            return logits, out
        return logits


def build_gbs_model(name, token_dim=ENCODER.token_dim, actions=NUM_ACTIONS):
    cfg = GBS_MODEL_CONFIGS[name]
    architecture = cfg["architecture"]
    if architecture == "late_fusion_mlp":
        return MeanPoolMLP(token_dim, cfg["hidden_dim"], actions)
    if architecture == "goal_cross_attention":
        return GoalCrossAttentionMEC(
            token_dim, cfg["hidden_dim"], cfg["num_heads"], actions
        )
    if architecture == "explicit_transformer":
        return ExplicitReasoningTransformer(
            token_dim=token_dim,
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            num_actions=actions,
            ff_mult=cfg.get("ff_mult", 4),
        )
    raise ValueError(f"Unknown MEC model/architecture: {name}/{architecture}")
