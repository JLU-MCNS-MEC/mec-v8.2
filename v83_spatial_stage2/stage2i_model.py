"""Stage-2i MEC input ablations T0-T5 (Transformer-M).

T0  Current RGB + Goal RGB + Heatmap + Network State 5D (original M0 inputs)
T1  Current RGB + Goal RGB + Network State 5D (heatmap branch removed)
T2  Current RGB + Goal RGB + Heatmap (network branch removed)
T3  Current RGB + Goal RGB (both branches removed)
T4  T3 + 16-step visual history + UAV relative-pose history + GAB
T5  T4 + anonymous global landmark retrieval graph

Branch removal is structural: the unused heatmap/network encoders, projections
and modality tokens are deleted from the module, not merely zero-filled.

Shared modules (visual projection, goal-current attention, fusion encoder,
movement head, context encoders) are built in one canonical seeded block, so
every variant starts from the same reproducible initialisation for every
parameter it shares.  Variant-specific branches use separate fixed seeds.

The frozen shared RGB encoder is not part of this module: token features are
pre-computed and handed in.  No simulator state, goal coordinate, distance,
oracle action or success label is read here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AUX, ENCODER, NETWORK, NUM_ACTIONS, NUM_MOVE_ACTIONS
from gbs_model_v3 import (
    ArrivalReasoningHead,
    HeatmapNetworkContextEncoder,
    combine_move_and_arrival,
    goal_current_similarity,
)


HISTORY_STEPS = 16
VISUAL_DESCRIPTOR_DIM = 2 * ENCODER.token_dim  # mean + max pooled tokens
POSE_FEATURE_DIM = 10  # 3 relative motion + 3 relative pose + 4 action one-hot
GAB_FEATURE_DIM = 20
MAP_FEATURE_DIM = 22  # 5 nodes x (sim, forward, right, distance) + 2 summary
MAP_TOP_K = 5

SHARED_INIT_SEED = 20264017
EXTRA_SEED_BASE = SHARED_INIT_SEED + 1000

VARIANTS = {
    "T0": dict(use_heatmap=True, use_network=True, use_history=False,
               use_gab=False, use_map=False),
    "T1": dict(use_heatmap=False, use_network=True, use_history=False,
               use_gab=False, use_map=False),
    "T2": dict(use_heatmap=True, use_network=False, use_history=False,
               use_gab=False, use_map=False),
    "T3": dict(use_heatmap=False, use_network=False, use_history=False,
               use_gab=False, use_map=False),
    "T4": dict(use_heatmap=False, use_network=False, use_history=True,
               use_gab=True, use_map=False),
    "T5": dict(use_heatmap=False, use_network=False, use_history=True,
               use_gab=True, use_map=True),
}

VARIANT_DESCRIPTIONS = {
    "T0": "Current RGB + Goal RGB + Heatmap + Network State 5D",
    "T1": "Current RGB + Goal RGB + Network State 5D (no heatmap branch)",
    "T2": "Current RGB + Goal RGB + Heatmap (no network branch)",
    "T3": "Current RGB + Goal RGB (no heatmap / network branches)",
    "T4": "T3 + 16-step visual history + relative pose + GAB",
    "T5": "T4 + anonymous global landmark retrieval graph",
}


def _fork(seed):
    """Deterministic CPU-RNG context for reproducible module initialisation."""
    return torch.random.fork_rng(devices=[], enabled=True)


def match_features(current_tokens, goal_tokens, temperature=None):
    """The 8 observable Current/Goal match features used by the Arrival head.

    This is an exact copy of ``ArrivalReasoningHead.extract_features``'s match
    block so T1-T5 keep the same arrival feature definition as T0.
    """
    if temperature is None:
        temperature = float(AUX.arrival_match_temperature)
    sim = goal_current_similarity(current_tokens, goal_tokens)
    patch_best = sim.max(dim=1).values
    b, n = patch_best.shape

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
    return torch.stack(
        [max_match, topk_mean, mean_match, entropy,
         weighted_x, weighted_y, best_x, best_y],
        dim=-1,
    )


class VariantArrivalHead(nn.Module):
    """Binary Arrival head with an explicitly sized observable context.

    ``context_dim`` is 32 for T1 (network only), 64 for T2 (heatmap only) and
    0 for T3-T5.  The removed modalities never reach this head.
    """

    def __init__(self, context_dim):
        super().__init__()
        self.context_dim = int(context_dim)
        hidden = int(AUX.arrival_hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(8 + self.context_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def extract_features(self, current_tokens, goal_tokens, context):
        feats = [match_features(current_tokens, goal_tokens)]
        if self.context_dim > 0:
            if context is None:
                raise ValueError("Variant arrival head requires context features")
            if context.shape[-1] != self.context_dim:
                raise ValueError(
                    f"Expected context dim {self.context_dim}, "
                    f"got {tuple(context.shape)}"
                )
            feats.append(context)
        elif context is not None:
            raise ValueError("This variant must not receive context features")
        return torch.cat(feats, dim=-1)

    def forward(self, current_tokens, goal_tokens, context):
        features = self.extract_features(current_tokens, goal_tokens, context)
        logit = self.net(features).squeeze(-1)
        return logit, features


class HistoryEncoder(nn.Module):
    """Causal 16-step visual + pose + action history encoder.

    Each history slot contains only past observations.  Missing slots are
    zeroed (mask=0) and therefore cannot leak future information.
    """

    def __init__(self, steps=HISTORY_STEPS, visual_dim=VISUAL_DESCRIPTOR_DIM,
                 pose_dim=POSE_FEATURE_DIM, action_dim=4,
                 hidden_dim=128, num_layers=2, num_heads=4):
        super().__init__()
        self.steps = int(steps)
        self.visual_proj = nn.Linear(int(visual_dim), hidden_dim)
        self.pose_proj = nn.Linear(int(pose_dim), hidden_dim)
        self.action_proj = nn.Linear(int(action_dim), hidden_dim)
        self.step_pos = nn.Parameter(torch.zeros(1, self.steps, hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, history_vis, history_pose, history_action, history_mask):
        if history_vis.ndim != 3 or history_vis.shape[1] != self.steps:
            raise ValueError(
                f"history_vis must be [B,{self.steps},D], "
                f"got {tuple(history_vis.shape)}"
            )
        mask = history_mask.to(history_vis.dtype).unsqueeze(-1)
        x = (
            self.visual_proj(history_vis.float())
            + self.pose_proj(history_pose.float())
            + self.action_proj(history_action.float())
        ) * mask
        x = x + self.step_pos * mask
        cls = self.cls_token.expand(x.size(0), -1, -1)
        encoded = self.encoder(torch.cat([cls, x], dim=1))
        return self.norm(encoded[:, 0])


class Stage2ITransformer(nn.Module):
    """Transformer-M sized MEC with structural Stage-2i modality ablations."""

    def __init__(self, variant, token_dim=ENCODER.token_dim,
                 hidden_dim=256, num_layers=2, num_heads=8, ff_mult=4):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown Stage-2i variant {variant}")
        self.variant = variant
        flags = VARIANTS[variant]
        self.use_heatmap = bool(flags["use_heatmap"])
        self.use_network = bool(flags["use_network"])
        self.use_history = bool(flags["use_history"])
        self.use_gab = bool(flags["use_gab"])
        self.use_map = bool(flags["use_map"])
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        # ------------------------------------------------------------------
        # Canonical seeded block: identical construction order for every
        # variant, exactly mirroring ExplicitReasoningTransformer so T0 is
        # the unmodified M0 architecture.
        # ------------------------------------------------------------------
        with _fork(SHARED_INIT_SEED):
            torch.manual_seed(SHARED_INIT_SEED)
            self.context_encoder = HeatmapNetworkContextEncoder()
            full_arrival_head = ArrivalReasoningHead()

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

        if variant == "T0":
            self.arrival_head = full_arrival_head
        else:
            del full_arrival_head
            context_dim = 0
            if self.use_heatmap:
                context_dim += int(AUX.heatmap_feature_dim)
            if self.use_network:
                context_dim += int(AUX.network_feature_dim)
            with _fork(EXTRA_SEED_BASE + 4):
                torch.manual_seed(EXTRA_SEED_BASE + 4)
                self.arrival_head = VariantArrivalHead(context_dim)

        # Structural removal of unused context branches and their projections.
        if not self.use_heatmap:
            del self.context_encoder.heatmap_net
            del self.heat_proj
            del self.heat_type
        if not self.use_network:
            del self.context_encoder.network_net
            del self.net_proj
            del self.net_type

        # ------------------------------------------------------------------
        # Variant-specific branches (deterministic, independent seeds).
        # T4 and T5 receive identical history/GAB initialisation.
        # ------------------------------------------------------------------
        if self.use_history:
            with _fork(EXTRA_SEED_BASE + 1):
                torch.manual_seed(EXTRA_SEED_BASE + 1)
                self.history_encoder = HistoryEncoder()
                self.hist_proj = nn.Linear(128, hidden_dim)
                self.hist_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        if self.use_gab:
            with _fork(EXTRA_SEED_BASE + 2):
                torch.manual_seed(EXTRA_SEED_BASE + 2)
                self.gab_proj = nn.Linear(GAB_FEATURE_DIM, hidden_dim)
                self.gab_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        if self.use_map:
            with _fork(EXTRA_SEED_BASE + 3):
                torch.manual_seed(EXTRA_SEED_BASE + 3)
                self.map_proj = nn.Linear(MAP_FEATURE_DIM, hidden_dim)
                self.map_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    # ------------------------------------------------------------------
    # Input validation: removed modalities must not be supplied at all.
    # ------------------------------------------------------------------
    def _check_inputs(self, current_tokens, goal_tokens, heatmap, network_state,
                      history_vis, history_pose, history_action, history_mask,
                      gab, map_feat):
        if current_tokens is None or goal_tokens is None:
            raise ValueError("Current RGB tokens and Goal tokens are required")
        if self.use_heatmap and heatmap is None:
            raise ValueError(f"{self.variant} requires the heatmap branch input")
        if not self.use_heatmap and heatmap is not None:
            raise ValueError(
                f"{self.variant} has no heatmap branch; heatmap must not be passed"
            )
        if self.use_network and network_state is None:
            raise ValueError(f"{self.variant} requires the network-state input")
        if not self.use_network and network_state is not None:
            raise ValueError(
                f"{self.variant} has no network branch; network_state must not be passed"
            )
        extras = (history_vis, history_pose, history_action, history_mask)
        if self.use_history and any(item is None for item in extras):
            raise ValueError(f"{self.variant} requires history tensors")
        if not self.use_history and any(item is not None for item in extras):
            raise ValueError(f"{self.variant} has no history branch")
        if self.use_gab and gab is None:
            raise ValueError(f"{self.variant} requires GAB features")
        if not self.use_gab and gab is not None:
            raise ValueError(f"{self.variant} has no GAB branch")
        if self.use_map and map_feat is None:
            raise ValueError(f"{self.variant} requires map features")
        if not self.use_map and map_feat is not None:
            raise ValueError(f"{self.variant} has no map branch")

    def forward_components(
        self,
        current_tokens,
        goal_tokens,
        heatmap=None,
        network_state=None,
        history_vis=None,
        history_pose=None,
        history_action=None,
        history_mask=None,
        gab=None,
        map_feat=None,
    ):
        self._check_inputs(
            current_tokens, goal_tokens, heatmap, network_state,
            history_vis, history_pose, history_action, history_mask,
            gab, map_feat,
        )
        current_tokens = current_tokens.float()
        goal_tokens = goal_tokens.float()
        b = current_tokens.size(0)

        current = self.visual_proj(current_tokens) + self.current_pos
        goal = self.visual_proj(goal_tokens) + self.goal_pos

        kv_current = torch.cat(
            [current, torch.zeros(b, 1, current.size(-1), device=current.device)],
            dim=1,
        )
        matched, _attn = self.goal_current_attn(
            query=goal, key=kv_current, value=kv_current,
            need_weights=False,
        )
        matched = self.goal_current_norm(goal + matched)
        matched = self.goal_current_ffn_norm(
            matched + self.goal_current_ffn(matched)
        )

        context_feats = []
        tokens = [
            self.cls_token.expand(b, -1, -1),
            current + self.current_type,
            matched + self.goal_type,
        ]

        if self.use_heatmap:
            if heatmap.ndim != 4:
                raise ValueError(
                    f"heatmap must be [B,3,8,8], got {tuple(heatmap.shape)}"
                )
            heat_feat = self.context_encoder.heatmap_net(heatmap.float())
            context_feats.append(heat_feat)
            tokens.append(
                self.heat_proj(heat_feat).unsqueeze(1) + self.heat_type
            )
        if self.use_network:
            if network_state.ndim != 2:
                raise ValueError(
                    f"network_state must be [B,5], got {tuple(network_state.shape)}"
                )
            net_feat = self.context_encoder.network_net(network_state.float())
            context_feats.append(net_feat)
            tokens.append(
                self.net_proj(net_feat).unsqueeze(1) + self.net_type
            )
        if self.use_history:
            hist_emb = self.history_encoder(
                history_vis, history_pose, history_action, history_mask
            )
            tokens.append(
                self.hist_proj(hist_emb).unsqueeze(1) + self.hist_type
            )
        if self.use_gab:
            tokens.append(
                self.gab_proj(gab.float()).unsqueeze(1) + self.gab_type
            )
        if self.use_map:
            tokens.append(
                self.map_proj(map_feat.float()).unsqueeze(1) + self.map_type
            )

        encoded = self.fusion_encoder(torch.cat(tokens, dim=1))
        move_logits = self.move_head(self.final_norm(encoded[:, 0]))

        context = (
            torch.cat(context_feats, dim=-1) if context_feats else None
        )
        if self.variant == "T0":
            # Exactly the original ArrivalReasoningHead signature.
            arrival_logit, arrival_features = self.arrival_head(
                current_tokens, goal_tokens,
                context_feats[0], context_feats[1],
            )
        else:
            arrival_logit, arrival_features = self.arrival_head(
                current_tokens, goal_tokens, context
            )

        return {
            "move_logits": move_logits,
            "arrival_logit": arrival_logit,
            "arrival_features": arrival_features,
            "attention": {},
        }

    def forward(self, *args, **kwargs):
        comp = self.forward_components(*args, **kwargs)
        return combine_move_and_arrival(
            comp["move_logits"], comp["arrival_logit"]
        )


def build_stage2i_model(variant):
    return Stage2ITransformer(variant)


def parameter_count(model):
    return int(sum(p.numel() for p in model.parameters()))
