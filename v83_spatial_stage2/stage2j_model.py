"""Stage-2j: five heterogeneous MEC architectures on the Stage-2i T5 inputs.

All five models receive exactly the same observable input set:

  * Current RGB tokens           [B, 576, 384]
  * Goal RGB tokens              [B, 3, 384]
  * 16-step visual history       [B, 16, 768]  mean+max pooled past tokens
  * UAV relative pose history    [B, 16, 10]
  * executed-action history      [B, 16, 4]
  * history mask                 [B, 16]
  * GAB target-bearing belief    [B, 20]
  * anonymous landmark graph     [B, 22]       Top-5 retrieval + 2 summaries

There is no 3x8x8 heatmap branch and no network-state branch.  GAB may be
computed from observable Current/Goal similarity, but only its 20-D belief
enters the MEC.

Architecture identity follows ``GBS_MODEL_CONFIGS`` exactly:

  Small          late_fusion_mlp       256
  Transformer-S  explicit_transformer  192 / 1 layer / 4 heads / ff x3
  Medium         goal_cross_attention  384 / 6 heads
  Transformer-M  explicit_transformer  256 / 2 layers / 8 heads / ff x4
  Large          explicit_transformer  352 / 2 layers / 8 heads / ff x4

Transformer-M is *exactly* ``Stage2ITransformer("T5")``: same class, same
configuration, same state-dict keys and values as the Stage-2i checkpoint.
Transformer-S and Large are the same explicit-transformer module at their
native widths.  Small and Medium keep their original movement backbones
(mean-pool late fusion, and Goal->Current cross-attention + late fusion);
only the removed heatmap/network context is replaced by the shared
history/GAB/map inputs.

The shared history encoder and the 8-feature observable Arrival head are
protocol-level input/inference components.  They are initialised from the
same fixed seeds as Stage-2i T5 for every architecture, so they are not part
of the architecture difference under test.  The Arrival head never receives
map geometry, true distance or target coordinates.

No simulator state, goal coordinate, distance, oracle action or success label
is read here.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

from config import AUX, ENCODER, GBS_MODEL_CONFIGS, NUM_MOVE_ACTIONS
from gbs_model_v3 import combine_move_and_arrival
from v83_spatial_stage2.stage2i_model import (
    EXTRA_SEED_BASE,
    GAB_FEATURE_DIM,
    HISTORY_STEPS,
    MAP_FEATURE_DIM,
    POSE_FEATURE_DIM,
    VISUAL_DESCRIPTOR_DIM,
    HistoryEncoder,
    Stage2ITransformer,
    VariantArrivalHead,
    parameter_count,
)


TABLE_ORDER = ("Small", "Transformer-S", "Medium", "Transformer-M", "Large")
EXECUTION_ORDER = ("Small", "Transformer-S", "Medium", "Large", "Transformer-M")

HISTORY_EMBED_DIM = 128

# These are the native architecture identities, not a new family of models.
STAGE2J_CONFIGS = {
    "Small": {
        "architecture": "late_fusion_mlp",
        "hidden_dim": 256,
        "num_layers": 0,
        "num_heads": 1,
    },
    "Transformer-S": {
        "architecture": "explicit_transformer",
        "hidden_dim": 192,
        "num_layers": 1,
        "num_heads": 4,
        "ff_mult": 3,
    },
    "Medium": {
        "architecture": "goal_cross_attention",
        "hidden_dim": 384,
        "num_layers": 0,
        "num_heads": 6,
    },
    "Transformer-M": {
        "architecture": "explicit_transformer",
        "hidden_dim": 256,
        "num_layers": 2,
        "num_heads": 8,
        "ff_mult": 4,
    },
    "Large": {
        "architecture": "explicit_transformer",
        "hidden_dim": 352,
        "num_layers": 2,
        "num_heads": 8,
        "ff_mult": 4,
    },
}


@contextlib.contextmanager
def _seeded(seed):
    """Deterministic CPU-RNG context (fork/restore), matching Stage-2i."""
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(int(seed))
        yield


def _validate_common_inputs(
    current_tokens,
    goal_tokens,
    history_vis,
    history_pose,
    history_action,
    history_mask,
    gab,
    map_feat,
    heatmap=None,
    network_state=None,
):
    """Enforce the single Stage-2j input contract for every MEC."""
    if current_tokens is None or goal_tokens is None:
        raise ValueError("Current RGB tokens and Goal tokens are required")
    if heatmap is not None:
        raise ValueError("Stage-2j has no heatmap branch; heatmap must not be passed")
    if network_state is not None:
        raise ValueError(
            "Stage-2j has no network-state branch; network_state must not be passed"
        )
    for name, value in (
        ("history_vis", history_vis),
        ("history_pose", history_pose),
        ("history_action", history_action),
        ("history_mask", history_mask),
        ("gab", gab),
        ("map_feat", map_feat),
    ):
        if value is None:
            raise ValueError(f"Stage-2j requires {name}")

    if current_tokens.ndim != 3:
        raise ValueError(
            f"current_tokens must be [B,{ENCODER.num_tokens},"
            f"{ENCODER.token_dim}], got {tuple(current_tokens.shape)}"
        )
    if current_tokens.shape[1:] != (ENCODER.num_tokens, ENCODER.token_dim):
        raise ValueError(
            f"current_tokens must be [B,{ENCODER.num_tokens},"
            f"{ENCODER.token_dim}], got {tuple(current_tokens.shape)}"
        )
    if goal_tokens.ndim != 3 or goal_tokens.shape[1:] != (
        AUX.goal_token_count,
        ENCODER.token_dim,
    ):
        raise ValueError(
            f"goal_tokens must be [B,{AUX.goal_token_count},"
            f"{ENCODER.token_dim}], got {tuple(goal_tokens.shape)}"
        )
    if history_vis.ndim != 3 or history_vis.shape[1:] != (
        HISTORY_STEPS,
        VISUAL_DESCRIPTOR_DIM,
    ):
        raise ValueError(
            f"history_vis must be [B,{HISTORY_STEPS},"
            f"{VISUAL_DESCRIPTOR_DIM}], got {tuple(history_vis.shape)}"
        )
    if history_pose.ndim != 3 or history_pose.shape[1:] != (
        HISTORY_STEPS,
        POSE_FEATURE_DIM,
    ):
        raise ValueError(
            f"history_pose must be [B,{HISTORY_STEPS},"
            f"{POSE_FEATURE_DIM}], got {tuple(history_pose.shape)}"
        )
    if history_action.ndim != 3 or history_action.shape[1:] != (HISTORY_STEPS, 4):
        raise ValueError(
            f"history_action must be [B,{HISTORY_STEPS},4], "
            f"got {tuple(history_action.shape)}"
        )
    if history_mask.ndim != 2 or history_mask.shape[1:] != (HISTORY_STEPS,):
        raise ValueError(
            f"history_mask must be [B,{HISTORY_STEPS}], "
            f"got {tuple(history_mask.shape)}"
        )
    if gab.ndim != 2 or gab.shape[-1] != GAB_FEATURE_DIM:
        raise ValueError(
            f"gab must be [B,{GAB_FEATURE_DIM}], got {tuple(gab.shape)}"
        )
    if map_feat.ndim != 2 or map_feat.shape[-1] != MAP_FEATURE_DIM:
        raise ValueError(
            f"map_feat must be [B,{MAP_FEATURE_DIM}], got {tuple(map_feat.shape)}"
        )
    batch = current_tokens.size(0)
    for name, value in (
        ("goal_tokens", goal_tokens),
        ("history_vis", history_vis),
        ("history_pose", history_pose),
        ("history_action", history_action),
        ("history_mask", history_mask),
        ("gab", gab),
        ("map_feat", map_feat),
    ):
        if value.size(0) != batch:
            raise ValueError(
                f"{name} batch {value.size(0)} != current_tokens batch {batch}"
            )


class _Stage2JCommon(nn.Module):
    """Shared observable input adapters and Arrival head.

    These modules use the same fixed seeds for all five architectures so the
    comparison changes the MEC decision backbone, not its input encoding.
    """

    def _init_common(self):
        with _seeded(EXTRA_SEED_BASE + 1):
            self.history_encoder = HistoryEncoder()
        with _seeded(EXTRA_SEED_BASE + 4):
            self.arrival_head = VariantArrivalHead(context_dim=0)

    def _encode_history(self, history_vis, history_pose, history_action,
                        history_mask):
        return self.history_encoder(
            history_vis, history_pose, history_action, history_mask
        )

    def _arrival(self, current_tokens, goal_tokens):
        # Only the 8 observable Current/Goal match features.  No map geometry,
        # no executed action, no true distance.
        return self.arrival_head(current_tokens.float(), goal_tokens.float(), None)


class Stage2JSmallMEC(_Stage2JCommon):
    """Small: native late-fusion MLP over pooled visual + modality features."""

    architecture = "late_fusion_mlp"

    def __init__(self, token_dim=ENCODER.token_dim, hidden_dim=256):
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self._init_common()
        fusion_dim = (
            2 * self.token_dim
            + HISTORY_EMBED_DIM
            + GAB_FEATURE_DIM
            + MAP_FEATURE_DIM
        )
        self.fusion_dim = int(fusion_dim)
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
        heatmap=None,
        network_state=None,
        history_vis=None,
        history_pose=None,
        history_action=None,
        history_mask=None,
        gab=None,
        map_feat=None,
    ):
        _validate_common_inputs(
            current_tokens, goal_tokens, history_vis, history_pose,
            history_action, history_mask, gab, map_feat,
            heatmap=heatmap, network_state=network_state,
        )
        current_tokens = current_tokens.float()
        goal_tokens = goal_tokens.float()
        hist_emb = self._encode_history(
            history_vis, history_pose, history_action, history_mask
        )
        move_logits = self.move_classifier(
            torch.cat(
                [
                    current_tokens.mean(dim=1),
                    goal_tokens.mean(dim=1),
                    hist_emb,
                    gab.float(),
                    map_feat.float(),
                ],
                dim=-1,
            )
        )
        arrival_logit, arrival_features = self._arrival(
            current_tokens, goal_tokens
        )
        return {
            "move_logits": move_logits,
            "arrival_logit": arrival_logit,
            "arrival_features": arrival_features,
            "attention": {},
        }

    def forward(self, *args, **kwargs):
        components = self.forward_components(*args, **kwargs)
        return combine_move_and_arrival(
            components["move_logits"], components["arrival_logit"]
        )


class Stage2JMediumMEC(_Stage2JCommon):
    """Medium: native Goal->Current cross-attention + late-fusion classifier."""

    architecture = "goal_cross_attention"

    def __init__(
        self,
        token_dim=ENCODER.token_dim,
        hidden_dim=384,
        num_heads=6,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        # The native Medium uses token_dim (384) as the attention width.
        if self.token_dim % self.num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self._init_common()

        self.current_pos = nn.Parameter(
            torch.zeros(1, ENCODER.num_tokens, self.token_dim)
        )
        self.goal_scale = nn.Parameter(
            torch.zeros(1, AUX.goal_token_count, self.token_dim)
        )
        self.no_match = nn.Parameter(torch.zeros(1, 1, self.token_dim))

        self.goal_current_attn = nn.MultiheadAttention(
            self.token_dim, self.num_heads, dropout=0.1, batch_first=True
        )
        self.goal_current_norm = nn.LayerNorm(self.token_dim)
        self.goal_current_ffn = nn.Sequential(
            nn.Linear(self.token_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.token_dim),
        )
        self.goal_current_ffn_norm = nn.LayerNorm(self.token_dim)

        fusion_dim = (
            3 * self.token_dim
            + HISTORY_EMBED_DIM
            + GAB_FEATURE_DIM
            + MAP_FEATURE_DIM
        )
        self.fusion_dim = int(fusion_dim)
        self.move_classifier = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, NUM_MOVE_ACTIONS),
        )

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
        _validate_common_inputs(
            current_tokens, goal_tokens, history_vis, history_pose,
            history_action, history_mask, gab, map_feat,
            heatmap=heatmap, network_state=network_state,
        )
        current_tokens = current_tokens.float()
        goal_tokens = goal_tokens.float()
        batch = current_tokens.size(0)

        current = current_tokens + self.current_pos
        goal = goal_tokens + self.goal_scale
        kv_current = torch.cat(
            [current, self.no_match.expand(batch, -1, -1)], dim=1
        )
        matched, _attn = self.goal_current_attn(
            query=goal,
            key=kv_current,
            value=kv_current,
            need_weights=False,
        )
        matched = self.goal_current_norm(goal + matched)
        matched = self.goal_current_ffn_norm(
            matched + self.goal_current_ffn(matched)
        )

        hist_emb = self._encode_history(
            history_vis, history_pose, history_action, history_mask
        )
        move_logits = self.move_classifier(
            torch.cat(
                [
                    current_tokens.mean(dim=1),
                    goal_tokens.mean(dim=1),
                    matched.mean(dim=1),
                    hist_emb,
                    gab.float(),
                    map_feat.float(),
                ],
                dim=-1,
            )
        )
        arrival_logit, arrival_features = self._arrival(
            current_tokens, goal_tokens
        )
        return {
            "move_logits": move_logits,
            "arrival_logit": arrival_logit,
            "arrival_features": arrival_features,
            "attention": {},
        }

    def forward(self, *args, **kwargs):
        components = self.forward_components(*args, **kwargs)
        return combine_move_and_arrival(
            components["move_logits"], components["arrival_logit"]
        )


class Stage2JTransformerMEC(Stage2ITransformer):
    """Explicit-transformer MEC with the Stage-2j input contract.

    This adds only shape validation in front of the unchanged Stage-2i T5
    module.  It introduces no parameters, so the state dict and outputs of
    Transformer-M remain exactly those of ``Stage2ITransformer("T5")``.
    """

    architecture = "explicit_transformer"

    def __init__(self, hidden_dim, num_layers, num_heads, ff_mult):
        super().__init__(
            "T5",
            token_dim=ENCODER.token_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_mult=ff_mult,
        )

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
        _validate_common_inputs(
            current_tokens, goal_tokens, history_vis, history_pose,
            history_action, history_mask, gab, map_feat,
            heatmap=heatmap, network_state=network_state,
        )
        return super().forward_components(
            current_tokens,
            goal_tokens,
            heatmap=heatmap,
            network_state=network_state,
            history_vis=history_vis,
            history_pose=history_pose,
            history_action=history_action,
            history_mask=history_mask,
            gab=gab,
            map_feat=map_feat,
        )


def build_stage2j_model(name):
    """Build one of the five native MEC architectures on the T5 inputs."""
    if name not in STAGE2J_CONFIGS:
        raise ValueError(f"Unknown Stage-2j MEC {name}")
    cfg = STAGE2J_CONFIGS[name]
    architecture = cfg["architecture"]
    if architecture == "late_fusion_mlp":
        return Stage2JSmallMEC(
            token_dim=ENCODER.token_dim, hidden_dim=cfg["hidden_dim"]
        )
    if architecture == "goal_cross_attention":
        return Stage2JMediumMEC(
            token_dim=ENCODER.token_dim,
            hidden_dim=cfg["hidden_dim"],
            num_heads=cfg["num_heads"],
        )
    if architecture == "explicit_transformer":
        # Transformer-M (256/2/8/4) is exactly Stage2ITransformer("T5"),
        # whose state dict is the Stage-2i T5 checkpoint.
        return Stage2JTransformerMEC(
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            ff_mult=cfg.get("ff_mult", 4),
        )
    raise ValueError(f"Unknown architecture {architecture} for {name}")


def model_metadata(name):
    """Architecture identity, resolved config and parameter counts."""
    cfg = STAGE2J_CONFIGS[name]
    model = build_stage2j_model(name)
    return {
        "model": name,
        "architecture": cfg["architecture"],
        "config": dict(cfg),
        "parameters": parameter_count(model),
        "trainable_parameters": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
    }


def assert_native_configs():
    """Fail if the Stage-2j table drifts from GBS_MODEL_CONFIGS."""
    for name, cfg in STAGE2J_CONFIGS.items():
        native = GBS_MODEL_CONFIGS[name]
        for key in ("architecture", "hidden_dim", "num_layers", "num_heads"):
            if cfg[key] != native[key]:
                raise AssertionError(
                    f"{name}.{key}: Stage-2j {cfg[key]} != native {native[key]}"
                )
        if cfg.get("ff_mult", 4) != native.get("ff_mult", 4):
            raise AssertionError(f"{name}.ff_mult drifted from native config")
    return True


assert_native_configs()
