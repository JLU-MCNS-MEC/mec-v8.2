from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    # V8.2 continues to use the V8.1 AerialVLN-S-like synthetic environment.
    world_size_m: float = 1500.0
    image_size: int = 336

    # Fixed altitude + yaw navigation.
    uav_altitude_m: float = 80.0
    camera_hfov_deg: float = 90.0
    camera_vfov_deg: float = 80.0
    camera_pitch_deg: float = -40.0

    forward_step_m: float = 5.0
    turn_step_deg: float = 15.0
    stop_radius_m: float = 20.0
    max_steps: int = 400

    min_start_goal_distance_m: float = 150.0
    max_start_goal_distance_m: float = 550.0

    boundary_margin_m: float = 100.0
    num_landmarks: int = 160
    distractor_ratio: float = 0.55
    initial_yaw_error_deg: float = 30.0

    landmark_size_min_m: float = 12.0
    landmark_size_max_m: float = 42.0
    goal_size_min_m: float = 18.0
    goal_size_max_m: float = 42.0

    brightness_min: float = 0.75
    brightness_max: float = 1.20
    sensor_noise_std: float = 4.0
    haze_start_m: float = 180.0
    haze_end_m: float = 700.0


@dataclass(frozen=True)
class EncoderConfig:
    # Reuse the already-trained shared RGB encoder.
    image_size: int = 336
    patch_size: int = 14
    token_dim: int = 384
    num_layers: int = 2
    num_heads: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.10

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_tokens(self) -> int:
        return self.grid_size ** 2


@dataclass(frozen=True)
class AuxiliaryConfig:
    # ------------------------------------------------------------------
    # Legacy environment helpers kept for diagnostics/backward compatibility.
    # These fields are NOT passed to the V8.2 MEC input.
    # ------------------------------------------------------------------
    binary_size: int = 32
    binary_channel_spread_threshold: float = 35.0
    binary_luma_margin: float = 30.0

    # V8 legacy depth helpers. Depth is NOT part of the V8.2 MEC input.
    depth_max_m: float = 250.0
    depth_token_dim: int = 3

    # Goal reference remains multi-scale. Each scale becomes one 384-D token.
    goal_reference_size: int = 96
    goal_reference_object_pixels: tuple = (14, 28, 56)

    # Explicit goal-conditioned spatial relation heatmap:
    # one channel per Goal reference scale, generated from Current/Goal
    # cosine similarity on the 24x24 RGB-token grid, then downsampled to 8x8.
    spatial_heatmap_grid: int = 8
    spatial_heatmap_channels: int = 3
    spatial_match_temperature: float = 0.10

    # Small shared context encoder dimensions.
    heatmap_feature_dim: int = 64
    network_feature_dim: int = 32

    # Arrival reasoning remains a binary head.
    arrival_match_temperature: float = 0.10
    arrival_hidden_dim: int = 64

    @property
    def goal_token_count(self) -> int:
        return len(self.goal_reference_object_pixels)

    @property
    def spatial_heatmap_dim(self) -> int:
        return self.spatial_heatmap_grid ** 2 * self.spatial_heatmap_channels

    @property
    def network_state_dim(self) -> int:
        return NETWORK.network_state_dim


@dataclass(frozen=True)
class TrainingConfig:
    # Keep the V8.1 training budget unchanged for a clean input-modality comparison.
    encoder_pretrain_updates: int = 3500
    encoder_pretrain_batch_size: int = 8
    encoder_pretrain_lr: float = 1e-4
    encoder_pretrain_weight_decay: float = 1e-4
    encoder_validate_every: int = 250
    encoder_validation_batches: int = 20
    encoder_match_loss_weight: float = 0.50
    encoder_match_temperature: float = 0.07

    token_train_episodes: int = 300
    token_val_episodes: int = 60
    token_test_episodes: int = 60
    token_shard_size: int = 100

    trajectory_mix: tuple = (
        ("short", 0.30, 150.0, 250.0, 30, 80),
        ("medium", 0.50, 250.0, 400.0, 50, 115),
        ("long", 0.20, 400.0, 550.0, 80, 160),
    )
    trajectory_max_candidate_attempts: int = 250

    # Same arrival hard-state supervision as V8.1.
    arrival_hard_positive_per_episode: int = 2
    arrival_hard_boundary_negative_per_episode: int = 2
    arrival_hard_distractor_negative_per_episode: int = 2
    arrival_positive_distance_range_m: tuple = (5.0, 19.0)
    arrival_boundary_negative_range_m: tuple = (21.0, 40.0)
    arrival_distractor_near_range_m: tuple = (5.0, 20.0)
    arrival_distractor_goal_min_distance_m: float = 80.0
    arrival_pose_yaw_jitter_deg: float = 25.0

    arrival_natural_near_radius_m: float = 60.0
    arrival_far_negative_keep_prob: float = 0.03

    # MEC training.
    gbs_epochs: int = 200
    gbs_lr: float = 1e-4
    use_amp: bool = True

    # Movement only learns F/L/R; Stop is learned by the binary Arrival head.
    movement_class_balance_gamma: float = 0.5
    arrival_loss_weight: float = 1.0

    checkpoint_macro_f1_weight: float = 0.70
    checkpoint_arrival_f1_weight: float = 0.30


@dataclass(frozen=True)
class NetworkConfig:
    # Global network context is derived from the same 20-MEC layout.
    num_gbs: int = 20
    coverage_radius_m: float = 500.0
    min_spacing_m: float = 120.0
    required_grid_coverage: int = 2
    max_layout_attempts: int = 10000

    gbs_height_m: float = 10.0
    uav_altitude_m: float = 80.0
    carrier_frequency_hz: float = 2.0e9

    # Per-node physical bandwidth classes used only by the network-state generator.
    # Model type / compute capacity remain hidden from the MEC input.
    bandwidth_hz: float = 5.0e6
    gbs_bandwidth_classes_hz: tuple = (5.0e6, 10.0e6, 20.0e6, 30.0e6)
    max_bandwidth_hz: float = 30.0e6

    default_tx_power_w: float = 0.1
    min_tx_power_w: float = 0.01
    max_tx_power_w: float = 0.2

    noise_psd_dbm_per_hz: float = -174.0
    noise_figure_db: float = 9.0
    los_a: float = 9.61
    los_b: float = 0.16
    excessive_loss_los_db: float = 1.0
    excessive_loss_nlos_db: float = 20.0

    visual_token_bits_per_element: int = 16
    depth_token_bits_per_element: int = 16
    decision_latency_budget_s: float = 0.100
    uav_compute_flops_per_s: float = 235.8e9
    mec_compute_classes_flops_per_s: tuple = (0.50e12, 1.00e12, 2.00e12)

    network_state_dim: int = 5

    # Network-state normalization constants.
    relative_distance_clip: float = 2.0
    snr_min_db: float = -20.0
    snr_max_db: float = 40.0
    tcom_clip_ms: float = 200.0

    # One fixed layout per episode; separate bases keep train/val/test independent.
    layout_seed_base: int = 820000


ENV = EnvConfig()
ENCODER = EncoderConfig()
# NETWORK must exist before AuxiliaryConfig.network_state_dim is evaluated at runtime.
NETWORK = NetworkConfig()
AUX = AuxiliaryConfig()
TRAIN = TrainingConfig()


# Five heterogeneous MEC architectures remain unchanged in scale.
GBS_MODEL_CONFIGS = {
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

NUM_ACTIONS = 4
NUM_MOVE_ACTIONS = 3
ARRIVAL_STOP_THRESHOLD = 0.50
