import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from config import AUX, ENCODER, ENV

FORWARD = 0
LEFT = 1
RIGHT = 2
STOP = 3

ACTION_NAMES = {
    FORWARD: "Forward",
    LEFT: "Left",
    RIGHT: "Right",
    STOP: "Stop",
}


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class Landmark:
    x: float
    y: float
    physical_size_m: float
    shape: str
    base_color: np.ndarray
    marker_type: str
    marker_color: np.ndarray
    signature: tuple
    is_distractor: bool

    @property
    def position(self):
        return np.asarray([self.x, self.y], dtype=np.float32)

    @property
    def height_m(self):
        return float(self.physical_size_m)


class ToyUAVEnv:
    """
    V8合成无人机Image-Goal导航环境。

    设计原则：
    1) 保持接近AerialVLN-S的较短空间尺度和F/L/R/Stop离散动作；
    2) 固定80m高度，但相机改为-40°向下斜视、80° VFOV，使20m Stop区域仍可见；
    3) Goal不是“唯一红色diamond”。所有landmark都具有shape/color/内部marker，
       Goal Reference给出目标自身外观，真实目标与参考图共享同一视觉签名；
    4) 为避免“为了复杂而复杂”，只提供适度、现实可解释的视觉身份标记，
       类似建筑/屋顶/设施上的颜色与几何标识；不会提供坐标、距离或oracle mask；
    5) simulator中的Goal投影/soft mask只允许用于数据诊断和shared encoder预训练，
       MEC在线推理绝不读取它们；
    6) V8新增普通深度传感：深度图对Goal和所有distractor一视同仁，仅提供可见表面的
       几何距离，不提供Goal ID、Goal坐标或“距离Goal还有多少米”。
    """

    def __init__(self, config=ENV):
        self.cfg = config
        self.world_size = float(config.world_size_m)
        self.img_size = int(config.image_size)
        self.forward_step = float(config.forward_step_m)
        self.turn_step = np.deg2rad(config.turn_step_deg)
        self.stop_radius = float(config.stop_radius_m)
        self.max_steps = int(config.max_steps)
        self.min_start_goal_distance = float(config.min_start_goal_distance_m)
        self.max_start_goal_distance = float(config.max_start_goal_distance_m)
        self.margin = float(config.boundary_margin_m)
        self.num_landmarks = int(config.num_landmarks)

        self.uav_altitude_m = float(config.uav_altitude_m)
        self.hfov_rad = np.deg2rad(config.camera_hfov_deg)
        self.vfov_rad = np.deg2rad(config.camera_vfov_deg)
        self.camera_pitch_rad = np.deg2rad(config.camera_pitch_deg)

        self.x = None
        self.y = None
        self.yaw = None
        self.start = None
        self.goal = None
        self.goal_landmark = None
        self.landmarks = []
        self.step_count = 0
        self.rng = np.random.default_rng()

        self._start_reference_mask = None
        self._goal_reference_rgb = None

        # 离散颜色基元 + 连续小扰动。每个地标还具有内部marker，
        # 组合后形成更丰富但仍可控的Image-Goal外观。
        self.base_palette = [
            np.asarray([185, 60, 75], dtype=np.float32),
            np.asarray([55, 105, 185], dtype=np.float32),
            np.asarray([70, 150, 110], dtype=np.float32),
            np.asarray([185, 135, 55], dtype=np.float32),
            np.asarray([135, 80, 175], dtype=np.float32),
            np.asarray([70, 155, 165], dtype=np.float32),
            np.asarray([165, 95, 105], dtype=np.float32),
            np.asarray([115, 125, 75], dtype=np.float32),
        ]
        self.marker_palette = [
            np.asarray([235, 235, 225], dtype=np.float32),
            np.asarray([30, 35, 45], dtype=np.float32),
            np.asarray([245, 205, 65], dtype=np.float32),
            np.asarray([80, 220, 230], dtype=np.float32),
        ]
        self.shapes = ["square", "circle", "triangle", "diamond", "cross", "bar"]
        self.marker_types = [
            "dot",
            "ring",
            "hbar",
            "vbar",
            "plus",
            "x",
            "double_dot",
            "frame",
        ]

    # ------------------------------------------------------------------
    # Episode / appearance sampling
    # ------------------------------------------------------------------
    def _sample_signature(self, forbidden=None):
        forbidden = set() if forbidden is None else set(forbidden)
        for _ in range(1000):
            shape_id = int(self.rng.integers(0, len(self.shapes)))
            base_id = int(self.rng.integers(0, len(self.base_palette)))
            marker_id = int(self.rng.integers(0, len(self.marker_types)))
            marker_color_id = int(self.rng.integers(0, len(self.marker_palette)))
            sig = (shape_id, base_id, marker_id, marker_color_id)
            if sig not in forbidden:
                return sig
        raise RuntimeError("Unable to sample a unique landmark signature")

    def _landmark_from_signature(self, x, y, size, signature, is_distractor):
        shape_id, base_id, marker_id, marker_color_id = signature
        base = self.base_palette[base_id].copy()
        base += self.rng.normal(0.0, 5.0, size=3)
        marker = self.marker_palette[marker_color_id].copy()
        marker += self.rng.normal(0.0, 3.0, size=3)
        return Landmark(
            x=float(x),
            y=float(y),
            physical_size_m=float(size),
            shape=self.shapes[shape_id],
            base_color=np.clip(base, 0, 255).astype(np.float32),
            marker_type=self.marker_types[marker_id],
            marker_color=np.clip(marker, 0, 255).astype(np.float32),
            signature=signature,
            is_distractor=bool(is_distractor),
        )

    def _sample_goal_by_distance(self, min_distance_m, max_distance_m):
        min_distance_m = float(min_distance_m)
        max_distance_m = float(max_distance_m)
        for _ in range(1000):
            distance = float(self.rng.uniform(min_distance_m, max_distance_m))
            bearing = float(self.rng.uniform(-np.pi, np.pi))
            goal = self.start + distance * np.asarray(
                [math.cos(bearing), math.sin(bearing)], dtype=np.float32
            )
            if (
                self.margin <= goal[0] <= self.world_size - self.margin
                and self.margin <= goal[1] <= self.world_size - self.margin
            ):
                return goal.astype(np.float32)
        raise RuntimeError("Unable to sample a valid goal in the requested distance range")

    def reset(self, seed=None, goal_distance_range_m=None, render_observation=True):
        self.rng = np.random.default_rng(seed)
        self.x = float(self.rng.uniform(self.margin, self.world_size - self.margin))
        self.y = float(self.rng.uniform(self.margin, self.world_size - self.margin))
        self.start = self.position.copy()

        if goal_distance_range_m is None:
            distance_min = self.min_start_goal_distance
            distance_max = self.max_start_goal_distance
        else:
            distance_min, distance_max = map(float, goal_distance_range_m)

        self.goal = self._sample_goal_by_distance(distance_min, distance_max)

        goal_sig = self._sample_signature()
        self.goal_landmark = self._landmark_from_signature(
            self.goal[0],
            self.goal[1],
            self.rng.uniform(self.cfg.goal_size_min_m, self.cfg.goal_size_max_m),
            goal_sig,
            is_distractor=False,
        )
        self._goal_reference_rgb = self._render_goal_reference_rgb()

        goal_bearing = math.atan2(self.goal[1] - self.y, self.goal[0] - self.x)
        yaw_error = np.deg2rad(
            self.rng.uniform(-self.cfg.initial_yaw_error_deg, self.cfg.initial_yaw_error_deg)
        )
        self.yaw = wrap_angle(goal_bearing + yaw_error)

        # 所有普通landmark也拥有内部marker；只禁止与Goal的离散signature完全相同，
        # 因而Goal仍需由reference识别，但不会出现完全不可区分的真重复目标。
        self.landmarks = []
        forbidden = {goal_sig}
        for _ in range(self.num_landmarks):
            x = float(self.rng.uniform(self.margin * 0.5, self.world_size - self.margin * 0.5))
            y = float(self.rng.uniform(self.margin * 0.5, self.world_size - self.margin * 0.5))
            sig = self._sample_signature(forbidden=forbidden)
            self.landmarks.append(
                self._landmark_from_signature(
                    x,
                    y,
                    self.rng.uniform(
                        self.cfg.landmark_size_min_m,
                        self.cfg.landmark_size_max_m,
                    ),
                    sig,
                    is_distractor=bool(self.rng.random() < self.cfg.distractor_ratio),
                )
            )

        self.step_count = 0
        if render_observation:
            obs = self.render()
            # V8.2 does not use the start mask as a MEC input.
            # Keep the legacy helper available for backward-compatible diagnostics,
            # but do not compute it during every reset.
            self._start_reference_mask = None
            return obs

        # 预训练脚本可先无渲染地找到某个oracle状态，再只渲染目标状态以提速。
        self._start_reference_mask = None
        return None

    # ------------------------------------------------------------------
    # Navigation dynamics
    # ------------------------------------------------------------------
    def oracle_action(self):
        dx = float(self.goal[0] - self.x)
        dy = float(self.goal[1] - self.y)
        distance = math.hypot(dx, dy)
        if distance < self.stop_radius:
            return STOP

        target_angle = math.atan2(dy, dx)
        heading_error = wrap_angle(target_angle - self.yaw)
        threshold = np.deg2rad(7.5)
        if heading_error > threshold:
            return LEFT
        if heading_error < -threshold:
            return RIGHT
        return FORWARD

    def expert_action(self):
        return self.oracle_action()

    def step(self, action, render_observation=True):
        if action == FORWARD:
            self.x += self.forward_step * math.cos(self.yaw)
            self.y += self.forward_step * math.sin(self.yaw)
        elif action == LEFT:
            self.yaw += self.turn_step
        elif action == RIGHT:
            self.yaw -= self.turn_step
        elif action == STOP:
            pass
        else:
            raise ValueError(f"Unknown action: {action}")

        self.x = float(np.clip(self.x, 0.0, self.world_size))
        self.y = float(np.clip(self.y, 0.0, self.world_size))
        self.yaw = wrap_angle(self.yaw)
        self.step_count += 1

        distance = self.distance_to_goal
        success = bool(action == STOP and distance < self.stop_radius)
        done = bool(success or action == STOP or self.step_count >= self.max_steps)
        obs = self.render() if render_observation else None
        return obs, success, done

    # ------------------------------------------------------------------
    # 3D-like visual projection
    # ------------------------------------------------------------------
    def _project(self, position, physical_size_m):
        dx = float(position[0] - self.x)
        dy = float(position[1] - self.y)
        horizontal_distance = max(math.hypot(dx, dy), 1e-6)
        relative_yaw = wrap_angle(math.atan2(dy, dx) - self.yaw)
        if abs(relative_yaw) > self.hfov_rad / 2.0:
            return None

        object_height = float(physical_size_m)
        object_center_z = 0.5 * object_height
        dz = object_center_z - self.uav_altitude_m
        elevation = math.atan2(dz, horizontal_distance)
        relative_pitch = elevation - self.camera_pitch_rad

        slant_distance = max(math.sqrt(horizontal_distance ** 2 + dz ** 2), 1e-6)
        angular_half_height = math.atan2(max(object_height * 0.5, 0.5), slant_distance)
        if abs(relative_pitch) > self.vfov_rad / 2.0 + angular_half_height:
            return None

        width = self.img_size
        height = self.img_size
        focal_x = (width / 2.0) / math.tan(self.hfov_rad / 2.0)
        focal_y = (height / 2.0) / math.tan(self.vfov_rad / 2.0)

        px = width * 0.5 - focal_x * math.tan(relative_yaw)
        py = height * 0.5 - focal_y * math.tan(relative_pitch)

        half_size_px = 0.5 * min(focal_x, focal_y) * object_height / slant_distance
        half_size_px = float(np.clip(half_size_px, 1.0, 40.0))

        if px < -half_size_px or px > width - 1 + half_size_px:
            return None
        if py < -half_size_px or py > height - 1 + half_size_px:
            return None

        return float(px), float(py), float(slant_distance), half_size_px

    def _haze_mix(self, color, distance, background_value):
        denom = max(self.cfg.haze_end_m - self.cfg.haze_start_m, 1.0)
        haze = np.clip((distance - self.cfg.haze_start_m) / denom, 0.0, 0.72)
        background = np.asarray([background_value] * 3, dtype=np.float32)
        return np.clip((1.0 - haze) * color + haze * background, 0, 255)

    @staticmethod
    def _bbox(cx, cy, half_size):
        return [cx - half_size, cy - half_size, cx + half_size, cy + half_size]

    def _draw_shape(self, draw, shape, cx, cy, half_size, color):
        c = tuple(int(v) for v in np.clip(color, 0, 255))
        hs = max(float(half_size), 1.0)
        if shape == "circle":
            draw.ellipse(self._bbox(cx, cy, hs), fill=c)
        elif shape == "triangle":
            draw.polygon(
                [(cx, cy - hs), (cx - hs, cy + hs), (cx + hs, cy + hs)],
                fill=c,
            )
        elif shape == "diamond":
            draw.polygon(
                [(cx, cy - hs), (cx - hs, cy), (cx, cy + hs), (cx + hs, cy)],
                fill=c,
            )
        elif shape == "cross":
            w = max(hs * 0.35, 1.0)
            draw.rectangle([cx - w, cy - hs, cx + w, cy + hs], fill=c)
            draw.rectangle([cx - hs, cy - w, cx + hs, cy + w], fill=c)
        elif shape == "bar":
            draw.rectangle([cx - hs, cy - hs * 0.45, cx + hs, cy + hs * 0.45], fill=c)
        else:
            draw.rectangle(self._bbox(cx, cy, hs), fill=c)

    def _draw_marker(self, draw, marker_type, cx, cy, half_size, color, base_color):
        """所有landmark都可拥有marker，因此marker是可匹配外观，而非Goal-only泄漏。"""
        hs = max(float(half_size), 1.0)
        c = tuple(int(v) for v in np.clip(color, 0, 255))
        bc = tuple(int(v) for v in np.clip(base_color, 0, 255))
        r = max(1.0, hs * 0.24)
        width = max(1, int(round(hs * 0.14)))

        if marker_type == "dot":
            draw.ellipse(self._bbox(cx, cy, r), fill=c)
        elif marker_type == "ring":
            draw.ellipse(self._bbox(cx, cy, r * 1.35), fill=c)
            draw.ellipse(self._bbox(cx, cy, r * 0.62), fill=bc)
        elif marker_type == "hbar":
            draw.rectangle([cx - r * 1.5, cy - width, cx + r * 1.5, cy + width], fill=c)
        elif marker_type == "vbar":
            draw.rectangle([cx - width, cy - r * 1.5, cx + width, cy + r * 1.5], fill=c)
        elif marker_type == "plus":
            draw.rectangle([cx - r * 1.4, cy - width, cx + r * 1.4, cy + width], fill=c)
            draw.rectangle([cx - width, cy - r * 1.4, cx + width, cy + r * 1.4], fill=c)
        elif marker_type == "x":
            draw.line([cx - r, cy - r, cx + r, cy + r], fill=c, width=width)
            draw.line([cx - r, cy + r, cx + r, cy - r], fill=c, width=width)
        elif marker_type == "double_dot":
            rr = max(1.0, r * 0.55)
            draw.ellipse(self._bbox(cx - r * 0.7, cy, rr), fill=c)
            draw.ellipse(self._bbox(cx + r * 0.7, cy, rr), fill=c)
        elif marker_type == "frame":
            rr = r * 1.35
            for k in range(width):
                draw.rectangle([cx - rr - k, cy - rr - k, cx + rr + k, cy + rr + k], outline=c)
        else:
            draw.ellipse(self._bbox(cx, cy, r), fill=c)

    def _draw_landmark(self, draw, obj, px, py, half_size, distance, background_value):
        base = obj.base_color + self.rng.normal(0.0, 4.0, size=3)
        marker = obj.marker_color + self.rng.normal(0.0, 2.0, size=3)
        base = self._haze_mix(base, distance, background_value)
        marker = self._haze_mix(marker, distance, background_value)
        self._draw_shape(draw, obj.shape, px, py, half_size, base)
        self._draw_marker(draw, obj.marker_type, px, py, half_size, marker, base)

    def render(self):
        H = W = self.img_size
        base_background = float(self.rng.uniform(24.0, 42.0))

        yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
        xx = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]
        background = base_background + 8.0 * (1.0 - yy) + 3.0 * np.sin(2.0 * np.pi * xx)
        background = np.repeat(background[:, :, None], 3, axis=2)
        image = Image.fromarray(np.clip(background, 0, 255).astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(image)

        visible = []
        for lm in self.landmarks:
            proj = self._project(lm.position, lm.physical_size_m)
            if proj is not None:
                visible.append((proj[2], lm, proj))

        goal_proj = self._project(self.goal_landmark.position, self.goal_landmark.physical_size_m)
        if goal_proj is not None:
            visible.append((goal_proj[2], self.goal_landmark, goal_proj))

        # 远处先画，近处后画。
        visible.sort(key=lambda item: item[0], reverse=True)
        for distance, obj, proj in visible:
            px, py, _, half_size = proj
            self._draw_landmark(draw, obj, px, py, half_size, distance, base_background)

        if self.rng.random() < 0.30:
            image = image.filter(ImageFilter.GaussianBlur(radius=float(self.rng.uniform(0.0, 0.45))))

        arr = np.asarray(image).astype(np.float32)
        arr *= float(self.rng.uniform(self.cfg.brightness_min, self.cfg.brightness_max))
        arr += self.rng.normal(0.0, self.cfg.sensor_noise_std, size=arr.shape)
        return np.clip(arr, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # V8 depth sensor
    # ------------------------------------------------------------------
    @staticmethod
    def _shape_mask_local(shape, xx, yy):
        """
        在归一化局部坐标[-1,1]内生成地标几何轮廓。

        深度传感不绘制颜色/marker，因此不会告诉模型“哪个是Goal”；
        它只模拟普通RGB-D相机/深度传感器对可见实体表面的距离观测。
        """
        ax = np.abs(xx)
        ay = np.abs(yy)
        if shape == "circle":
            return xx * xx + yy * yy <= 1.0
        if shape == "triangle":
            # 顶点(0,-1)、左下(-1,1)、右下(1,1)
            return (yy >= -1.0) & (yy <= 1.0) & (ax <= 0.5 * (yy + 1.0))
        if shape == "diamond":
            return ax + ay <= 1.0
        if shape == "cross":
            return (ax <= 0.35) | (ay <= 0.35)
        if shape == "bar":
            return (ax <= 1.0) & (ay <= 0.45)
        return (ax <= 1.0) & (ay <= 1.0)

    def render_depth(self, max_depth_m=None):
        """
        返回[H,W] float32深度图，单位米；0表示无有效深度回波。

        重要：
        - Goal与普通landmark使用完全相同的z-buffer规则；
        - 深度值是相机到可见实体中心/表面的近似slant range；
        - 不包含Goal ID、世界坐标、horizontal goal distance或oracle mask；
        - 因此可作为类似AerialVLN RGB-D中的普通Depth输入，而不是测试时作弊。
        """
        if max_depth_m is None:
            max_depth_m = float(AUX.depth_max_m)
        max_depth_m = float(max_depth_m)

        H = W = self.img_size
        depth = np.zeros((H, W), dtype=np.float32)
        zbuf = np.full((H, W), np.inf, dtype=np.float32)

        objects = list(self.landmarks)
        if self.goal_landmark is not None:
            objects.append(self.goal_landmark)

        for obj in objects:
            proj = self._project(obj.position, obj.physical_size_m)
            if proj is None:
                continue
            px, py, distance, half_size = proj
            if distance > max_depth_m:
                continue

            hs = max(float(half_size), 1.0)
            x0 = max(0, int(math.floor(px - hs - 1)))
            x1 = min(W - 1, int(math.ceil(px + hs + 1)))
            y0 = max(0, int(math.floor(py - hs - 1)))
            y1 = min(H - 1, int(math.ceil(py + hs + 1)))
            if x1 < x0 or y1 < y0:
                continue

            xs = np.arange(x0, x1 + 1, dtype=np.float32)
            ys = np.arange(y0, y1 + 1, dtype=np.float32)
            xx, yy = np.meshgrid(xs, ys)
            local_x = (xx - float(px)) / hs
            local_y = (yy - float(py)) / hs
            mask = self._shape_mask_local(obj.shape, local_x, local_y)
            if not np.any(mask):
                continue

            region_z = zbuf[y0:y1 + 1, x0:x1 + 1]
            update = mask & (float(distance) < region_z)
            if np.any(update):
                region_z[update] = float(distance)
                region_depth = depth[y0:y1 + 1, x0:x1 + 1]
                region_depth[update] = float(distance)

        return depth

    # ------------------------------------------------------------------
    # Goal reference / diagnostic helpers
    # ------------------------------------------------------------------
    def _render_goal_reference_canvas(self, canvas_size, object_width_px):
        background = 34
        image = Image.new("RGB", (canvas_size, canvas_size), color=(background,) * 3)
        draw = ImageDraw.Draw(image)
        cx = cy = (canvas_size - 1) / 2.0
        hs = max(float(object_width_px) / 2.0, 1.0)
        self._draw_shape(
            draw,
            self.goal_landmark.shape,
            cx,
            cy,
            hs,
            self.goal_landmark.base_color,
        )
        self._draw_marker(
            draw,
            self.goal_landmark.marker_type,
            cx,
            cy,
            hs,
            self.goal_landmark.marker_color,
            self.goal_landmark.base_color,
        )
        return np.asarray(image, dtype=np.uint8)

    def _render_goal_reference_rgb(self):
        size = int(AUX.goal_reference_size)
        return self._render_goal_reference_canvas(size, size * 0.54)

    def _render_goal_reference_scale(self, object_width_px):
        return self._render_goal_reference_canvas(self.img_size, object_width_px)

    def get_goal_reference_scale_rgb(self, object_width_px):
        return self._render_goal_reference_scale(object_width_px).copy()

    def get_goal_reference_rgb(self):
        if self._goal_reference_rgb is None:
            raise RuntimeError("Call reset() before requesting the goal reference image")
        return self._goal_reference_rgb.copy()

    def get_goal_reference_multiscale_rgbs(self):
        return np.stack(
            [self._render_goal_reference_scale(px) for px in AUX.goal_reference_object_pixels],
            axis=0,
        ).astype(np.uint8)

    def binary_mask_from_rgb(self, rgb_image, output_size=None):
        size = int(AUX.binary_size if output_size is None else output_size)
        arr = np.asarray(rgb_image, dtype=np.float32)
        channel_spread = arr.max(axis=2) - arr.min(axis=2)
        luminance = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        background_level = float(np.median(luminance))
        mask = (
            (channel_spread >= AUX.binary_channel_spread_threshold)
            | (luminance >= background_level + AUX.binary_luma_margin)
        )
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        mask_image = mask_image.filter(ImageFilter.MaxFilter(size=3))
        mask_image = mask_image.resize((size, size), resample=Image.Resampling.BOX)
        return (np.asarray(mask_image, dtype=np.uint8) >= 8).astype(np.uint8)

    def get_start_reference_mask(self):
        if self._start_reference_mask is None:
            # 如果reset(render_observation=False)，这里按当前初始状态补渲染一次。
            obs = self.render()
            self._start_reference_mask = self.binary_mask_from_rgb(obs)
        return self._start_reference_mask.copy()

    def get_goal_reference_mask(self, output_size=None):
        return self.binary_mask_from_rgb(self.get_goal_reference_rgb(), output_size=output_size)

    def goal_projection(self):
        """仅用于诊断；不作为学习策略输入。"""
        if self.goal_landmark is None:
            return None
        return self._project(self.goal_landmark.position, self.goal_landmark.physical_size_m)

    def goal_token_soft_mask(self, grid_size=ENCODER.grid_size):
        """
        simulator监督用的Goal patch soft mask，[grid_size^2]。
        只用于shared encoder预训练/诊断，不写入MEC token dataset，也不用于在线推理。
        """
        proj = self.goal_projection()
        if proj is None:
            return np.zeros((grid_size * grid_size,), dtype=np.float32)

        px, py, _, half_size = proj
        cell = self.img_size / float(grid_size)
        xs = (np.arange(grid_size, dtype=np.float32) + 0.5) * cell
        ys = (np.arange(grid_size, dtype=np.float32) + 0.5) * cell
        xx, yy = np.meshgrid(xs, ys)

        # 小目标至少覆盖约一个patch的高斯支持；近距离目标按真实投影尺寸扩大。
        sigma = max(float(half_size) * 0.75, cell * 0.55)
        w = np.exp(-0.5 * (((xx - px) / sigma) ** 2 + ((yy - py) / sigma) ** 2))
        w[w < 0.03] = 0.0
        if w.sum() <= 1e-8:
            ix = int(np.clip(px / cell, 0, grid_size - 1))
            iy = int(np.clip(py / cell, 0, grid_size - 1))
            w[iy, ix] = 1.0
        return w.astype(np.float32).reshape(-1)

    @property
    def position(self):
        return np.asarray([self.x, self.y], dtype=np.float32)

    @property
    def pose(self):
        return np.asarray([self.x, self.y, self.yaw], dtype=np.float32)

    @property
    def distance_to_goal(self):
        return float(np.linalg.norm(self.position - self.goal))
