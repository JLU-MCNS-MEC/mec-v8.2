import torch
import torch.nn as nn

from config import ENCODER


class VisualTokenEncoder(nn.Module):
    """
    UAV端共享视觉编码器。

    输入:
        [B,3,336,336]
    输出:
        [B,576,384]

    V7需要重新训练shared_encoder.pt，因为：
    1) 相机投影/俯角发生变化；
    2) Goal外观和目标标记规则变化；
    3) 编码器需要额外具备Goal-reference <-> Current-target匹配能力。
    """

    def __init__(
        self,
        image_size=ENCODER.image_size,
        patch_size=ENCODER.patch_size,
        token_dim=ENCODER.token_dim,
        num_layers=ENCODER.num_layers,
        num_heads=ENCODER.num_heads,
        mlp_ratio=ENCODER.mlp_ratio,
        dropout=ENCODER.dropout,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.token_dim = int(token_dim)
        self.grid_size = self.image_size // self.patch_size
        self.num_tokens = self.grid_size ** 2

        self.patch_embed = nn.Conv2d(
            3,
            token_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, token_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * int(mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(token_dim)

    def forward(self, x):
        z = self.patch_embed(x)                    # [B,D,24,24]
        z = z.flatten(2).transpose(1, 2)          # [B,576,D]
        z = z + self.pos_embed
        z = self.encoder(z)
        return self.output_norm(z)
