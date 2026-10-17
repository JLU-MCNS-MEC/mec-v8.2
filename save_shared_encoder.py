"""
V7兼容脚本。
train_shared_encoder.py / train_navigation.py已经直接把最佳encoder保存为shared_encoder.pt，
因此不再需要从navigation_model.pt中二次拆出encoder。
"""

from pathlib import Path


if __name__ == "__main__":
    path = Path("shared_encoder.pt")
    if not path.exists():
        raise FileNotFoundError(
            "shared_encoder.pt不存在。请先运行 python train_shared_encoder.py"
        )
    print("shared_encoder.pt already exists; no extraction is needed in V7.")
