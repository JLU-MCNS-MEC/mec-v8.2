"""兼容旧入口：V8复用V7的shared encoder训练已由train_shared_encoder.py接管。"""

from train_shared_encoder import main


if __name__ == "__main__":
    main()
