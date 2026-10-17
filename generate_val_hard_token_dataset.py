from config import TRAIN
from token_data import generate_token_dataset


if __name__ == "__main__":
    # Separate validation trajectories and seeds; the existing natural val/test
    # directories are not modified. Hard states supervise Arrival only.
    generate_token_dataset(
        output_dir="token_dataset_val_hard",
        target_episodes=TRAIN.token_val_episodes,
        seed_start=70000,
        include_hard_states=True,
    )
