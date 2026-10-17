from config import TRAIN
from token_data import generate_token_dataset

if __name__ == "__main__":
    generate_token_dataset(
        output_dir="token_dataset_test",
        target_episodes=TRAIN.token_test_episodes,
        seed_start=50000,
        include_hard_states=False,
    )
