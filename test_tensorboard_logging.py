"""CPU-only TensorBoard integration smoke tests; no datasets/checkpoints/GPU."""

import csv
import tempfile
import unittest
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from csv_to_tensorboard import convert_csv_file
from tensorboard_logging import make_writer, write_epoch_metrics


def read_scalars(folder):
    acc = EventAccumulator(str(folder), size_guidance={"scalars": 0})
    acc.Reload()
    return {tag: acc.Scalars(tag) for tag in acc.Tags()["scalars"]}


class TensorBoardLoggingTest(unittest.TestCase):
    def test_live_metrics_and_run_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with make_writer("test_exp", "Small", run_id="one", log_root=root) as writer:
                write_epoch_metrics(
                    writer,
                    {
                        "epoch": 1, "model_seed": 42,
                        "train_total_loss": 1.234,
                        "train_move_loss": 0.456,
                        "val_macro_f1": 0.7,
                        "checkpoint_score": 0.6,
                        "lr": 0.0001,
                        "ignored_metric": "not-a-number",
                    },
                    1,
                )
            d1 = root / "test_exp" / "one" / "small"
            tags = read_scalars(d1)
            self.assertAlmostEqual(tags["train/total_loss"][0].value, 1.234, places=5)
            self.assertAlmostEqual(tags["train/move_loss"][0].value, 0.456, places=5)
            self.assertAlmostEqual(tags["val/macro_f1"][0].value, 0.7, places=5)
            self.assertAlmostEqual(tags["selection/checkpoint_score"][0].value, 0.6, places=5)
            self.assertIn("optimizer/lr", tags)
            self.assertNotIn("metrics/model_seed", tags)
            self.assertNotIn("metrics/ignored_metric", tags)

            with make_writer("test_exp", "Medium", run_id="two", log_root=root) as writer:
                write_epoch_metrics(writer, {"train_total_loss": 0.321}, 2)
            d2 = root / "test_exp" / "two" / "medium"
            self.assertNotEqual(d1, d2)
            self.assertAlmostEqual(
                read_scalars(d2)["train/total_loss"][0].value, 0.321, places=5
            )

    def test_historical_import_idempotent_and_non_destructive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result_dir = root / "gbs_v82_epoch_compare_results"
            result_dir.mkdir()
            csv_path = result_dir / "small_training_history.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                w = csv.DictWriter(
                    stream, fieldnames=[
                        "epoch", "train_total_loss", "val_natural_arrival_f1",
                        "val_hard_arrival_f1", "val_hard_negative_false_stop_rate"
                    ]
                )
                w.writeheader()
                w.writerow({
                    "epoch": 1, "train_total_loss": 1.5,
                    "val_natural_arrival_f1": 0.2,
                    "val_hard_arrival_f1": 0.4,
                    "val_hard_negative_false_stop_rate": 0.05
                })
            before = csv_path.read_bytes()
            log_root = root / "tb"
            events_path, created = convert_csv_file(csv_path, log_root)
            self.assertTrue(created)
            tags = read_scalars(events_path)
            self.assertAlmostEqual(tags["train/total_loss"][0].value, 1.5)
            self.assertIn("val/natural_arrival_f1", tags)
            self.assertIn("val/hard_arrival_f1", tags)
            self.assertIn("val/hard_negative_false_stop_rate", tags)
            same_path, created_again = convert_csv_file(csv_path, log_root)
            self.assertEqual(events_path, same_path)
            self.assertFalse(created_again)
            self.assertEqual(before, csv_path.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
