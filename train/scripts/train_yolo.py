"""CLI entry point for training the YOLO detector."""
from pathlib import Path
import argparse

from train.yolo_trainer import YOLOTrainer, YOLOTrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO model for ANPR")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("train/train_config/yolo_train_config.yaml"),
        help="Path to the training configuration YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = YOLOTrainingConfig.from_yaml(args.config)
    trainer = YOLOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
