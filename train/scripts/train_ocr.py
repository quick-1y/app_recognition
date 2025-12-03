"""CLI wrapper for the OCR training pipeline."""
from pathlib import Path
import argparse

from train.ocr_trainer import OCRTrainingConfig, train_ocr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OCR model for ANPR")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("train/train_config/ocr_config.yaml"),
        help="Path to the OCR training configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OCRTrainingConfig.from_yaml(args.config)
    train_ocr(config)


if __name__ == "__main__":
    main()
