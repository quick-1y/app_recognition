"""
Utilities for training YOLO detection models for the ANPR system.

The trainer is configured via a YAML file that mirrors the layout used in the
reference project. The main entry point can be used directly or imported from
other scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import argparse
import yaml
from ultralytics import YOLO


@dataclass
class YOLOTrainingConfig:
    """Configuration for YOLO training."""

    model: str
    data_config: str
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = ""
    project: str = "runs/train"
    name: str = "anpr_yolo"
    workers: int = 8
    resume: bool = False
    multi_scale: bool = True
    patience: Optional[int] = None
    lr0: Optional[float] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "YOLOTrainingConfig":
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls(**raw)

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "data": self.data_config,
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "project": self.project,
            "name": self.name,
            "workers": self.workers,
            "resume": self.resume,
            "multi_scale": self.multi_scale,
        }
        if self.device:
            kwargs["device"] = self.device
        if self.patience is not None:
            kwargs["patience"] = self.patience
        if self.lr0 is not None:
            kwargs["lr0"] = self.lr0
        return kwargs


class YOLOTrainer:
    """Helper that wraps Ultralytics' training API with a typed config."""

    def __init__(self, config: YOLOTrainingConfig) -> None:
        self.config = config

    def train(self) -> Any:
        model = YOLO(self.config.model)
        train_kwargs = self.config.to_kwargs()
        print("Launching YOLO training with the following options:")
        for key, value in train_kwargs.items():
            print(f"  {key}: {value}")
        results = model.train(**train_kwargs)
        print("Training complete. Review run artifacts for metrics and weights.")
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO detector for ANPR")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("train/train_config/yolo_train_config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        help="Override the dataset configuration file path.",
    )
    parser.add_argument("--epochs", type=int, help="Override the number of epochs.")
    parser.add_argument("--batch", type=int, help="Override the batch size.")
    parser.add_argument("--imgsz", type=int, help="Override the training image size.")
    parser.add_argument("--device", type=str, help="Force a specific compute device.")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> YOLOTrainingConfig:
    config = YOLOTrainingConfig.from_yaml(args.config)
    if args.data_config:
        config.data_config = args.data_config
    if args.epochs:
        config.epochs = args.epochs
    if args.batch:
        config.batch = args.batch
    if args.imgsz:
        config.imgsz = args.imgsz
    if args.device:
        config.device = args.device
    return config


if __name__ == "__main__":
    cli_args = parse_args()
    trainer = YOLOTrainer(load_config(cli_args))
    trainer.train()
