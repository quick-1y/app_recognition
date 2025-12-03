"""
Simple OCR training pipeline inspired by the reference ANPR project.

The script defines a compact CRNN model and expects a dataset layout with a
``labels.csv`` file describing image paths and their transcription. Images are
resized, normalized to ``img_height``/``img_width`` from the config and training
uses CTC loss for sequence alignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Tokenizer:
    """Maps between characters and integer indices for CTC training."""

    def __init__(self, vocab: str) -> None:
        self.vocab = vocab
        self.char_to_idx: Dict[str, int] = {ch: i + 1 for i, ch in enumerate(vocab)}
        self.idx_to_char: Dict[int, str] = {i + 1: ch for i, ch in enumerate(vocab)}

    @property
    def blank(self) -> int:
        return 0

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + 1  # +1 for the CTC blank symbol

    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]

    def decode(self, labels: List[int]) -> str:
        return "".join(self.idx_to_char[idx] for idx in labels if idx in self.idx_to_char)


class OCRDataset(Dataset):
    """Dataset expecting a labels.csv with two columns: image_path,label."""

    def __init__(
        self,
        root: Path,
        labels_file: Path,
        img_size: Tuple[int, int],
        max_label_length: int,
    ) -> None:
        super().__init__()
        self.root = root
        self.img_size = img_size
        self.max_label_length = max_label_length
        with labels_file.open("r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            self.samples = [(row[0], row[1]) for row in reader if len(row) >= 2]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        rel_path, label = self.samples[idx]
        label = label[: self.max_label_length]
        img_path = self.root / rel_path
        with Image.open(img_path) as img:
            img = img.convert("L").resize(self.img_size)
        img_np = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(img_np).unsqueeze(0)  # C x H x W
        return tensor, label


class CRNN(nn.Module):
    """Compact convolutional recurrent network for OCR."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.gru = nn.GRU(256 * 4, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B x C x H x W
        features = self.features(x)
        b, c, h, w = features.size()
        features = features.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        recurrent, _ = self.gru(features)
        logits = self.classifier(recurrent)
        return logits  # B x T x num_classes


def collate_fn(
    batch: List[Tuple[torch.Tensor, str]], tokenizer: Tokenizer
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images, texts = zip(*batch)
    image_batch = torch.stack(images)
    encoded = [torch.tensor(tokenizer.encode(text), dtype=torch.long) for text in texts]
    lengths = torch.tensor([t.numel() for t in encoded], dtype=torch.long)
    targets = torch.cat(encoded)
    input_lengths = torch.full(size=(len(batch),), fill_value=image_batch.shape[-1] // 4, dtype=torch.long)
    return image_batch, targets, input_lengths, lengths


@dataclass
class OCRTrainingConfig:
    dataset_root: Path
    labels_file: Path
    vocab: str
    img_height: int
    img_width: int
    max_label_length: int
    batch_size: int
    num_workers: int
    epochs: int
    learning_rate: float
    device: str
    checkpoint_dir: Path
    checkpoint_interval: int = 1
    log_interval: int = 10
    resume_from: str = ""

    @classmethod
    def from_yaml(cls, path: Path) -> "OCRTrainingConfig":
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        raw["dataset_root"] = Path(raw["dataset_root"])
        raw["labels_file"] = Path(raw["labels_file"])
        raw["checkpoint_dir"] = Path(raw["checkpoint_dir"])
        return cls(**raw)


def train_ocr(config: OCRTrainingConfig) -> None:
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    seed_everything()
    tokenizer = Tokenizer(config.vocab)

    dataset = OCRDataset(
        root=config.dataset_root,
        labels_file=config.labels_file,
        img_size=(config.img_width, config.img_height),
        max_label_length=config.max_label_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )

    model = CRNN(num_classes=tokenizer.vocab_size).to(device)
    criterion = nn.CTCLoss(blank=tokenizer.blank, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    start_epoch = 0
    if config.resume_from:
        ckpt = torch.load(config.resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed training from epoch {start_epoch}")

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, config.epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, targets, input_lengths, target_lengths) in enumerate(dataloader, start=1):
            images = images.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            logits = model(images)  # B x T x C
            logits = logits.log_softmax(2).permute(1, 0, 2)  # T x B x C
            loss = criterion(logits, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % config.log_interval == 0:
                avg_loss = running_loss / config.log_interval
                print(f"Epoch {epoch+1}/{config.epochs} | Step {batch_idx} | Loss: {avg_loss:.4f}")
                running_loss = 0.0

        if (epoch + 1) % config.checkpoint_interval == 0:
            ckpt_path = config.checkpoint_dir / f"ocr_epoch_{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OCR model for ANPR")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("train/train_config/ocr_config.yaml"),
        help="Path to the OCR training YAML file.",
    )
    args = parser.parse_args()
    cfg = OCRTrainingConfig.from_yaml(args.config)
    train_ocr(cfg)
