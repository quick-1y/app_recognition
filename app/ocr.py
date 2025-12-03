"""CRNN-based OCR pipeline for licence plate recognition."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class PlateCandidate:
    """Represents recognized plate text with optional region metadata."""

    text: str
    confidence: float
    region: str = ""


class PlateRecognizer(Protocol):
    def recognize(self, img: np.ndarray) -> Optional[PlateCandidate]:
        """Recognize plate text from an image snippet."""


def _load_patterns(path: Path) -> List[dict]:
    if not path.exists():
        logger.warning("Pattern file %s not found. Proceeding without region patterns.", path)
        return []

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
        return config.get("plate_patterns", [])


class PatternValidator:
    """Validates recognized text against configured plate patterns."""

    def __init__(self, patterns_path: Path):
        self.patterns = _load_patterns(patterns_path)

    def match(self, text: str) -> Optional[PlateCandidate]:
        cleaned_text = re.sub(r"[^A-Z0-9]", "", text.upper())
        for pattern in self.patterns:
            regex = re.compile(pattern.get("pattern", ""))
            if regex.match(cleaned_text):
                return PlateCandidate(
                    text=cleaned_text,
                    confidence=1.0,
                    region=pattern.get("region", ""),
                )
        return None


class CRNN(nn.Module):
    """Standard CRNN architecture for scene text recognition."""

    def __init__(self, img_height: int, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),

            nn.Conv2d(512, 512, 2, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        conv_out_h = img_height // 4
        self.rnn = nn.Sequential(
            nn.LSTM(512 * conv_out_h, 256, bidirectional=True, num_layers=2, batch_first=False),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):  # (B, 1, H, W)
        features = self.cnn(x)
        b, c, h, w = features.size()
        features = features.permute(3, 0, 1, 2).contiguous().view(w, b, c * h)
        recurrent, _ = self.rnn[0](features)
        output = self.rnn[1](recurrent)
        return output  # (T, B, num_classes)


class CRNNPreprocessor:
    """Handles preprocessing steps before sending image to the CRNN."""

    def __init__(self, settings: Settings, device: torch.device):
        self.settings = settings
        self.device = device

    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(
            gray,
            (self.settings.ocr.crnn_img_width, self.settings.ocr.crnn_img_height),
            interpolation=cv2.INTER_CUBIC,
        )
        normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return tensor.to(self.device)


class CRNNDecoder:
    """Decodes raw network logits into human-readable text and confidence."""

    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        self.blank_idx = len(alphabet)

    def decode(self, logits: torch.Tensor) -> tuple[str, float]:
        probs = torch.softmax(logits, dim=2)
        best_path = probs.argmax(2).squeeze(1)

        collapsed: List[int] = []
        last_char = None
        for idx in best_path.tolist():
            if idx == self.blank_idx:
                last_char = None
                continue
            if idx != last_char:
                collapsed.append(idx)
                last_char = idx

        text = "".join(self.alphabet[i] for i in collapsed if i < len(self.alphabet))
        confidence = float(probs.max(2)[0].mean().item()) if text else 0.0
        return text, confidence


class CRNNPredictor:
    """Encapsulates model loading and inference to keep PlateRecognizer lean."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = torch.device("cuda" if settings.ocr.gpu and torch.cuda.is_available() else "cpu")
        self.decoder = CRNNDecoder(settings.ocr.alphabet)
        self.preprocessor = CRNNPreprocessor(settings, self.device)

        num_classes = len(settings.ocr.alphabet) + 1  # CTC blank
        self.model = CRNN(settings.ocr.crnn_img_height, num_classes).to(self.device)
        if not settings.ocr.crnn_weights.exists():
            raise FileNotFoundError(
                f"CRNN weights not found at {settings.ocr.crnn_weights}. Provide trained weights to enable OCR."
            )

        state_dict = torch.load(settings.ocr.crnn_weights, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def predict(self, img: np.ndarray) -> tuple[str, float]:
        with torch.no_grad():
            tensor = self.preprocessor.preprocess(img)
            logits = self.model(tensor)
        return self.decoder.decode(logits)


class CRNNPlateRecognizer:
    """High-level OCR service that combines validation and CRNN prediction."""

    def __init__(self, settings: Settings, validator: PatternValidator | None = None):
        self.settings = settings
        self.validator = validator or PatternValidator(settings.app.plate_patterns_path)
        self.predictor = CRNNPredictor(settings)

    def recognize(self, img: np.ndarray) -> Optional[PlateCandidate]:
        text, confidence = self.predictor.predict(img)
        if not text or confidence < self.settings.ocr.min_confidence:
            return None

        matched = self.validator.match(text)
        if matched:
            matched.confidence = confidence
            return matched

        return PlateCandidate(text=text, confidence=confidence)


class PlateRecognitionService:
    """Facade for OCR to simplify usage in pipelines and GUIs."""

    def __init__(self, settings: Settings):
        self._recognizer: PlateRecognizer = CRNNPlateRecognizer(settings)

    def recognize(self, img: np.ndarray) -> Optional[PlateCandidate]:
        return self._recognizer.recognize(img)


__all__ = [
    "PlateCandidate",
    "PlateRecognizer",
    "PlateRecognitionService",
]
