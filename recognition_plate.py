"""OCR pipeline for number plate recognition with EasyOCR or CRNN."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol

import cv2
import easyocr
import numpy as np
import torch
import torch.nn as nn
import yaml

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class PlateCandidate:
    text: str
    confidence: float
    region: str = ""


class _Recognizer(Protocol):
    def recognize(self, img) -> Optional[PlateCandidate]:
        ...


def _load_patterns(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
        return config.get("plate_patterns", [])


class _PatternValidator:
    def __init__(self, patterns_path: Path):
        self.patterns = _load_patterns(patterns_path)

    def filter_by_pattern(self, text: str) -> Optional[PlateCandidate]:
        cleaned_text = re.sub(r"[^A-Z0-9]", "", text.upper())
        for pattern in self.patterns:
            regex = re.compile(pattern.get("pattern", ""))
            if regex.match(cleaned_text):
                return PlateCandidate(text=cleaned_text, confidence=1.0, region=pattern.get("region", ""))
        return None


class EasyOCRPlateRecognizer(_PatternValidator):
    """High level wrapper around EasyOCR with additional preprocessing."""

    def __init__(self, settings: Settings):
        super().__init__(settings.app.plate_patterns_path)
        self.settings = settings
        self.reader = easyocr.Reader(self.settings.ocr.languages, gpu=self.settings.ocr.gpu)

    def preprocess_image(self, img) -> np.ndarray:
        """Aggressive preprocessing to stabilise OCR results."""
        resize_factor = max(self.settings.ocr.resize_factor, 1.0)
        img = cv2.resize(img, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, sharpen_kernel)
        gray = cv2.bilateralFilter(gray, self.settings.ocr.denoise_diameter, 75, 75)

        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            max(self.settings.ocr.threshold_block_size, 3),
            self.settings.ocr.threshold_c,
        )
        return thresh

    def recognize(self, img) -> Optional[PlateCandidate]:
        processed = self.preprocess_image(img)
        results = self.reader.readtext(
            processed,
            detail=1,
            allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            width_ths=0.6,
            ycenter_ths=0.5,
            height_ths=0.2,
        )

        candidates: List[PlateCandidate] = []
        for _, text, conf in results:
            if conf < self.settings.ocr.min_confidence:
                continue
            cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
            if not cleaned:
                continue
            candidates.append(PlateCandidate(text=cleaned, confidence=float(conf)))

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        candidates = candidates[: self.settings.ocr.max_candidates]

        for candidate in candidates:
            matched = self.filter_by_pattern(candidate.text)
            if matched:
                matched.confidence = candidate.confidence
                return matched

        return candidates[0] if candidates else None


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


class CRNNPlateRecognizer(_PatternValidator):
    """CRNN-based OCR recognizer that mirrors the reference project."""

    def __init__(self, settings: Settings):
        super().__init__(settings.app.plate_patterns_path)
        self.settings = settings
        self.alphabet = settings.ocr.alphabet
        num_classes = len(self.alphabet) + 1  # CTC blank
        self.device = torch.device("cuda" if settings.ocr.gpu and torch.cuda.is_available() else "cpu")

        self.model = CRNN(settings.ocr.crnn_img_height, num_classes).to(self.device)
        if not settings.ocr.crnn_weights.exists():
            raise FileNotFoundError(
                f"CRNN weights not found at {settings.ocr.crnn_weights}. Provide trained weights to enable CRNN OCR."
            )

        state_dict = torch.load(settings.ocr.crnn_weights, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def _preprocess(self, img) -> torch.Tensor:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(
            gray,
            (self.settings.ocr.crnn_img_width, self.settings.ocr.crnn_img_height),
            interpolation=cv2.INTER_CUBIC,
        )
        normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return tensor.to(self.device)

    def _decode(self, logits: torch.Tensor) -> tuple[str, float]:
        # logits: (T, 1, C)
        probs = torch.softmax(logits, dim=2)
        best_path = probs.argmax(2).squeeze(1)

        blank_idx = len(self.alphabet)
        collapsed: List[int] = []
        last_char = None
        for idx in best_path.tolist():
            if idx == blank_idx:
                last_char = None
                continue
            if idx != last_char:
                collapsed.append(idx)
                last_char = idx

        text = "".join(self.alphabet[i] for i in collapsed if i < len(self.alphabet))
        confidence = float(probs.max(2)[0].mean().item()) if text else 0.0
        return text, confidence

    def recognize(self, img) -> Optional[PlateCandidate]:
        with torch.no_grad():
            tensor = self._preprocess(img)
            logits = self.model(tensor)  # (T, B, C)

        text, confidence = self._decode(logits)
        if not text or confidence < self.settings.ocr.min_confidence:
            return None

        matched = self.filter_by_pattern(text)
        if matched:
            matched.confidence = confidence
            return matched
        return PlateCandidate(text=text, confidence=confidence)


class PlateRecognizer:
    """Facade that instantiates OCR backend specified in config."""

    def __init__(self, settings: Settings):
        backend = settings.ocr.backend.lower()
        if backend == "crnn":
            logger.info("Using CRNN OCR backend")
            self._impl: _Recognizer = CRNNPlateRecognizer(settings)
        else:
            logger.info("Using EasyOCR backend")
            self._impl = EasyOCRPlateRecognizer(settings)

    def recognize(self, img) -> Optional[PlateCandidate]:
        return self._impl.recognize(img)

