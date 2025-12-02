"""OCR pipeline for number plate recognition."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import easyocr
import numpy as np
import yaml

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class PlateCandidate:
    text: str
    confidence: float
    region: str = ""


class PlateRecognizer:
    """High level wrapper around EasyOCR with additional preprocessing."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.reader = easyocr.Reader(self.settings.ocr.languages, gpu=self.settings.ocr.gpu)
        self.patterns = self._load_patterns(self.settings.app.plate_patterns_path)

    @staticmethod
    def _load_patterns(path: Path) -> List[dict]:
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
            return config.get("plate_patterns", [])

    def preprocess_image(self, img) -> np.ndarray:
        """Aggressive preprocessing to stabilise OCR results."""
        resize_factor = max(self.settings.ocr.resize_factor, 1.0)
        img = cv2.resize(img, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Contrast Limited Adaptive Histogram Equalization for uneven lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Sharpening and denoising to clarify characters
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, sharpen_kernel)
        gray = cv2.bilateralFilter(gray, self.settings.ocr.denoise_diameter, 75, 75)

        # Adaptive thresholding helps with varying illumination
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            max(self.settings.ocr.threshold_block_size, 3),
            self.settings.ocr.threshold_c,
        )
        return thresh

    def filter_by_pattern(self, text: str) -> Optional[PlateCandidate]:
        cleaned_text = re.sub(r"[^A-Z0-9]", "", text.upper())
        for pattern in self.patterns:
            regex = re.compile(pattern.get("pattern", ""))
            if regex.match(cleaned_text):
                return PlateCandidate(text=cleaned_text, confidence=1.0, region=pattern.get("region", ""))
        return None

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
