"""YOLO-based licence plate detector service."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from ultralytics import YOLO

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float


class PlateDetector:
    """Wrapper around YOLO to keep pipeline logic simple and testable."""

    def __init__(self, settings: Settings):
        self.settings = settings
        weights_path = Path(settings.model.detector_weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"Detector weights not found at {weights_path}")
        self.model = YOLO(weights_path)

    def detect(self, frame_rgb: np.ndarray) -> List[Detection]:
        results = self.model.track(
            frame_rgb,
            persist=True,
            conf=self.settings.model.confidence_threshold,
            iou=self.settings.model.iou_threshold,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(box.conf[0]),
                    )
                )
        return detections


__all__ = ["Detection", "PlateDetector"]
