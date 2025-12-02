from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PyQt5.QtGui import QImage
from ultralytics import YOLO

from app.config import Settings
from app.database import PlateDatabase
from recognition_plate import PlateRecognizer, PlateCandidate

logger = logging.getLogger(__name__)


@dataclass
class FrameCallbacks:
    frame: Callable[[QImage], None]
    text: Callable[[str], None]


class PlatePipeline:
    def __init__(self, settings: Settings, database: PlateDatabase):
        self.settings = settings
        self.database = database
        self.model = YOLO(Path(self.settings.model.detector_weights))
        self.plate_recognizer = PlateRecognizer(settings)
        self.last_recognized_plate: Optional[str] = None

    def _to_qimage(self, frame: np.ndarray) -> QImage:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        return QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

    def _annotate(self, frame: np.ndarray, box, class_name: str, color, label: str) -> None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _process_plate(self, frame: np.ndarray, bbox, source: str) -> Optional[PlateCandidate]:
        x1, y1, x2, y2 = bbox
        plate_region = frame[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)]
        if plate_region.size == 0:
            return None

        candidate = self.plate_recognizer.recognize(plate_region)
        if candidate and candidate.text:
            label = f"{candidate.text} {candidate.region}".strip()
            if label != self.last_recognized_plate:
                self.database.add_plate(candidate.text, candidate.region, candidate.confidence, source)
                self.last_recognized_plate = label
            return candidate
        return None

    def process_video(self, video_path: str, callbacks: FrameCallbacks) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Could not open video file %s", video_path)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_delay = 1.0 / fps

        class_colors = {"licence": (255, 255, 255)}

        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model.track(
                frame_rgb,
                persist=True,
                conf=self.settings.model.confidence_threshold,
                iou=self.settings.model.iou_threshold,
                verbose=False,
            )

            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    color = class_colors.get(class_name, (0, 255, 0))

                    label = None
                    if class_name == "licence":
                        candidate = self._process_plate(frame, (x1, y1, x2, y2), source=video_path)
                        if candidate:
                            label = f"{candidate.text} {candidate.region}".strip()
                            callbacks.text(label)

                    if self.settings.processing.draw_tracks:
                        self._annotate(frame, (x1, y1, x2, y2), class_name, color, label or class_name)

            callbacks.frame(self._to_qimage(frame))

            elapsed_time = time.time() - start_time
            if elapsed_time < frame_delay:
                time.sleep(frame_delay - elapsed_time)

        cap.release()
