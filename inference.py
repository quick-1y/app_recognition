"""CLI inference pipeline using YOLO detection and CRNN OCR.

This script mirrors the end-to-end recognition flow from the reference
ANPR implementation: a YOLO detector isolates licence plates and a compact
CRNN model transcribes the cropped plates. It supports single-image and
video processing with optional result saving.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from train.ocr_trainer import CRNN, Tokenizer


@dataclass
class PlateDetection:
    bbox: Tuple[int, int, int, int]
    score: float
    class_name: str


@dataclass
class RecognitionResult:
    detection: PlateDetection
    text: str
    confidence: float


class OCRCRNNRecognizer:
    def __init__(
        self,
        weights_path: Path,
        vocab: str,
        img_height: int,
        img_width: int,
        device: str,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = Tokenizer(vocab)
        self.model = CRNN(num_classes=self.tokenizer.vocab_size)
        self._load_weights(weights_path)
        self.model.to(self.device)
        self.model.eval()
        self.img_height = img_height
        self.img_width = img_width

    def _load_weights(self, weights_path: Path) -> None:
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint.get("model_state", checkpoint)
        self.model.load_state_dict(state_dict)

    def _preprocess(self, plate_image: np.ndarray) -> torch.Tensor:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.img_width, self.img_height))
        normalized = resized.astype("float32") / 255.0
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)  # B x C x H x W
        return tensor.to(self.device)

    def _decode(self, logits: torch.Tensor) -> Tuple[str, float]:
        probs = torch.softmax(logits, dim=2)
        max_probs, indices = probs.max(2)
        indices = indices.squeeze(0).tolist()
        max_probs = max_probs.squeeze(0).tolist()

        decoded_chars: List[str] = []
        char_confs: List[float] = []
        prev_idx: Optional[int] = None
        for idx, conf in zip(indices, max_probs):
            if idx == self.tokenizer.blank:
                prev_idx = None
                continue
            if idx == prev_idx:
                continue
            decoded_chars.append(self.tokenizer.idx_to_char.get(idx, ""))
            char_confs.append(float(conf))
            prev_idx = idx

        text = "".join(decoded_chars)
        confidence = float(np.mean(char_confs)) if char_confs else 0.0
        return text, confidence

    @torch.no_grad()
    def recognize(self, plate_image: np.ndarray) -> Tuple[str, float]:
        tensor = self._preprocess(plate_image)
        logits = self.model(tensor)
        return self._decode(logits)


class ANPRInference:
    def __init__(
        self,
        detector_weights: Path,
        ocr_weights: Path,
        vocab: str,
        img_height: int,
        img_width: int,
        conf_thres: float,
        iou_thres: float,
        device: str,
    ) -> None:
        self.detector = YOLO(detector_weights)
        self.ocr = OCRCRNNRecognizer(ocr_weights, vocab, img_height, img_width, device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def _select_plate_boxes(self, results) -> List[PlateDetection]:
        detections: List[PlateDetection] = []
        for result in results:
            names = result.names
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = names[class_id] if isinstance(names, Sequence) else names.get(class_id, str(class_id))
                label = class_name.lower()
                if "plate" not in label and "licence" not in label and "license" not in label:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                detections.append(PlateDetection(bbox=(x1, y1, x2, y2), score=float(box.conf[0]), class_name=class_name))
        return detections

    def detect(self, frame_bgr: np.ndarray) -> List[PlateDetection]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.predict(
            frame_rgb,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
        )
        return self._select_plate_boxes(results)

    def recognize_from_frame(self, frame_bgr: np.ndarray) -> List[RecognitionResult]:
        detections = self.detect(frame_bgr)
        results: List[RecognitionResult] = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            plate = frame_bgr[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)]
            if plate.size == 0:
                continue
            text, confidence = self.ocr.recognize(plate)
            if text:
                results.append(RecognitionResult(detection=det, text=text, confidence=confidence))
        return results


def draw_annotations(frame: np.ndarray, results: List[RecognitionResult]) -> None:
    for res in results:
        x1, y1, x2, y2 = res.detection.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{res.text} | d:{res.detection.score:.2f} o:{res.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def process_image(pipeline: ANPRInference, image_path: Path, save_dir: Path) -> List[RecognitionResult]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    results = pipeline.recognize_from_frame(image)
    draw_annotations(image, results)

    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"annotated_{image_path.name}"
    cv2.imwrite(str(output_path), image)
    return results


def process_video(
    pipeline: ANPRInference,
    video_path: Path,
    save_dir: Path,
    save_video: bool,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    writer: Optional[cv2.VideoWriter] = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        save_dir.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(save_dir / "annotated.mp4"), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = pipeline.recognize_from_frame(frame)
        draw_annotations(frame, results)
        for res in results:
            print(f"Frame {frame_idx}: {res.text} | det={res.detection.score:.3f} ocr={res.confidence:.3f}")

        if writer:
            writer.write(frame)
        frame_idx += 1

    cap.release()
    if writer:
        writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ANPR inference with YOLO + CRNN")
    parser.add_argument("--image", type=Path, help="Path to image file", default=None)
    parser.add_argument("--video", type=Path, help="Path to video file", default=None)
    parser.add_argument("--detector-weights", type=Path, default=Path("models/yolo11n.pt"))
    parser.add_argument("--ocr-weights", type=Path, default=Path("models/ocr_crnn.pt"))
    parser.add_argument("--vocab", type=str, default="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    parser.add_argument("--img-height", type=int, default=64)
    parser.add_argument("--img-width", type=int, default=160)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=Path, default=Path("runs/inference"))
    parser.add_argument("--save-video", action="store_true", help="Save annotated video output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image is None and args.video is None:
        raise SystemExit("Please provide either --image or --video path")

    pipeline = ANPRInference(
        detector_weights=args.detector_weights,
        ocr_weights=args.ocr_weights,
        vocab=args.vocab,
        img_height=args.img_height,
        img_width=args.img_width,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device=args.device,
    )

    if args.image:
        results = process_image(pipeline, args.image, args.save_dir)
        for res in results:
            print(f"Image: {res.text} | det={res.detection.score:.3f} ocr={res.confidence:.3f}")
    if args.video:
        process_video(pipeline, args.video, args.save_dir, save_video=args.save_video)


if __name__ == "__main__":
    main()
