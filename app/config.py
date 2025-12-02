from __future__ import annotations

import yaml
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List


@dataclass
class OCRConfig:
    """Configuration for OCR reader and preprocessing."""

    gpu: bool = False
    languages: List[str] = field(default_factory=lambda: ["en"])
    min_confidence: float = 0.35
    resize_factor: float = 2.5
    contrast_alpha: float = 1.6
    contrast_beta: int = 0
    denoise_diameter: int = 7
    threshold_block_size: int = 25
    threshold_c: int = 7
    max_candidates: int = 5


@dataclass
class ModelConfig:
    """YOLO detection model configuration."""

    detector_weights: Path = Path("models/best.pt")
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45


@dataclass
class ProcessingConfig:
    """General processing configuration."""

    plate_image_send_interval: int = 20
    tracking_history: int = 32
    draw_tracks: bool = True


@dataclass
class DatabaseConfig:
    """SQLite storage configuration."""

    path: Path = Path("data/plates.db")
    vacuum_on_start: bool = False
    retention_days: int = 30


@dataclass
class AppConfig:
    """Application configuration container."""

    video_paths: List[str] = field(default_factory=list)
    plate_patterns_path: Path = Path("configs/plate_patterns.yaml")


@dataclass
class Settings:
    """Bundled settings for the application."""

    app: AppConfig = field(default_factory=AppConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)


def _load_section(source: dict | None, section: str) -> dict:
    return (source or {}).get(section, {})


def _apply_dataclass(defaults, values: dict):
    merged = defaults.__class__(**{**defaults.__dict__, **values})
    return merged


def load_config(path: str | Path = "config.yaml") -> Settings:
    """Load YAML configuration and merge with defaults."""

    with open(path, "r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}

    defaults = Settings()
    app_cfg = _apply_dataclass(defaults.app, _load_section(raw_config, "app"))
    ocr_cfg = _apply_dataclass(defaults.ocr, _load_section(raw_config, "ocr"))
    model_cfg = _apply_dataclass(defaults.model, _load_section(raw_config, "model"))
    processing_cfg = _apply_dataclass(defaults.processing, _load_section(raw_config, "processing"))
    database_cfg = _apply_dataclass(defaults.database, _load_section(raw_config, "database"))

    # Ensure paths are Path objects
    app_cfg.plate_patterns_path = Path(app_cfg.plate_patterns_path)
    model_cfg.detector_weights = Path(model_cfg.detector_weights)
    database_cfg.path = Path(database_cfg.path)

    return Settings(
        app=app_cfg,
        ocr=ocr_cfg,
        model=model_cfg,
        processing=processing_cfg,
        database=database_cfg,
    )


def save_config(settings: Settings, path: str | Path = "config.yaml") -> None:
    """Persist configuration back to disk."""

    serialized = {
        "app": asdict(settings.app),
        "ocr": asdict(settings.ocr),
        "model": asdict(settings.model),
        "processing": asdict(settings.processing),
        "database": asdict(settings.database),
    }
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(serialized, file, allow_unicode=True, sort_keys=False)
