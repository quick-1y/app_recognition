"""Realtime video processing entrypoint."""
from __future__ import annotations

import logging
from typing import Callable, Optional

from app.config import Settings, load_config
from app.database import PlateDatabase
from app.pipeline import FrameCallbacks, PlatePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(
        self,
        frame_callback: Callable,
        text_callback: Callable,
        settings: Optional[Settings] = None,
        database: Optional[PlateDatabase] = None,
    ):
        self.settings = settings or load_config()
        self.database = database or PlateDatabase(self.settings.database)
        self.pipeline = PlatePipeline(self.settings, self.database)
        self.frame_callback = frame_callback
        self.text_callback = text_callback

    def run_for_path(self, video_path: str) -> None:
        callbacks = FrameCallbacks(frame=self.frame_callback, text=self.text_callback)
        self.pipeline.process_video(video_path, callbacks)


processor_cache: VideoProcessor | None = None


def process_video_realtime(
    video_path: str,
    frame_callback: Callable,
    text_callback: Callable,
    settings: Optional[Settings] = None,
    database: Optional[PlateDatabase] = None,
) -> None:
    """Compatibility wrapper used by the GUI thread."""
    global processor_cache
    if processor_cache is None:
        processor_cache = VideoProcessor(frame_callback, text_callback, settings=settings, database=database)
    processor_cache.run_for_path(video_path)
