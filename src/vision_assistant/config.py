from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    camera_source: str = "0"
    camera_index: int = 0
    cam_width: int = 1280
    cam_height: int = 720
    display_scale: float = 1.15
    camera_open_timeout_ms: int = 8000
    camera_read_timeout_ms: int = 12000
    camera_reconnect_delay_seconds: float = 2.0
    process_every_n_frames: int = 1
    detector_model: str = "yolo11s.pt"
    detector_confidence: float = 0.4
    detector_classes: list[int] = field(default_factory=list)
    tracker_config: str = "bytetrack.yaml"
    vlm_runtime: str = "lmstudio"
    vlm_model: str = "google/gemma-4-26b-a4b"
    lmstudio_base_url: str = "http://localhost:1234"
    ollama_base_url: str = "http://localhost:11434"
    scene_cooldown_seconds: float = 6.0
    scene_change_min_interval_seconds: float = 2.5
    max_description_chars: int = 180
    vlm_image_max_edge: int = 448
    vlm_jpeg_quality: int = 55
    vlm_max_labels: int = 8
    max_recent_events: int = 4
    enable_hand_signals: bool = True
    hand_model_path: Path = Path("models/hand_landmarker.task")
    screenshot_dir: Path = Path("screenshots")
