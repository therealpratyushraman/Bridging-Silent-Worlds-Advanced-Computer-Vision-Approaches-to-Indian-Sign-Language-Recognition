"""Central configuration for the Indian Sign Language Recognition system.

All project-wide settings are defined here as dataclasses. Use ``get_config()``
to obtain a single ``AppConfig`` instance that aggregates every sub-config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    """Return 'cuda' when a GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Paths and loader settings for the dataset."""

    raw_dir: str = str(_PROJECT_ROOT / "data" / "raw")
    processed_dir: str = str(_PROJECT_ROOT / "data" / "processed")
    image_size: int = 640
    batch_size: int = 16
    num_workers: int = 4


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Architecture and detection thresholds."""

    model_variant: str = "yolov8s"
    num_classes: int = 6
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    gesture_classes: List[str] = field(
        default_factory=lambda: ["Hello", "Help", "Home", "No", "Please", "Yes"]
    )
    device: str = field(default_factory=_detect_device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Hyperparameters for model training."""

    epochs: int = 100
    learning_rate: float = 0.01
    optimizer: str = "SGD"
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    freeze_layers: int = 10
    transfer_learning: bool = True
    device: str = field(default_factory=_detect_device)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """Real-time inference parameters."""

    sustained_detection_seconds: float = 3.0
    webcam_resolution: Tuple[int, int] = (640, 480)
    fps_target: int = 30
    device: str = field(default_factory=_detect_device)


# ---------------------------------------------------------------------------
# Emotion analysis
# ---------------------------------------------------------------------------

@dataclass
class EmotionConfig:
    """Sentiment / emotion classification settings."""

    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    labels: List[str] = field(
        default_factory=lambda: ["positive", "negative", "neutral"]
    )


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@dataclass
class APIConfig:
    """FastAPI / Flask server settings."""

    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


# ---------------------------------------------------------------------------
# Combined application config
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """Top-level configuration that aggregates all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    api: APIConfig = field(default_factory=APIConfig)
    project_root: str = str(_PROJECT_ROOT)
    device: str = field(default_factory=_detect_device)


# ---------------------------------------------------------------------------
# Public accessor
# ---------------------------------------------------------------------------

_config_instance: AppConfig | None = None


def get_config() -> AppConfig:
    """Return the singleton ``AppConfig`` instance.

    The configuration is created once and reused for the lifetime of the
    process.  Environment variable overrides can be applied here if needed
    in the future.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance
