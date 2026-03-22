"""Unified detection interface for ISL gesture recognition."""

from typing import Dict, List, Optional
import numpy as np

from config.settings import get_config


class SignLanguageDetector:
    """Unified interface for sign language detection.

    Supports YOLOv5, YOLOv8, and ensemble modes.

    Args:
        model_type: One of 'yolov5', 'yolov8', or 'ensemble'.
        weights_path: Path to model weights. If None, uses pretrained defaults.
        conf_threshold: Confidence threshold for detections.
        iou_threshold: IoU threshold for NMS.
    """

    def __init__(
        self,
        model_type: str = "yolov8",
        weights_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ):
        self.model_type = model_type
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None

        self._load_model(weights_path)

    def _load_model(self, weights_path: Optional[str]):
        """Load the detection model based on type."""
        cfg = get_config()

        if self.model_type == "yolov5":
            from models.yolov5_detector import YOLOv5Detector
            self.model = YOLOv5Detector(
                num_classes=cfg.model.num_classes, pretrained=(weights_path is None)
            )
            if weights_path:
                self.model.load_weights(weights_path)

        elif self.model_type == "yolov8":
            from models.yolov8_detector import YOLOv8Detector
            self.model = YOLOv8Detector(
                num_classes=cfg.model.num_classes, pretrained=(weights_path is None)
            )
            if weights_path:
                self.model.load_weights(weights_path)

        elif self.model_type == "ensemble":
            from models.ensemble import EnsembleDetector
            self.model = EnsembleDetector(num_classes=cfg.model.num_classes)

        else:
            raise ValueError(
                f"Unknown model type: {self.model_type}. "
                "Use 'yolov5', 'yolov8', or 'ensemble'."
            )

    def detect(self, image: np.ndarray) -> List[Dict]:
        """Run detection on an image.

        Args:
            image: Input image as numpy array (BGR format from OpenCV).

        Returns:
            List of detection dictionaries with keys:
                - bbox: [x1, y1, x2, y2] bounding box coordinates
                - class_id: Integer class ID
                - class_name: String class name
                - confidence: Float confidence score
        """
        return self.model.detect(
            image,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )

    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """Run detection on a batch of images.

        Args:
            images: List of input images.

        Returns:
            List of detection lists, one per image.
        """
        return [self.detect(img) for img in images]

    def set_threshold(self, conf: float = None, iou: float = None):
        """Update detection thresholds.

        Args:
            conf: New confidence threshold.
            iou: New IoU threshold.
        """
        if conf is not None:
            self.conf_threshold = conf
        if iou is not None:
            self.iou_threshold = iou
