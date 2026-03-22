"""YOLOv5 wrapper for Indian Sign Language gesture detection."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from .gesture_vocabulary import GESTURE_CLASSES

logger = logging.getLogger(__name__)

_VALID_SIZES = ("n", "s", "m", "l", "x")


class YOLOv5Detector:
    """Thin wrapper around the Ultralytics YOLOv5 model loaded via torch.hub.

    This class provides a consistent interface for loading, training, running
    inference, and exporting a YOLOv5 model fine-tuned for ISL gesture
    detection.
    """

    def __init__(
        self,
        model_size: str = "s",
        num_classes: int = 6,
        pretrained: bool = True,
    ) -> None:
        """Initialise the YOLOv5 detector.

        Args:
            model_size: One of 'n', 's', 'm', 'l', 'x' (YOLOv5 variants).
            num_classes: Number of output classes (default 6 for ISL gestures).
            pretrained: If ``True``, load COCO-pretrained weights from
                Ultralytics via ``torch.hub``.
        """
        if model_size not in _VALID_SIZES:
            raise ValueError(
                f"model_size must be one of {_VALID_SIZES}, got '{model_size}'"
            )

        self.model_size = model_size
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.class_names: Dict[int, str] = dict(GESTURE_CLASSES)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = f"yolov5{model_size}"
        logger.info("Loading %s (pretrained=%s) via torch.hub", model_name, pretrained)

        if pretrained:
            self.model = torch.hub.load(
                "ultralytics/yolov5",
                model_name,
                pretrained=True,
                trust_repo=True,
            )
        else:
            self.model = torch.hub.load(
                "ultralytics/yolov5",
                model_name,
                pretrained=False,
                classes=num_classes,
                trust_repo=True,
            )

        self.model.to(self.device)
        logger.info("YOLOv5 model loaded on %s", self.device)

    # ------------------------------------------------------------------
    # Transfer learning
    # ------------------------------------------------------------------

    def configure_transfer_learning(self, freeze_layers: int = 10) -> None:
        """Freeze the first *freeze_layers* layers of the backbone.

        This is the standard approach for fine-tuning a COCO-pretrained
        detector on a smaller domain-specific dataset.

        Args:
            freeze_layers: Number of backbone layers to freeze.
        """
        frozen = 0
        for i, (name, param) in enumerate(self.model.model.named_parameters()):
            if i < freeze_layers:
                param.requires_grad = False
                frozen += 1
            else:
                param.requires_grad = True

        trainable = sum(p.requires_grad for p in self.model.model.parameters())
        total = sum(1 for _ in self.model.model.parameters())
        logger.info(
            "Froze %d / %d parameters (%d trainable)",
            frozen,
            total,
            trainable,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """Run inference on a single image.

        Args:
            image: File path (str / Path) or a BGR/RGB numpy array.
            conf_threshold: Minimum confidence to keep a detection.
            iou_threshold: IoU threshold for non-maximum suppression.

        Returns:
            List of detection dicts, each with keys:
                - ``bbox`` (list[float]): ``[x1, y1, x2, y2]``
                - ``class_id`` (int): Predicted class index.
                - ``class_name`` (str): Human-readable class name.
                - ``confidence`` (float): Detection confidence score.
        """
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold

        if isinstance(image, Path):
            image = str(image)

        results = self.model(image)
        detections: List[Dict[str, Any]] = []

        # results.xyxy[0] is a tensor of shape (N, 6): x1 y1 x2 y2 conf cls
        preds = results.xyxy[0].cpu().numpy()
        for *xyxy, conf, cls_id in preds:
            cls_id = int(cls_id)
            detections.append(
                {
                    "bbox": [float(c) for c in xyxy],
                    "class_id": cls_id,
                    "class_name": self.class_names.get(cls_id, f"class_{cls_id}"),
                    "confidence": float(conf),
                }
            )

        return detections

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        data_yaml: Union[str, Path],
        epochs: int = 100,
        batch_size: int = 16,
        **kwargs: Any,
    ) -> Any:
        """Launch a YOLOv5 training run.

        This shells out to the YOLOv5 ``train.py`` script via
        ``torch.hub`` utilities, which is the officially supported way to
        train YOLOv5.

        Args:
            data_yaml: Path to a YOLO-format ``data.yaml`` file.
            epochs: Number of training epochs.
            batch_size: Batch size per device.
            **kwargs: Forwarded to the training script (e.g. ``imgsz``,
                ``lr0``, ``project``).

        Returns:
            The subprocess result or training results object.
        """
        import subprocess
        import sys

        data_yaml = str(data_yaml)
        model_name = f"yolov5{self.model_size}.pt"

        cmd = [
            sys.executable,
            "-m",
            "yolov5.train",
            "--img",
            str(kwargs.pop("imgsz", 640)),
            "--batch",
            str(batch_size),
            "--epochs",
            str(epochs),
            "--data",
            data_yaml,
            "--weights",
            model_name,
        ]

        for key, value in kwargs.items():
            cmd.extend([f"--{key}", str(value)])

        logger.info("Starting YOLOv5 training: %s", " ".join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result

    # ------------------------------------------------------------------
    # Export / Load
    # ------------------------------------------------------------------

    def export(self, format: str = "onnx") -> Path:
        """Export the model to the specified format.

        Args:
            format: Target format, e.g. ``'onnx'``, ``'torchscript'``,
                ``'coreml'``.

        Returns:
            Path to the exported model file.
        """
        import subprocess
        import sys

        supported = ("onnx", "torchscript", "coreml", "tflite", "engine")
        if format not in supported:
            raise ValueError(f"Unsupported export format '{format}'. Choose from {supported}")

        export_path = Path(f"yolov5{self.model_size}.{format}")

        cmd = [
            sys.executable,
            "-m",
            "yolov5.export",
            "--weights",
            f"yolov5{self.model_size}.pt",
            "--include",
            format,
        ]

        logger.info("Exporting YOLOv5 to %s", format)
        subprocess.run(cmd, check=True, capture_output=False)
        logger.info("Model exported to %s", export_path)
        return export_path

    def load_weights(self, weights_path: Union[str, Path]) -> None:
        """Load custom-trained weights from disk.

        Args:
            weights_path: Path to a ``.pt`` weights file.

        Raises:
            FileNotFoundError: If *weights_path* does not exist.
        """
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        logger.info("Loading custom weights from %s", weights_path)
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=str(weights_path),
            trust_repo=True,
        )
        self.model.to(self.device)
        logger.info("Custom weights loaded successfully")
