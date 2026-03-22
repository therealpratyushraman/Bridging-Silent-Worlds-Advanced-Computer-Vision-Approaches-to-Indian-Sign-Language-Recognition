"""YOLOv8 wrapper for Indian Sign Language gesture detection (primary model)."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from ultralytics import YOLO

from .gesture_vocabulary import GESTURE_CLASSES

logger = logging.getLogger(__name__)

_VALID_SIZES = ("n", "s", "m", "l", "x")


class YOLOv8Detector:
    """Primary detector built on Ultralytics YOLOv8.

    Provides a unified API for training, inference, export, and metrics
    retrieval on the ISL gesture detection task.
    """

    def __init__(
        self,
        model_size: str = "s",
        num_classes: int = 6,
        pretrained: bool = True,
    ) -> None:
        """Initialise the YOLOv8 detector.

        Args:
            model_size: One of 'n', 's', 'm', 'l', 'x'.
            num_classes: Number of output gesture classes.
            pretrained: If ``True``, load COCO-pretrained weights.
        """
        if model_size not in _VALID_SIZES:
            raise ValueError(
                f"model_size must be one of {_VALID_SIZES}, got '{model_size}'"
            )

        self.model_size = model_size
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.class_names: Dict[int, str] = dict(GESTURE_CLASSES)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._train_metrics: Optional[Any] = None

        weight_file = f"yolov8{model_size}.pt" if pretrained else f"yolov8{model_size}.yaml"
        logger.info("Loading YOLOv8 from %s (pretrained=%s)", weight_file, pretrained)

        self.model = YOLO(weight_file)
        logger.info("YOLOv8 model ready on %s", self.device)

    # ------------------------------------------------------------------
    # Transfer learning
    # ------------------------------------------------------------------

    def configure_transfer_learning(self, freeze_layers: int = 10) -> None:
        """Freeze the first *freeze_layers* parameters of the backbone.

        Args:
            freeze_layers: Number of parameter groups to freeze.
        """
        frozen = 0
        for i, (name, param) in enumerate(self.model.model.named_parameters()):
            if i < freeze_layers:
                param.requires_grad = False
                frozen += 1
            else:
                param.requires_grad = True

        total = sum(1 for _ in self.model.model.parameters())
        trainable = sum(p.requires_grad for p in self.model.model.parameters())
        logger.info(
            "Froze %d / %d parameters (%d remain trainable)",
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
            image: File path or a numpy array (BGR or RGB).
            conf_threshold: Minimum confidence for a detection.
            iou_threshold: IoU threshold for NMS.

        Returns:
            List of dicts with keys ``bbox``, ``class_id``,
            ``class_name``, ``confidence``.
        """
        if isinstance(image, Path):
            image = str(image)

        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                detections.append(
                    {
                        "bbox": xyxy,
                        "class_id": cls_id,
                        "class_name": self.class_names.get(cls_id, f"class_{cls_id}"),
                        "confidence": conf,
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
        imgsz: int = 640,
        **kwargs: Any,
    ) -> Any:
        """Train (or fine-tune) the model using the Ultralytics training API.

        Args:
            data_yaml: Path to a YOLO-format ``data.yaml``.
            epochs: Training epochs.
            batch_size: Images per batch.
            imgsz: Training image size (pixels).
            **kwargs: Extra keyword arguments forwarded to
                ``model.train()`` (e.g. ``lr0``, ``optimizer``,
                ``project``, ``name``).

        Returns:
            The Ultralytics ``Results`` object from training.
        """
        data_yaml = str(data_yaml)
        logger.info(
            "Starting YOLOv8 training: data=%s, epochs=%d, batch=%d, imgsz=%d",
            data_yaml,
            epochs,
            batch_size,
            imgsz,
        )

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=self.device,
            **kwargs,
        )

        self._train_metrics = results
        return results

    # ------------------------------------------------------------------
    # Export / Load / Metrics
    # ------------------------------------------------------------------

    def export(self, format: str = "onnx") -> Path:
        """Export the model to the given format.

        Args:
            format: One of ``'onnx'``, ``'torchscript'``, ``'coreml'``,
                ``'tflite'``, ``'engine'``, etc.

        Returns:
            Path to the exported file.
        """
        supported = ("onnx", "torchscript", "coreml", "tflite", "engine",
                      "openvino", "saved_model", "pb")
        if format not in supported:
            raise ValueError(
                f"Unsupported export format '{format}'. Choose from {supported}"
            )

        logger.info("Exporting YOLOv8 model to %s", format)
        export_path = self.model.export(format=format)
        logger.info("Model exported to %s", export_path)
        return Path(export_path)

    def load_weights(self, weights_path: Union[str, Path]) -> None:
        """Load custom-trained weights.

        Args:
            weights_path: Path to a ``.pt`` checkpoint.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        logger.info("Loading custom weights from %s", weights_path)
        self.model = YOLO(str(weights_path))
        logger.info("Custom weights loaded successfully")

    def get_metrics(self) -> Dict[str, Any]:
        """Return training metrics from the last ``train()`` call.

        Returns:
            Dictionary containing mAP50, mAP50-95, precision, recall,
            and per-class metrics when available.  Returns an empty dict
            if training has not been run.
        """
        if self._train_metrics is None:
            logger.warning("No training metrics available; run train() first.")
            return {}

        results = self._train_metrics
        metrics: Dict[str, Any] = {}

        # The Ultralytics Results object exposes .results_dict after training
        if hasattr(results, "results_dict"):
            metrics.update(results.results_dict)
        else:
            # Fallback: pull common attributes
            for attr in (
                "box",
                "maps",
                "fitness",
                "ap",
                "ap_class_index",
                "ap50",
            ):
                val = getattr(results, attr, None)
                if val is not None:
                    if isinstance(val, (np.ndarray, torch.Tensor)):
                        val = val.tolist()
                    metrics[attr] = val

        return metrics
