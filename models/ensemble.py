"""Ensemble detector combining YOLOv5 and YOLOv8 for robust ISL gesture detection."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .gesture_vocabulary import GESTURE_CLASSES
from .yolov5_detector import YOLOv5Detector
from .yolov8_detector import YOLOv8Detector

logger = logging.getLogger(__name__)


def _iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute Intersection-over-Union between two ``[x1, y1, x2, y2]`` boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class EnsembleDetector:
    """Combines YOLOv5 and YOLOv8 predictions using weighted box fusion or
    consensus voting for improved detection reliability.

    Example::

        ensemble = EnsembleDetector(
            yolov5_weights="runs/yolov5/best.pt",
            yolov8_weights="runs/yolov8/best.pt",
        )
        detections = ensemble.detect(image, method="weighted_average")
    """

    def __init__(
        self,
        yolov5_weights: Optional[Union[str, Path]] = None,
        yolov8_weights: Optional[Union[str, Path]] = None,
        num_classes: int = 6,
    ) -> None:
        """Initialise both detectors and optionally load custom weights.

        Args:
            yolov5_weights: Path to custom YOLOv5 ``.pt`` weights.  If
                ``None``, COCO-pretrained weights are used.
            yolov8_weights: Path to custom YOLOv8 ``.pt`` weights.  If
                ``None``, COCO-pretrained weights are used.
            num_classes: Number of gesture classes.
        """
        self.num_classes = num_classes
        self.class_names: Dict[int, str] = dict(GESTURE_CLASSES)

        # --- YOLOv5 ---
        logger.info("Initialising YOLOv5 detector for ensemble")
        self.yolov5 = YOLOv5Detector(
            model_size="s", num_classes=num_classes, pretrained=True
        )
        if yolov5_weights is not None:
            self.yolov5.load_weights(yolov5_weights)

        # --- YOLOv8 ---
        logger.info("Initialising YOLOv8 detector for ensemble")
        self.yolov8 = YOLOv8Detector(
            model_size="s", num_classes=num_classes, pretrained=True
        )
        if yolov8_weights is not None:
            self.yolov8.load_weights(yolov8_weights)

        # Default model weights for WBF (YOLOv8 weighted higher as primary)
        self.model_weights: List[float] = [0.4, 0.6]

        logger.info("Ensemble detector ready (YOLOv5 + YOLOv8)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        image: Union[str, Path, "np.ndarray"],
        conf_threshold: float = 0.5,
        method: str = "weighted_average",
    ) -> List[Dict[str, Any]]:
        """Run both models on *image* and fuse their predictions.

        Args:
            image: File path or numpy array (BGR/RGB).
            conf_threshold: Minimum confidence for individual model
                detections.
            method: Fusion strategy — ``'weighted_average'`` for weighted
                box fusion or ``'consensus'`` for consensus voting.

        Returns:
            Merged list of detection dicts (``bbox``, ``class_id``,
            ``class_name``, ``confidence``).
        """
        if method not in ("weighted_average", "consensus"):
            raise ValueError(
                f"Unknown fusion method '{method}'. "
                "Choose 'weighted_average' or 'consensus'."
            )

        # Gather predictions from each model
        v5_dets = self.yolov5.detect(image, conf_threshold=conf_threshold)
        v8_dets = self.yolov8.detect(image, conf_threshold=conf_threshold)

        detections_list = [v5_dets, v8_dets]

        if method == "weighted_average":
            return self._weighted_box_fusion(
                detections_list, weights=self.model_weights
            )
        else:
            return self._consensus_voting(detections_list, min_agreement=2)

    # ------------------------------------------------------------------
    # Fusion strategies
    # ------------------------------------------------------------------

    def _weighted_box_fusion(
        self,
        detections_list: List[List[Dict[str, Any]]],
        weights: List[float],
        iou_threshold: float = 0.55,
    ) -> List[Dict[str, Any]]:
        """Merge detections from multiple models via Weighted Box Fusion.

        For every cluster of overlapping boxes (same class, IoU above
        threshold), a single fused box is produced whose coordinates and
        confidence are weighted averages of the contributing boxes.

        Args:
            detections_list: Per-model list of detection dicts.
            weights: Per-model importance weight (should sum to 1 but
                will be normalised internally).
            iou_threshold: Minimum IoU to consider two boxes as matching
                the same object.

        Returns:
            Fused detection list.
        """
        # Normalise weights
        total_w = sum(weights) or 1.0
        norm_weights = [w / total_w for w in weights]

        # Flatten all detections and tag each with its source weight
        tagged: List[Dict[str, Any]] = []
        for model_idx, dets in enumerate(detections_list):
            for det in dets:
                entry = dict(det)
                entry["_weight"] = norm_weights[model_idx]
                tagged.append(entry)

        # Group by class
        class_groups: Dict[int, List[Dict[str, Any]]] = {}
        for det in tagged:
            cls_id = det["class_id"]
            class_groups.setdefault(cls_id, []).append(det)

        fused: List[Dict[str, Any]] = []

        for cls_id, dets in class_groups.items():
            # Sort by confidence (descending) to seed clusters with best boxes
            dets.sort(key=lambda d: d["confidence"], reverse=True)
            used = [False] * len(dets)

            for i, det_i in enumerate(dets):
                if used[i]:
                    continue

                # Start a new cluster with det_i
                cluster = [det_i]
                used[i] = True

                for j in range(i + 1, len(dets)):
                    if used[j]:
                        continue
                    if _iou(det_i["bbox"], dets[j]["bbox"]) >= iou_threshold:
                        cluster.append(dets[j])
                        used[j] = True

                # Fuse cluster into a single box
                weighted_bbox = [0.0, 0.0, 0.0, 0.0]
                weighted_conf = 0.0
                sum_w = 0.0

                for det in cluster:
                    w = det["_weight"] * det["confidence"]
                    for k in range(4):
                        weighted_bbox[k] += det["bbox"][k] * w
                    weighted_conf += det["confidence"] * det["_weight"]
                    sum_w += w

                if sum_w > 0:
                    weighted_bbox = [c / sum_w for c in weighted_bbox]

                # Average confidence across contributing models
                avg_conf = weighted_conf / len(cluster) if cluster else 0.0

                fused.append(
                    {
                        "bbox": weighted_bbox,
                        "class_id": cls_id,
                        "class_name": self.class_names.get(cls_id, f"class_{cls_id}"),
                        "confidence": float(avg_conf),
                        "num_models": len(cluster),
                    }
                )

        # Sort final detections by confidence
        fused.sort(key=lambda d: d["confidence"], reverse=True)
        return fused

    def _consensus_voting(
        self,
        detections_list: List[List[Dict[str, Any]]],
        min_agreement: int = 2,
        iou_threshold: float = 0.50,
    ) -> List[Dict[str, Any]]:
        """Keep only detections agreed upon by at least *min_agreement* models.

        Two detections from different models are considered to agree when
        they predict the same class and their bounding boxes overlap with
        IoU >= *iou_threshold*.

        Args:
            detections_list: Per-model list of detection dicts.
            min_agreement: Minimum number of models that must detect an
                object for it to be retained.
            iou_threshold: IoU threshold for matching boxes across models.

        Returns:
            Filtered and averaged detection list.
        """
        if len(detections_list) < min_agreement:
            logger.warning(
                "Only %d model(s) provided but min_agreement=%d; "
                "returning empty results.",
                len(detections_list),
                min_agreement,
            )
            return []

        # Use the first model's detections as reference, then check agreement
        reference = detections_list[0]
        other_models = detections_list[1:]

        agreed: List[Dict[str, Any]] = []

        for ref_det in reference:
            agreement_count = 1  # reference model itself
            matching_dets = [ref_det]

            for model_dets in other_models:
                for det in model_dets:
                    if det["class_id"] != ref_det["class_id"]:
                        continue
                    if _iou(ref_det["bbox"], det["bbox"]) >= iou_threshold:
                        agreement_count += 1
                        matching_dets.append(det)
                        break  # one match per model is enough

            if agreement_count >= min_agreement:
                # Average the matching detections
                avg_bbox = [0.0, 0.0, 0.0, 0.0]
                avg_conf = 0.0
                n = len(matching_dets)

                for det in matching_dets:
                    for k in range(4):
                        avg_bbox[k] += det["bbox"][k]
                    avg_conf += det["confidence"]

                avg_bbox = [c / n for c in avg_bbox]
                avg_conf /= n

                cls_id = ref_det["class_id"]
                agreed.append(
                    {
                        "bbox": avg_bbox,
                        "class_id": cls_id,
                        "class_name": self.class_names.get(cls_id, f"class_{cls_id}"),
                        "confidence": float(avg_conf),
                        "num_models": agreement_count,
                    }
                )

        agreed.sort(key=lambda d: d["confidence"], reverse=True)
        return agreed
