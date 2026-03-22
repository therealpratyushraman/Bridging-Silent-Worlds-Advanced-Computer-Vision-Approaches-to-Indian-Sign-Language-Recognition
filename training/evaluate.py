"""Model evaluation utilities for ISL gesture recognition."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from config.settings import get_config


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Optional list of class names.

    Returns:
        Dictionary of computed metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    if class_names:
        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True,
            zero_division=0,
        )
        metrics["per_class"] = {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in class_names
            if name in report
        }

    return metrics


def evaluate_yolo_model(
    model_type: str,
    weights_path: str,
    data_yaml: str,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.45,
) -> Dict:
    """Evaluate a YOLO model on a test dataset.

    Args:
        model_type: Either 'yolov5' or 'yolov8'.
        weights_path: Path to model weights.
        data_yaml: Path to dataset YAML.
        conf_threshold: Confidence threshold.
        iou_threshold: IoU threshold for NMS.

    Returns:
        Dictionary of evaluation metrics.
    """
    if model_type == "yolov5":
        from models.yolov5_detector import YOLOv5Detector
        detector = YOLOv5Detector(pretrained=False)
        detector.load_weights(weights_path)
    elif model_type == "yolov8":
        from models.yolov8_detector import YOLOv8Detector
        detector = YOLOv8Detector(pretrained=False)
        detector.load_weights(weights_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    results = detector.model.val(
        data=data_yaml,
        conf=conf_threshold,
        iou=iou_threshold,
    )

    metrics = {
        "model_type": model_type,
        "weights": weights_path,
        "mAP50": float(results.box.map50) if hasattr(results.box, "map50") else 0.0,
        "mAP50_95": float(results.box.map) if hasattr(results.box, "map") else 0.0,
        "precision": float(results.box.mp) if hasattr(results.box, "mp") else 0.0,
        "recall": float(results.box.mr) if hasattr(results.box, "mr") else 0.0,
    }

    return metrics


def compare_models(
    yolov5_weights: Optional[str] = None,
    yolov8_weights: Optional[str] = None,
    data_yaml: str = None,
) -> Dict:
    """Compare YOLOv5 and YOLOv8 performance.

    Args:
        yolov5_weights: Path to YOLOv5 weights.
        yolov8_weights: Path to YOLOv8 weights.
        data_yaml: Path to dataset YAML.

    Returns:
        Comparison dictionary with metrics for each model.
    """
    comparison = {}

    if yolov5_weights and os.path.exists(yolov5_weights):
        print("Evaluating YOLOv5...")
        comparison["yolov5"] = evaluate_yolo_model("yolov5", yolov5_weights, data_yaml)

    if yolov8_weights and os.path.exists(yolov8_weights):
        print("Evaluating YOLOv8...")
        comparison["yolov8"] = evaluate_yolo_model("yolov8", yolov8_weights, data_yaml)

    if len(comparison) == 2:
        print("\n--- Model Comparison ---")
        print(f"{'Metric':<20} {'YOLOv5':>10} {'YOLOv8':>10} {'Winner':>10}")
        print("-" * 52)
        for metric in ["mAP50", "mAP50_95", "precision", "recall"]:
            v5 = comparison["yolov5"].get(metric, 0)
            v8 = comparison["yolov8"].get(metric, 0)
            winner = "YOLOv8" if v8 >= v5 else "YOLOv5"
            print(f"{metric:<20} {v5:>10.4f} {v8:>10.4f} {winner:>10}")

    return comparison


def evaluate_model(
    model_type: str = "yolov8",
    weights_path: str = None,
    data_yaml: str = None,
    output_dir: str = "runs/evaluate",
) -> Dict:
    """Main evaluation function.

    Args:
        model_type: Model type ('yolov5', 'yolov8', or 'compare').
        weights_path: Path to model weights.
        data_yaml: Path to dataset YAML.
        output_dir: Directory to save results.

    Returns:
        Evaluation results dictionary.
    """
    cfg = get_config()
    os.makedirs(output_dir, exist_ok=True)

    if data_yaml is None:
        data_yaml = os.path.join(cfg.data.processed_dir, "dataset.yaml")

    if model_type == "compare":
        results = compare_models(
            yolov5_weights=os.path.join("runs/yolov5/isl_training/weights/best.pt"),
            yolov8_weights=os.path.join("runs/yolov8/isl_training/weights/best.pt"),
            data_yaml=data_yaml,
        )
    else:
        if weights_path is None:
            weights_path = os.path.join(
                f"runs/{model_type}/isl_training/weights/best.pt"
            )
        results = evaluate_yolo_model(model_type, weights_path, data_yaml)

    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ISL recognition models")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8",
        choices=["yolov5", "yolov8", "compare"],
        help="Model to evaluate",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/evaluate",
        help="Output directory for results",
    )
    args = parser.parse_args()

    evaluate_model(args.model, args.weights, args.data, args.output_dir)


if __name__ == "__main__":
    main()
