"""YOLOv5 training script for ISL gesture recognition."""

import argparse
import os
import yaml
from pathlib import Path

import torch

from models.yolov5_detector import YOLOv5Detector
from config.settings import get_config


def train(config_path: str, data_yaml: str = None, resume: bool = False):
    """Train YOLOv5 on ISL gesture dataset.

    Args:
        config_path: Path to YOLOv5 training config YAML.
        data_yaml: Path to dataset YAML. If None, uses default from config.
        resume: Whether to resume from last checkpoint.
    """
    cfg = get_config()

    with open(config_path, "r") as f:
        train_config = yaml.safe_load(f)

    if data_yaml is None:
        data_yaml = os.path.join(cfg.data.processed_dir, "dataset.yaml")

    detector = YOLOv5Detector(
        model_size=train_config.get("model_size", "s"),
        num_classes=cfg.model.num_classes,
        pretrained=True,
    )

    if cfg.training.transfer_learning:
        detector.configure_transfer_learning(
            freeze_layers=cfg.training.freeze_layers
        )
        print(
            f"Transfer learning enabled: freezing first "
            f"{cfg.training.freeze_layers} layers"
        )

    epochs = train_config.get("epochs", cfg.training.epochs)
    batch_size = train_config.get("batch_size", cfg.data.batch_size)
    imgsz = train_config.get("imgsz", cfg.data.image_size)
    lr0 = train_config.get("lr0", cfg.training.learning_rate)

    print(f"Starting YOLOv5 training:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {imgsz}")
    print(f"  Learning rate: {lr0}")
    print(f"  Device: {cfg.device}")
    print(f"  Dataset: {data_yaml}")

    results = detector.train(
        data_yaml=data_yaml,
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz,
        lr0=lr0,
        device=cfg.device,
        project="runs/yolov5",
        name="isl_training",
        exist_ok=True,
    )

    print("\nTraining complete!")
    if results is not None:
        print(f"Results saved to: runs/yolov5/isl_training")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv5 on ISL dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config/yolov5_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset YAML",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    args = parser.parse_args()

    train(args.config, args.data, args.resume)


if __name__ == "__main__":
    main()
