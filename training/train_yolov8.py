"""YOLOv8 training script for ISL gesture recognition (primary model)."""

import argparse
import os
import yaml
from pathlib import Path

from models.yolov8_detector import YOLOv8Detector
from config.settings import get_config


def train(config_path: str, data_yaml: str = None, resume: bool = False):
    """Train YOLOv8 on ISL gesture dataset.

    Args:
        config_path: Path to YOLOv8 training config YAML.
        data_yaml: Path to dataset YAML. If None, uses default from config.
        resume: Whether to resume from last checkpoint.
    """
    cfg = get_config()

    with open(config_path, "r") as f:
        train_config = yaml.safe_load(f)

    if data_yaml is None:
        data_yaml = os.path.join(cfg.data.processed_dir, "dataset.yaml")

    model_size = train_config.get("model", "yolov8s.pt").replace("yolov8", "").replace(".pt", "")
    if not model_size:
        model_size = "s"

    detector = YOLOv8Detector(
        model_size=model_size,
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
    batch_size = train_config.get("batch", cfg.data.batch_size)
    imgsz = train_config.get("imgsz", cfg.data.image_size)
    lr0 = train_config.get("lr0", cfg.training.learning_rate)

    print(f"Starting YOLOv8 training:")
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
        project="runs/yolov8",
        name="isl_training",
        exist_ok=True,
    )

    print("\nTraining complete!")

    metrics = detector.get_metrics()
    if metrics:
        print(f"\nFinal Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on ISL dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config/yolov8_config.yaml",
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
