"""Dataset exploration and visualization utilities."""

import os
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2


def plot_class_distribution(
    data_dir: str,
    output_path: str = "runs/plots/class_distribution.png",
):
    """Plot the class distribution of a YOLO-format dataset.

    Args:
        data_dir: Path to dataset directory containing label files.
        output_path: Path to save the plot.
    """
    from models.gesture_vocabulary import GESTURE_CLASSES

    class_counts = Counter()
    labels_dir = os.path.join(data_dir, "labels")

    if not os.path.isdir(labels_dir):
        # Try train subdirectory
        labels_dir = os.path.join(data_dir, "labels", "train")

    if not os.path.isdir(labels_dir):
        print(f"Labels directory not found: {labels_dir}")
        return

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue
        with open(os.path.join(labels_dir, label_file), "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_name = GESTURE_CLASSES.get(class_id, f"Class {class_id}")
                    class_counts[class_name] += 1

    if not class_counts:
        print("No labels found.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(classes, counts, color="steelblue", edgecolor="black")
    ax.bar_label(bars, fontsize=10)
    ax.set_xlabel("Gesture Class")
    ax.set_ylabel("Number of Instances")
    ax.set_title("Dataset Class Distribution")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Class distribution plot saved to: {output_path}")


def visualize_samples(
    data_dir: str,
    num_samples: int = 16,
    output_path: str = "runs/plots/sample_images.png",
):
    """Visualize a grid of sample images from the dataset.

    Args:
        data_dir: Path to dataset directory with images/.
        num_samples: Number of samples to display.
        output_path: Path to save the grid image.
    """
    images_dir = os.path.join(data_dir, "images")
    if not os.path.isdir(images_dir):
        images_dir = os.path.join(data_dir, "images", "train")

    if not os.path.isdir(images_dir):
        print(f"Images directory not found: {images_dir}")
        return

    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    if not image_files:
        print("No images found.")
        return

    num_samples = min(num_samples, len(image_files))
    selected = np.random.choice(image_files, num_samples, replace=False)

    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, img_name in enumerate(selected):
        row, col = divmod(idx, cols)
        img_path = os.path.join(images_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(img)
        axes[row, col].set_title(img_name[:20], fontsize=8)
        axes[row, col].axis("off")

    # Hide empty subplots
    for idx in range(num_samples, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].axis("off")

    plt.suptitle("Dataset Sample Images", fontsize=14)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sample images grid saved to: {output_path}")


def print_dataset_stats(data_dir: str):
    """Print dataset statistics to console.

    Args:
        data_dir: Path to dataset directory.
    """
    stats = {"splits": {}}

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(data_dir, "images", split)
        lbl_dir = os.path.join(data_dir, "labels", split)

        if os.path.isdir(img_dir):
            num_images = len([
                f for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ])
        else:
            num_images = 0

        if os.path.isdir(lbl_dir):
            num_labels = len([
                f for f in os.listdir(lbl_dir)
                if f.endswith(".txt")
            ])
        else:
            num_labels = 0

        stats["splits"][split] = {
            "images": num_images,
            "labels": num_labels,
        }

    print("\n=== Dataset Statistics ===")
    total_images = 0
    for split, info in stats["splits"].items():
        print(f"  {split}: {info['images']} images, {info['labels']} labels")
        total_images += info["images"]
    print(f"  Total: {total_images} images")
    print("=" * 28)

    return stats
