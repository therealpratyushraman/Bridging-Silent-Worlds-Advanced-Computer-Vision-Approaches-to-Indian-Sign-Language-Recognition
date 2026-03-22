"""Visualization of training metrics and evaluation results."""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(
    log_path: str,
    output_path: str = "runs/plots/training_curves.png",
    metrics: List[str] = None,
):
    """Plot training loss and metric curves from a training log.

    Args:
        log_path: Path to training_log.json file.
        output_path: Path to save the plot.
        metrics: List of metric names to plot. If None, plots all.
    """
    with open(log_path, "r") as f:
        history = json.load(f)

    if not history:
        print("No training history found.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    epochs = [e["epoch"] for e in history]

    if metrics is None:
        metrics = [k for k in history[0].keys() if k != "epoch"]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        values = [e.get(metric, None) for e in history]
        valid = [(ep, v) for ep, v in zip(epochs, values) if v is not None]

        if valid:
            ep_vals, met_vals = zip(*valid)
            ax.plot(ep_vals, met_vals, "b-", linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric)
            ax.set_title(metric.replace("_", " ").title())
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to: {output_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str = "runs/plots/confusion_matrix.png",
    normalize: bool = True,
):
    """Plot a confusion matrix heatmap.

    Args:
        cm: Confusion matrix as numpy array.
        class_names: List of class names.
        output_path: Path to save the plot.
        normalize: Whether to normalize the matrix.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm.astype(float) / row_sums
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def plot_precision_recall_curve(
    precisions: List[float],
    recalls: List[float],
    class_names: List[str] = None,
    output_path: str = "runs/plots/precision_recall.png",
):
    """Plot precision-recall curves.

    Args:
        precisions: Per-class precision values.
        recalls: Per-class recall values.
        class_names: Class names for labels.
        output_path: Path to save the plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    if class_names:
        for i, name in enumerate(class_names):
            if i < len(precisions) and i < len(recalls):
                ax.scatter(recalls[i], precisions[i], s=100, label=name, zorder=5)
    else:
        ax.scatter(recalls, precisions, s=100, zorder=5)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall per Class")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    if class_names:
        ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Precision-recall plot saved to: {output_path}")


def plot_model_comparison(
    comparison: Dict,
    output_path: str = "runs/plots/model_comparison.png",
):
    """Plot comparison bar chart between YOLOv5 and YOLOv8.

    Args:
        comparison: Dictionary with model names as keys and metric dicts as values.
        output_path: Path to save the plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    metrics = ["mAP50", "mAP50_95", "precision", "recall"]
    models = list(comparison.keys())

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):
        values = [comparison[model].get(m, 0) for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model)
        ax.bar_label(bars, fmt="%.3f", fontsize=8)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("YOLOv5 vs YOLOv8 Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Model comparison plot saved to: {output_path}")
