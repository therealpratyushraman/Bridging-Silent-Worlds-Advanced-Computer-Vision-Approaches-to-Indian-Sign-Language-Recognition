"""Dataset download and directory setup utilities.

Supports:
- Sign Language MNIST from Kaggle
- Custom 2000-image ISL gesture dataset
- ISL CSLTR continuous signing video dataset
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SIGN_LANGUAGE_MNIST_SLUG = "datamunge/sign-language-mnist"

_ISL_GESTURE_CLASSES = ["Hello", "Help", "Home", "No", "Please", "Yes"]

_MNIST_CLASSES = [
    chr(ord("A") + i) for i in range(26) if chr(ord("A") + i) not in ("J", "Z")
]


# ---------------------------------------------------------------------------
# Sign Language MNIST
# ---------------------------------------------------------------------------

def download_sign_language_mnist(output_dir: str | Path) -> Path:
    """Download Sign Language MNIST from Kaggle and create YOLO directory layout.

    Parameters
    ----------
    output_dir:
        Root directory where the dataset will be stored.  A subfolder
        ``sign_language_mnist/`` is created beneath it.

    Returns
    -------
    Path
        The path to the created dataset directory.

    Notes
    -----
    Requires the ``kaggle`` CLI to be installed and configured with valid
    API credentials (``~/.kaggle/kaggle.json``).
    """
    output_dir = Path(output_dir)
    dataset_dir = output_dir / "sign_language_mnist"

    # Skip download if CSVs already present
    if (dataset_dir / "sign_mnist_train.csv").exists() and (
        dataset_dir / "sign_mnist_test.csv"
    ).exists():
        print(f"[download] Sign Language MNIST already exists at {dataset_dir}")
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] Downloading Sign Language MNIST to {dataset_dir} ...")
    try:
        subprocess.check_call(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                _SIGN_LANGUAGE_MNIST_SLUG,
                "-p",
                str(dataset_dir),
                "--unzip",
            ]
        )
    except FileNotFoundError:
        print(
            "[download] ERROR: 'kaggle' CLI not found. "
            "Install it with: pip install kaggle\n"
            "Then place your API token at ~/.kaggle/kaggle.json",
            file=sys.stderr,
        )
        raise
    except subprocess.CalledProcessError as exc:
        print(
            f"[download] ERROR: kaggle download failed (exit {exc.returncode}). "
            "Check your Kaggle credentials and network connection.",
            file=sys.stderr,
        )
        raise

    # Create YOLO-format directory structure
    for split in ("train", "val", "test"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    print(f"[download] YOLO directory structure created under {dataset_dir}")
    return dataset_dir


# ---------------------------------------------------------------------------
# Custom ISL dataset (2000 images, 6 classes)
# ---------------------------------------------------------------------------

def setup_custom_dataset(base_dir: str | Path) -> Path:
    """Create the expected directory layout for the custom 2000-image ISL dataset.

    Parameters
    ----------
    base_dir:
        Root directory for the custom dataset.

    Returns
    -------
    Path
        The path to the dataset root.
    """
    base_dir = Path(base_dir)
    dataset_dir = base_dir / "isl_custom"

    for split in ("train", "val", "test"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Create per-class placeholder directories for raw collection
    for cls in _ISL_GESTURE_CLASSES:
        (dataset_dir / "raw" / cls).mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Custom ISL Dataset — Directory Structure Created")
    print("=" * 65)
    print(f"  Root : {dataset_dir}")
    print()
    print("  Gesture classes:")
    for i, cls in enumerate(_ISL_GESTURE_CLASSES):
        print(f"    {i}: {cls}")
    print()
    print("  Place raw images into:")
    for cls in _ISL_GESTURE_CLASSES:
        print(f"    {dataset_dir / 'raw' / cls}/")
    print()
    print("  After collecting images, run data/preprocess.py to convert them")
    print("  to YOLO format, then data/split.py to create train/val/test splits.")
    print("=" * 65)
    return dataset_dir


# ---------------------------------------------------------------------------
# ISL CSLTR (video-based continuous signing)
# ---------------------------------------------------------------------------

def setup_isl_csltr(base_dir: str | Path) -> Path:
    """Create the directory structure for ISL CSLTR video frame extraction.

    Parameters
    ----------
    base_dir:
        Root directory for the CSLTR dataset.

    Returns
    -------
    Path
        The path to the dataset root.
    """
    base_dir = Path(base_dir)
    dataset_dir = base_dir / "isl_csltr"

    (dataset_dir / "videos").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "annotations").mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        (dataset_dir / "frames" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ISL CSLTR Dataset — Directory Structure Created")
    print("=" * 65)
    print(f"  Root : {dataset_dir}")
    print()
    print("  Place raw videos into:")
    print(f"    {dataset_dir / 'videos'}/")
    print()
    print("  Place annotation files (CSV/JSON) into:")
    print(f"    {dataset_dir / 'annotations'}/")
    print()
    print("  Run data/preprocess.py --extract-frames to extract video frames.")
    print("=" * 65)
    return dataset_dir


# ---------------------------------------------------------------------------
# YOLO dataset.yaml generation
# ---------------------------------------------------------------------------

def generate_dataset_yaml(
    data_dir: str | Path,
    classes: List[str],
    output_path: Optional[str | Path] = None,
) -> Path:
    """Generate a ``dataset.yaml`` file for Ultralytics YOLO training.

    Parameters
    ----------
    data_dir:
        Root of the YOLO-format dataset (must contain ``images/train``, etc.).
    classes:
        Ordered list of class names.
    output_path:
        Where to write the YAML file.  Defaults to ``<data_dir>/dataset.yaml``.

    Returns
    -------
    Path
        The path to the generated YAML file.
    """
    data_dir = Path(data_dir).resolve()
    if output_path is None:
        output_path = data_dir / "dataset.yaml"
    else:
        output_path = Path(output_path)

    dataset_config = {
        "path": str(data_dir),
        "train": str(data_dir / "images" / "train"),
        "val": str(data_dir / "images" / "val"),
        "test": str(data_dir / "images" / "test"),
        "nc": len(classes),
        "names": classes,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        yaml.dump(dataset_config, fh, default_flow_style=False, sort_keys=False)

    print(f"[download] dataset.yaml written to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and set up ISL recognition datasets."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "raw"),
        help="Root directory for dataset storage (default: data/raw/).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "custom", "csltr", "all"],
        default="all",
        help="Which dataset to set up.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output = Path(args.output_dir)

    if args.dataset in ("mnist", "all"):
        mnist_dir = download_sign_language_mnist(output)
        generate_dataset_yaml(mnist_dir, _MNIST_CLASSES)

    if args.dataset in ("custom", "all"):
        custom_dir = setup_custom_dataset(output)
        generate_dataset_yaml(custom_dir, _ISL_GESTURE_CLASSES)

    if args.dataset in ("csltr", "all"):
        setup_isl_csltr(output)

    print("\n[download] Done.")
