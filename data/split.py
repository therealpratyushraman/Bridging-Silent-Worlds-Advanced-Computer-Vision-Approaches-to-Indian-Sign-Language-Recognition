"""Dataset splitting utilities for ISL recognition.

Provides stratified train/val/test splitting that preserves class balance
across all partitions, and file-copy helpers for organising YOLO-format
directory layouts.
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


# ---------------------------------------------------------------------------
# Stratified splitting
# ---------------------------------------------------------------------------

def stratified_split(
    data_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Path]]:
    """Split images into train/val/test sets while maintaining class balance.

    The function supports two common layouts:

    1. **Class-folder layout** -- images organised in per-class
       subdirectories (``data_dir/<class>/image.png``).
    2. **YOLO label layout** -- a flat image directory with corresponding
       YOLO ``.txt`` label files whose first token is the class index.

    The function auto-detects which layout is present.

    Parameters
    ----------
    data_dir:
        Root directory containing images.  For class-folder layout this
        is the parent of the class subdirectories.  For YOLO layout this
        is the ``images/`` directory with a sibling ``labels/`` directory.
    train_ratio:
        Fraction of data for training.
    val_ratio:
        Fraction of data for validation.
    test_ratio:
        Fraction of data for testing.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{'train': [...], 'val': [...], 'test': [...]}`` mapping split
        names to lists of image ``Path`` objects.

    Raises
    ------
    ValueError
        If the ratios do not sum to approximately 1.0, or if no images
        are found.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got "
            f"{train_ratio} + {val_ratio} + {test_ratio} = "
            f"{train_ratio + val_ratio + test_ratio}"
        )

    data_dir = Path(data_dir)

    # Detect layout and group images by class
    class_to_images = _collect_by_class(data_dir)

    total_images = sum(len(v) for v in class_to_images.values())
    if total_images == 0:
        raise ValueError(f"No images found in {data_dir}")

    rng = random.Random(seed)

    splits: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}

    for cls_name in sorted(class_to_images.keys()):
        images = class_to_images[cls_name]
        rng.shuffle(images)

        n = len(images)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n > 2 else 0
        # Ensure we don't exceed total
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val

        splits["train"].extend(images[:n_train])
        splits["val"].extend(images[n_train : n_train + n_val])
        splits["test"].extend(images[n_train + n_val :])

    for split_name, file_list in splits.items():
        print(f"[split] {split_name}: {len(file_list)} images")

    return splits


def _collect_by_class(data_dir: Path) -> Dict[str, List[Path]]:
    """Group images by class, auto-detecting the directory layout."""
    class_to_images: Dict[str, List[Path]] = defaultdict(list)

    # Check for class-folder layout: subdirs containing images
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    has_class_folders = any(
        any(f.suffix.lower() in _IMAGE_EXTENSIONS for f in sd.iterdir() if f.is_file())
        for sd in subdirs
        if sd.name not in ("images", "labels")
    ) if subdirs else False

    if has_class_folders:
        for subdir in subdirs:
            if subdir.name in ("images", "labels"):
                continue
            cls_name = subdir.name
            for img_path in sorted(subdir.iterdir()):
                if img_path.suffix.lower() in _IMAGE_EXTENSIONS:
                    class_to_images[cls_name].append(img_path)
        return dict(class_to_images)

    # YOLO layout: flat image directory with sibling labels directory
    label_dir = data_dir.parent / "labels" / data_dir.name
    if not label_dir.is_dir():
        # Also check for labels at the same level
        label_dir = data_dir.parent / "labels"

    image_files = sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )

    for img_path in image_files:
        cls_name = _get_class_from_label(label_dir, img_path.stem)
        class_to_images[cls_name].append(img_path)

    return dict(class_to_images)


def _get_class_from_label(label_dir: Path, stem: str) -> str:
    """Read the class index from a YOLO label file."""
    label_path = label_dir / f"{stem}.txt"
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as fh:
            first_line = fh.readline().strip()
            if first_line:
                return first_line.split()[0]
    return "unknown"


# ---------------------------------------------------------------------------
# File copying
# ---------------------------------------------------------------------------

def copy_split_files(
    file_list: List[Path],
    src_dir: str | Path,
    dst_dir: str | Path,
    copy_labels: bool = True,
) -> None:
    """Copy image files (and optionally matching labels) to a destination.

    Parameters
    ----------
    file_list:
        List of image file paths to copy.
    src_dir:
        Source root directory (used to resolve relative label paths).
    dst_dir:
        Destination directory.  ``images/`` and ``labels/`` subdirectories
        are created automatically when *copy_labels* is True.
    copy_labels:
        Whether to look for and copy corresponding YOLO ``.txt`` label
        files.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    img_dst = dst_dir / "images"
    img_dst.mkdir(parents=True, exist_ok=True)

    lbl_dst: Optional[Path] = None
    if copy_labels:
        lbl_dst = dst_dir / "labels"
        lbl_dst.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(file_list, desc=f"Copying to {dst_dir.name}"):
        img_path = Path(img_path)

        # Copy image
        shutil.copy2(str(img_path), str(img_dst / img_path.name))

        # Copy matching label
        if lbl_dst is not None:
            # Try several common label locations
            for candidate_dir in [
                img_path.parent.parent / "labels" / img_path.parent.name,
                img_path.parent.parent / "labels",
                src_dir / "labels",
            ]:
                label_src = candidate_dir / f"{img_path.stem}.txt"
                if label_src.exists():
                    shutil.copy2(str(label_src), str(lbl_dst / label_src.name))
                    break

    print(f"[split] Copied {len(file_list)} files to {dst_dir}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split an ISL dataset into train/val/test partitions."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory containing images (class-folder or YOLO layout).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root output directory for the split dataset.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set fraction (default: 0.7).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set fraction (default: 0.15).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set fraction (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Skip copying label files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    splits = stratified_split(
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    output_root = Path(args.output_dir)
    for split_name, file_list in splits.items():
        copy_split_files(
            file_list=file_list,
            src_dir=args.data_dir,
            dst_dir=output_root / split_name,
            copy_labels=not args.no_labels,
        )

    print(f"\n[split] Done. Output written to {output_root}")
