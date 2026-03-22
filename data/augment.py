"""Data augmentation pipelines for ISL recognition datasets.

Uses albumentations for efficient, GPU-friendly image augmentations.
HorizontalFlip is intentionally excluded because mirroring would change
the meaning of hand-specific sign language gestures.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = 640) -> A.Compose:
    """Return the training augmentation pipeline.

    Includes brightness/contrast jitter, Gaussian noise, motion blur,
    shift-scale-rotate (up to 15 degrees), resize, and normalisation.
    HorizontalFlip is deliberately omitted to preserve gesture semantics.

    Parameters
    ----------
    image_size:
        Target square image dimension.

    Returns
    -------
    albumentations.Compose
        Composed augmentation pipeline.
    """
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        A.MotionBlur(blur_limit=7, p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            value=114,
            p=0.5,
        ),
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def get_val_transforms(image_size: int = 640) -> A.Compose:
    """Return the validation / test augmentation pipeline.

    Only resizes and normalises -- no stochastic augmentations.

    Parameters
    ----------
    image_size:
        Target square image dimension.

    Returns
    -------
    albumentations.Compose
        Composed transform pipeline.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


# ---------------------------------------------------------------------------
# Single-image augmentation
# ---------------------------------------------------------------------------

def augment_single_image(
    image: np.ndarray,
    transform: A.Compose,
) -> np.ndarray:
    """Apply an albumentations transform to a single image.

    Parameters
    ----------
    image:
        Input BGR image as a numpy array (uint8).
    transform:
        An ``albumentations.Compose`` pipeline.

    Returns
    -------
    np.ndarray
        Transformed image.  If the pipeline includes ``Normalize`` the
        result will be float32; otherwise it retains the input dtype.
    """
    result = transform(image=image)
    return result["image"]


# ---------------------------------------------------------------------------
# Batch dataset augmentation
# ---------------------------------------------------------------------------

def augment_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    multiplier: int = 3,
    image_size: int = 640,
    label_dir: Optional[str | Path] = None,
) -> None:
    """Expand a dataset by generating augmented copies of every image.

    For each source image, *multiplier* augmented variants are produced
    using the training transform pipeline.  If a matching YOLO label
    file exists in *label_dir* (or ``<input_dir>/../labels/<split>``),
    it is copied alongside each augmented image.

    Parameters
    ----------
    input_dir:
        Directory containing source images (PNG, JPG, JPEG, BMP).
    output_dir:
        Destination for augmented images.
    multiplier:
        Number of augmented copies per original image.
    image_size:
        Target image size passed to the transform pipeline.
    label_dir:
        Optional directory containing YOLO ``.txt`` labels.  When
        ``None`` the function attempts to infer it from the conventional
        YOLO layout (``<input_dir>/../../labels/<split>``).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Attempt to resolve label directories
    if label_dir is not None:
        label_dir = Path(label_dir)
    else:
        # Conventional YOLO layout: images/<split> -> labels/<split>
        candidate = input_dir.parent.parent / "labels" / input_dir.name
        if candidate.is_dir():
            label_dir = candidate

    label_out_dir: Optional[Path] = None
    if label_dir is not None:
        label_out_dir = output_dir.parent.parent / "labels" / output_dir.name
        label_out_dir.mkdir(parents=True, exist_ok=True)

    transform = get_train_transforms(image_size)

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in image_extensions
    )

    if not image_files:
        print(f"[augment] No images found in {input_dir}")
        return

    total_written = 0
    for img_path in tqdm(image_files, desc="Augmenting"):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[augment] WARNING: could not read {img_path}, skipping.")
            continue

        stem = img_path.stem
        suffix = img_path.suffix

        # Read label once if available
        label_content: Optional[str] = None
        if label_dir is not None:
            label_path = label_dir / f"{stem}.txt"
            if label_path.exists():
                label_content = label_path.read_text(encoding="utf-8")

        for aug_idx in range(multiplier):
            augmented = augment_single_image(image, transform)

            # Convert back to uint8 BGR for saving if normalised
            if augmented.dtype != np.uint8:
                # Reverse ImageNet normalisation for saving
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                restored = augmented * std + mean
                restored = np.clip(restored * 255.0, 0, 255).astype(np.uint8)
                # albumentations outputs RGB after Normalize; convert to BGR for cv2
                if restored.ndim == 3 and restored.shape[2] == 3:
                    restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
                save_img = restored
            else:
                save_img = augmented

            out_name = f"{stem}_aug{aug_idx:02d}{suffix}"
            cv2.imwrite(str(output_dir / out_name), save_img)

            # Copy label file for the augmented image
            if label_content is not None and label_out_dir is not None:
                lbl_out = label_out_dir / f"{stem}_aug{aug_idx:02d}.txt"
                lbl_out.write_text(label_content, encoding="utf-8")

            total_written += 1

    print(
        f"[augment] Generated {total_written} augmented images "
        f"({len(image_files)} originals x {multiplier}) in {output_dir}"
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Augment an image dataset for ISL recognition training."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for augmented output images.",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=3,
        help="Number of augmented copies per image (default: 3).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Target image size (default: 640).",
    )
    parser.add_argument(
        "--label-dir",
        type=str,
        default=None,
        help="Optional YOLO labels directory to copy alongside images.",
    )

    args = parser.parse_args()
    augment_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        multiplier=args.multiplier,
        image_size=args.image_size,
        label_dir=args.label_dir,
    )
